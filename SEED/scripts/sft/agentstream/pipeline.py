#!/usr/bin/env python3
"""Generate episode-skill SFT data from AgentStream policy rollouts.

Stage-1 (hindsight-skill SFT) pipeline for the AgentStream benchmark suite,
mirroring scripts/sft/alfworld/pipeline.py:

  1. sample tasks   — the same seed-42 stream selection used by RL training
                      (agent_system/environments/env_package/agentstream),
                      so SFT data never leaks into the RL holdout;
  2. rollouts       — drive exgentic sessions with an OpenAI-compatible policy
                      endpoint (no Ray / GPU in this process);
  3. skill gen      — one episode-level skill per trajectory via the SEED
                      analyzer prompt (reuses _common.build_candidate_skill_record);
  4. export         — parse-ok candidates -> train/val parquet for
                      scripts/sft/_common/trainer.sh.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.environments.env_package.agentstream.exgentic_client import (  # noqa: E402
    BenchmarkHub,
    SessionDriver,
)
from agent_system.environments.env_package.agentstream.projection import (  # noqa: E402
    agentstream_projection_detailed,
)
from agent_system.environments.env_package.agentstream.prompts import render_prompt  # noqa: E402
from agent_system.environments.env_package.agentstream.task_stream import select_tasks  # noqa: E402
from scripts.sft._common.pipeline import (  # noqa: E402
    OpenAITextClient,
    append_jsonl,
    build_candidate_skill_record,
    load_env_file,
    log_stage,
    read_jsonl,
    resolve_endpoint,
    setup_logging,
    write_json,
)

OUTPUT_FILES = [
    "sampled_tasks.jsonl",
    "baseline_rollouts.jsonl",
    "candidate_skills.jsonl",
    "sft_episode_skill_all.jsonl",
    "sft_episode_skill_train.parquet",
    "sft_episode_skill_val.parquet",
    "metrics.json",
    "run_config.json",
    "inspection/rollouts.json",
    "inspection/skills.json",
]

_SECRET_RE = re.compile(
    r"(?i)(api[-_ ]?key|authorization|bearer|token|password|secret)(\s*[:=]\s*|\s+)([^\s,;]+)"
)
_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{256,}={0,2}(?![A-Za-z0-9+/=])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="outputs/agentstream_episode_skill_pipeline")
    parser.add_argument("--exgentic-root", required=True)
    parser.add_argument("--benchmarks", default="bfcl", help="comma list, e.g. bfcl,tau2,appworld")
    parser.add_argument("--benchmark-kwargs-json", default="{}")
    parser.add_argument("--runner", default="venv")
    # Same selection knob as RL (env.agentstream.num_tasks_per_benchmark).
    parser.add_argument("--num-tasks-per-benchmark", type=int, default=50)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument("--parallel-sessions", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--no-require-think", action="store_true")
    parser.add_argument("--accept-tool-call", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--skill-gen-workers", type=int, default=64)
    parser.add_argument("--sft-val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--inspection-samples", type=int, default=3)
    parser.add_argument("--inspection-max-chars", type=int, default=4000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-after-baseline-rollouts", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    for prefix, temp, tokens, retries in (("policy", 0.4, 1024, 2), ("skill", 0.0, 1024, 5)):
        parser.add_argument(f"--{prefix}-base-url", default=None)
        parser.add_argument(f"--{prefix}-api-key", default=None)
        parser.add_argument(f"--{prefix}-model", default=None)
        parser.add_argument(f"--{prefix}-temperature", type=float, default=temp)
        parser.add_argument(f"--{prefix}-max-completion-tokens", type=int, default=tokens)
        parser.add_argument(f"--{prefix}-timeout", type=float, default=120.0)
        parser.add_argument(f"--{prefix}-retries", type=int, default=retries)
        parser.add_argument(f"--{prefix}-retry-delay", type=float, default=1.0)
        parser.add_argument(f"--{prefix}-extra-body-json", default=None)
    return parser.parse_args()


def _inspection_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    text = _SECRET_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", text)
    text = _BASE64_RE.sub("[REDACTED_BASE64]", text)
    if len(text) > max_chars:
        text = f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"
    return text


def _select_inspection_rows(rows: Sequence[Dict[str, Any]], sample_count: int) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("task_type", "")),
            str(item.get("task_id", item.get("skill_id", ""))),
            int(item.get("rollout_id", item.get("source_rollout_id", 0))),
        ),
    ):
        by_type.setdefault(str(row.get("task_type", "")), []).append(row)

    selected = []
    while len(selected) < sample_count and any(by_type.values()):
        for task_type in sorted(by_type):
            if by_type[task_type]:
                selected.append(by_type[task_type].pop(0))
                if len(selected) == sample_count:
                    break
    return selected


def write_inspection_samples(
    output_dir: Path,
    rollouts: Sequence[Dict[str, Any]],
    skills: Sequence[Dict[str, Any]],
    *,
    sample_count: int,
    max_chars: int,
) -> None:
    if sample_count <= 0:
        return

    inspection_dir = output_dir / "inspection"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    selected_rollouts = _select_inspection_rows(rollouts, sample_count)
    rollout_samples = []
    for rollout in selected_rollouts:
        rollout_samples.append(
            {
                "task_id": rollout.get("task_id"),
                "rollout_id": rollout.get("rollout_id"),
                "task_type": rollout.get("task_type"),
                "success": bool(rollout.get("success", False)),
                "score": float(rollout.get("score", 0.0)),
                "num_steps": int(rollout.get("num_steps", 0)),
                "task_description": _inspection_text(rollout.get("task_description"), max_chars),
                "steps": [
                    {
                        "step_idx": step.get("step_idx"),
                        "observation": _inspection_text(step.get("observation"), max_chars),
                        "model_response": _inspection_text(step.get("model_response"), max_chars),
                        "is_action_valid": bool(step.get("info", {}).get("is_action_valid", False)),
                        "invalid_reason": str(step.get("info", {}).get("invalid_reason", "")),
                        "policy_api_error": bool(step.get("info", {}).get("policy_api_error")),
                    }
                    for step in rollout.get("steps", [])
                ],
            }
        )

    selected_skills = _select_inspection_rows(skills, sample_count)
    skill_samples = [
        {
            "skill_id": skill.get("skill_id"),
            "task_type": skill.get("task_type"),
            "source_success": bool(skill.get("source_success", False)),
            "source_num_steps": int(skill.get("source_num_steps", 0)),
            "parse_ok": bool(skill.get("parse_ok", False)),
            "episode_summary": _inspection_text(skill.get("episode_summary"), max_chars),
            "episode_skill": _inspection_text(skill.get("episode_skill"), max_chars),
            "analysis_error": _inspection_text(skill.get("analysis_error"), max_chars),
        }
        for skill in selected_skills
    ]
    write_json(inspection_dir / "rollouts.json", rollout_samples)
    write_json(inspection_dir / "skills.json", skill_samples)


def prepare_output_dir(output_dir: Path, *, resume: bool, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / n for n in OUTPUT_FILES if (output_dir / n).exists()]
    if overwrite:
        for path in existing:
            path.unlink()
    elif existing and not resume:
        names = ", ".join(p.name for p in existing)
        raise FileExistsError(
            f"Output files already exist in {output_dir}: {names}. Use --resume or --overwrite."
        )


# ---------------------------------------------------------------------------
# Stage 1: task sampling (identical selection to RL training)
# ---------------------------------------------------------------------------

def sample_tasks(args: argparse.Namespace, output_dir: Path, hub: BenchmarkHub) -> List[Dict[str, Any]]:
    path = output_dir / "sampled_tasks.jsonl"
    if args.resume and path.exists():
        return read_jsonl(path)
    selected = select_tasks(hub.list_all_tasks(), args.num_tasks_per_benchmark)
    tasks = [{"slug": slug, "task_id": tid} for slug in sorted(selected) for tid in selected[slug]]
    for task in tasks:
        append_jsonl(path, task)
    return tasks


# ---------------------------------------------------------------------------
# Stage 2: baseline rollouts
# ---------------------------------------------------------------------------

_TLS = threading.local()


def _thread_driver(args: argparse.Namespace, drivers: List[SessionDriver], lock: threading.Lock) -> SessionDriver:
    driver = getattr(_TLS, "driver", None)
    if driver is None:
        driver = SessionDriver(
            exgentic_root=args.exgentic_root,
            runner=args.runner,
            output_dir=str(Path(args.output_dir) / "exgentic_sessions"),
            run_id=f"sft_t{threading.get_ident()}",
            max_steps=args.max_steps,
        )
        with lock:
            drivers.append(driver)
        _TLS.driver = driver
    return driver


def _history_text(history: List[Dict[str, str]], window: int) -> str:
    recent = history[-window:]
    start = len(history) - len(recent)
    return "\n".join(
        f"[Observation {start + j + 1}: '{h['obs']}', Action {start + j + 1}: '{h['act']}']"
        for j, h in enumerate(recent)
    )


def run_one_rollout(
    spec: Dict[str, Any],
    args: argparse.Namespace,
    policy_client: OpenAITextClient,
    bm_kwargs: Dict[str, Any],
    drivers: List[SessionDriver],
    dlock: threading.Lock,
) -> Dict[str, Any]:
    slug, task_id, rollout_id = spec["slug"], spec["task_id"], spec["rollout_id"]
    driver = _thread_driver(args, drivers, dlock)
    payload = driver.reset(slug, task_id, bm_kwargs, spec["session_kwargs"])

    steps: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []
    obs = payload["observation"]
    success, score = False, 0.0

    if not payload.get("reset_error"):
        for step_idx in range(args.max_steps):
            prompt = render_prompt(
                slug=slug,
                task=payload["task"],
                context=payload["context"],
                actions_text=payload["actions_text"],
                observation=obs,
                step_count=len(history),
                history_text=_history_text(history, args.history_length),
                history_len=min(len(history), args.history_length),
            )
            response, api_error = policy_client.complete([{"role": "user", "content": prompt}])
            response = response or ""
            payloads, valids, extras = agentstream_projection_detailed(
                [response],
                require_think=not args.no_require_think,
                accept_tool_call=args.accept_tool_call,
            )
            obs_next, done, info = driver.step(payloads[0])
            is_valid = bool(valids[0]) and not info.get("action_error", False)
            reason = extras[0]["reason"]
            if not reason and not is_valid:
                reason = "env_action_error"
            steps.append(
                {
                    "step_idx": step_idx,
                    "observation": obs,
                    "observation_prompt": prompt,
                    "model_response": response,
                    "info": {
                        "is_action_valid": is_valid,
                        "invalid_reason": reason,
                        "think_present": extras[0]["think_present"],
                        "used_tool_call_alias": extras[0]["used_tool_call_alias"],
                        "policy_api_error": api_error,
                    },
                }
            )
            history.append(
                {
                    "obs": obs,
                    "act": json.dumps(payloads[0], ensure_ascii=False)
                    if valids[0]
                    else "[invalid action format]",
                }
            )
            obs = obs_next or "[episode finished]"
            if done:
                success = bool(info.get("won", False))
                score = float(info.get("score", 0.0))
                break

    return {
        # Slug-qualified task_id keeps skill_ids unique across benchmarks.
        "task_id": f"{slug}/{task_id}",
        "benchmark_task_id": str(task_id),
        "rollout_id": rollout_id,
        "task_type": slug,
        "game_file": f"{slug}:{task_id}",
        "task_description": payload["task"],
        "success": success,
        "score": score,
        "num_steps": len(steps),
        "reset_error": bool(payload.get("reset_error", False)),
        "steps": steps,
    }


def collect_rollouts(
    tasks: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    hub: BenchmarkHub,
    policy_endpoint,
    bm_kwargs_by_slug: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    path = output_dir / "baseline_rollouts.jsonl"
    records = read_jsonl(path) if args.resume and path.exists() else []
    done_keys = {f"{r['task_id']}:{r['rollout_id']}" for r in records}

    specs = []
    for task in tasks:
        for rollout_id in range(args.rollouts_per_task):
            if f"{task['slug']}/{task['task_id']}:{rollout_id}" in done_keys:
                continue
            # Fresh session kwargs per rollout (main thread; evaluator proxies
            # are not assumed thread-safe).
            specs.append(
                {
                    **task,
                    "rollout_id": rollout_id,
                    "session_kwargs": hub.session_kwargs(task["slug"], task["task_id"]),
                }
            )

    log_stage(output_dir, "baseline_rollouts", "running", pending=len(specs), existing=len(records))
    if not specs:
        return records

    policy_client = OpenAITextClient(policy_endpoint)
    drivers: List[SessionDriver] = []
    dlock = threading.Lock()
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel_sessions)) as executor:
            futures = {
                executor.submit(
                    run_one_rollout,
                    spec,
                    args,
                    policy_client,
                    bm_kwargs_by_slug.get(spec["slug"], {}),
                    drivers,
                    dlock,
                ): spec
                for spec in specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                completed += 1
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "task_id": f"{spec['slug']}/{spec['task_id']}",
                        "benchmark_task_id": str(spec["task_id"]),
                        "rollout_id": spec["rollout_id"],
                        "task_type": spec["slug"],
                        "game_file": f"{spec['slug']}:{spec['task_id']}",
                        "task_description": "",
                        "success": False,
                        "score": 0.0,
                        "num_steps": 0,
                        "reset_error": True,
                        "rollout_error": f"{type(exc).__name__}: {exc}",
                        "steps": [],
                    }
                append_jsonl(path, record)
                records.append(record)
                logging.info(
                    "Rollout %d/%d: %s success=%s steps=%d",
                    completed,
                    len(specs),
                    f"{record['task_id']}:{record['rollout_id']}",
                    record["success"],
                    record["num_steps"],
                )
    finally:
        for driver in drivers:
            driver.close()

    log_stage(
        output_dir,
        "baseline_rollouts",
        "complete",
        baseline_rollouts=len(records),
        successes=sum(1 for r in records if r.get("success")),
    )
    return records


_INVALID_REASONS = ("no_action_tag", "bad_action_json", "missing_think", "env_action_error")


def summarize_action_validity(
    rollouts: Sequence[Dict[str, Any]],
    *,
    require_think: bool,
    accept_tool_call: bool,
) -> Dict[str, Any]:
    """Per-step validity breakdown, re-classified from stored model responses.

    Re-parsing (instead of trusting per-step info) also covers resumed rollouts
    recorded before the detailed fields existed. A step that parses fine but was
    recorded invalid means the env rejected it (env_action_error).
    """
    overall: Counter = Counter()
    by_type: Dict[str, Counter] = {}
    for rollout in rollouts:
        counter = by_type.setdefault(str(rollout.get("task_type", "")), Counter())
        for step in rollout.get("steps", []):
            _, valids, extras = agentstream_projection_detailed(
                [str(step.get("model_response", ""))],
                require_think=require_think,
                accept_tool_call=accept_tool_call,
            )
            stored_valid = bool(step.get("info", {}).get("is_action_valid", False))
            for c in (counter, overall):
                c["steps"] += 1
                c["think_present"] += int(extras[0]["think_present"])
                c["tool_call_alias"] += int(extras[0]["used_tool_call_alias"])
                if stored_valid:
                    c["valid"] += 1
                else:
                    c[f"invalid_{extras[0]['reason'] or 'env_action_error'}"] += 1

    def _ratios(counter: Counter) -> Dict[str, Any]:
        steps = counter["steps"]
        out: Dict[str, Any] = {"steps": steps}
        if steps:
            out["valid_action_ratio"] = counter["valid"] / steps
            out["think_present_ratio"] = counter["think_present"] / steps
            out["tool_call_alias_ratio"] = counter["tool_call_alias"] / steps
            for reason in _INVALID_REASONS:
                out[f"invalid_{reason}"] = counter[f"invalid_{reason}"] / steps
        return out

    return {
        "require_think": require_think,
        "accept_tool_call": accept_tool_call,
        "overall": _ratios(overall),
        "by_benchmark": {slug: _ratios(c) for slug, c in sorted(by_type.items())},
    }


# ---------------------------------------------------------------------------
# Stage 3: candidate skill generation (reuses the shared SEED analyzer path)
# ---------------------------------------------------------------------------

def generate_skills(
    rollouts: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    skill_endpoint,
) -> List[Dict[str, Any]]:
    path = output_dir / "candidate_skills.jsonl"
    records = read_jsonl(path) if args.resume and path.exists() else []
    done_ids = {r["skill_id"] for r in records}

    # Skip rollouts with no usable trajectory.
    pending = [
        t
        for t in rollouts
        if t.get("num_steps", 0) > 0 and f"{t['task_id']}:{t['rollout_id']}" not in done_ids
    ]
    if args.max_candidates is not None:
        pending = pending[: max(0, args.max_candidates)]

    log_stage(output_dir, "skill_generation", "running", pending=len(pending), existing=len(records))
    if not pending:
        return records

    with ThreadPoolExecutor(max_workers=max(1, min(args.skill_gen_workers, len(pending)))) as executor:
        futures = {
            executor.submit(
                build_candidate_skill_record, trajectory=t, skill_endpoint=skill_endpoint
            ): t
            for t in pending
        }
        for i, future in enumerate(as_completed(futures), 1):
            trajectory = futures[future]
            skill_id = f"{trajectory['task_id']}:{trajectory['rollout_id']}"
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "skill_id": skill_id,
                    "task_id": trajectory["task_id"],
                    "task_type": trajectory["task_type"],
                    "game_file": trajectory["game_file"],
                    "source_rollout_id": trajectory["rollout_id"],
                    "source_success": bool(trajectory.get("success", False)),
                    "source_num_steps": int(trajectory.get("num_steps", 0)),
                    "task_description": trajectory.get("task_description", ""),
                    "analysis_prompt": None,
                    "llm_raw_output": "",
                    "episode_summary": "",
                    "episode_skill": "",
                    "parse_ok": False,
                    "analysis_error": f"{type(exc).__name__}: {exc}",
                }
            append_jsonl(path, record)
            records.append(record)
            logging.info("Skill %d/%d: %s parse_ok=%s", i, len(pending), skill_id, record.get("parse_ok"))

    log_stage(
        output_dir,
        "skill_generation",
        "complete",
        candidate_skills=len(records),
        parse_ok_skills=sum(1 for r in records if r.get("parse_ok")),
    )
    return records


# ---------------------------------------------------------------------------
# Stage 4: SFT export (parse-ok candidates, alfworld-compatible schema)
# ---------------------------------------------------------------------------

def export_sft(
    candidates: Sequence[Dict[str, Any]],
    output_dir: Path,
    val_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    sft_records = []
    for candidate in candidates:
        if not candidate.get("parse_ok"):
            continue
        prompt = candidate.get("analysis_prompt") or {}
        messages = prompt.get("messages", []) if isinstance(prompt, dict) else []
        if not messages:
            continue
        sft_records.append(
            {
                "prompt": str(messages[-1].get("content", "")),
                "response": json.dumps(
                    {
                        "episode_summary": candidate.get("episode_summary", ""),
                        "episode_skill": candidate.get("episode_skill", ""),
                    },
                    ensure_ascii=False,
                ),
                "skill_id": candidate["skill_id"],
                "task_id": candidate["task_id"],
                "task_type": candidate["task_type"],
                "source_success": bool(candidate.get("source_success", False)),
                "source_num_steps": int(candidate.get("source_num_steps", 0)),
            }
        )

    all_jsonl = output_dir / "sft_episode_skill_all.jsonl"
    if all_jsonl.exists():
        all_jsonl.unlink()
    for record in sft_records:
        append_jsonl(all_jsonl, record)

    if not sft_records:
        logging.warning("No parseable candidate skills; SFT parquet export skipped.")
        return sft_records

    shuffled = list(sft_records)
    random.Random(seed).shuffle(shuffled)
    val_size = int(round(len(shuffled) * val_ratio))
    if len(shuffled) > 1:
        val_size = max(1, min(val_size, len(shuffled) - 1))

    import pandas as pd

    pd.DataFrame(shuffled[val_size:]).to_parquet(output_dir / "sft_episode_skill_train.parquet")
    pd.DataFrame(shuffled[:val_size]).to_parquet(output_dir / "sft_episode_skill_val.parquet")
    return sft_records


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, resume=args.resume, overwrite=args.overwrite)
    setup_logging(output_dir, args.log_level)

    policy_endpoint = resolve_endpoint(
        prefix="policy",
        args=args,
        default_base_url_env="POLICY_OPENAI_BASE_URL",
        default_model_env="POLICY_OPENAI_MODEL",
        default_model="Qwen2.5-3B-Instruct",
        temperature=args.policy_temperature,
        max_completion_tokens=args.policy_max_completion_tokens,
        timeout=args.policy_timeout,
        retries=args.policy_retries,
        retry_delay=args.policy_retry_delay,
        extra_body_json=args.policy_extra_body_json,
    )
    skill_endpoint = resolve_endpoint(
        prefix="skill",
        args=args,
        default_base_url_env="SKILL_OPENAI_BASE_URL",
        default_model_env="SKILL_OPENAI_MODEL",
        default_model="Qwen2.5-3B-Instruct",
        temperature=args.skill_temperature,
        max_completion_tokens=args.skill_max_completion_tokens,
        timeout=args.skill_timeout,
        retries=args.skill_retries,
        retry_delay=args.skill_retry_delay,
        extra_body_json=args.skill_extra_body_json,
    )

    slugs = sorted(s.strip() for s in args.benchmarks.split(",") if s.strip())
    bm_kwargs_by_slug = {str(k): dict(v or {}) for k, v in json.loads(args.benchmark_kwargs_json).items()}
    hub = BenchmarkHub(
        exgentic_root=args.exgentic_root,
        slugs=slugs,
        benchmark_kwargs=bm_kwargs_by_slug,
        runner=args.runner,
        output_dir=str(output_dir / "exgentic_sessions"),
        run_id="sft_hub",
    )

    try:
        tasks = sample_tasks(args, output_dir, hub)
        logging.info("Sampled %d tasks across %s.", len(tasks), slugs)
        rollouts = collect_rollouts(tasks, args, output_dir, hub, policy_endpoint, bm_kwargs_by_slug)
    finally:
        hub.close()

    write_inspection_samples(
        output_dir,
        rollouts,
        [],
        sample_count=args.inspection_samples,
        max_chars=args.inspection_max_chars,
    )

    if args.stop_after_baseline_rollouts:
        logging.info("Baseline rollouts complete; stopping before skill generation.")
        return

    candidates = generate_skills(rollouts, args, output_dir, skill_endpoint)
    write_inspection_samples(
        output_dir,
        rollouts,
        candidates,
        sample_count=args.inspection_samples,
        max_chars=args.inspection_max_chars,
    )
    sft_records = export_sft(candidates, output_dir, args.sft_val_ratio, args.seed)

    write_json(
        output_dir / "metrics.json",
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "benchmarks": slugs,
            "sampled_tasks": len(tasks),
            "baseline_rollouts": len(rollouts),
            "rollout_successes": sum(1 for r in rollouts if r.get("success")),
            "candidate_skills": len(candidates),
            "parse_ok_skills": sum(1 for r in candidates if r.get("parse_ok")),
            "sft_records": len(sft_records),
            "rollouts_by_benchmark": dict(Counter(r["task_type"] for r in rollouts)),
            "success_by_benchmark": dict(
                Counter(r["task_type"] for r in rollouts if r.get("success"))
            ),
            "action_validity": summarize_action_validity(
                rollouts,
                require_think=not args.no_require_think,
                accept_tool_call=args.accept_tool_call,
            ),
        },
    )
    logging.info("Pipeline complete. Outputs in %s.", output_dir)


if __name__ == "__main__":
    main()
