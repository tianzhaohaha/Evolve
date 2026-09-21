#!/usr/bin/env python3
"""Minimal offline delta test: does an in-context demonstration raise the frozen SFT policy's success rate?

Two conditions on the same tasks, same frozen checkpoint, same sampling settings as Stage-3 RL:
  C0  plain prompt;
  C2  on tasks whose C0 rollouts were mixed (some successes, some failures), the shortest successful
      C0 rollout is injected as a "Reference Solution" (reasoning + action per step) through
      seed.prompting.build_augmented_observation_text -- the very section the trainer's teacher
      prompt uses.
delta = success(C2) - success(C0) per task, paired over tasks (mean, standard error, by benchmark).

Building blocks are the Stage-1 SFT pipeline (scripts/sft/agentstream/pipeline.py: task selection
from the RL stream's seed-42 shuffle minus the holdout, the OpenAI-compatible policy client, the
threaded resumable rollout collector) plus a per-step ``prompt_hook`` that pipeline exposes; nothing
on the training path is touched. Sampling mirrors RL: temperature 1.0, 512 response tokens, history
window 3, the RL max-steps / observation caps, and the RL projection flags (think optional,
tool_call accepted).

Needs a running OpenAI-compatible endpoint for the SFT checkpoint, e.g. (seed conda env):
  vllm serve "$AGENTSTREAM_SFT_MODEL_DIR" --port 60001 --served-model-name sft --max-model-len 40960
and the exgentic checkout (AGENTSTREAM_EXGENTIC_ROOT). The wrapper examples/agentstream_trainer/
run_delta_test.sh (one PBS line) sources the public config, starts the endpoint and passes the RL
settings; direct use: source agentstream_full.env (set -a) then
  python examples/agentstream_trainer/delta_test.py --phase all --output-dir outputs/delta_test_v5
(--policy-base-url / --policy-model default to that local endpoint and served name.)
Phases (each resumes from its outputs): c0 -> materials -> c2 -> report.
Outputs under --output-dir: sampled_tasks.jsonl, C0/baseline_rollouts.jsonl, materials.jsonl,
C2/baseline_rollouts.jsonl, delta_report.md, delta_per_task.csv, run_config.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed.prompting import build_augmented_observation_text  # noqa: E402
from seed.sibling import build_reference_solution  # noqa: E402

PHASES = ("c0", "materials", "c2", "report")
PromptHook = Callable[[str, Dict[str, Any], int], str]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested, no environment / network)
# ---------------------------------------------------------------------------

def record_to_steps(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pipeline rollout record -> the step dicts seed.sibling renders (index, observation, response, validity)."""
    steps = []
    for step in record.get("steps", []):
        info = step.get("info") or {}
        steps.append(
            {
                "step_index": int(step.get("step_idx", len(steps))),
                "observation": step.get("observation", ""),
                "response": step.get("model_response", ""),
                "action_valid": bool(info.get("is_action_valid", True)),
            }
        )
    return steps


def usable(record: Dict[str, Any]) -> bool:
    return not record.get("reset_error") and not record.get("rollout_error") and int(record.get("num_steps", 0)) > 0


def select_reference(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Shortest successful rollout of one task (ties: higher score, then lower rollout_id)."""
    successes = [r for r in records if usable(r) and r.get("success")]
    if not successes:
        return None
    return min(successes, key=lambda r: (int(r["num_steps"]), -float(r.get("score", 0.0)), int(r["rollout_id"])))


def build_materials(
    c0_records: Sequence[Dict[str, Any]],
    *,
    obs_chars: int = 160,
    response_chars: int = 1200,
    max_chars: int = 6000,
) -> List[Dict[str, Any]]:
    """One reference solution per mixed-outcome task (0 < successes < usable rollouts)."""
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in c0_records:
        by_task[record["task_id"]].append(record)
    materials = []
    for task_id in sorted(by_task):
        records = [r for r in by_task[task_id] if usable(r)]
        successes = sum(1 for r in records if r.get("success"))
        if not (0 < successes < len(records)):
            continue
        reference = select_reference(records)
        text, rendered = build_reference_solution(
            record_to_steps(reference), obs_chars=obs_chars, response_chars=response_chars, max_chars=max_chars
        )
        if not text:
            continue
        materials.append(
            {
                "task_id": task_id,
                "slug": reference["task_type"],
                "benchmark_task_id": reference["benchmark_task_id"],
                "c0_rollouts": len(records),
                "c0_successes": successes,
                "source_rollout_id": reference["rollout_id"],
                "source_num_steps": int(reference["num_steps"]),
                "rendered_steps": rendered,
                "chars": len(text),
                "text": text,
            }
        )
    return materials


def make_prompt_hook(materials: Sequence[Dict[str, Any]]) -> PromptHook:
    """Inject the task's reference solution exactly as the trainer builds its teacher prompt."""
    texts = {m["task_id"]: m["text"] for m in materials}

    def hook(prompt: str, spec: Dict[str, Any], step_idx: int) -> str:
        key = f"{spec['slug']}/{spec['task_id']}"
        if key not in texts:
            raise KeyError(f"no reference solution for {key}; C2 must run on the materials' tasks only")
        return build_augmented_observation_text(observation=prompt, reference_solution=texts[key])

    return hook


def episode_stats(record: Dict[str, Any]) -> Dict[str, float]:
    steps = record.get("steps", [])
    invalid = sum(1 for s in steps if not (s.get("info") or {}).get("is_action_valid", True))
    chars = [len(s.get("model_response", "") or "") for s in steps]
    return {
        "success": 1.0 if record.get("success") else 0.0,
        "num_steps": float(len(steps)),
        "invalid_ratio": invalid / len(steps) if steps else 0.0,
        "response_chars": sum(chars) / len(chars) if chars else 0.0,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _se(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1) / len(values))


def summarize_delta(c0_records: Sequence[Dict[str, Any]], c2_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Paired per-task comparison on the tasks present in both conditions."""
    per_task: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: {"C0": [], "C2": []})
    slug_of: Dict[str, str] = {}
    for cond, records in (("C0", c0_records), ("C2", c2_records)):
        for r in records:
            if usable(r):
                per_task[r["task_id"]][cond].append(episode_stats(r))
                slug_of[r["task_id"]] = r["task_type"]
    tasks = []
    for task_id in sorted(per_task):
        c0, c2 = per_task[task_id]["C0"], per_task[task_id]["C2"]
        if not c0 or not c2:
            continue
        row = {"task_id": task_id, "slug": slug_of[task_id], "n_c0": len(c0), "n_c2": len(c2)}
        for key in ("success", "num_steps", "invalid_ratio", "response_chars"):
            row[f"{key}_c0"] = _mean([s[key] for s in c0])
            row[f"{key}_c2"] = _mean([s[key] for s in c2])
        row["delta"] = row["success_c2"] - row["success_c0"]
        tasks.append(row)
    groups = [("all", tasks)] + [(slug, [t for t in tasks if t["slug"] == slug]) for slug in sorted({t["slug"] for t in tasks})]
    rows = []
    for name, group in groups:
        deltas = [t["delta"] for t in group]
        rows.append(
            {
                "group": name,
                "tasks": len(group),
                "success_c0": _mean([t["success_c0"] for t in group]),
                "success_c2": _mean([t["success_c2"] for t in group]),
                "delta": _mean(deltas),
                "se": _se(deltas),
                "tasks_up": sum(1 for d in deltas if d > 0),
                "tasks_down": sum(1 for d in deltas if d < 0),
                "num_steps_c0": _mean([t["num_steps_c0"] for t in group]),
                "num_steps_c2": _mean([t["num_steps_c2"] for t in group]),
                "invalid_ratio_c0": _mean([t["invalid_ratio_c0"] for t in group]),
                "invalid_ratio_c2": _mean([t["invalid_ratio_c2"] for t in group]),
                "response_chars_c0": _mean([t["response_chars_c0"] for t in group]),
                "response_chars_c2": _mean([t["response_chars_c2"] for t in group]),
            }
        )
    return {"rows": rows, "per_task": tasks}


def format_report(summary: Dict[str, Any], materials: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Offline delta test: C2 (reference solution in context) vs C0 (plain)",
        "",
        f"Mixed-outcome tasks with a reference: {len(materials)}; mean reference length "
        f"{_mean([m['chars'] for m in materials]):.0f} chars, {_mean([m['rendered_steps'] for m in materials]):.1f} steps.",
        "",
        "| group | tasks | success C0 | success C2 | delta | SE | up / down | steps C0 -> C2 | invalid C0 -> C2 | resp chars C0 -> C2 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(
            f"| {r['group']} | {r['tasks']} | {r['success_c0']:.3f} | {r['success_c2']:.3f} | {r['delta']:+.3f} | {r['se']:.3f} "
            f"| {r['tasks_up']} / {r['tasks_down']} | {r['num_steps_c0']:.1f} -> {r['num_steps_c2']:.1f} "
            f"| {r['invalid_ratio_c0']:.3f} -> {r['invalid_ratio_c2']:.3f} | {r['response_chars_c0']:.0f} -> {r['response_chars_c2']:.0f} |"
        )
    lines += [
        "",
        "Reading: delta within about one SE of zero = the context does not change the outcome; a positive delta with a large",
        "drop in response chars or steps is style imitation, not help. Compare 'all' first, then the per-benchmark rows.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Phases (need the Stage-1 pipeline, exgentic and a policy endpoint)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    # .env first (machine paths, keys, the per-benchmark step / observation caps), so the env-derived
    # defaults below see it; load_env_file never overrides variables already exported.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=".env")
    from scripts.sft._common.pipeline import load_env_file

    load_env_file(pre.parse_known_args()[0].env_file)
    env = os.environ.get
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[pre]
    )
    parser.add_argument("--phase", default="all", choices=PHASES + ("all",))
    parser.add_argument("--output-dir", default="outputs/delta_test")
    parser.add_argument("--exgentic-root", default=env("AGENTSTREAM_EXGENTIC_ROOT", ""))
    parser.add_argument("--benchmarks", default=env("AGENTSTREAM_BENCHMARKS", "bfcl,tau2,browsecompplus"))
    parser.add_argument("--benchmark-kwargs-json", default=env("AGENTSTREAM_BENCHMARK_KWARGS_JSON", "{}"))
    parser.add_argument("--runner", default="venv")
    # Tasks: the first N of the RL training stream (seed-42 shuffle minus the RL holdout).
    parser.add_argument("--num-tasks-per-benchmark", type=int, default=16)
    parser.add_argument("--holdout-after-tasks", type=int, default=int(env("AGENTSTREAM_NUM_TASKS", "64")))
    parser.add_argument("--holdout-tasks-per-benchmark", type=int, default=int(env("AGENTSTREAM_VAL_TASKS", "32")))
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument("--parallel-sessions", type=int, default=8)
    # Episode limits and prompt rendering: the RL values.
    parser.add_argument("--max-steps", type=int, default=int(env("AGENTSTREAM_MAX_STEPS", "40")))
    # RL applies uniform caps (env.max_steps, the package observation default); the Stage-1 per-benchmark
    # JSONs of agentstream_full.env are not part of the RL launch, so they are opt-in here.
    parser.add_argument("--max-steps-json", default="{}")
    parser.add_argument("--observation-max-chars", type=int, default=None, help="default: package default")
    parser.add_argument("--observation-max-chars-json", default="{}")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--require-think", action="store_true", help="RL runs with projection_require_think=false")
    parser.add_argument("--no-accept-tool-call", action="store_true", help="RL runs with projection_accept_tool_call=true")
    # Reference rendering caps (seed.sibling.build_reference_solution).
    parser.add_argument("--ref-obs-chars", type=int, default=160)
    parser.add_argument("--ref-response-chars", type=int, default=1200)
    parser.add_argument("--ref-max-chars", type=int, default=6000)
    # Policy endpoint: RL rollout sampling (temperature 1.0, 512 tokens). Explicit local defaults so the
    # resolver never falls back to the OPENAI_BASE_URL of .env (the skill-teacher API).
    parser.add_argument("--policy-base-url", default="http://127.0.0.1:60001/v1")
    parser.add_argument("--policy-api-key", default="EMPTY")
    parser.add_argument("--policy-model", default="sft")
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--policy-max-completion-tokens", type=int, default=512)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--policy-retries", type=int, default=2)
    parser.add_argument("--policy-retry-delay", type=float, default=1.0)
    parser.add_argument("--policy-extra-body-json", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def rollout_args(args: argparse.Namespace, output_dir: Path, prompt_hook: Optional[PromptHook] = None) -> SimpleNamespace:
    """The namespace scripts.sft.agentstream.pipeline.collect_rollouts / run_one_rollout read."""
    from agent_system.environments.env_package.agentstream.as_config import DEFAULT_OBSERVATION_MAX_CHARS, parse_int_mapping

    return SimpleNamespace(
        output_dir=str(output_dir),
        exgentic_root=args.exgentic_root,
        runner=args.runner,
        resume=True,
        rollouts_per_task=args.rollouts_per_task,
        parallel_sessions=args.parallel_sessions,
        max_steps=args.max_steps,
        max_steps_by_slug=parse_int_mapping(args.max_steps_json, "--max-steps-json"),
        observation_max_chars=args.observation_max_chars or DEFAULT_OBSERVATION_MAX_CHARS,
        observation_max_chars_by_slug=parse_int_mapping(args.observation_max_chars_json, "--observation-max-chars-json"),
        history_length=args.history_length,
        no_require_think=not args.require_think,
        accept_tool_call=not args.no_accept_tool_call,
        prompt_hook=prompt_hook,
        # sample_tasks reads these
        num_tasks_per_benchmark=args.num_tasks_per_benchmark,
        holdout_after_tasks=args.holdout_after_tasks,
        holdout_tasks_per_benchmark=args.holdout_tasks_per_benchmark,
    )


class Runner:
    """Shares the exgentic hub / policy endpoint between the rollout phases."""

    def __init__(self, args: argparse.Namespace, out: Path):
        from agent_system.environments.env_package.agentstream.as_config import resolve_benchmark_kwargs
        from agent_system.environments.env_package.agentstream.exgentic_client import BenchmarkHub
        from scripts.sft._common.pipeline import resolve_endpoint

        if not args.exgentic_root:
            raise SystemExit("--exgentic-root (or AGENTSTREAM_EXGENTIC_ROOT) is required")
        self.args, self.out = args, out
        self.slugs = sorted(s.strip() for s in args.benchmarks.split(",") if s.strip())
        overrides = json.loads(args.benchmark_kwargs_json)
        self.bm_kwargs = {slug: resolve_benchmark_kwargs(slug, overrides.get(slug)) for slug in self.slugs}
        self.endpoint = resolve_endpoint(
            prefix="policy", args=args, default_base_url_env="POLICY_OPENAI_BASE_URL",
            default_model_env="POLICY_OPENAI_MODEL", default_model="sft",
            temperature=args.policy_temperature, max_completion_tokens=args.policy_max_completion_tokens,
            timeout=args.policy_timeout, retries=args.policy_retries, retry_delay=args.policy_retry_delay,
            extra_body_json=args.policy_extra_body_json,
        )
        self.hub = BenchmarkHub(
            exgentic_root=args.exgentic_root, slugs=self.slugs, benchmark_kwargs=self.bm_kwargs,
            runner=args.runner, output_dir=str(out / "exgentic_sessions"), run_id="delta_hub",
        )

    def close(self) -> None:
        self.hub.close()

    def tasks(self) -> List[Dict[str, Any]]:
        from scripts.sft.agentstream.pipeline import sample_tasks

        return sample_tasks(rollout_args(self.args, self.out), self.out, self.hub)

    def collect(self, condition: str, tasks: Sequence[Dict[str, Any]], prompt_hook: Optional[PromptHook]) -> List[Dict[str, Any]]:
        from scripts.sft.agentstream.pipeline import collect_rollouts

        cond_dir = self.out / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        logging.info("[%s] %d tasks x %d rollouts -> %s", condition, len(tasks), self.args.rollouts_per_task, cond_dir)
        return collect_rollouts(list(tasks), rollout_args(self.args, cond_dir, prompt_hook), cond_dir, self.hub, self.endpoint, self.bm_kwargs)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def phase_materials(args: argparse.Namespace, out: Path) -> List[Dict[str, Any]]:
    c0 = _read_jsonl(out / "C0" / "baseline_rollouts.jsonl")
    if not c0:
        raise SystemExit("materials: run --phase c0 first")
    materials = build_materials(c0, obs_chars=args.ref_obs_chars, response_chars=args.ref_response_chars, max_chars=args.ref_max_chars)
    _write_jsonl(out / "materials.jsonl", materials)
    by_slug = defaultdict(int)
    for m in materials:
        by_slug[m["slug"]] += 1
    logging.info("materials: %d mixed-outcome tasks with a reference (%s)", len(materials), dict(by_slug))
    return materials


def phase_report(out: Path) -> None:
    c0, c2 = _read_jsonl(out / "C0" / "baseline_rollouts.jsonl"), _read_jsonl(out / "C2" / "baseline_rollouts.jsonl")
    materials = _read_jsonl(out / "materials.jsonl")
    if not c0 or not c2:
        raise SystemExit("report: C0 and C2 rollouts are both required")
    summary = summarize_delta(c0, c2)
    (out / "delta_report.md").write_text(format_report(summary, materials), encoding="utf-8")
    if summary["per_task"]:
        with open(out / "delta_per_task.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary["per_task"][0].keys()))
            writer.writeheader()
            writer.writerows(summary["per_task"])
    print((out / "delta_report.md").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    from scripts.sft._common.pipeline import setup_logging

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out, args.log_level)
    phases = PHASES if args.phase == "all" else (args.phase,)
    (out / "run_config.json").write_text(json.dumps({k: v for k, v in vars(args).items() if "api_key" not in k}, indent=2, default=str))

    runner: Optional[Runner] = None
    try:
        if "c0" in phases or "c2" in phases:
            runner = Runner(args, out)
        if "c0" in phases:
            runner.collect("C0", runner.tasks(), prompt_hook=None)
        materials = phase_materials(args, out) if "materials" in phases else _read_jsonl(out / "materials.jsonl")
        if "c2" in phases:
            if not materials:
                raise SystemExit("c2: no materials (run --phase materials; needs mixed-outcome tasks in C0)")
            tasks = [{"slug": m["slug"], "task_id": m["benchmark_task_id"]} for m in materials]
            runner.collect("C2", tasks, prompt_hook=make_prompt_hook(materials))
        if "report" in phases:
            phase_report(out)
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
