#!/usr/bin/env python3
"""Offline delta test: does an in-context demonstration raise the frozen SFT policy's success rate?

Conditions on the same tasks, same frozen checkpoint, same sampling settings as Stage-3 RL:
  C0  plain prompt (every sampled task);
  C2  the task's OWN shortest successful C0 rollout injected as a "Reference Solution" (reasoning +
      action per step) -- tasks whose C0 rollouts were mixed. Measures whether the model can follow
      a demonstration at all; it is an upper bound (same task, answer included).
  C4  the demonstration of the most similar OTHER task of the same benchmark that has a C0 success,
      final step dropped (no answer / submit leakage) -- every task not already solved every time.
      Measures transferable information, i.e. what a global experience pool could deliver; read the
      "c0_all_fail" stratum first: that is where GRPO has no signal of its own.
The injection goes through seed.prompting.build_augmented_observation_text, the very section the
trainer's teacher prompt uses. delta = success(Cx) - success(C0) per task, paired over tasks (mean,
standard error, by benchmark, by C0 stratum).

Building blocks are the Stage-1 SFT pipeline (scripts/sft/agentstream/pipeline.py: task selection
from the RL stream's seed-42 shuffle minus the holdout, the OpenAI-compatible policy client, the
threaded resumable rollout collector) plus the per-step ``prompt_hook`` that pipeline exposes;
nothing on the training path is touched. Sampling mirrors RL: temperature 1.0, 512 response tokens,
history window 3, the RL max-steps cap and projection flags (think optional, tool_call accepted).

Needs a running OpenAI-compatible endpoint for the SFT checkpoint and the exgentic checkout. The
wrapper examples/agentstream_trainer/run_delta_test.sh (one PBS line) sources the public config,
starts the endpoint and passes the RL settings; direct use after `set -a; source
examples/agentstream_trainer/agentstream_full.env; set +a`:
  python examples/agentstream_trainer/delta_test.py --phase all --conditions C2,C4 --output-dir outputs/delta_test_v5
Phases (each resumes from its outputs): c0 -> materials -> conditions -> report.
Outputs under --output-dir: sampled_tasks.jsonl, C0/baseline_rollouts.jsonl, materials_<C>.jsonl,
<C>/baseline_rollouts.jsonl, delta_report.md, delta_per_task_<C>.csv, run_config.json.
The token-level companion (does the reference raise the likelihood of the C0 trajectories?) is
examples/agentstream_trainer/delta_gap.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed.prompting import build_augmented_observation_text  # noqa: E402
from seed.sibling import build_reference_solution  # noqa: E402

PHASES = ("c0", "materials", "conditions", "report")
# source: whose successful rollout is the reference; drop_final_step: strip the answer / submit step.
CONDITIONS: Dict[str, Dict[str, Any]] = {
    "C2": {"source": "self", "drop_final_step": False},
    "C4": {"source": "neighbor", "drop_final_step": True},
}
PromptHook = Callable[[str, Dict[str, Any], int], str]
_WORD_RE = re.compile(r"[a-z0-9]+")


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


def group_by_task(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """task_id -> usable rollouts, tasks in sorted order."""
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if usable(record):
            by_task[record["task_id"]].append(record)
    return dict(sorted(by_task.items()))


def success_count(records: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for r in records if r.get("success"))


def select_reference(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Shortest successful rollout of one task (ties: higher score, then lower rollout_id)."""
    successes = [r for r in records if r.get("success")]
    if not successes:
        return None
    return min(successes, key=lambda r: (int(r["num_steps"]), -float(r.get("score", 0.0)), int(r["rollout_id"])))


def _bag(text: object) -> Counter:
    return Counter(_WORD_RE.findall(str(text or "").lower()))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def select_neighbor(task_id: str, by_task: Dict[str, List[Dict[str, Any]]]) -> Optional[Tuple[str, float]]:
    """Most similar other task of the same benchmark with at least one C0 success (task-description
    bag-of-words cosine; ties -> lower task_id). Returns (task_id, similarity) or None."""
    own = by_task[task_id][0]
    desc = _bag(own.get("task_description"))
    candidates = [
        (cosine(desc, _bag(recs[0].get("task_description"))), other)
        for other, recs in by_task.items()
        if other != task_id and recs[0]["task_type"] == own["task_type"] and success_count(recs) > 0
    ]
    if not candidates:
        return None
    sim, other = min(candidates, key=lambda c: (-c[0], c[1]))
    return other, sim


def build_materials(
    c0_records: Sequence[Dict[str, Any]],
    *,
    source: str = "self",
    drop_final_step: bool = False,
    obs_chars: int = 160,
    response_chars: int = 1200,
    max_chars: int = 6000,
) -> List[Dict[str, Any]]:
    """One reference solution per target task.

    ``source="self"``: mixed-outcome tasks (0 < successes < rollouts), reference = own shortest success.
    ``source="neighbor"``: every task not solved every time, reference = the most similar other task's
    shortest success (see :func:`select_neighbor`). ``drop_final_step`` removes the reference's last
    step (the answer / submit action) so a transferred demonstration carries procedure, not the answer.
    """
    if source not in ("self", "neighbor"):
        raise ValueError(f"source must be 'self' or 'neighbor', got {source!r}")
    by_task = group_by_task(c0_records)
    materials = []
    for task_id, records in by_task.items():
        n, s = len(records), success_count(records)
        if source == "self":
            if not (0 < s < n):
                continue
            source_task, similarity = task_id, 1.0
        else:
            if s >= n:
                continue
            neighbor = select_neighbor(task_id, by_task)
            if neighbor is None:
                continue
            source_task, similarity = neighbor
        reference = select_reference(by_task[source_task])
        steps = record_to_steps(reference)
        if drop_final_step and len(steps) > 1:
            steps = steps[:-1]
        text, rendered = build_reference_solution(steps, obs_chars=obs_chars, response_chars=response_chars, max_chars=max_chars)
        if not text:
            continue
        materials.append(
            {
                "task_id": task_id,
                "slug": records[0]["task_type"],
                "benchmark_task_id": records[0]["benchmark_task_id"],
                "c0_rollouts": n,
                "c0_successes": s,
                "source": source,
                "source_task_id": source_task,
                "similarity": round(similarity, 4),
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
            raise KeyError(f"no reference solution for {key}; a condition must run on its materials' tasks only")
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


STATS = ("success", "num_steps", "invalid_ratio", "response_chars")


def summarize_delta(c0_records: Sequence[Dict[str, Any]], cx_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Paired per-task comparison on the tasks present in both conditions; groups = all, per benchmark,
    and the C0 strata (all-fail / mixed / all-success in C0)."""
    per_task: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: {"C0": [], "CX": []})
    slug_of: Dict[str, str] = {}
    for cond, records in (("C0", c0_records), ("CX", cx_records)):
        for r in records:
            if usable(r):
                per_task[r["task_id"]][cond].append(episode_stats(r))
                slug_of[r["task_id"]] = r["task_type"]
    tasks = []
    for task_id in sorted(per_task):
        c0, cx = per_task[task_id]["C0"], per_task[task_id]["CX"]
        if not c0 or not cx:
            continue
        row = {"task_id": task_id, "slug": slug_of[task_id], "n_c0": len(c0), "n_cx": len(cx)}
        for key in STATS:
            row[f"{key}_c0"] = _mean([s[key] for s in c0])
            row[f"{key}_cx"] = _mean([s[key] for s in cx])
        row["delta"] = row["success_cx"] - row["success_c0"]
        row["stratum"] = "c0_all_fail" if row["success_c0"] == 0 else ("c0_all_success" if row["success_c0"] == 1 else "c0_mixed")
        tasks.append(row)
    groups = [("all", tasks)]
    groups += [(slug, [t for t in tasks if t["slug"] == slug]) for slug in sorted({t["slug"] for t in tasks})]
    groups += [(name, [t for t in tasks if t["stratum"] == name]) for name in ("c0_all_fail", "c0_mixed", "c0_all_success")]
    rows = []
    for name, group in groups:
        if not group:
            continue
        deltas = [t["delta"] for t in group]
        row = {"group": name, "tasks": len(group), "delta": _mean(deltas), "se": _se(deltas),
               "tasks_up": sum(1 for d in deltas if d > 0), "tasks_down": sum(1 for d in deltas if d < 0)}
        for key in STATS:
            row[f"{key}_c0"] = _mean([t[f"{key}_c0"] for t in group])
            row[f"{key}_cx"] = _mean([t[f"{key}_cx"] for t in group])
        rows.append(row)
    return {"rows": rows, "per_task": tasks}


def format_report(sections: Sequence[Tuple[str, Dict[str, Any], Sequence[Dict[str, Any]]]]) -> str:
    """Markdown report: one section per (condition, summary, materials)."""
    lines = ["# Offline delta test: demonstration in context vs plain prompt (C0)", ""]
    for cond, summary, materials in sections:
        spec = CONDITIONS.get(cond, {})
        lines += [
            f"## {cond}: source={spec.get('source', '?')}, final step dropped={spec.get('drop_final_step', '?')}",
            "",
            f"Tasks with a reference: {len(materials)}; mean reference length {_mean([m['chars'] for m in materials]):.0f} chars, "
            f"{_mean([m['rendered_steps'] for m in materials]):.1f} steps, mean task similarity {_mean([m.get('similarity', 1.0) for m in materials]):.2f}.",
            "",
            "| group | tasks | success C0 | success Cx | delta | SE | up / down | steps C0 -> Cx | invalid C0 -> Cx | resp chars C0 -> Cx |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in summary["rows"]:
            lines.append(
                f"| {r['group']} | {r['tasks']} | {r['success_c0']:.3f} | {r['success_cx']:.3f} | {r['delta']:+.3f} | {r['se']:.3f} "
                f"| {r['tasks_up']} / {r['tasks_down']} | {r['num_steps_c0']:.1f} -> {r['num_steps_cx']:.1f} "
                f"| {r['invalid_ratio_c0']:.3f} -> {r['invalid_ratio_cx']:.3f} | {r['response_chars_c0']:.0f} -> {r['response_chars_cx']:.0f} |"
            )
        lines.append("")
    lines += [
        "Reading: delta within about one SE of zero = the context does not change the outcome; a positive delta with a large",
        "drop in response chars is style imitation, not help. C2 is the same-task upper bound (answer included); C4 is",
        "transfer -- judge it on the c0_all_fail row, the tasks where GRPO has no signal of its own.",
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
    parser.add_argument("--conditions", default="C2,C4", help="comma list drawn from " + ",".join(CONDITIONS))
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
    # Episode limits and prompt rendering: the RL values. RL applies uniform caps (env.max_steps, the
    # package observation default); the Stage-1 per-benchmark JSONs are not part of the RL launch.
    parser.add_argument("--max-steps", type=int, default=int(env("AGENTSTREAM_MAX_STEPS", "40")))
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
    args = parser.parse_args()
    args.condition_list = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in args.condition_list if c not in CONDITIONS]
    if unknown:
        parser.error(f"unknown conditions {unknown}; choose from {list(CONDITIONS)}")
    return args


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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def materials_path(out: Path, cond: str) -> Path:
    """materials_<C>.jsonl; the first version of this script wrote C2's file as materials.jsonl."""
    path = out / f"materials_{cond}.jsonl"
    legacy = out / "materials.jsonl"
    return legacy if (cond == "C2" and not path.exists() and legacy.exists()) else path


def phase_materials(args: argparse.Namespace, out: Path, cond: str) -> List[Dict[str, Any]]:
    c0 = read_jsonl(out / "C0" / "baseline_rollouts.jsonl")
    if not c0:
        raise SystemExit("materials: run --phase c0 first")
    materials = build_materials(
        c0, **CONDITIONS[cond], obs_chars=args.ref_obs_chars, response_chars=args.ref_response_chars, max_chars=args.ref_max_chars
    )
    write_jsonl(materials_path(out, cond), materials)
    by_slug = Counter(m["slug"] for m in materials)
    logging.info("materials[%s]: %d tasks with a reference (%s)", cond, len(materials), dict(by_slug))
    return materials


def phase_report(out: Path, conds: Sequence[str]) -> None:
    c0 = read_jsonl(out / "C0" / "baseline_rollouts.jsonl")
    if not c0:
        raise SystemExit("report: C0 rollouts are required")
    sections = []
    for cond in conds:
        cx = read_jsonl(out / cond / "baseline_rollouts.jsonl")
        if not cx:
            logging.warning("report: no %s rollouts, skipping", cond)
            continue
        summary = summarize_delta(c0, cx)
        sections.append((cond, summary, read_jsonl(materials_path(out, cond))))
        if summary["per_task"]:
            with open(out / f"delta_per_task_{cond}.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary["per_task"][0].keys()))
                writer.writeheader()
                writer.writerows(summary["per_task"])
    if not sections:
        raise SystemExit("report: no condition has rollouts yet")
    (out / "delta_report.md").write_text(format_report(sections), encoding="utf-8")
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
        if "c0" in phases or "conditions" in phases:
            runner = Runner(args, out)
        if "c0" in phases:
            runner.collect("C0", runner.tasks(), prompt_hook=None)
        for cond in args.condition_list:
            materials = phase_materials(args, out, cond) if "materials" in phases else read_jsonl(materials_path(out, cond))
            if "conditions" in phases:
                if not materials:
                    logging.warning("[%s] no materials (run --phase materials); skipping", cond)
                    continue
                tasks = [{"slug": m["slug"], "task_id": m["benchmark_task_id"]} for m in materials]
                runner.collect(cond, tasks, prompt_hook=make_prompt_hook(materials))
        if "report" in phases:
            phase_report(out, args.condition_list)
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
