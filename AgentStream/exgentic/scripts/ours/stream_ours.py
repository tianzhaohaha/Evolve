# SPDX-License-Identifier: Apache-2.0
"""One runner for the AgentStream baselines on the SEED task stream.

    python stream_ours.py online  --agent {tool_calling,reasoning_bank,ace} --model M --mode {interleaved,isolated} --output-dir D
    python stream_ours.py holdout --agent ... --model M --mode ... --output-dir D   # after the online pass

``online`` walks the 192-task stream one task at a time (the same order as the SEED RL runs), lets
the agent learn between tasks, checkpoints its memory store and appends one record per task to
``online_metrics.jsonl``; ``holdout`` reloads the final store, freezes learning and scores the 96
held-out tasks into ``holdout_metrics.jsonl``. Both log to one wandb run with the SEED metric names.
Progress is kept in ``progress.json`` / ``holdout_progress.json`` so a killed process resumes at the
next task. ``tool_calling`` is the no-learning reference; the two learning agents differ only in
their store class and constructor kwargs (``AGENTS``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "utils"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from registry_ours import (  # noqa: E402
    BENCHMARKS, HOLDOUT_TEMPERATURE, MAX_STEPS, NUM_HOLDOUT_TASKS, NUM_STREAM_TASKS, ONLINE_TEMPERATURE,
    STREAM_SEED, build_configs, holdout_task_ids,
)
from wandb_stream import OnlineTally, WandbStream, holdout_summary  # noqa: E402


@dataclass(frozen=True)
class AgentSpec:
    slug: str
    store_prefix: Optional[str]  # None = no memory store (the no-learning reference)
    store_module: Optional[str]
    store_class: Optional[str]

    def store_cls(self):
        if self.store_module is None:
            return None
        import importlib

        return getattr(importlib.import_module(self.store_module), self.store_class)

    def store_key(self, mode: str, bm_slug: str) -> str:
        return f"{self.store_prefix}_isolated_{bm_slug}" if mode == "isolated" else f"{self.store_prefix}_{mode}_global"

    def build(self, model: str, mode: str, bm_slug: str, settings, learning_enabled: bool, **agent_kwargs):
        from exgentic.interfaces.registry import load_agent

        common = dict(model=model, runner="direct", model_settings=settings, **agent_kwargs)
        if self.slug == "tool_calling":
            return load_agent(self.slug)(allow_truncated_messages=True, **common)
        common.update(shuffle_mode=mode, benchmark_id=bm_slug, learning_enabled=learning_enabled)
        if self.slug == "reasoning_bank":
            return load_agent(self.slug)(memory_model=model, eval_model=model, **common)
        return load_agent(self.slug)(curator_model=model, **common)  # ace


AGENTS = {
    "tool_calling": AgentSpec("tool_calling", None, None, None),
    "reasoning_bank": AgentSpec("reasoning_bank", "rb", "exgentic.agents.reasoning_bank.rb_store", "ReasoningBankStore"),
    "ace": AgentSpec("ace", "ace", "exgentic.agents.ace.playbook_store", "PlaybookStore"),
}


# ----------------------------------------------------------------------------- memory stores

def store_keys(spec: AgentSpec, mode: str, benchmarks: Iterable[str]) -> List[str]:
    if spec.store_prefix is None:
        return []
    return [spec.store_key(mode, bm) for bm in benchmarks] if mode == "isolated" else [spec.store_key(mode, "")]


def get_store(spec: AgentSpec, mode: str, bm_slug: str):
    return spec.store_cls().get_or_create(shuffle_mode=mode, benchmark_id=bm_slug)


def save_stores(spec: AgentSpec, output_dir: Path) -> None:
    if spec.store_prefix is None:
        return
    for key, store in spec.store_cls().list_stores().items():
        store.save_checkpoint(str(output_dir / f"store_{key}.json"))


def load_stores(spec: AgentSpec, mode: str, benchmarks: Iterable[str], output_dir: Path, *, required: bool) -> int:
    """Restore every store checkpoint of the run; returns how many were found."""
    if spec.store_prefix is None:
        return 0
    spec.store_cls().reset_all()
    found = 0
    for bm in benchmarks:
        key = spec.store_key(mode, bm)
        path = output_dir / f"store_{key}.json"
        if path.exists():
            get_store(spec, mode, bm).load_checkpoint(str(path))
            found += 1
        elif required:
            raise FileNotFoundError(f"store checkpoint missing: {path} (run the online pass first)")
        if mode != "isolated":
            break
    return found


def memory_stats(spec: AgentSpec, mode: str, bm_slug: str) -> Tuple[int, int]:
    """(entries, ~tokens) of the store the task used; (0, 0) without a store."""
    if spec.store_prefix is None:
        return 0, 0
    store = spec.store_cls().list_stores().get(spec.store_key(mode, bm_slug))
    if store is None:
        return 0, 0
    if hasattr(store, "get_entries"):  # ReasoningBank
        entries = store.get_entries()
        return len(entries), sum(len("\n\n".join(e.memory_items)) for e in entries) // 4
    from exgentic.agents.ace.playbook_utils import get_playbook_stats  # ACE

    text = getattr(store, "playbook", "")
    return int(get_playbook_stats(text)["total_bullets"]), len(text) // 4


# ----------------------------------------------------------------------------- one task

def token_counts(cost_reports: Dict[str, Any]) -> Tuple[int, int]:
    total_in = total_out = 0
    for report in cost_reports.values():
        get = report.get if isinstance(report, dict) else lambda key, default=None: getattr(report, key, default)
        total_in += int(get("input_tokens") or 0)
        total_out += int(get("output_tokens") or 0)
    return total_in, total_out


def run_task(spec: AgentSpec, configs, bm_slug: str, task_id: str, *, model: str, mode: str, settings,
             learning_enabled: bool, sessions_dir: Path):
    from exgentic.interfaces.lib.api import evaluate
    from exgentic.interfaces.registry import load_benchmark

    benchmark = load_benchmark(bm_slug)(**configs[bm_slug]["bm_kwargs"])
    agent = spec.build(model, mode, bm_slug, settings, learning_enabled, **configs[bm_slug].get("agent_kwargs", {}))
    results = evaluate(
        benchmark=benchmark, agent=agent, task_ids=[task_id], max_workers=1, max_steps=MAX_STEPS,
        max_actions=MAX_STEPS * 10,  # steps are the only cap, as in the SEED environment (parallel tool calls count as actions)
        output_dir=str(sessions_dir), overwrite_sessions=True,
    )
    return results.session_results[0]


def task_record(sr, *, index: int, bm_slug: str, task_id: str, spec: AgentSpec, mode: str, args) -> Dict[str, Any]:
    score = sr.score if sr.score is not None else (1.0 if sr.success else 0.0)
    tokens_in, tokens_out = token_counts(sr.cost_reports)
    entries, mem_tokens = memory_stats(spec, mode, bm_slug)
    return {
        "session_index": index, "seed": args.seed, "mode": mode, "agent": spec.slug, "model": args.model,
        "benchmark_slug": bm_slug, "task_id": task_id, "score": float(score), "success": bool(sr.success),
        "steps": sr.steps, "action_count": sr.action_count, "agent_cost": sr.agent_cost,
        "input_tokens": tokens_in, "output_tokens": tokens_out, "memory_entries": entries, "memory_tokens": mem_tokens,
        "execution_time": sr.execution_time, "status": getattr(sr.status, "value", str(sr.status)),
        "timestamp": datetime.now().isoformat(),
    }


# ----------------------------------------------------------------------------- progress files

class Progress:
    """``<name>.jsonl`` records plus ``<name>_progress.json`` {"done": n}: the records file is
    truncated to ``done`` lines on load, so a crash between the two writes never double counts."""

    def __init__(self, output_dir: Path, name: str) -> None:
        self.records_path = output_dir / f"{name}.jsonl"
        self.progress_path = output_dir / f"{name.replace('_metrics', '')}_progress.json"

    def load(self) -> List[Dict[str, Any]]:
        done = json.loads(self.progress_path.read_text())["done"] if self.progress_path.exists() else 0
        lines = self.records_path.read_text().splitlines() if self.records_path.exists() else []
        kept = lines[:done]
        if len(lines) != len(kept):
            self.records_path.write_text("".join(line + "\n" for line in kept))
        return [json.loads(line) for line in kept]

    def append(self, record: Dict[str, Any], done: int) -> None:
        with open(self.records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.progress_path.write_text(json.dumps({"done": done}))


# ----------------------------------------------------------------------------- passes

def model_settings(args, temperature: float):
    from exgentic.core.types import ModelSettings

    kwargs = {"temperature": temperature}
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    if args.reasoning_effort is not None:
        kwargs["reasoning_effort"] = args.reasoning_effort
    return ModelSettings(**kwargs)


def run_name(args) -> str:
    model_short = args.model.split("/")[-1]
    return args.run_name or f"as_{args.agent}_{model_short}_{args.mode}_s{args.seed}"


def setup(args):
    if args.api_base:
        os.environ["OPENAI_API_BASE"] = args.api_base
    benchmarks = [s.strip() for s in args.benchmarks.split(",")]
    configs = build_configs(args.judge_model, args.retriever_url, benchmarks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "agent": args.agent, "model": args.model, "mode": args.mode, "seed": args.seed, "num_tasks": args.num_tasks,
        "num_holdout": args.num_holdout, "benchmarks": benchmarks, "max_steps": MAX_STEPS,
        "online_temperature": ONLINE_TEMPERATURE, "holdout_temperature": HOLDOUT_TEMPERATURE,
        "reasoning_effort": args.reasoning_effort, "registry": {k: v["bm_kwargs"] for k, v in configs.items()},
    }
    stream = WandbStream(run_name(args), config, output_dir, enabled=not args.no_wandb, project=args.wandb_project)
    return AGENTS[args.agent], configs, benchmarks, output_dir, stream


def run_online(args) -> None:
    from task_ordering import get_unified_task_order

    spec, configs, benchmarks, output_dir, stream = setup(args)
    task_order = get_unified_task_order(configs, args.num_tasks, args.seed, args.mode)
    (output_dir / "experiment_config.json").write_text(json.dumps({**stream_config(args, benchmarks), "task_order": task_order}, indent=2))
    progress = Progress(output_dir, "online_metrics")
    done = progress.load()
    tally = OnlineTally()
    for rec in done:
        tally.add(rec["benchmark_slug"], rec["score"], rec["success"])
    load_stores(spec, args.mode, benchmarks, output_dir, required=False)
    print(f"[online] {run_name(args)}: {len(task_order)} tasks, resuming at {len(done)}")
    settings = model_settings(args, ONLINE_TEMPERATURE)
    for index in range(len(done), len(task_order)):
        bm_slug, task_id = task_order[index]
        sr = run_task(spec, configs, bm_slug, task_id, model=args.model, mode=args.mode, settings=settings,
                      learning_enabled=True, sessions_dir=output_dir / "sessions_online")
        record = task_record(sr, index=index, bm_slug=bm_slug, task_id=task_id, spec=spec, mode=args.mode, args=args)
        tally.add(bm_slug, record["score"], record["success"])
        save_stores(spec, output_dir)
        progress.append(record, index + 1)
        stream.log_task(record, tally)
        print(f"  [{index + 1}/{len(task_order)}] {bm_slug}::{task_id} score={record['score']:.2f} "
              f"success={record['success']} cum={tally.metrics()['online/cumulative_avg_score']:.3f} steps={record['steps']}")
    (output_dir / "online_summary.json").write_text(json.dumps(tally.metrics(), indent=2))
    print(json.dumps(tally.metrics(), indent=2))
    stream.finish()


def run_holdout(args) -> None:
    spec, configs, benchmarks, output_dir, stream = setup(args)
    holdout = holdout_task_ids(configs, args.num_tasks, args.num_holdout)
    tasks = [(bm, tid) for bm in benchmarks for tid in holdout[bm]]
    online_done = len(Progress(output_dir, "online_metrics").load())
    load_stores(spec, args.mode, benchmarks, output_dir, required=True)
    progress = Progress(output_dir, "holdout_metrics")
    done = progress.load()
    print(f"[holdout] {run_name(args)}: {len(tasks)} tasks (frozen memory after {online_done} stream tasks), resuming at {len(done)}")
    settings = model_settings(args, HOLDOUT_TEMPERATURE)
    for index in range(len(done), len(tasks)):
        bm_slug, task_id = tasks[index]
        sr = run_task(spec, configs, bm_slug, task_id, model=args.model, mode=args.mode, settings=settings,
                      learning_enabled=False, sessions_dir=output_dir / "sessions_holdout")
        record = task_record(sr, index=index, bm_slug=bm_slug, task_id=task_id, spec=spec, mode=args.mode, args=args)
        done.append(record)
        progress.append(record, index + 1)
        print(f"  [{index + 1}/{len(tasks)}] {bm_slug}::{task_id} score={record['score']:.2f} success={record['success']}")
    summary = holdout_summary(done)
    (output_dir / "holdout_summary.json").write_text(json.dumps(summary, indent=2))
    stream.log_holdout(summary, task_index=max(online_done, 1))
    print(json.dumps(summary, indent=2))
    stream.finish()


def stream_config(args, benchmarks) -> Dict[str, Any]:
    return {"agent": args.agent, "model": args.model, "mode": args.mode, "seed": args.seed,
            "num_tasks": args.num_tasks, "num_holdout": args.num_holdout, "benchmarks": benchmarks, "max_steps": MAX_STEPS}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pass_", metavar="PASS", choices=["online", "holdout"])
    parser.add_argument("--agent", required=True, choices=sorted(AGENTS))
    parser.add_argument("--model", required=True, help="litellm model id, e.g. openai/Qwen3-4B-Instruct-2507 or openrouter/openai/gpt-6-luna")
    parser.add_argument("--mode", required=True, choices=["interleaved", "isolated", "sequential"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=STREAM_SEED)
    parser.add_argument("--num-tasks", type=int, default=NUM_STREAM_TASKS, help="stream tasks per benchmark")
    parser.add_argument("--num-holdout", type=int, default=NUM_HOLDOUT_TASKS, help="held-out tasks per benchmark")
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible endpoint for openai/<model> (the local vLLM policy server)")
    parser.add_argument("--judge-model", default=None, help="tau2 user simulator / browsecomp judge (default: GLM)")
    parser.add_argument("--retriever-url", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None, help="e.g. low for reasoning-capable API models")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default="agentic_agentstream")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)
    (run_online if args.pass_ == "online" else run_holdout)(args)


if __name__ == "__main__":
    main()
