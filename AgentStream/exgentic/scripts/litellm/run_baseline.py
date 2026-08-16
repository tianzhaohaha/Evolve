# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ace"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from exgentic.interfaces.lib.api import evaluate
from exgentic.interfaces.registry import load_agent, load_benchmark
from exgentic.core.types import ModelSettings

from task_ordering import get_unified_task_order, group_by_benchmark


BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    "browsecompplus": {
        "bm_kwargs": {
            "searcher_type": "faiss",
            "include_get_document": True,
            "eval_model_id": "openai/gpt-5.4",
        },
        "agent_kwargs": {},
    },
    "swebench": {
        "bm_kwargs": {
            "subset": "princeton-nlp/SWE-bench_Verified",
        },
        "agent_kwargs": {},
    },
    "appworld": {
        "bm_kwargs": {
            "subset": "test_challenge",
        },
        "agent_kwargs": {
            "enable_tool_shortlisting": True,
            "max_selected_tools": 30,
        },
    },
    "bfcl": {
        "bm_kwargs": {
            "subset": "multi_turn_base",
        },
        "agent_kwargs": {},
    },
    "tau2": {
        "bm_kwargs": {
            "subset": "telecom",
            "user_simulator_model": "openai/gpt-5.4",
        },
        "agent_kwargs": {},
    },
    "hle": {
        "bm_kwargs": {
            "judge_model": "openai/gpt-5.4",
        },
        "agent_kwargs": {},
    },
}

def extract_token_counts(cost_reports: dict) -> tuple[int, int]:
    total_in, total_out = 0, 0
    for report in cost_reports.values():
        if isinstance(report, dict):
            total_in += report.get("input_tokens", 0)
            total_out += report.get("output_tokens", 0)
        elif hasattr(report, "input_tokens"):
            total_in += report.input_tokens
            total_out += report.output_tokens
    return total_in, total_out


def record_online_metrics(
    metrics_path: Path,
    session_index: int,
    bm_slug: str,
    task_id: str,
    sr: Any,
    seed: int,
    model: str,
    all_scores: list[float],
    bm_scores: dict[str, list[float]],
):
    score = sr.score if sr.score is not None else (1.0 if sr.success else 0.0)
    all_scores.append(score)
    bm_scores[bm_slug].append(score)

    input_tokens, output_tokens = extract_token_counts(sr.cost_reports)

    record = {
        "session_index": session_index,
        "seed": seed,
        "mode": "baseline",
        "agent": "tool_calling",
        "model": model,
        "benchmark_slug": bm_slug,
        "task_id": task_id,
        "score": score,
        "cumulative_avg_score": sum(all_scores) / len(all_scores),
        "benchmark_cumulative_avg_score": (
            sum(bm_scores[bm_slug]) / len(bm_scores[bm_slug])
        ),
        "steps": sr.steps,
        "action_count": sr.action_count,
        "agent_cost": sr.agent_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "memory_tokens": 0,
        "playbook_bullets": 0,
        "execution_time": sr.execution_time,
        "status": sr.status.value if hasattr(sr.status, "value") else str(sr.status),
        "timestamp": datetime.now().isoformat(),
    }

    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record

def run_baseline(args):
    benchmarks_to_run = [s.strip() for s in args.benchmarks.split(",")]
    configs = {k: BENCHMARK_REGISTRY[k] for k in benchmarks_to_run}

    settings_kwargs = {}
    if args.max_tokens is not None:
        settings_kwargs["max_tokens"] = args.max_tokens
    if args.reasoning_effort is not None:
        settings_kwargs["reasoning_effort"] = args.reasoning_effort
    model_settings = ModelSettings(**settings_kwargs)

    print(f"\n{'=' * 70}")
    print(f"  Baseline: tool_calling agent (no learning)")
    print(f"  seed={args.seed}  model={args.model}  num_tasks={args.num_tasks}")
    print(f"  model_settings={settings_kwargs or 'default'}")
    print(f"  benchmarks={benchmarks_to_run}")
    print(f"  output_dir={args.output_dir}")
    print(f"{'=' * 70}\n")

    task_order = get_unified_task_order(configs, args.num_tasks, args.seed, "isolated")
    print(f"Total tasks: {len(task_order)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "online_metrics.jsonl"

    exp_config = {
        "mode": "baseline",
        "seed": args.seed,
        "agent": "tool_calling",
        "model": args.model,
        "num_tasks": args.num_tasks,
        "benchmarks": benchmarks_to_run,
        "task_order": [(s, t) for s, t in task_order],
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    all_scores: list[float] = []
    bm_scores: defaultdict[str, list[float]] = defaultdict(list)
    session_index = 0

    for bm_slug, task_ids in group_by_benchmark(task_order):
        print(f"\n{'=' * 60}")
        print(f"  BASELINE — {bm_slug} ({len(task_ids)} tasks)")
        print(f"{'=' * 60}\n")

        bm_kwargs = configs[bm_slug]["bm_kwargs"]
        agent_kwargs = configs[bm_slug].get("agent_kwargs", {})

        benchmark = load_benchmark(bm_slug)(**bm_kwargs)
        agent = load_agent("tool_calling")(
            model=args.model,
            runner="direct",
            model_settings=model_settings,
            allow_truncated_messages=True,
            **agent_kwargs,
        )

        results = evaluate(
            benchmark=benchmark,
            agent=agent,
            task_ids=task_ids,
            max_workers=5,
            output_dir=str(output_dir),
        )

        print(f"  {bm_slug} score={results.benchmark_score}")

        for i, sr in enumerate(results.session_results):
            tid = task_ids[i] if i < len(task_ids) else sr.task_id or "?"
            rec = record_online_metrics(
                metrics_path, session_index, bm_slug, tid, sr,
                args.seed, args.model, all_scores, bm_scores,
            )
            print(f"    [{session_index}] {bm_slug}::{tid}  "
                  f"score={rec['score']:.2f}  cum={rec['cumulative_avg_score']:.3f}  "
                  f"steps={rec['steps']}")
            session_index += 1

    print(f"\n{'=' * 70}")
    print(f"  Baseline Summary")
    print(f"{'=' * 70}")
    print(f"  Overall avg score: {sum(all_scores)/len(all_scores):.3f}")
    for bm, scores in sorted(bm_scores.items()):
        print(f"  {bm:20s}: avg={sum(scores)/len(scores):.3f}  n={len(scores)}")
    print(f"\n  Metrics: {metrics_path}")
    print(f"  Output:  {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline (tool_calling agent, no learning)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, default=50, help="Tasks per benchmark")
    parser.add_argument("--model", default="openai/gpt-5.4")
    parser.add_argument("--benchmarks", default="browsecompplus,swebench,bfcl,tau2",
                        help="Comma-separated benchmark slugs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--reasoning-effort", default=None, help="Reasoning effort (low/medium/high)")
    args = parser.parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
