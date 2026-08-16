# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

# --- ExGentic imports ---
from exgentic.interfaces.lib.api import evaluate
from exgentic.interfaces.registry import load_agent, load_benchmark
from exgentic.agents.harness.harness_store import HarnessStore
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
            "runner": "direct",
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


def get_memory_tokens(mode: str, bm_slug: str) -> tuple[int, int]:
    stores = HarnessStore.list_stores()
    if mode == "isolated":
        store = stores.get(f"harness_isolated_{bm_slug}")
    elif mode == "sequential":
        store = stores.get("harness_sequential_global")
    elif mode == "interleaved":
        store = stores.get("harness_interleaved_global")
    else:
        return 0, 0
    if store is None:
        return 0, 0
    total_chars = len(store.system_prompt) + len(store.memory)
    for skill in store.list_skills():
        total_chars += len(skill.description) + len(skill.body)
    mem_tokens = total_chars // 4
    return mem_tokens, store.skill_count


def record_online_metrics(
    metrics_path: Path,
    session_index: int,
    bm_slug: str,
    task_id: str,
    sr: Any,
    mode: str,
    seed: int,
    model: str,
    all_scores: list[float],
    bm_scores: dict[str, list[float]],
):
    score = sr.score if sr.score is not None else (1.0 if sr.success else 0.0)
    all_scores.append(score)
    bm_scores[bm_slug].append(score)

    input_tokens, output_tokens = extract_token_counts(sr.cost_reports)
    memory_tokens, skill_count = get_memory_tokens(mode, bm_slug)

    record = {
        "session_index": session_index,
        "seed": seed,
        "mode": mode,
        "agent": "harness",
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
        "memory_tokens": memory_tokens,
        "skill_count": skill_count,
        "execution_time": sr.execution_time,
        "status": sr.status.value if hasattr(sr.status, "value") else str(sr.status),
        "timestamp": datetime.now().isoformat(),
    }

    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record

def run_experiment(args):
    benchmarks_to_run = [s.strip() for s in args.benchmarks.split(",")]
    configs = {k: BENCHMARK_REGISTRY[k] for k in benchmarks_to_run}

    settings_kwargs = {}
    if args.max_tokens is not None:
        settings_kwargs["max_tokens"] = args.max_tokens
    if args.reasoning_effort is not None:
        settings_kwargs["reasoning_effort"] = args.reasoning_effort
    model_settings = ModelSettings(**settings_kwargs)

    print(f"\n{'=' * 70}")
    print(f"  Harness Experiment: mode={args.mode}  seed={args.seed}")
    print(f"  model={args.model}  num_tasks={args.num_tasks}")
    print(f"  model_settings={settings_kwargs or 'default'}")
    print(f"  benchmarks={benchmarks_to_run}")
    print(f"  output_dir={args.output_dir}")
    print(f"{'=' * 70}\n")

    task_order = get_unified_task_order(configs, args.num_tasks, args.seed, args.mode)
    print(f"Total tasks: {len(task_order)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "online_metrics.jsonl"

    exp_config = {
        "mode": args.mode,
        "seed": args.seed,
        "agent": "harness",
        "model": args.model,
        "num_tasks": args.num_tasks,
        "benchmarks": benchmarks_to_run,
        "task_order": [(s, t) for s, t in task_order],
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    HarnessStore.reset_all()
    if args.mode == "interleaved":
        _ckpt_ids = ["harness_interleaved_global"]
    elif args.mode == "sequential":
        _ckpt_ids = ["harness_sequential_global"]
    else:
        _ckpt_ids = [f"harness_isolated_{b}" for b in benchmarks_to_run]

    _restored = False
    _restored_session_count = 0
    for sid in _ckpt_ids:
        ckpt_path = output_dir / f"harness_{sid}.json"
        if ckpt_path.exists():
            store = HarnessStore.get_or_create(
                shuffle_mode=args.mode,
                benchmark_id=sid.replace("harness_isolated_", "") if args.mode == "isolated" else None,
            )
            store.load_checkpoint(str(ckpt_path))
            _restored_session_count = max(_restored_session_count, store.session_count)
            _restored = True
    if _restored:
        print(f"  ♻️  Restored harness store from checkpoint (session_count={_restored_session_count})")

    all_scores: list[float] = []
    bm_scores: defaultdict[str, list[float]] = defaultdict(list)
    session_index = 0

    if metrics_path.exists():
        kept_lines: list[str] = []
        with open(metrics_path, "r") as f:
            for line in f:
                if _restored and session_index >= _restored_session_count:
                    break
                rec = json.loads(line)
                all_scores.append(rec["score"])
                bm_scores[rec["benchmark_slug"]].append(rec["score"])
                kept_lines.append(line)
                session_index += 1
        with open(metrics_path, "w") as f:
            f.writelines(kept_lines)
        if session_index > 0:
            print(f"  ♻️  Restored {session_index} metrics records (cum_avg={sum(all_scores)/len(all_scores):.3f})")

    if args.mode in ("isolated", "sequential"):
        _completed_benchmarks: set[str] = set()
        if _restored and session_index > 0:
            _bm_counts: dict[str, int] = defaultdict(int)
            with open(metrics_path, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    _bm_counts[rec["benchmark_slug"]] += 1
            for bm_slug, task_ids in group_by_benchmark(task_order):
                if _bm_counts.get(bm_slug, 0) >= len(task_ids):
                    _completed_benchmarks.add(bm_slug)
            if _completed_benchmarks:
                print(f"  ⏭️  Skipping completed benchmarks: {sorted(_completed_benchmarks)}")

        for bm_slug, task_ids in group_by_benchmark(task_order):
            if bm_slug in _completed_benchmarks:
                continue

            print(f"\n{'=' * 60}")
            print(f"  {args.mode.upper()} — {bm_slug} ({len(task_ids)} tasks)")
            print(f"{'=' * 60}\n")

            bm_kwargs = configs[bm_slug]["bm_kwargs"]
            agent_kwargs = configs[bm_slug].get("agent_kwargs", {})

            benchmark = load_benchmark(bm_slug)(**bm_kwargs)
            agent = load_agent("harness")(
                model=args.model,
                evolver_model=args.model,
                shuffle_mode=args.mode,
                benchmark_id=bm_slug,
                embedding_model="all-MiniLM-L6-v2",
                runner="direct",
                model_settings=model_settings,
                **agent_kwargs,
            )

            results = evaluate(
                benchmark=benchmark,
                agent=agent,
                task_ids=task_ids,
                max_workers=1,
                output_dir=str(output_dir),
            )

            print(f"  {bm_slug} score={results.benchmark_score}")
            stores = HarnessStore.list_stores()
            store_key = (f"harness_isolated_{bm_slug}" if args.mode == "isolated"
                         else "harness_sequential_global")
            store = stores.get(store_key)
            if store:
                print(f"  harness: sessions={store.session_count}, "
                      f"skills={store.skill_count}")
                store.save_checkpoint(str(output_dir / f"harness_{store_key}.json"))

            for i, sr in enumerate(results.session_results):
                tid = task_ids[i] if i < len(task_ids) else sr.task_id or "?"
                rec = record_online_metrics(
                    metrics_path, session_index, bm_slug, tid, sr,
                    args.mode, args.seed, args.model,
                    all_scores, bm_scores,
                )
                print(f"    [{session_index}] {bm_slug}::{tid}  "
                      f"score={rec['score']:.2f}  cum={rec['cumulative_avg_score']:.3f}  "
                      f"steps={rec['steps']}")
                session_index += 1

    elif args.mode == "interleaved":
        for i, (bm_slug, task_id) in enumerate(task_order):
            if _restored and i < _restored_session_count:
                print(f"  ⏭️  Skipping Interleaved [{i+1}/{len(task_order)}] {bm_slug}::{task_id} (cached)")
                continue

            print(f"\n--- Interleaved [{i+1}/{len(task_order)}] {bm_slug}::{task_id} ---")

            bm_kwargs = configs[bm_slug]["bm_kwargs"]
            agent_kwargs = configs[bm_slug].get("agent_kwargs", {})

            benchmark = load_benchmark(bm_slug)(**bm_kwargs)
            agent = load_agent("harness")(
                model=args.model,
                evolver_model=args.model,
                shuffle_mode="interleaved",
                benchmark_id=bm_slug,
                embedding_model="all-MiniLM-L6-v2",
                runner="direct",
                model_settings=model_settings,
                **agent_kwargs,
            )

            results = evaluate(
                benchmark=benchmark,
                agent=agent,
                task_ids=[task_id],
                max_workers=1,
                output_dir=str(output_dir),
            )

            sr = results.session_results[0]

            rec = record_online_metrics(
                metrics_path, session_index, bm_slug, task_id, sr,
                "interleaved", args.seed, args.model,
                all_scores, bm_scores,
            )
            print(f"  score={rec['score']:.2f}  cum={rec['cumulative_avg_score']:.3f}  "
                  f"steps={rec['steps']}")

            stores = HarnessStore.list_stores()
            store = stores.get("harness_interleaved_global")
            if store:
                print(f"  harness: sessions={store.session_count}, "
                      f"skills={store.skill_count}")
                store.save_checkpoint(str(output_dir / "harness_harness_interleaved_global.json"))
            session_index += 1

    print(f"\n{'=' * 70}")
    print(f"  Final Summary")
    print(f"{'=' * 70}")
    if all_scores:
        print(f"  Overall avg score: {sum(all_scores)/len(all_scores):.3f}")
    for bm, scores in sorted(bm_scores.items()):
        print(f"  {bm:20s}: avg={sum(scores)/len(scores):.3f}  n={len(scores)}")

    stores = HarnessStore.list_stores()
    for store_id, store in stores.items():
        print(f"  harness[{store_id}]: sessions={store.session_count}, "
              f"skills={store.skill_count}")

    print(f"\n  Metrics: {metrics_path}")
    print(f"  Output:  {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Harness test-time learning experiment")
    parser.add_argument("--mode", required=True, choices=["isolated", "sequential", "interleaved"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, default=50, help="Tasks per benchmark")
    parser.add_argument("--model", default="openai/gpt-5.4")
    parser.add_argument("--benchmarks", default="browsecompplus,swebench,bfcl,tau2",
                        help="Comma-separated benchmark slugs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--reasoning-effort", default=None, help="Reasoning effort (low/medium/high)")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
