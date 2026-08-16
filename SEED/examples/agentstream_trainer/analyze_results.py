#!/usr/bin/env python3
"""Compare stream-mode runs from agentstream_online_metrics.jsonl files.

Works for both AgentStream-suite runs and alfworld_stream runs (same schema).
Only first-pass episodes count toward online metrics, matching the AgentStream
protocol. GRPO group rollouts of the same task are averaged per task first, so
group_n does not bias the online success rate.

Usage:
  python analyze_results.py run1/agentstream_online_metrics.jsonl [run2/... ...]
  python analyze_results.py <files...> --csv combined.csv   # long-format export
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_run(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(path: Path, rows: list[dict]) -> dict:
    first = [r for r in rows if r.get("first_pass", True)]
    # Average GRPO group rollouts per (benchmark, task) first.
    per_task: dict[tuple, list[float]] = defaultdict(list)
    for r in first:
        per_task[(r["benchmark_slug"], r["task_id"])].append(float(r.get("score", 0.0)))
    task_scores = {k: sum(v) / len(v) for k, v in per_task.items()}

    per_bm: dict[str, list[float]] = defaultdict(list)
    for (slug, _tid), score in task_scores.items():
        per_bm[slug].append(score)

    meta = rows[0] if rows else {}
    return {
        "run": path.parent.name or str(path),
        "experiment": meta.get("experiment", ""),
        "protocol": meta.get("protocol", ""),
        "stream_mode": meta.get("stream_mode", ""),
        "stream_seed": meta.get("stream_seed", ""),
        "episodes": len(rows),
        "first_pass_tasks": len(task_scores),
        "online_avg_score": (
            sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
        ),
        "per_benchmark": {
            slug: (sum(v) / len(v), len(v)) for slug, v in sorted(per_bm.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, default=None, help="export per-episode long-format CSV")
    args = parser.parse_args()

    summaries = []
    all_rows: list[dict] = []
    for path in args.files:
        if not path.exists():
            print(f"warning: {path} not found, skipped", file=sys.stderr)
            continue
        rows = load_run(path)
        if not rows:
            print(f"warning: {path} is empty, skipped", file=sys.stderr)
            continue
        summaries.append(summarize(path, rows))
        for r in rows:
            all_rows.append({"source": str(path), **r})

    if not summaries:
        print("No data.", file=sys.stderr)
        return 1

    header = f"{'run':<40} {'mode':<12} {'protocol':<8} {'seed':<5} {'tasks':<6} {'online_avg':<10}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['run'][:39]:<40} {s['stream_mode']:<12} {s['protocol']:<8} "
            f"{str(s['stream_seed']):<5} {s['first_pass_tasks']:<6} {s['online_avg_score']:<10.4f}"
        )
        for slug, (avg, n) in s["per_benchmark"].items():
            print(f"    {slug:<28} avg={avg:.4f}  (n={n})")

    if args.csv:
        keys = sorted({k for r in all_rows for k in r})
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nExported {len(all_rows)} episode rows to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
