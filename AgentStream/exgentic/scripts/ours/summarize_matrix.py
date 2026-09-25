# SPDX-License-Identifier: Apache-2.0
"""Table of the AgentStream-baseline matrix from the local run directories (no wandb needed):
online cumulative score / success (overall and per benchmark), holdout success, gain over the
same-model tool_calling reference, tokens per task.

    python scripts/ours/summarize_matrix.py [outputs_ours] [--csv table.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BENCHMARKS = ("bfcl", "tau2", "browsecompplus")


def load_run(run_dir: Path) -> dict | None:
    cfg_path, online_path = run_dir / "experiment_config.json", run_dir / "online_metrics.jsonl"
    if not (cfg_path.exists() and online_path.exists()):
        return None
    cfg = json.loads(cfg_path.read_text())
    rows = [json.loads(l) for l in online_path.read_text().splitlines()]
    hold = json.loads((run_dir / "holdout_summary.json").read_text()) if (run_dir / "holdout_summary.json").exists() else {}
    out = {"run": run_dir.name, "agent": cfg["agent"], "model": cfg["model"].split("/")[-1], "mode": cfg["mode"], "tasks": len(rows),
           "online_score": sum(r["score"] for r in rows) / len(rows) if rows else float("nan"),
           "online_success": sum(r["success"] for r in rows) / len(rows) if rows else float("nan"),
           "tokens_per_task": sum(r["input_tokens"] + r["output_tokens"] for r in rows) / len(rows) if rows else float("nan"),
           "val_success": hold.get("val/success_rate", float("nan"))}
    for bm in BENCHMARKS:
        sub = [r for r in rows if r["benchmark_slug"] == bm]
        out[f"online_{bm}"] = sum(r["score"] for r in sub) / len(sub) if sub else float("nan")
        out[f"val_{bm}"] = hold.get(f"val/{bm}_success_rate", float("nan"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="outputs_ours")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    runs = [r for r in (load_run(d) for d in sorted(Path(args.root).iterdir()) if d.is_dir()) if r]
    ref = {(r["model"], r["mode"]): r for r in runs if r["agent"] == "tool_calling"}
    for r in runs:
        base = ref.get((r["model"], r["mode"]))
        r["gain_vs_ref"] = r["online_score"] - base["online_score"] if base else float("nan")
    cols = ["agent", "model", "mode", "tasks", "online_score", "online_success", "gain_vs_ref", *[f"online_{b}" for b in BENCHMARKS],
            "val_success", *[f"val_{b}" for b in BENCHMARKS], "tokens_per_task"]
    print(" | ".join(f"{c:>14s}" for c in cols))
    for r in sorted(runs, key=lambda r: (r["model"], r["mode"], r["agent"])):
        print(" | ".join(f"{r[c]:>14.3f}" if isinstance(r[c], float) else f"{str(r[c]):>14s}" for c in cols))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run", *cols]); w.writeheader(); w.writerows({k: r[k] for k in ["run", *cols]} for r in runs)


if __name__ == "__main__":
    main()
