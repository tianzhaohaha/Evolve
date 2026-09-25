# SPDX-License-Identifier: Apache-2.0
"""Real-time wandb logging for AgentStream baseline runs, with the SEED RL runs' metric names so
both families plot on the same panels (``online/*`` per task, ``val/*`` for the holdout).

``WandbStream`` is a no-op when wandb is unavailable or ``enabled=False``; the run id is kept in
``<output_dir>/wandb_run.json`` so a restarted process continues the same curve.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

TASKS_PER_RL_STEP = 10  # the RL runs train on 10 tasks per step: online/stream_step aligns the x-axes


class OnlineTally:
    """Cumulative score / success, overall and per benchmark, in task order (SEED ``_Tally``)."""

    def __init__(self) -> None:
        self.scores: list[float] = []
        self.successes: list[float] = []
        self.by_bm: Dict[str, list[tuple[float, float]]] = defaultdict(list)

    def add(self, slug: str, score: float, success: bool) -> None:
        self.scores.append(float(score))
        self.successes.append(1.0 if success else 0.0)
        self.by_bm[slug].append((float(score), 1.0 if success else 0.0))

    @property
    def count(self) -> int:
        return len(self.scores)

    def metrics(self, prefix: str = "online/") -> Dict[str, float]:
        out = {
            f"{prefix}cumulative_avg_score": sum(self.scores) / len(self.scores) if self.scores else 0.0,
            f"{prefix}cumulative_success_rate": sum(self.successes) / len(self.successes) if self.successes else 0.0,
            f"{prefix}tasks_done": float(len(self.scores)),
        }
        for slug, rows in self.by_bm.items():
            out[f"{prefix}{slug}/cumulative_avg_score"] = sum(s for s, _ in rows) / len(rows)
            out[f"{prefix}{slug}/cumulative_success_rate"] = sum(w for _, w in rows) / len(rows)
        return out


def holdout_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """``val/*`` metrics of a frozen holdout pass (SEED validation names)."""
    by_bm: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_bm[rec["benchmark_slug"]].append(rec)
    rows = [r for rs in by_bm.values() for r in rs]
    if not rows:
        return {}
    out = {
        "val/success_rate": sum(1.0 if r["success"] else 0.0 for r in rows) / len(rows),
        "val/score": sum(float(r["score"]) for r in rows) / len(rows),
        "val/env_error_rate": sum(1.0 if r.get("status") == "error" else 0.0 for r in rows) / len(rows),
        "val/tasks": float(len(rows)),
    }
    for slug, rs in by_bm.items():
        out[f"val/{slug}_success_rate"] = sum(1.0 if r["success"] else 0.0 for r in rs) / len(rs)
        out[f"val/{slug}_score"] = sum(float(r["score"]) for r in rs) / len(rs)
    return out


class WandbStream:
    """One wandb run per (agent, model, mode); ``log_task`` per finished stream task, ``log_holdout`` once."""

    def __init__(self, run_name: str, config: Dict[str, Any], output_dir: Path, *, enabled: bool = True,
                 project: str = "agentic_agentstream", entity: Optional[str] = None) -> None:
        self.run = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError:
            print("wandb not installed; metrics stay in the local jsonl files only.")
            return
        id_path = Path(output_dir) / "wandb_run.json"
        run_id = json.loads(id_path.read_text())["id"] if id_path.exists() else None
        self.run = wandb.init(
            project=project, entity=entity, name=run_name, config=config, dir=str(output_dir),
            id=run_id, resume="allow" if run_id else None,
        )
        id_path.write_text(json.dumps({"id": self.run.id, "name": run_name}))
        self.run.define_metric("online/*", step_metric="online/task_index")
        self.run.define_metric("val/*", step_metric="online/task_index")

    def log_task(self, record: Dict[str, Any], tally: OnlineTally) -> None:
        if self.run is None:
            return
        index = int(record["session_index"]) + 1
        payload = dict(tally.metrics())
        payload.update({
            "online/task_index": index,
            "online/stream_step": (index - 1) // TASKS_PER_RL_STEP + 1,
            "online/task_score": float(record["score"]),
            "online/task_success": 1.0 if record["success"] else 0.0,
            "online/steps": float(record.get("steps") or 0),
            "online/input_tokens": float(record.get("input_tokens") or 0),
            "online/output_tokens": float(record.get("output_tokens") or 0),
            "online/memory_entries": float(record.get("memory_entries") or 0),
            "online/memory_tokens": float(record.get("memory_tokens") or 0),
        })
        self.run.log(payload, step=index)

    def log_holdout(self, summary: Dict[str, float], task_index: int) -> None:
        """Holdout of the memory frozen after ``task_index`` stream tasks; logged one wandb step
        later than that task so a resumed run never writes to an already committed step."""
        if self.run is None or not summary:
            return
        self.run.log({**summary, "online/task_index": task_index}, step=task_index + 1)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
