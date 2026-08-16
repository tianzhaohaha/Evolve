# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""
Seed-controlled task selection and ordering for evaluation experiments.

Guarantees:
  - Same seed → same task set for every benchmark, regardless of mode.
  - Within-benchmark task order is identical across isolated / sequential / interleaved.
  - Interleaved only interleaves *between* benchmarks; within-benchmark order is preserved.
"""

from __future__ import annotations

import hashlib
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

# Allow importing exgentic from the repo source tree
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from exgentic.interfaces.registry import load_benchmark


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def get_unified_task_order(
    benchmark_configs: dict[str, dict[str, Any]],
    num_tasks_per_benchmark: int,
    seed: int,
    mode: str,
) -> list[tuple[str, str]]:
    """Return a deterministic, mode-aware task ordering.

    Parameters
    ----------
    benchmark_configs : dict
        ``{slug: {"bm_kwargs": {...}, "agent_kwargs": {...}}}``
    num_tasks_per_benchmark : int
        How many tasks to select from each benchmark (e.g. 50).
    seed : int
        Ordering seed.  Controls within-benchmark task order and
        interleaved interleaving.  Task *selection* is always fixed at
        seed=42 so all experiments use the same task set.
    mode : str
        ``"isolated"`` | ``"sequential"`` | ``"interleaved"``.

    Returns
    -------
    list of (benchmark_slug, task_id)
        Ordered task sequence.  For isolated/sequential the tasks are grouped
        by benchmark (sorted alphabetically by slug).  For interleaved the
        tasks are interleaved across benchmarks while preserving
        within-benchmark order.
    """
    # Always use seed=42 for task SELECTION (which tasks to pick),
    # use the provided seed only for ORDERING (task sequence).
    _SELECTION_SEED = 42
    per_bm_tasks = _select_tasks(benchmark_configs, num_tasks_per_benchmark, _SELECTION_SEED)

    # Re-shuffle within-benchmark order using the provided seed
    if seed != _SELECTION_SEED:
        for slug in per_bm_tasks:
            order_seed = _derive_seed(seed, slug)
            rng = random.Random(order_seed)
            rng.shuffle(per_bm_tasks[slug])

    if mode in ("isolated", "sequential"):
        result: list[tuple[str, str]] = []
        for slug in sorted(per_bm_tasks):
            for tid in per_bm_tasks[slug]:
                result.append((slug, tid))
        return result

    if mode == "interleaved":
        return _interleave_preserving_order(per_bm_tasks, seed)

    raise ValueError(f"Unknown mode: {mode!r}")


def select_tasks_only(
    benchmark_configs: dict[str, dict[str, Any]],
    num_tasks_per_benchmark: int,
    seed: int,
) -> dict[str, list[str]]:
    """Return the selected tasks per benchmark (no ordering applied)."""
    return _select_tasks(benchmark_configs, num_tasks_per_benchmark, seed)


# ──────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────

def _select_tasks(
    benchmark_configs: dict[str, dict[str, Any]],
    num_tasks: int,
    seed: int,
) -> dict[str, list[str]]:
    """Select *num_tasks* tasks per benchmark using per-benchmark derived seeds."""
    per_bm: dict[str, list[str]] = {}
    for slug in sorted(benchmark_configs):
        bm_kwargs = benchmark_configs[slug].get("bm_kwargs", {})
        bm = load_benchmark(slug)(**bm_kwargs)
        evaluator = bm.get_evaluator()
        try:
            all_ids = [str(t) for t in evaluator.list_tasks()]
        finally:
            try:
                evaluator.close()
            except Exception:
                pass
            bm.close()

        bm_seed = _derive_seed(seed, slug)
        rng = random.Random(bm_seed)
        rng.shuffle(all_ids)
        per_bm[slug] = all_ids[:num_tasks]
    return per_bm


def _interleave_preserving_order(
    per_bm_tasks: dict[str, list[str]],
    seed: int,
) -> list[tuple[str, str]]:
    """Interleave tasks across benchmarks, preserving within-benchmark order.

    At each step, randomly pick a non-empty benchmark queue and pop
    its next task.  This ensures the relative order within each
    benchmark is the same as in isolated/sequential mode.
    """
    queues = {slug: deque(tasks) for slug, tasks in per_bm_tasks.items()}
    rng = random.Random(seed)
    result: list[tuple[str, str]] = []
    while any(queues.values()):
        available = sorted(s for s, q in queues.items() if q)
        slug = rng.choice(available)
        result.append((slug, queues[slug].popleft()))
    return result


def _derive_seed(master_seed: int, slug: str) -> int:
    """Derive a deterministic per-benchmark seed from master seed + slug."""
    h = hashlib.md5(f"{master_seed}_{slug}".encode()).hexdigest()
    return int(h, 16) % (2**31)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def group_by_benchmark(
    task_order: list[tuple[str, str]],
) -> list[tuple[str, list[str]]]:
    """Group a task order list into (slug, [task_ids]) preserving order."""
    groups: list[tuple[str, list[str]]] = []
    current_slug: str | None = None
    current_ids: list[str] = []
    for slug, tid in task_order:
        if slug != current_slug:
            if current_slug is not None:
                groups.append((current_slug, current_ids))
            current_slug = slug
            current_ids = [tid]
        else:
            current_ids.append(tid)
    if current_slug is not None:
        groups.append((current_slug, current_ids))
    return groups


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick sanity check without requiring benchmark data
    print("=== task_ordering.py self-test ===\n")

    # Simulate with fake data
    fake_per_bm = {
        "bfcl": ["b1", "b2", "b3", "b4", "b5"],
        "tau2": ["t1", "t2", "t3", "t4", "t5"],
        "browsecompplus": ["c1", "c2", "c3", "c4", "c5"],
    }

    # Test interleave preserving order
    interleaved = _interleave_preserving_order(fake_per_bm, seed=42)
    print("Interleaved interleave (seed=42):")
    for slug, tid in interleaved:
        print(f"  {slug}: {tid}")

    # Verify within-benchmark order is preserved
    for slug in fake_per_bm:
        original = fake_per_bm[slug]
        fused = [tid for s, tid in interleaved if s == slug]
        assert fused == original, f"{slug}: order changed! {original} → {fused}"
        print(f"  ✓ {slug} order preserved: {fused}")

    # Verify different seeds produce different interleaving
    interleaved2 = _interleave_preserving_order(fake_per_bm, seed=123)
    order1 = [(s, t) for s, t in interleaved]
    order2 = [(s, t) for s, t in interleaved2]
    print(f"\n  seed=42 vs seed=123 differ: {order1 != order2}")

    print("\nAll checks passed.")
