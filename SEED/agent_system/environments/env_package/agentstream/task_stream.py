# Copyright 2026 SEED x AgentStream integration.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic task-stream construction and batch-granular scheduling.

The ordering logic is a faithful port of
``AgentStream/exgentic/scripts/utils/task_ordering.py`` (standard library
only), so streams produced here match the AgentStream paper protocol:

  * task *selection* is always performed at seed 42 -> identical task sets
    across modes and stream seeds;
  * within-benchmark task order is identical across isolated / sequential /
    interleaved for a given stream seed;
  * interleaved only interleaves *between* benchmarks, preserving the
    within-benchmark order.

On top of that, this module adds what SEED training needs and AgentStream's
test-time harness does not have:

  * a train/eval split (``protocol='split'``) with a held-out validation set
    drawn from the seed-42 shuffled remainder of each benchmark;
  * a batch-granular cursor: SEED consumes ``env_num`` tasks per RL step
    (each replicated ``group_n`` times by the caller for GRPO), while
    AgentStream streams tasks one by one;
  * pass tracking so the online protocol can distinguish first-encounter
    (online metric) episodes from repeat passes;
  * a ``random`` mode that replicates SEED's original per-reset uniform
    sampling, keeping the original SEED data-processing logic available for
    like-for-like comparisons.

This module is dependency-free (numpy optional) and unit-testable on a
machine without benchmark environments.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

if TYPE_CHECKING:  # annotation-only; keeps this module runnable standalone
    from .as_config import AgentStreamConfig

# AgentStream fixes task *selection* at seed 42 (scripts/utils/task_ordering.py);
# mirrored here so selected task sets match the paper protocol exactly.
AGENTSTREAM_SELECTION_SEED = 42

TaskRef = Tuple[str, str]  # (benchmark_slug, task_id)


# ---------------------------------------------------------------------------
# Ordering primitives (ported from AgentStream scripts/utils/task_ordering.py)
# ---------------------------------------------------------------------------

def derive_seed(master_seed: int, slug: str) -> int:
    """Deterministic per-benchmark seed from master seed + slug (verbatim port)."""
    h = hashlib.md5(f"{master_seed}_{slug}".encode()).hexdigest()
    return int(h, 16) % (2 ** 31)


def select_tasks(
    task_universe: Dict[str, Sequence[str]],
    num_tasks: int,
    seed: int = AGENTSTREAM_SELECTION_SEED,
) -> Dict[str, List[str]]:
    """Select ``num_tasks`` per benchmark with per-benchmark derived seeds.

    Matches AgentStream ``_select_tasks``: shuffle the full id list with the
    derived seed, take the first ``num_tasks``.
    """
    per_bm: Dict[str, List[str]] = {}
    for slug in sorted(task_universe):
        all_ids = [str(t) for t in task_universe[slug]]
        rng = random.Random(derive_seed(seed, slug))
        rng.shuffle(all_ids)
        per_bm[slug] = all_ids[:num_tasks]
    return per_bm


def select_holdout_tasks(
    task_universe: Dict[str, Sequence[str]],
    num_stream_tasks: int,
    num_val_tasks: int,
    seed: int = AGENTSTREAM_SELECTION_SEED,
) -> Dict[str, List[str]]:
    """Held-out validation tasks: the *next* ``num_val_tasks`` ids after the
    stream selection in the same seed-42 shuffle, guaranteeing disjointness."""
    per_bm: Dict[str, List[str]] = {}
    for slug in sorted(task_universe):
        all_ids = [str(t) for t in task_universe[slug]]
        rng = random.Random(derive_seed(seed, slug))
        rng.shuffle(all_ids)
        holdout = all_ids[num_stream_tasks:num_stream_tasks + num_val_tasks]
        if not holdout:
            # Benchmark too small for a disjoint holdout; fall back to the
            # tail of the stream set (flagged so callers can warn).
            holdout = all_ids[-num_val_tasks:]
        per_bm[slug] = holdout
    return per_bm


def order_tasks(
    per_bm_tasks: Dict[str, List[str]],
    mode: str,
    seed: int,
) -> List[TaskRef]:
    """Mode-aware ordering (verbatim semantics of ``get_unified_task_order``).

    ``seed`` reshuffles the within-benchmark order (unless it equals the
    selection seed) and drives the interleaving pattern.
    """
    per_bm = {slug: list(ids) for slug, ids in per_bm_tasks.items()}

    if seed != AGENTSTREAM_SELECTION_SEED:
        for slug in per_bm:
            rng = random.Random(derive_seed(seed, slug))
            rng.shuffle(per_bm[slug])

    if mode in ("isolated", "sequential"):
        result: List[TaskRef] = []
        for slug in sorted(per_bm):
            result.extend((slug, tid) for tid in per_bm[slug])
        return result

    if mode == "interleaved":
        return _interleave_preserving_order(per_bm, seed)

    raise ValueError(f"Unknown ordering mode: {mode!r}")


def _interleave_preserving_order(
    per_bm_tasks: Dict[str, List[str]],
    seed: int,
) -> List[TaskRef]:
    """Interleave across benchmarks, preserving within-benchmark order."""
    queues = {slug: deque(tasks) for slug, tasks in per_bm_tasks.items()}
    rng = random.Random(seed)
    result: List[TaskRef] = []
    while any(queues.values()):
        available = sorted(s for s, q in queues.items() if q)
        slug = rng.choice(available)
        result.append((slug, queues[slug].popleft()))
    return result


def expand_block_passes(ordered: List[TaskRef], block_passes: int) -> List[TaskRef]:
    """Repeat each contiguous same-benchmark block ``block_passes`` times.

    Only meaningful for sequential/isolated streams; a no-op for
    ``block_passes == 1``. Keeps within-block order on every pass.
    """
    if block_passes <= 1 or not ordered:
        return list(ordered)
    result: List[TaskRef] = []
    block: List[TaskRef] = [ordered[0]]
    for ref in ordered[1:]:
        if ref[0] == block[-1][0]:
            block.append(ref)
        else:
            result.extend(block * block_passes)
            block = [ref]
    result.extend(block * block_passes)
    return result


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class StreamBatch:
    """One reset()'s worth of tasks (before group_n replication)."""

    refs: List[TaskRef]
    # Global position of each task in the ordered stream (-1 for random mode).
    stream_indices: List[int]
    # Pass number of each task (0 = first encounter -> counts toward online metrics).
    passes: List[int]

    def replicate(self, group_n: int) -> "StreamBatch":
        refs: List[TaskRef] = []
        idx: List[int] = []
        passes: List[int] = []
        for r, i, p in zip(self.refs, self.stream_indices, self.passes):
            refs.extend([r] * group_n)
            idx.extend([i] * group_n)
            passes.extend([p] * group_n)
        return StreamBatch(refs=refs, stream_indices=idx, passes=passes)


@dataclass
class TaskStreamScheduler:
    """Serves per-reset task batches according to the configured stream.

    Parameters
    ----------
    train_tasks : per-benchmark stream task ids (already selected at seed 42).
    mode : 'random' | 'isolated' | 'sequential' | 'interleaved'.
    stream_seed : ordering seed (AgentStream's run seed).
    block_passes : sequential/isolated block repetition factor.
    on_exhausted : 'cycle' | 'stop'.
    sample_seed : RNG seed for 'random' mode (SEED-compatible sampling).
    """

    train_tasks: Dict[str, List[str]]
    mode: str = "random"
    stream_seed: int = 44
    block_passes: int = 1
    on_exhausted: str = "cycle"
    sample_seed: int = 0

    _ordered: List[TaskRef] = field(init=False, default_factory=list)
    _cursor: int = field(init=False, default=0)
    _pass: int = field(init=False, default=0)
    _rng: random.Random = field(init=False, default=None)  # type: ignore[assignment]
    _seen: set = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.sample_seed)
        if self.mode != "random":
            ordered = order_tasks(self.train_tasks, self.mode, self.stream_seed)
            self._ordered = expand_block_passes(ordered, self.block_passes)
            if not self._ordered:
                raise ValueError("Task stream is empty; check benchmark task lists.")

    # -- introspection -----------------------------------------------------

    @property
    def stream_length(self) -> int:
        return len(self._ordered)

    @property
    def all_refs(self) -> List[TaskRef]:
        if self.mode == "random":
            return [(slug, tid) for slug in sorted(self.train_tasks) for tid in self.train_tasks[slug]]
        return list(self._ordered)

    def state_dict(self) -> Dict[str, int]:
        return {"cursor": self._cursor, "pass": self._pass}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self._cursor = int(state.get("cursor", 0))
        self._pass = int(state.get("pass", 0))

    # -- batch serving --------------------------------------------------------

    def next_batch(self, env_num: int) -> StreamBatch:
        if self.mode == "random":
            return self._next_random_batch(env_num)
        return self._next_stream_batch(env_num)

    def _next_random_batch(self, env_num: int) -> StreamBatch:
        """SEED-original behavior: uniform sample without replacement per reset."""
        pool = self.all_refs
        if env_num <= len(pool):
            refs = self._rng.sample(pool, env_num)
        else:
            refs = [self._rng.choice(pool) for _ in range(env_num)]
        return StreamBatch(refs=refs, stream_indices=[-1] * env_num, passes=[0] * env_num)

    def _next_stream_batch(self, env_num: int) -> StreamBatch:
        refs: List[TaskRef] = []
        idx: List[int] = []
        passes: List[int] = []
        for _ in range(env_num):
            if self._cursor >= len(self._ordered):
                if self.on_exhausted == "stop":
                    self._cursor = len(self._ordered) - env_num
                    self._cursor = max(0, self._cursor)
                else:  # cycle
                    self._cursor = 0
                    self._pass += 1
            position = self._cursor
            refs.append(self._ordered[position])
            idx.append(position + self._pass * len(self._ordered))
            passes.append(self._pass)
            self._cursor += 1
        return StreamBatch(refs=refs, stream_indices=idx, passes=passes)


@dataclass
class ValTaskCycler:
    """Fixed validation task set served deterministically, batch by batch.

    Cycles the per-benchmark validation tasks in a fixed round-robin order so
    every val pass covers the same task mix; with val_batch_size >= total task
    count each validation covers every task at least once, making the
    per-benchmark success curves comparable across trainer.test_freq points
    (forgetting / transfer measurement).
    """

    val_tasks: Dict[str, List[str]]
    _flat: List[TaskRef] = field(init=False, default_factory=list)
    _cursor: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # Round-robin across benchmarks for balanced coverage in partial batches.
        queues = {slug: deque(ids) for slug, ids in sorted(self.val_tasks.items()) if ids}
        while any(queues.values()):
            for slug in sorted(queues):
                if queues[slug]:
                    self._flat.append((slug, queues[slug].popleft()))
        if not self._flat:
            raise ValueError("Validation task set is empty.")

    @property
    def total(self) -> int:
        return len(self._flat)

    def next_batch(self, env_num: int) -> StreamBatch:
        refs: List[TaskRef] = []
        for _ in range(env_num):
            refs.append(self._flat[self._cursor % len(self._flat)])
            self._cursor += 1
        return StreamBatch(refs=refs, stream_indices=[-1] * env_num, passes=[0] * env_num)

    def reset_cursor(self) -> None:
        self._cursor = 0


# ---------------------------------------------------------------------------
# Split construction from config
# ---------------------------------------------------------------------------

def build_splits(
    task_universe: Dict[str, Sequence[str]],
    cfg: "AgentStreamConfig",
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (train_tasks, val_tasks) per benchmark for the configured protocol.

    split protocol:
        train = seed-42 stream selection (num_tasks_per_benchmark)
        val   = next val_tasks_per_benchmark ids of the same shuffle (disjoint)
    online protocol:
        train = seed-42 stream selection (the stream is the evaluation)
        val   = per val_source: holdout | stream (retention) | both
    """
    stream_set = select_tasks(task_universe, cfg.num_tasks_per_benchmark)
    holdout = select_holdout_tasks(
        task_universe, cfg.num_tasks_per_benchmark, cfg.val_tasks_per_benchmark
    )

    if cfg.protocol == "split":
        return stream_set, holdout

    # online protocol
    if cfg.val_source == "holdout":
        val = holdout
    elif cfg.val_source == "stream":
        val = {slug: list(ids) for slug, ids in stream_set.items()}
    else:  # both
        val = {
            slug: list(stream_set.get(slug, [])) + list(holdout.get(slug, []))
            for slug in sorted(set(stream_set) | set(holdout))
        }
    return stream_set, val


if __name__ == "__main__":
    # Pure-logic self-test (no benchmark data needed).
    universe = {
        "bfcl": [f"b{i}" for i in range(100)],
        "tau2": [f"t{i}" for i in range(100)],
        "appworld": [f"a{i}" for i in range(100)],
    }

    print("=== selection determinism ===")
    s1 = select_tasks(universe, 10)
    s2 = select_tasks(universe, 10)
    assert s1 == s2
    print({k: v[:3] for k, v in s1.items()})

    print("\n=== split disjointness ===")
    hold = select_holdout_tasks(universe, 10, 5)
    for slug in universe:
        assert not (set(s1[slug]) & set(hold[slug])), slug
    print("disjoint OK")

    print("\n=== ordering invariants ===")
    seq = order_tasks(s1, "sequential", seed=44)
    inter = order_tasks(s1, "interleaved", seed=44)
    assert sorted(seq) == sorted(inter)
    for slug in universe:
        seq_order = [t for s, t in seq if s == slug]
        inter_order = [t for s, t in inter if s == slug]
        assert seq_order == inter_order, f"within-benchmark order broken for {slug}"
    print("within-benchmark order preserved across modes OK")

    print("\n=== scheduler cursor/passes ===")
    sched = TaskStreamScheduler(train_tasks=s1, mode="interleaved", stream_seed=44)
    n = sched.stream_length
    seen = []
    b = sched.next_batch(16)
    seen.extend(b.refs)
    assert b.passes == [0] * 16
    for _ in range((n // 16)):
        b = sched.next_batch(16)
        seen.extend(b.refs)
    assert max(b.passes) == 1  # wrapped into pass 1
    assert set(seen[:n]) == set(inter)
    print(f"stream length={n}, wrap-around pass tracking OK")

    print("\n=== block passes ===")
    blocks = expand_block_passes(order_tasks(s1, "sequential", 44), 2)
    assert len(blocks) == 2 * len(seq)
    print("block repetition OK")

    print("\n=== random mode (SEED-compatible) ===")
    rand_sched = TaskStreamScheduler(train_tasks=s1, mode="random", sample_seed=7)
    rb = rand_sched.next_batch(8)
    assert len(rb.refs) == 8 and rb.stream_indices == [-1] * 8
    grouped = rb.replicate(4)
    assert len(grouped.refs) == 32 and grouped.refs[0] == grouped.refs[3]
    print("random sampling + group replication OK")

    print("\nAll task_stream self-tests passed.")
