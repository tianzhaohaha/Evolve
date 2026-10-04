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

"""Online (AgentStream-protocol) episode metrics recorder.

Mirrors ``record_online_metrics`` in
AgentStream/exgentic/scripts/*/run_experiment.py: one JSONL row per finished
episode with cumulative averages, so results are directly comparable with
AgentStream harness outputs. Rows for repeat passes (pass_idx > 0) are
recorded with ``first_pass=false`` and excluded from the cumulative averages,
matching the single-pass semantics of the AgentStream online protocol. With
``track_repeat_passes`` enabled (multi-pass streams) the episodes of every
pass are additionally pooled into one ``multipass/...`` family (see
``snapshot()``): cumulative averages over all passes plus a sliding window over
the most recent stream cycle. The first-pass ``online/...`` metrics are
unaffected by the switch.

Two estimators of the per-pass score are maintained side by side:

* ``cumulative_avg_score`` — every first-pass attempt counts. With
  ``env.rollout.n = group_n`` copies per task this is a mean@group_n estimate.
* ``cumulative_avg_score_single`` — only the first copy of each task group
  (``rollout_slot % group_n == 0``; ``StreamBatch.replicate`` lays copies out
  task-major, so that is exactly one attempt per task). This matches the
  one-attempt-per-task protocol of the AgentStream baselines.

``snapshot()`` exposes both (globally and per benchmark) as trainer metrics so
they can be pushed to wandb every step under the ``online/`` prefix.

Episodes the environment itself broke (``env_error``: reset failure or worker
timeout) are still written to the JSONL for auditing but never enter the
averages; ``online/env_error_episodes`` (and the per-benchmark variant) counts
them so runs remain comparable.

Resume semantics (``restore_up_to_step``): when the trainer resumes from a
checkpoint the accumulators are rebuilt from the existing JSONL, taking only
rows written at or before the checkpointed global step (later rows belong to
the crashed window and will be replayed). When the run starts from scratch
nothing is restored: an existing JSONL is left untouched and simply appended
to, exactly as before. Accumulated rows are additionally de-duplicated on
``(benchmark, task, pass, rollout_slot)`` as a safety net against replays.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class _CumulativeStats:
    """Running score/success sums, overall and per benchmark."""

    __slots__ = ("sum_scores", "sum_success", "num", "bm_sum", "bm_success", "bm_num")

    def __init__(self) -> None:
        self.sum_scores = 0.0
        self.sum_success = 0.0
        self.num = 0
        self.bm_sum: Dict[str, float] = defaultdict(float)
        self.bm_success: Dict[str, float] = defaultdict(float)
        self.bm_num: Dict[str, int] = defaultdict(int)

    def add(self, slug: str, score: float, success: bool) -> None:
        self.sum_scores += score
        self.sum_success += float(success)
        self.num += 1
        self.bm_sum[slug] += score
        self.bm_success[slug] += float(success)
        self.bm_num[slug] += 1

    def avg_score(self, slug: Optional[str] = None) -> float:
        if slug is None:
            return self.sum_scores / self.num if self.num else 0.0
        n = self.bm_num.get(slug, 0)
        return self.bm_sum[slug] / n if n else 0.0

    def emit(
        self, out: Dict[str, float], prefix: str, n_key: str, slug: Optional[str] = None
    ) -> None:
        """Write cumulative averages under ``prefix`` into ``out`` (no-op when empty)."""
        if slug is None:
            scores, success, n = self.sum_scores, self.sum_success, self.num
        else:
            scores = self.bm_sum.get(slug, 0.0)
            success = self.bm_success.get(slug, 0.0)
            n = self.bm_num.get(slug, 0)
        if n <= 0:
            return
        out[f"{prefix}cumulative_avg_score"] = scores / n
        out[f"{prefix}cumulative_success_rate"] = success / n
        out[f"{prefix}{n_key}"] = float(n)


class OnlineMetricsRecorder:
    def __init__(
        self,
        path: str,
        run_meta: Optional[Dict[str, Any]] = None,
        group_n: int = 1,
        restore_up_to_step: Optional[int] = None,
        track_repeat_passes: bool = False,
        window_episodes: int = 0,
    ) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.run_meta = dict(run_meta or {})
        self.group_n = max(int(group_n), 1)
        self.track_repeat_passes = bool(track_repeat_passes)
        self._lock = threading.Lock()
        self._episode_counter = 0

        # Each accumulator pair keeps the two estimators described above:
        # [0] all attempts (group_n per task), [1] one attempt per task (first
        # copy of each group). ``_first`` is the AgentStream online metric (pass
        # 0 only); ``_pooled`` covers every pass (``multipass/*``) and is only
        # fed when ``track_repeat_passes`` is on, as is the window, which holds
        # the most recent ``window_episodes`` episodes (<= 0 disables it) as
        # (score, success).
        self._first: Tuple[_CumulativeStats, _CumulativeStats] = (
            _CumulativeStats(),
            _CumulativeStats(),
        )
        self._pooled: Tuple[_CumulativeStats, _CumulativeStats] = (
            _CumulativeStats(),
            _CumulativeStats(),
        )
        self._window: Optional[Deque[Tuple[float, bool]]] = (
            deque(maxlen=int(window_episodes)) if int(window_episodes) > 0 else None
        )

        self._env_errors: Dict[str, int] = defaultdict(int)  # per benchmark
        self._seen: Set[Tuple[str, str, int, int]] = set()
        if restore_up_to_step is not None:
            self._restore_from_file(int(restore_up_to_step))
        elif os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            logger.info(
                "Online metrics file %s already exists; starting cumulative averages fresh "
                "(training is not resuming) and appending to it.",
                self.path,
            )

    # ------------------------------------------------------------------ core
    def _accumulate(
        self,
        *,
        slug: str,
        task_id: str,
        pass_idx: int,
        rollout_slot: int,
        score: float,
        success: bool,
        env_error: bool = False,
    ) -> bool:
        """Fold one episode into the first-pass and/or pooled accumulators.

        Returns False (and changes nothing) for a duplicate key, which only
        happens when steps are replayed after a resume. Environment failures
        are counted per benchmark instead of averaged.
        """
        key = (slug, str(task_id), int(pass_idx), int(rollout_slot))
        if key in self._seen:
            return False
        self._seen.add(key)

        if env_error:
            self._env_errors[slug] += 1
            return True
        first_attempt = int(rollout_slot) % self.group_n == 0
        targets = [self._first] if int(pass_idx) == 0 else []
        if self.track_repeat_passes:
            targets.append(self._pooled)
            if self._window is not None:
                self._window.append((score, success))
        for stats_all, stats_single in targets:
            stats_all.add(slug, score, success)
            if first_attempt:
                stats_single.add(slug, score, success)
        return True

    def _restore_from_file(self, up_to_step: int) -> None:
        if not os.path.exists(self.path):
            return
        restored = 0
        skipped_after_ckpt = 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            continue
                        self._episode_counter = max(
                            self._episode_counter, int(row.get("episode_index", 0) or 0)
                        )
                        pass_idx = int(row.get("pass_idx", 0) or 0)
                        if pass_idx > 0 and not self.track_repeat_passes:
                            continue
                        row_step = row.get("global_step")
                        if row_step is not None and int(row_step) > up_to_step:
                            skipped_after_ckpt += 1
                            continue
                        if self._accumulate(
                            slug=str(row.get("benchmark_slug", "")),
                            task_id=str(row.get("task_id", "")),
                            pass_idx=pass_idx,
                            rollout_slot=int(row.get("rollout_slot", 0) or 0),
                            score=float(row.get("score", 0.0) or 0.0),
                            success=bool(row.get("success", False)),
                            env_error=bool(row.get("env_error", False)),
                        ):
                            restored += 1
                    except (ValueError, TypeError):
                        continue  # malformed row: skip it, keep restoring
        except Exception as exc:  # best effort: never block training on metrics
            logger.warning("Could not restore online metrics from %s: %s", self.path, exc)
            return
        if restored or skipped_after_ckpt:
            logger.info(
                "Restored %d episodes (<= step %d) from %s; skipped %d rows "
                "after the checkpoint (first-pass cumulative_avg_score=%.4f)",
                restored,
                up_to_step,
                self.path,
                skipped_after_ckpt,
                self._first[0].avg_score(),
            )

    # --------------------------------------------------------------- public
    def record_episode(
        self,
        *,
        slug: str,
        task_id: str,
        stream_index: int,
        pass_idx: int,
        rollout_slot: int,
        success: bool,
        score: float,
        episode_steps: int,
        global_step: Optional[int] = None,
        action_stats: Optional[Dict[str, int]] = None,
        env_error: bool = False,
    ) -> None:
        first_pass = pass_idx == 0
        first_attempt = int(rollout_slot) % self.group_n == 0
        with self._lock:
            self._episode_counter += 1
            if first_pass or self.track_repeat_passes:
                self._accumulate(
                    slug=slug,
                    task_id=task_id,
                    pass_idx=pass_idx,
                    rollout_slot=rollout_slot,
                    score=float(score),
                    success=bool(success),
                    env_error=bool(env_error),
                )

            # Row-level cumulative fields keep their historical meaning: the
            # first-pass (AgentStream online) estimate, whatever pass the row
            # itself belongs to.
            fp_all, fp_single = self._first
            row = {
                **self.run_meta,
                "episode_index": self._episode_counter,
                "global_step": global_step,
                "benchmark_slug": slug,
                "task_id": task_id,
                "stream_index": stream_index,
                "pass_idx": pass_idx,
                "first_pass": first_pass,
                "first_attempt": first_attempt,
                "rollout_slot": rollout_slot,
                "success": bool(success),
                "score": float(score),
                "env_error": bool(env_error),
                "episode_steps": int(episode_steps),
                "cumulative_avg_score": fp_all.avg_score(),
                "benchmark_cumulative_avg_score": fp_all.avg_score(slug),
                "cumulative_avg_score_single": fp_single.avg_score(),
                "benchmark_cumulative_avg_score_single": fp_single.avg_score(slug),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if action_stats:
                row["action_stats"] = dict(action_stats)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def snapshot(self) -> Dict[str, float]:
        """Current cumulative averages as flat trainer metrics.

        Keys (``<bm>`` is a benchmark slug)::

            online/cumulative_avg_score            mean@group_n estimate
            online/cumulative_success_rate
            online/first_pass_episodes
            online/single/cumulative_avg_score     one attempt per task
            online/single/cumulative_success_rate
            online/single/first_pass_episodes
            online/<bm>/...  and  online/<bm>/single/...   same, per benchmark
            online/env_error_episodes  and  online/<bm>/env_error_episodes
                                                   environment failures (excluded above)

        With ``track_repeat_passes`` the episodes of all passes (the first one
        included) are also pooled into::

            multipass/global/cumulative_avg_score        all passes, all attempts
            multipass/global/cumulative_success_rate
            multipass/global/episodes
            multipass/global/single/cumulative_*         one attempt per task
            multipass/<bm>/...  and  multipass/<bm>/single/...   per benchmark
                                                         (only once the stream holds
                                                         more than one benchmark)
            multipass/global/window_avg_score            most recent stream cycle
            multipass/global/window_success_rate         (``window_episodes`` episodes)
            multipass/global/window_episodes

        The cumulative curves react slowly once many passes have been seen, so
        the window tracks the current policy. Until a full cycle has been seen
        the window simply covers everything so far.

        Benchmarks with no episode yet are omitted, so the curves start when
        the stream first reaches them.
        """
        with self._lock:
            out: Dict[str, float] = {}
            stats_all, stats_single = self._first
            stats_all.emit(out, "online/", "first_pass_episodes")
            stats_single.emit(out, "online/single/", "first_pass_episodes")
            for slug in sorted(stats_all.bm_num):
                stats_all.emit(out, f"online/{slug}/", "first_pass_episodes", slug)
                stats_single.emit(out, f"online/{slug}/single/", "first_pass_episodes", slug)
            if self.track_repeat_passes:
                pooled_all, pooled_single = self._pooled
                pooled_all.emit(out, "multipass/global/", "episodes")
                pooled_single.emit(out, "multipass/global/single/", "episodes")
                # with a single benchmark the per-benchmark curves would duplicate global
                if len(pooled_all.bm_num) > 1:
                    for slug in sorted(pooled_all.bm_num):
                        pooled_all.emit(out, f"multipass/{slug}/", "episodes", slug)
                        pooled_single.emit(out, f"multipass/{slug}/single/", "episodes", slug)
                if self._window:
                    n = len(self._window)
                    out["multipass/global/window_avg_score"] = sum(sc for sc, _ in self._window) / n
                    out["multipass/global/window_success_rate"] = (
                        sum(1.0 for _, ok in self._window if ok) / n
                    )
                    out["multipass/global/window_episodes"] = float(n)
            if self._env_errors:
                out["online/env_error_episodes"] = float(sum(self._env_errors.values()))
                for slug, n in sorted(self._env_errors.items()):
                    out[f"online/{slug}/env_error_episodes"] = float(n)
            return out
