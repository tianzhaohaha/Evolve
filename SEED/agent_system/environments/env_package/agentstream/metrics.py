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
matching the single-pass semantics of the AgentStream online protocol.

Two estimators of the first-pass score are maintained side by side:

* ``cumulative_avg_score`` — every first-pass attempt counts. With
  ``env.rollout.n = group_n`` copies per task this is a mean@group_n estimate.
* ``cumulative_avg_score_single`` — only the first copy of each task group
  (``rollout_slot % group_n == 0``; ``StreamBatch.replicate`` lays copies out
  task-major, so that is exactly one attempt per task). This matches the
  one-attempt-per-task protocol of the AgentStream baselines.

``snapshot()`` exposes both (globally and per benchmark) as trainer metrics so
they can be pushed to wandb every step under the ``online/`` prefix.

Resume semantics (``restore_up_to_step``): when the trainer resumes from a
checkpoint the accumulators are rebuilt from the existing JSONL, taking only
rows written at or before the checkpointed global step (later rows belong to
the crashed window and will be replayed). When the run starts from scratch
nothing is restored: an existing JSONL is left untouched and simply appended
to, exactly as before. First-pass rows are additionally de-duplicated on
``(benchmark, task, pass, rollout_slot)`` as a safety net against replays.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class OnlineMetricsRecorder:
    def __init__(
        self,
        path: str,
        run_meta: Optional[Dict[str, Any]] = None,
        group_n: int = 1,
        restore_up_to_step: Optional[int] = None,
    ) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.run_meta = dict(run_meta or {})
        self.group_n = max(int(group_n), 1)
        self._lock = threading.Lock()
        self._episode_counter = 0

        # All first-pass attempts (group_n per task).
        self._sum_scores = 0.0
        self._sum_success = 0.0
        self._num_scores = 0
        self._bm_sum: Dict[str, float] = defaultdict(float)
        self._bm_success: Dict[str, float] = defaultdict(float)
        self._bm_num: Dict[str, int] = defaultdict(int)

        # One attempt per task (first copy of each group).
        self._sum_scores_single = 0.0
        self._sum_success_single = 0.0
        self._num_scores_single = 0
        self._bm_sum_single: Dict[str, float] = defaultdict(float)
        self._bm_success_single: Dict[str, float] = defaultdict(float)
        self._bm_num_single: Dict[str, int] = defaultdict(int)

        self._seen_first_pass: Set[Tuple[str, str, int, int]] = set()
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
    ) -> bool:
        """Fold one first-pass episode into the accumulators.

        Returns False (and changes nothing) for a duplicate key, which only
        happens when steps are replayed after a resume.
        """
        key = (slug, str(task_id), int(pass_idx), int(rollout_slot))
        if key in self._seen_first_pass:
            return False
        self._seen_first_pass.add(key)

        self._sum_scores += score
        self._sum_success += float(success)
        self._num_scores += 1
        self._bm_sum[slug] += score
        self._bm_success[slug] += float(success)
        self._bm_num[slug] += 1

        if int(rollout_slot) % self.group_n == 0:
            self._sum_scores_single += score
            self._sum_success_single += float(success)
            self._num_scores_single += 1
            self._bm_sum_single[slug] += score
            self._bm_success_single[slug] += float(success)
            self._bm_num_single[slug] += 1
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
                        if not row.get("first_pass"):
                            continue
                        row_step = row.get("global_step")
                        if row_step is not None and int(row_step) > up_to_step:
                            skipped_after_ckpt += 1
                            continue
                        if self._accumulate(
                            slug=str(row.get("benchmark_slug", "")),
                            task_id=str(row.get("task_id", "")),
                            pass_idx=int(row.get("pass_idx", 0) or 0),
                            rollout_slot=int(row.get("rollout_slot", 0) or 0),
                            score=float(row.get("score", 0.0) or 0.0),
                            success=bool(row.get("success", False)),
                        ):
                            restored += 1
                    except (ValueError, TypeError):
                        continue  # malformed row: skip it, keep restoring
        except Exception as exc:  # best effort: never block training on metrics
            logger.warning("Could not restore online metrics from %s: %s", self.path, exc)
            return
        if restored or skipped_after_ckpt:
            logger.info(
                "Restored %d first-pass episodes (<= step %d) from %s; skipped %d rows "
                "after the checkpoint (cumulative_avg_score=%.4f)",
                restored,
                up_to_step,
                self.path,
                skipped_after_ckpt,
                self._sum_scores / max(self._num_scores, 1),
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
    ) -> None:
        first_pass = pass_idx == 0
        first_attempt = int(rollout_slot) % self.group_n == 0
        with self._lock:
            self._episode_counter += 1
            if first_pass:
                self._accumulate(
                    slug=slug,
                    task_id=task_id,
                    pass_idx=pass_idx,
                    rollout_slot=rollout_slot,
                    score=float(score),
                    success=bool(success),
                )

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
                "episode_steps": int(episode_steps),
                "cumulative_avg_score": (
                    self._sum_scores / self._num_scores if self._num_scores else 0.0
                ),
                "benchmark_cumulative_avg_score": (
                    self._bm_sum[slug] / self._bm_num[slug] if self._bm_num[slug] else 0.0
                ),
                "cumulative_avg_score_single": (
                    self._sum_scores_single / self._num_scores_single
                    if self._num_scores_single
                    else 0.0
                ),
                "benchmark_cumulative_avg_score_single": (
                    self._bm_sum_single[slug] / self._bm_num_single[slug]
                    if self._bm_num_single[slug]
                    else 0.0
                ),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if action_stats:
                row["action_stats"] = dict(action_stats)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def snapshot(self) -> Dict[str, float]:
        """Current first-pass cumulative averages as flat trainer metrics.

        Keys (``<bm>`` is a benchmark slug)::

            online/cumulative_avg_score            mean@group_n estimate
            online/cumulative_success_rate
            online/first_pass_episodes
            online/single/cumulative_avg_score     one attempt per task
            online/single/cumulative_success_rate
            online/single/first_pass_episodes
            online/<bm>/...  and  online/<bm>/single/...   same, per benchmark

        Benchmarks with no first-pass episode yet are omitted, so the curves
        start when the stream first reaches them.
        """

        def _put(out: Dict[str, float], prefix: str, s: float, succ: float, n: int) -> None:
            if n <= 0:
                return
            out[f"{prefix}cumulative_avg_score"] = s / n
            out[f"{prefix}cumulative_success_rate"] = succ / n
            out[f"{prefix}first_pass_episodes"] = float(n)

        with self._lock:
            out: Dict[str, float] = {}
            _put(out, "online/", self._sum_scores, self._sum_success, self._num_scores)
            _put(
                out,
                "online/single/",
                self._sum_scores_single,
                self._sum_success_single,
                self._num_scores_single,
            )
            for slug in sorted(self._bm_num):
                _put(
                    out,
                    f"online/{slug}/",
                    self._bm_sum[slug],
                    self._bm_success[slug],
                    self._bm_num[slug],
                )
                _put(
                    out,
                    f"online/{slug}/single/",
                    self._bm_sum_single[slug],
                    self._bm_success_single[slug],
                    self._bm_num_single[slug],
                )
            return out
