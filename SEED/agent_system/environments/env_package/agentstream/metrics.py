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
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Optional


class OnlineMetricsRecorder:
    def __init__(
        self,
        path: str,
        run_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.run_meta = dict(run_meta or {})
        self._lock = threading.Lock()
        self._episode_counter = 0
        self._sum_scores = 0.0
        self._num_scores = 0
        self._bm_sum: Dict[str, float] = defaultdict(float)
        self._bm_num: Dict[str, int] = defaultdict(int)

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
    ) -> None:
        first_pass = pass_idx == 0
        with self._lock:
            self._episode_counter += 1
            if first_pass:
                self._sum_scores += score
                self._num_scores += 1
                self._bm_sum[slug] += score
                self._bm_num[slug] += 1

            row = {
                **self.run_meta,
                "episode_index": self._episode_counter,
                "global_step": global_step,
                "benchmark_slug": slug,
                "task_id": task_id,
                "stream_index": stream_index,
                "pass_idx": pass_idx,
                "first_pass": first_pass,
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
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
