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

"""Ray-vectorized AgentStream environments (gym-style reset/step/close).

Follows the SEED env_package convention (see env_package/appworld/envs.py):
one Ray actor per parallel environment slot, ``reset()`` assigns tasks and
``step()`` fans actions out to all workers. The difference from the appworld
package is that task assignment is *pluggable*: a TaskStreamScheduler (train)
or ValTaskCycler (validation) decides which (benchmark, task) each slot runs,
which is how the Isolated / Sequential / Interleaved streaming scenarios and
the SEED-original random sampling are realized with the same machinery.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import ray

from .as_config import AgentStreamConfig
from .exgentic_client import BenchmarkHub, SessionDriver
from .task_stream import StreamBatch, TaskStreamScheduler, ValTaskCycler


class AgentStreamWorker:
    """Ray remote actor holding one exgentic SessionDriver."""

    def __init__(
        self,
        worker_id: int,
        exgentic_root: str,
        runner: Optional[str],
        output_dir: str,
        run_id: str,
        max_steps: int,
    ) -> None:
        self.worker_id = worker_id
        self.driver = SessionDriver(
            exgentic_root=exgentic_root,
            runner=runner,
            output_dir=output_dir,
            run_id=f"{run_id}_w{worker_id}",
            max_steps=max_steps,
        )

    def reset(
        self,
        slug: str,
        task_id: str,
        bm_kwargs: Dict[str, Any],
        session_kwargs: Dict[str, Any],
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            if max_steps:
                self.driver.set_max_steps(max_steps)
            return self.driver.reset(slug, task_id, bm_kwargs, session_kwargs)
        except Exception as exc:
            # Never kill the vectorized loop: surface the failure as a payload
            # the manager can render; the episode terminates with zero reward.
            # Tear the half-open session down so the first step() call sees a
            # finished driver instead of an exgentic session that may hang.
            try:
                self.driver.mark_failed()
            except Exception:
                pass
            return {
                "slug": slug,
                "task_id": str(task_id),
                "task": "",
                "context": "",
                "actions_text": "",
                "observation": f"[environment error during reset] {type(exc).__name__}: {exc}",
                "reset_error": True,
            }

    def step(self, action_payload: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs_text, done, info = self.driver.step(action_payload)
        return obs_text, done, info

    def close(self) -> None:
        self.driver.close()


class AgentStreamEnvs:
    """Vectorized environment over ``env_num * group_n`` worker slots."""

    def __init__(
        self,
        cfg: AgentStreamConfig,
        hub: BenchmarkHub,
        task_source: Any,  # TaskStreamScheduler | ValTaskCycler
        env_num: int,
        group_n: int,
        max_steps: int,
        resources_per_worker: Dict[str, Any],
        run_id: str,
    ) -> None:
        self.cfg = cfg
        self.hub = hub
        self.task_source = task_source
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self._current_batch: Optional[StreamBatch] = None

        if not ray.is_initialized():
            ray.init()

        self._worker_cls = ray.remote(**resources_per_worker)(AgentStreamWorker)
        output_dir = os.path.abspath(os.path.expanduser(cfg.exgentic_output_dir))
        self.default_max_steps = max_steps
        self._worker_kwargs = dict(
            exgentic_root=cfg.exgentic_root,
            runner=cfg.runner,
            output_dir=output_dir,
            run_id=run_id,
            max_steps=max_steps,
        )
        self.workers = [self._make_worker(i) for i in range(self.num_processes)]

    def _make_worker(self, worker_id: int):
        return self._worker_cls.remote(worker_id=worker_id, **self._worker_kwargs)

    def _replace_worker(self, i: int, reason: str) -> None:
        """Kill an unresponsive worker actor and start a fresh one in its slot.

        ray.kill also tears down the actor's exgentic serve subprocess tree, so
        a wedged benchmark server cannot poison later resets on this slot.
        """
        print(f"[agentstream] worker {i} unresponsive ({reason}); killing actor and recreating")
        try:
            ray.kill(self.workers[i], no_restart=True)
        except Exception:
            pass
        self.workers[i] = self._make_worker(i)

    def _gather(self, futures: List[Any], timeout_s: float, fallback) -> List[Any]:
        """Collect worker futures without letting one straggler freeze the batch.

        Workers that miss ``timeout_s`` (or died) are killed + recreated and
        their slot degrades to ``fallback(i)``; ``timeout_s <= 0`` restores the
        original unbounded ray.get.
        """
        if not timeout_s or timeout_s <= 0:
            return ray.get(futures)
        index_of = {ref: i for i, ref in enumerate(futures)}
        ready, pending = ray.wait(futures, num_returns=len(futures), timeout=float(timeout_s))
        results: List[Any] = [None] * len(futures)
        for ref in ready:
            i = index_of[ref]
            try:
                results[i] = ray.get(ref)
            except Exception as exc:
                self._replace_worker(i, f"{type(exc).__name__}: {exc}")
                results[i] = fallback(i)
        for ref in pending:
            i = index_of[ref]
            self._replace_worker(i, f"no result within {timeout_s}s")
            results[i] = fallback(i)
        return results

    # -- gym-style API ---------------------------------------------------------

    def reset(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Assign the next task batch and start sessions on all workers.

        Returns (payloads, infos): payloads carry everything the manager needs
        to build prompts; infos carry stream bookkeeping for metrics.
        """
        if isinstance(self.task_source, ValTaskCycler):
            # Every validation pass evaluates the same deterministic prefix,
            # keeping test_freq curves comparable over training.
            self.task_source.reset_cursor()

        batch = self.task_source.next_batch(self.env_num).replicate(self.group_n)
        self._current_batch = batch

        futures = []
        for i, worker in enumerate(self.workers):
            slug, task_id = batch.refs[i]
            bm_kwargs = self.cfg.resolved_benchmark_kwargs(slug)
            session_kwargs = self.hub.session_kwargs(slug, task_id)
            slug_max_steps = self.cfg.resolved_max_steps(slug, self.default_max_steps)
            futures.append(
                worker.reset.remote(slug, task_id, bm_kwargs, session_kwargs, slug_max_steps)
            )

        timeout_s = self.cfg.reset_timeout_s

        def timeout_payload(i: int) -> Dict[str, Any]:
            slug, task_id = batch.refs[i]
            return {
                "slug": slug,
                "task_id": str(task_id),
                "task": "",
                "context": "",
                "actions_text": "",
                "observation": f"[environment error during reset] worker timed out after {timeout_s}s",
                "reset_error": True,
            }

        payloads = self._gather(futures, timeout_s, timeout_payload)

        infos: List[Dict[str, Any]] = []
        for i, payload in enumerate(payloads):
            slug, task_id = batch.refs[i]
            infos.append(
                {
                    "slug": slug,
                    "task_id": str(task_id),
                    "stream_index": batch.stream_indices[i],
                    "pass_idx": batch.passes[i],
                    "rollout_slot": i,
                    "won": False,
                    "reset_error": bool(payload.get("reset_error", False)),
                }
            )
        return payloads, infos

    def step(
        self, action_payloads: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        assert len(action_payloads) == self.num_processes, (
            f"Expected {self.num_processes} actions, got {len(action_payloads)}"
        )
        futures = [
            worker.step.remote(action_payloads[i]) for i, worker in enumerate(self.workers)
        ]

        timeout_s = self.cfg.step_timeout_s

        def timeout_result(i: int) -> Tuple[str, bool, Dict[str, Any]]:
            slug, task_id = "", ""
            if self._current_batch is not None:
                slug, task_id = self._current_batch.refs[i]
            info = {
                "won": False,
                "score": 0.0,
                "slug": slug,
                "task_id": str(task_id),
                "step_count": 0,
                "action_error": False,
                "post_done": False,
                "env_timeout": True,
            }
            return "", True, info

        results = self._gather(futures, timeout_s, timeout_result)

        obs_list: List[str] = []
        reward_list: List[float] = []
        done_list: List[bool] = []
        info_list: List[Dict[str, Any]] = []

        batch = self._current_batch
        for i, (obs_text, done, info) in enumerate(results):
            reward = 0.0
            if done and not info.get("post_done", False):
                reward = self.cfg.reward_success * float(info.get("won", False))
                if self.cfg.reward_use_score:
                    reward += float(info.get("score", 0.0))
            if batch is not None:
                info.setdefault("stream_index", batch.stream_indices[i])
                info.setdefault("pass_idx", batch.passes[i])
                info.setdefault("rollout_slot", i)
            obs_list.append(obs_text)
            reward_list.append(reward)
            done_list.append(bool(done))
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def close(self) -> None:
        futures = []
        for worker in self.workers:
            try:
                futures.append(worker.close.remote())
            except Exception:
                pass
        try:
            ray.get(futures, timeout=300)
        except Exception:
            pass
        try:
            self.hub.close()
        except Exception:
            pass
