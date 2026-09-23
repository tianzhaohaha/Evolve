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

"""Environment manager gluing AgentStream envs into SEED's rollout contract.

Implements the same interface as the managers in
``agent_system/environments/env_manager.py`` (reset/step/success_evaluator/
build_text_obs) but lives inside this package so no existing SEED module needs
to change. Only two stable SEED APIs are imported:
``EnvironmentManagerBase``/``to_numpy`` and ``SimpleMemory``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from seed.resample import augment_observations

from .envs import AgentStreamEnvs
from .metrics import OnlineMetricsRecorder
from .prompts import render_prompt

_TERMINAL_OBS = "[episode finished]"


class AgentStreamEnvironmentManager(EnvironmentManagerBase):
    """SEED-compatible manager over :class:`AgentStreamEnvs`."""

    # reset(kwargs=[{task_slug, task_id, reference_solution}, ...]) re-runs explicit tasks with a
    # per-slot "Reference Solution" in the prompt (SEED sibling resample, seed/resample.py).
    supports_task_refs = True

    def __init__(
        self,
        envs: AgentStreamEnvs,
        projection_f,
        config,
        phase: str = "train",
        recorder: Optional[OnlineMetricsRecorder] = None,
    ) -> None:
        self.memory = SimpleMemory()
        self.phase = phase
        self.recorder = recorder
        self.retrieval_memory = None  # parity with other SEED managers
        self._global_step: Optional[int] = None  # set by the trainer before each rollout
        # Sibling resample pass: per-slot reference text ("" = plain prompt) and whether the
        # current episode batch is such a pass (its episodes stay out of the online metrics).
        self._reference_solutions: List[str] = []
        self._resample_mode = False

        self._slugs: List[str] = []
        self._task_ids: List[str] = []
        self._stream_indices: List[int] = []
        self._pass_indices: List[int] = []
        self._recorded: List[bool] = []
        self._env_error: List[bool] = []  # reset failed / worker timed out: episode unusable
        self._episode_steps: List[int] = []
        self._episode_action_stats: List[Dict[str, int]] = []
        self.tasks: List[str] = []
        self._contexts: List[str] = []
        self._actions_texts: List[str] = []
        self._first_obs: List[str] = []
        self.pre_text_obs: List[str] = []

        super().__init__(envs, projection_f, config)

    # ------------------------------------------------------------------ reset

    def reset(self, kwargs=None):
        # Resample requests are per-slot dicts carrying task_slug; any other env_kwargs are
        # ignored, as this manager always did.
        requests = list(kwargs) if kwargs is not None and len(kwargs) else []
        if not (requests and isinstance(requests[0], dict) and "task_slug" in requests[0]):
            payloads, infos = self.envs.reset()
            self._reference_solutions = [""] * len(payloads)
            self._resample_mode = False
        else:
            payloads, infos = self.envs.reset_refs([(str(r["task_slug"]), str(r["task_id"])) for r in requests])
            self._reference_solutions = [str(r.get("reference_solution", "")) for r in requests]
            self._resample_mode = True
        batch_size = len(payloads)

        self.memory.reset(batch_size=batch_size)
        self._slugs = [str(p.get("slug", "")) for p in payloads]
        self._task_ids = [str(p.get("task_id", "")) for p in payloads]
        self._stream_indices = [int(info.get("stream_index", -1)) for info in infos]
        self._pass_indices = [int(info.get("pass_idx", 0)) for info in infos]
        self._recorded = [False] * batch_size
        self._env_error = [bool(info.get("reset_error", False)) for info in infos]
        self._episode_steps = [0] * batch_size
        self._episode_action_stats = [
            {"steps": 0, "valid": 0, "think_present": 0, "tool_call_alias": 0}
            for _ in range(batch_size)
        ]

        self.tasks = [str(p.get("task", "")) for p in payloads]
        self._contexts = [str(p.get("context", "")) for p in payloads]
        self._actions_texts = [str(p.get("actions_text", "")) for p in payloads]
        raw_obs = [str(p.get("observation", "")) for p in payloads]
        self._first_obs = list(raw_obs)
        self.pre_text_obs = raw_obs

        full_text_obs = self.build_text_obs(raw_obs, init=True)
        observations = {
            "text": augment_observations(full_text_obs, self._reference_solutions),
            "text_base": full_text_obs,
            "image": None,
            "anchor": raw_obs,
        }
        return observations, infos

    # ------------------------------------------------------------------- step

    def step(self, text_actions: List[str]):
        payloads, valids, extras = self.projection_f(text_actions)
        obs_texts, rewards, dones, infos = self.envs.step(payloads)

        action_strs = [
            json.dumps(p, ensure_ascii=False) if v else "[invalid action format]"
            for p, v in zip(payloads, valids)
        ]
        self.memory.store({"text_obs": self.pre_text_obs, "action": action_strs})

        display_obs = [t if t else _TERMINAL_OBS for t in obs_texts]
        self.pre_text_obs = display_obs

        for i in range(len(infos)):
            post_done = bool(infos[i].get("post_done", False))
            self._episode_steps[i] += 0 if post_done else 1
            action_valid = bool(valids[i]) and not bool(infos[i].get("action_error", False))
            infos[i]["is_action_valid"] = to_numpy(action_valid)
            reason = extras[i]["reason"]
            if not reason and not action_valid:
                reason = "env_action_error"
            infos[i]["action_invalid_reason"] = reason
            infos[i]["think_present"] = bool(extras[i]["think_present"])
            infos[i]["used_tool_call_alias"] = bool(extras[i]["used_tool_call_alias"])
            if not post_done:
                stats = self._episode_action_stats[i]
                stats["steps"] += 1
                stats["valid"] += int(action_valid)
                stats["think_present"] += int(extras[i]["think_present"])
                stats["tool_call_alias"] += int(extras[i]["used_tool_call_alias"])
                if reason:
                    key = f"invalid_{reason}"
                    stats[key] = stats.get(key, 0) + 1
            infos[i].setdefault("won", False)
            # Sticky: once the environment itself failed (reset error or step timeout) every
            # later row of the trajectory carries the flag, so the rollout loop drops the
            # whole trajectory and the metrics below report it instead of scoring it.
            self._env_error[i] = self._env_error[i] or bool(infos[i].get("env_timeout", False))
            infos[i]["env_error"] = self._env_error[i]

        self._record_finished_episodes(dones, infos)

        full_text_obs = self.build_text_obs(display_obs, init=False)
        references = getattr(self, "_reference_solutions", None) or [""] * len(full_text_obs)
        next_observations = {
            "text": augment_observations(full_text_obs, references),
            "text_base": full_text_obs,
            "image": None,
            "anchor": display_obs,
        }
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def _record_finished_episodes(self, dones, infos) -> None:
        if self.recorder is None or self.phase != "train" or getattr(self, "_resample_mode", False):
            return
        for i, done in enumerate(dones):
            if not done or self._recorded[i]:
                continue
            self._recorded[i] = True
            info = infos[i]
            self.recorder.record_episode(
                slug=self._slugs[i],
                task_id=self._task_ids[i],
                stream_index=self._stream_indices[i],
                pass_idx=self._pass_indices[i],
                rollout_slot=int(info.get("rollout_slot", i)),
                success=bool(info.get("won", False)),
                score=float(info.get("score", 0.0)),
                episode_steps=int(info.get("step_count", self._episode_steps[i])),
                global_step=self._global_step,
                action_stats=dict(self._episode_action_stats[i]),
                env_error=self._env_error[i],
            )

    def stream_state_dict(self) -> Optional[Dict[str, object]]:
        """Task-stream scheduler state for the trainer checkpoint (None if unavailable)."""
        fn = getattr(self.envs, "stream_state_dict", None)
        return fn() if callable(fn) else None

    def load_stream_state(self, state: Dict[str, object]) -> None:
        fn = getattr(self.envs, "load_stream_state", None)
        if callable(fn):
            fn(state)

    def set_global_step(self, step: int) -> None:
        """Stamp subsequent episode rows with the trainer step (used on resume)."""
        self._global_step = int(step)

    def task_metadata(self) -> Dict[str, List[str]]:
        """Per-slot task identity for the current episode batch (constant across
        steps): benchmark slug, task id, full task text, first raw observation.

        The rollout loop forwards these as non-tensor columns so the SEED global
        skill pool can build its retrieval key (slug::task_id) and query (task
        text + first observation) without parsing prompt text — the prompt's
        "Your task is:" line is a benchmark-level constant on tau2 and the
        question-answering benchmarks, so text parsing cannot identify tasks.
        """
        return {
            "task_slug": list(self._slugs),
            "task_id": list(self._task_ids),
            "task_text": list(self.tasks),
            "task_first_obs": list(self._first_obs),
        }

    def online_metrics_snapshot(self) -> Dict[str, float]:
        """AgentStream cumulative averages for wandb (train phase only): first
        pass, plus per-pass subtrees when the recorder tracks repeat passes."""
        if self.recorder is None:
            return {}
        return self.recorder.snapshot()

    # ------------------------------------------------------------ observations

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        history_length = int(self.config.env.history_length)
        memory_contexts = [""] * len(text_obs)
        valid_lens = [0] * len(text_obs)
        if not init and history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                history_length, obs_key="text_obs", action_key="action"
            )

        return [
            render_prompt(
                slug=self._slugs[i],
                task=self.tasks[i],
                context=self._contexts[i],
                actions_text=self._actions_texts[i],
                observation=obs,
                step_count=0 if init else len(self.memory[i]),
                history_text=memory_contexts[i],
                history_len=valid_lens[i],
            )
            for i, obs in enumerate(text_obs)
        ]

    # ------------------------------------------------------------------ metrics

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Base contract minus its ``len(success_rate) == batch_size`` assertion:
        episodes broken by the environment are reported under ``env_error_rate``
        only, so the score keys stay model-only (see ``_process_batch``)."""
        total_batch_list, total_infos = kwargs["total_batch_list"], kwargs["total_infos"]
        success: Dict[str, list] = defaultdict(list)
        for bs in range(len(total_batch_list)):
            self._process_batch(bs, total_batch_list, total_infos, success)
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """Per-episode success plus per-benchmark breakdown.

        Emits ``success_rate``, ``<slug>_success_rate`` / ``<slug>_score`` and
        ``<slug>_score_error_rate`` (episodes whose benchmark scoring failed,
        counted as zero reward) so validation curves per benchmark are logged at
        every trainer.test_freq, which is exactly the forgetting / transfer
        measurement for sequential and interleaved streams.

        ``env_error_rate`` / ``<slug>_env_error_rate`` count episodes the
        environment itself broke (reset error, worker timeout). Those episodes
        are excluded from every other key: their rows never reach the training
        batch (rollout loop) and a zero there would measure the infrastructure,
        not the policy.
        """
        infos = total_infos[batch_idx]
        env_error = bool(infos[-1].get("env_error", False)) if infos else False
        slug = (str(infos[-1].get("slug", "")) or "unknown") if infos else "unknown"
        success["env_error_rate"].append(float(env_error))
        success[f"{slug}_env_error_rate"].append(float(env_error))
        if env_error:
            return
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info.get("won", False))
                success["success_rate"].append(won_value)
                success[f"{slug}_success_rate"].append(won_value)
                score = info.get("score", None)
                if score is not None:
                    success[f"{slug}_score"].append(float(score))
                success[f"{slug}_score_error_rate"].append(float(bool(info.get("score_error"))))
                return
