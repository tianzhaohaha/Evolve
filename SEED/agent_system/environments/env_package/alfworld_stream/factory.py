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

"""Factory + manager for stream-controlled ALFWorld (task types as domains).

Reuses, rather than reimplements:
  * TaskStreamScheduler / ValTaskCycler / seed-42 selection  (agentstream pkg)
  * OnlineMetricsRecorder (same JSONL schema -> same analyzer)
  * AlfWorldEnvironmentManager (prompts, memory, per-task-type val metrics)

Train stream and validation tasks are disjoint game-file sets drawn from the
ALFWorld *train* split (per-type seed-42 selection). Note: validation here is
held-out train-split games, not the standard eval_in_distribution split.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Tuple

from agent_system.environments.env_package.agentstream.as_config import _select
from agent_system.environments.env_package.agentstream.metrics import OnlineMetricsRecorder
from agent_system.environments.env_package.agentstream.task_stream import (
    TaskStreamScheduler,
    ValTaskCycler,
    select_holdout_tasks,
    select_tasks,
)

from .envs import AlfWorldStreamEnvs, collect_game_files


def _make_manager_class():
    """Deferred import: env_manager also imports this package's factory."""
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager

    class AlfWorldStreamEnvironmentManager(AlfWorldEnvironmentManager):
        """Adds AgentStream-style online episode metrics on top of the parent."""

        def __init__(self, envs, projection_f, config, phase="train", recorder=None):
            self.phase = phase
            self.recorder = recorder
            self._recorded = []
            super().__init__(envs, projection_f, config)

        def reset(self, kwargs=None):
            observations, infos = super().reset(kwargs)
            self._recorded = [False] * len(infos)
            return observations, infos

        def step(self, text_actions):
            next_observations, rewards, dones, infos = super().step(text_actions)
            if self.recorder is not None and self.phase == "train":
                meta = self.envs.last_batch_meta
                for i, done in enumerate(dones):
                    if not done or self._recorded[i]:
                        continue
                    self._recorded[i] = True
                    task_type, game_file = meta.refs[i]
                    self.recorder.record_episode(
                        slug=task_type,
                        task_id=os.path.basename(os.path.dirname(game_file)),
                        stream_index=meta.stream_indices[i],
                        pass_idx=meta.passes[i],
                        rollout_slot=i,
                        success=bool(infos[i].get("won", False)),
                        score=float(infos[i].get("won", False)),
                        episode_steps=len(self.memory[i]),
                    )
            return next_observations, rewards, dones, infos

    return AlfWorldStreamEnvironmentManager


def make_alfworld_stream_envs(config, require_think: bool = True) -> Tuple[object, object]:
    """Build (train_envs, val_envs) for ``env.env_name=alfworld_stream/*``."""
    from agent_system.environments.env_package.alfworld import alfworld_projection

    node_get = lambda key, default: _select(config, f"env.alfworld_stream.{key}", default)  # noqa: E731

    stream_mode = str(node_get("stream_mode", "random")).lower()
    stream_seed = int(node_get("stream_seed", 44))
    num_tasks_per_type = int(node_get("num_tasks_per_type", 30))
    val_tasks_per_type = int(node_get("val_tasks_per_type", 8))
    block_passes = max(1, int(node_get("block_passes", 1)))
    on_exhausted = str(node_get("on_exhausted", "cycle")).lower()

    alf_config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "alfworld/configs/config_tw.yaml"
    )
    universe = collect_game_files(alf_config_path)
    task_types = node_get("task_types", None)  # subset filter; used for isolated runs
    if task_types:
        if isinstance(task_types, str):
            task_types = [t.strip() for t in task_types.split(",") if t.strip()]
        universe = {k: v for k, v in universe.items() if k in set(task_types)}
    universe = {k: v for k, v in universe.items() if v}
    if not universe:
        raise ValueError("alfworld_stream: no game files found (check ALFWORLD_DATA / task_types).")
    train_tasks = select_tasks(universe, num_tasks_per_type)
    val_tasks = select_holdout_tasks(universe, num_tasks_per_type, val_tasks_per_type)

    env_seed = int(_select(config, "env.seed", 0))
    scheduler = TaskStreamScheduler(
        train_tasks=train_tasks,
        mode=stream_mode,
        stream_seed=stream_seed,
        block_passes=block_passes,
        on_exhausted=on_exhausted,
        sample_seed=env_seed,
    )
    val_cycler = ValTaskCycler(val_tasks=val_tasks)

    try:
        from omegaconf import OmegaConf

        resources_per_worker = OmegaConf.to_container(
            _select(config, "env.resources_per_worker"), resolve=True
        )
    except Exception:
        resources_per_worker = dict(_select(config, "env.resources_per_worker", {"num_cpus": 0.1}))

    group_n = int(_select(config, "env.rollout.n", 1)) or 1
    _envs = AlfWorldStreamEnvs(
        alf_config_path=alf_config_path,
        task_source=scheduler,
        seed=env_seed,
        env_num=int(_select(config, "data.train_batch_size")),
        group_n=group_n,
        resources_per_worker=resources_per_worker,
    )
    _val_envs = AlfWorldStreamEnvs(
        alf_config_path=alf_config_path,
        task_source=val_cycler,
        seed=env_seed + 1000,
        env_num=int(_select(config, "data.val_batch_size")),
        group_n=1,
        resources_per_worker=resources_per_worker,
    )

    recorder = None
    if bool(node_get("online_metrics_enable", True)):
        metrics_path = str(node_get("online_metrics_path", "")) or os.path.join(
            str(_select(config, "trainer.default_local_dir", "outputs")),
            "agentstream_online_metrics.jsonl",
        )
        recorder = OnlineMetricsRecorder(
            path=metrics_path,
            run_meta={
                "experiment": str(_select(config, "trainer.experiment_name", "alfworld_stream")),
                "protocol": "online",
                "stream_mode": stream_mode,
                "stream_seed": stream_seed,
                "benchmarks": ",".join(sorted(train_tasks)),
                "group_n": group_n,
            },
        )

    manager_cls = _make_manager_class()
    projection_f = partial(alfworld_projection, require_think=require_think)
    envs = manager_cls(_envs, projection_f, config, phase="train", recorder=recorder)
    val_envs = manager_cls(_val_envs, projection_f, config, phase="val", recorder=None)

    print(
        f"[alfworld_stream] stream_mode={stream_mode} stream_seed={stream_seed} "
        f"train_tasks={ {k: len(v) for k, v in train_tasks.items()} } "
        f"val_tasks={ {k: len(v) for k, v in val_tasks.items()} } "
        f"stream_length={scheduler.stream_length if stream_mode != 'random' else 'n/a (random)'}"
    )
    return envs, val_envs
