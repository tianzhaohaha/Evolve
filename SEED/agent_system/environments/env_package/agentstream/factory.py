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

"""Factory: build train/val AgentStream environment managers from the trainer config.

This is the single entry point used by the (one-line) hook in
``agent_system/environments/env_manager.py::make_envs``. Everything else in
the integration lives inside this package.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Optional, Tuple

from .as_config import _select, parse_agentstream_config
from .envs import AgentStreamEnvs
from .exgentic_client import BenchmarkHub
from .manager import AgentStreamEnvironmentManager
from .metrics import OnlineMetricsRecorder
from .projection import agentstream_projection
from .task_stream import TaskStreamScheduler, ValTaskCycler, build_splits


def make_agentstream_envs(
    config,
    require_think: Optional[bool] = None,
) -> Tuple[AgentStreamEnvironmentManager, AgentStreamEnvironmentManager]:
    """Build (train_envs, val_envs) for ``env.env_name=agentstream/*``.

    Mirrors the signature/behavior of the branches in ``make_envs``:
    train envs use ``data.train_batch_size`` x ``env.rollout.n`` slots,
    validation envs use ``data.val_batch_size`` x 1 slots.
    """
    cfg = parse_agentstream_config(config)

    experiment_name = str(_select(config, "trainer.experiment_name", "seed_agentstream"))
    run_id = f"{experiment_name}_{cfg.protocol}_{cfg.stream_mode}_s{cfg.stream_seed}"

    hub = BenchmarkHub(
        exgentic_root=cfg.exgentic_root,
        slugs=cfg.benchmarks,
        benchmark_kwargs={slug: cfg.resolved_benchmark_kwargs(slug) for slug in cfg.benchmarks},
        runner=cfg.runner,
        output_dir=os.path.abspath(os.path.expanduser(cfg.exgentic_output_dir)),
        run_id=run_id,
    )

    # Task universe -> AgentStream-compatible selection -> protocol splits.
    task_universe = hub.list_all_tasks()
    train_tasks, val_tasks = build_splits(task_universe, cfg)

    env_seed = int(_select(config, "env.seed", 0))
    scheduler = TaskStreamScheduler(
        train_tasks=train_tasks,
        mode=cfg.stream_mode,
        stream_seed=cfg.stream_seed,
        block_passes=cfg.block_passes,
        on_exhausted=cfg.on_exhausted,
        sample_seed=env_seed,
    )
    val_cycler = ValTaskCycler(val_tasks=val_tasks)

    group_n = int(_select(config, "env.rollout.n", 1)) or 1
    train_batch_size = int(_select(config, "data.train_batch_size"))
    val_batch_size = int(_select(config, "data.val_batch_size"))
    max_steps = cfg.max_steps if cfg.max_steps is not None else int(_select(config, "env.max_steps", 50))

    try:
        from omegaconf import OmegaConf

        resources_per_worker = OmegaConf.to_container(
            _select(config, "env.resources_per_worker"), resolve=True
        )
    except Exception:
        resources_per_worker = dict(_select(config, "env.resources_per_worker", {"num_cpus": 0.1}))

    _envs = AgentStreamEnvs(
        cfg=cfg,
        hub=hub,
        task_source=scheduler,
        env_num=train_batch_size,
        group_n=group_n,
        max_steps=max_steps,
        resources_per_worker=resources_per_worker,
        run_id=f"{run_id}_train",
    )
    _val_envs = AgentStreamEnvs(
        cfg=cfg,
        hub=hub,
        task_source=val_cycler,
        env_num=val_batch_size,
        group_n=1,
        max_steps=max_steps,
        resources_per_worker=resources_per_worker,
        run_id=f"{run_id}_val",
    )

    recorder = None
    if cfg.online_metrics_enable:
        metrics_path = cfg.online_metrics_path
        if not metrics_path:
            local_dir = str(_select(config, "trainer.default_local_dir", "outputs"))
            metrics_path = os.path.join(local_dir, "agentstream_online_metrics.jsonl")
        recorder = OnlineMetricsRecorder(
            path=metrics_path,
            run_meta={
                "experiment": experiment_name,
                "protocol": cfg.protocol,
                "stream_mode": cfg.stream_mode,
                "stream_seed": cfg.stream_seed,
                "benchmarks": ",".join(cfg.benchmarks),
                "group_n": group_n,
            },
        )

    think = True if require_think is None else bool(require_think)
    projection_f = partial(agentstream_projection, require_think=think)

    envs = AgentStreamEnvironmentManager(
        _envs, projection_f, config, phase="train", recorder=recorder
    )
    val_envs = AgentStreamEnvironmentManager(
        _val_envs, projection_f, config, phase="val", recorder=None
    )

    print(
        f"[agentstream] protocol={cfg.protocol} stream_mode={cfg.stream_mode} "
        f"stream_seed={cfg.stream_seed} benchmarks={cfg.benchmarks} "
        f"train_tasks={ {k: len(v) for k, v in train_tasks.items()} } "
        f"val_tasks={ {k: len(v) for k, v in val_tasks.items()} } "
        f"stream_length={scheduler.stream_length if cfg.stream_mode != 'random' else 'n/a (random)'}"
    )
    return envs, val_envs
