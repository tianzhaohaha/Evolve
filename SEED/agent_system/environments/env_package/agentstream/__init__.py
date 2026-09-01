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

"""AgentStream benchmark suite support for SEED (self-contained package).

Entry point: :func:`make_agentstream_envs` — see factory.py. The only hook in
existing SEED code is one ``elif`` branch in
``agent_system/environments/env_manager.py::make_envs`` delegating here.

Config lives under ``env.agentstream`` (see verl/trainer/config/ppo_trainer.yaml
and as_config.py for the full key reference).
"""

from .as_config import AgentStreamConfig, parse_agentstream_config
from .metrics import OnlineMetricsRecorder
from .projection import agentstream_projection
from .task_stream import (
    TaskStreamScheduler,
    ValTaskCycler,
    build_splits,
    order_tasks,
    select_tasks,
)

# Trainer-side entry points pull in ray/torch; resolve them lazily so the
# light modules (config, projection, task stream, exgentic bridge) stay
# importable in the SFT pipeline, smoke tools and unit tests. They remain in
# __all__, so ``from ... import *`` still needs the full trainer environment.
_LAZY = {
    "make_agentstream_envs": ".factory",
    "AgentStreamEnvironmentManager": ".manager",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentStreamConfig",
    "parse_agentstream_config",
    "make_agentstream_envs",
    "AgentStreamEnvironmentManager",
    "OnlineMetricsRecorder",
    "agentstream_projection",
    "TaskStreamScheduler",
    "ValTaskCycler",
    "build_splits",
    "order_tasks",
    "select_tasks",
]
