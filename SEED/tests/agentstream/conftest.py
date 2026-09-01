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

"""Fixtures for the AgentStream integration tests.

The pure modules (projection, task stream, config) need nothing beyond the
standard library. ``exgentic_root`` additionally requires the exgentic host
layer (pydantic, cloudpickle, ...) and skips otherwise, so the suite runs both
in the SEED conda env and in a minimal venv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SEED_ROOT = Path(__file__).resolve().parents[2]
if str(SEED_ROOT) not in sys.path:
    sys.path.insert(0, str(SEED_ROOT))


def _light_namespace() -> None:
    """Let ``agent_system.environments.env_package.agentstream`` import without torch/ray.

    ``agent_system/environments/__init__.py`` eagerly imports the trainer-side
    env manager. When torch is absent (minimal test venv) register the parent
    packages as plain namespace modules so only the pure agentstream modules
    are executed; in the SEED conda env the real packages are used untouched.
    """
    import importlib.util
    import types

    if importlib.util.find_spec("torch") is not None:
        return
    for name in ("agent_system", "agent_system.environments", "agent_system.environments.env_package"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(SEED_ROOT / name.replace(".", "/"))]
            sys.modules[name] = module


_light_namespace()


@pytest.fixture(scope="session")
def exgentic_root() -> str:
    root = os.environ.get("AGENTSTREAM_EXGENTIC_ROOT") or str(SEED_ROOT.parent / "AgentStream" / "exgentic")
    if not (Path(root) / "src" / "exgentic").is_dir():
        pytest.skip("AgentStream/exgentic checkout not found (set AGENTSTREAM_EXGENTIC_ROOT)")
    from agent_system.environments.env_package.agentstream.exgentic_client import bootstrap_exgentic

    bootstrap_exgentic(root)
    pytest.importorskip("exgentic.testing", reason="exgentic host-layer dependencies missing")
    return root
