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

"""The manager's task_metadata() feeds the SEED global pool's retrieval key
(slug::task_id) and query (task text + first observation); the rollout loop
attaches it only when every column matches the batch size."""

import pytest

torch = pytest.importorskip("torch", reason="manager module imports the trainer-side env base")

from agent_system.environments.env_package.agentstream.manager import AgentStreamEnvironmentManager


def _bare_manager() -> AgentStreamEnvironmentManager:
    manager = AgentStreamEnvironmentManager.__new__(AgentStreamEnvironmentManager)
    manager._slugs = ["tau2", "bfcl"]
    manager._task_ids = ["airline_3", "case_9"]
    manager.tasks = ["You are a customer service agent...", "Book a flight to Tokyo"]
    manager._first_obs = ["Hi, I need to change my reservation", ""]
    return manager


def test_task_metadata_exposes_per_slot_identity():
    metadata = _bare_manager().task_metadata()
    assert set(metadata) == {"task_slug", "task_id", "task_text", "task_first_obs"}
    assert metadata["task_slug"] == ["tau2", "bfcl"]
    assert metadata["task_id"] == ["airline_3", "case_9"]
    assert metadata["task_first_obs"][0].startswith("Hi, I need")
    assert all(len(values) == 2 for values in metadata.values())


def test_task_metadata_returns_copies():
    manager = _bare_manager()
    metadata = manager.task_metadata()
    metadata["task_slug"].append("intruder")
    assert manager.task_metadata()["task_slug"] == ["tau2", "bfcl"]
