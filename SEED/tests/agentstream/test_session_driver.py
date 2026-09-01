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

"""End-to-end bridge test on exgentic's in-process ``TestBenchmark`` fixture.

Covers the episode shapes the real benchmarks produce without their heavy
dependencies: multi-step (bfcl/tau2/appworld), single-step finish (hle),
step-budget exhaustion, unknown actions and per-benchmark observation caps.
"""

from __future__ import annotations

import pytest

from agent_system.environments.env_package.agentstream.exgentic_client import BenchmarkHub, SessionDriver

SLUG = "exgentic.testing.benchmark:TestBenchmark"


@pytest.fixture
def hub(exgentic_root, tmp_path):
    hub = BenchmarkHub(exgentic_root, [SLUG], {}, runner=None, output_dir=str(tmp_path / "hub"), run_id="t")
    yield hub
    hub.close()


@pytest.fixture
def driver(exgentic_root, tmp_path):
    driver = SessionDriver(exgentic_root, runner=None, output_dir=str(tmp_path / "w"), run_id="t", max_steps=3)
    yield driver
    driver.close()


def _reset(hub, driver, task_id="task-1"):
    return driver.reset(SLUG, task_id, {}, hub.session_kwargs(SLUG, task_id))


def test_hub_lists_tasks_and_driver_renders_payload(hub, driver):
    assert hub.list_all_tasks() == {SLUG: ["task-1", "task-2", "task-3"]}
    payload = _reset(hub, driver)
    assert payload["task"] == "Task task-1"
    assert '"task_id": "task-1"' in payload["context"]
    assert all(name in payload["actions_text"] for name in ("good", "bad", "finish"))
    assert payload["observation"] == "start"


def test_finish_scores_and_post_done_steps_are_shape_stable(hub, driver):
    _reset(hub, driver)
    obs, done, info = driver.step({"name": "good", "arguments": {}})
    assert (obs, done, info["action_error"]) == ("step", False, False)
    obs, done, info = driver.step({"name": "finish", "arguments": {}})
    assert done and info["won"] and info["score"] == 1.0 and not info["limit_reached"]
    _, done, info = driver.step({"name": "good", "arguments": {}})
    assert done and info["post_done"] and info["won"]


def test_single_step_finish_matches_hle_shape(hub, driver):
    driver.set_episode_limits(max_steps=1)
    _reset(hub, driver)
    _, done, info = driver.step({"name": "finish", "arguments": {}})
    assert done and info["step_count"] == 1
    assert info["won"] is False  # TestSession needs one good action before finish


def test_unknown_action_and_step_budget(hub, driver):
    driver.set_episode_limits(max_steps=2)
    _reset(hub, driver)
    obs, done, info = driver.step({"name": "nope", "arguments": {}})
    assert not done and info["action_error"] and obs.startswith("[invalid action]")
    _, done, info = driver.step({"name": "good", "arguments": {}})
    assert done and info["limit_reached"] and info["won"] is False


def test_observation_cap_applies_per_episode(hub, driver):
    driver.set_episode_limits(observation_max_chars=2)
    assert _reset(hub, driver)["observation"].startswith("st\n...[observation truncated]")
    driver.set_episode_limits(observation_max_chars=4096)
    assert _reset(hub, driver)["observation"] == "start"
