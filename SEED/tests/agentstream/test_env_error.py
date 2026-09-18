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

"""Episodes the environment itself breaks (reset failure, worker timeout) are
flagged ``env_error`` once by the manager and then (1) counted instead of scored
in the online JSONL / wandb metrics, (2) reported as ``env_error_rate`` by the
success evaluator, (3) dropped from the training batch by the rollout loop."""

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="manager module imports the trainer-side env base")

from agent_system.environments.env_package.agentstream.manager import AgentStreamEnvironmentManager
from agent_system.environments.env_package.agentstream.metrics import OnlineMetricsRecorder
from agent_system.memory import SimpleMemory
from agent_system.multi_turn_rollout.utils import drop_env_error_trajectories


# ----------------------------------------------------------------- recorder
def _record(rec, slot, *, slug="hle", task_id="930", score=1.0, env_error=False, step=1):
    rec.record_episode(
        slug=slug, task_id=task_id, stream_index=slot, pass_idx=0, rollout_slot=slot,
        success=score > 0, score=score, episode_steps=0 if env_error else 3,
        global_step=step, env_error=env_error,
    )


def test_recorder_counts_env_errors_instead_of_scoring_them(tmp_path):
    rec = OnlineMetricsRecorder(str(tmp_path / "online.jsonl"), group_n=2)
    _record(rec, 0, score=1.0)
    _record(rec, 1, score=0.0)
    _record(rec, 2, slug="bfcl", task_id="7", score=0.0, env_error=True)

    snap = rec.snapshot()
    assert snap["online/cumulative_avg_score"] == 0.5          # the error episode is not a zero
    assert snap["online/first_pass_episodes"] == 2
    assert snap["online/env_error_episodes"] == 1
    assert snap["online/bfcl/env_error_episodes"] == 1
    assert "online/bfcl/cumulative_avg_score" not in snap     # bfcl has no scored episode

    rows = [json.loads(l) for l in (tmp_path / "online.jsonl").read_text().splitlines()]
    assert [r["env_error"] for r in rows] == [False, False, True]  # audit trail kept
    assert rows[2]["cumulative_avg_score"] == 0.5


def test_recorder_restore_keeps_env_error_rows_out_of_the_averages(tmp_path):
    path = str(tmp_path / "online.jsonl")
    rec = OnlineMetricsRecorder(path, group_n=1)
    _record(rec, 0, score=1.0, step=1)
    _record(rec, 1, task_id="931", score=0.0, env_error=True, step=1)
    _record(rec, 2, task_id="932", score=0.0, step=2)  # after the checkpoint: replayed later

    resumed = OnlineMetricsRecorder(path, group_n=1, restore_up_to_step=1)
    snap = resumed.snapshot()
    assert snap["online/cumulative_avg_score"] == 1.0
    assert snap["online/env_error_episodes"] == 1
    _record(resumed, 1, task_id="931", score=0.0, env_error=True, step=1)  # replayed duplicate
    assert resumed.snapshot()["online/env_error_episodes"] == 1


# ------------------------------------------------------------------ manager
class _FakeEnvs:
    """One env whose first step times out (worker died), then post-done steps."""

    def __init__(self):
        self.calls = 0

    def step(self, payloads):
        self.calls += 1
        if self.calls == 1:
            info = {"won": False, "score": 0.0, "slug": "appworld", "task_id": "t1",
                    "step_count": 0, "action_error": False, "post_done": False, "env_timeout": True}
            return [""], [0.0], [True], [info]
        info = {"won": False, "score": 0.0, "slug": "appworld", "task_id": "t1",
                "step_count": 0, "action_error": False, "post_done": True}
        return [""], [0.0], [True], [info]


def _manager_with_one_env(tmp_path, envs):
    m = AgentStreamEnvironmentManager.__new__(AgentStreamEnvironmentManager)
    m.envs = envs
    m.config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    m.phase = "train"
    m.recorder = OnlineMetricsRecorder(str(tmp_path / "online.jsonl"), group_n=1)
    m._global_step = 4
    m.projection_f = lambda texts: (
        [{"name": "noop", "arguments": {}}] * len(texts), [True] * len(texts),
        [{"reason": "", "think_present": True, "used_tool_call_alias": False}] * len(texts),
    )
    m.memory = SimpleMemory()
    m.memory.reset(batch_size=1)
    m._slugs, m._task_ids = ["appworld"], ["t1"]
    m._stream_indices, m._pass_indices = [0], [0]
    m._recorded, m._env_error, m._episode_steps = [False], [False], [0]
    m._episode_action_stats = [{"steps": 0, "valid": 0, "think_present": 0, "tool_call_alias": 0}]
    m.tasks, m._contexts, m._actions_texts, m._first_obs = ["task"], [""], ["- noop"], ["obs"]
    m.pre_text_obs = ["obs"]
    return m


def test_manager_flags_timeouts_sticky_and_records_them_as_env_error(tmp_path):
    m = _manager_with_one_env(tmp_path, _FakeEnvs())
    _, _, dones, infos = m.step(["<action>noop</action>"])
    assert dones[0] and infos[0]["env_error"] is True
    _, _, _, infos = m.step(["<action>noop</action>"])           # post-done row keeps the flag
    assert infos[0]["env_error"] is True

    snap = m.recorder.snapshot()
    assert snap["online/appworld/env_error_episodes"] == 1
    assert "online/cumulative_avg_score" not in snap             # nothing was scored


def test_success_evaluator_reports_env_error_rate_and_skips_score_keys():
    m = AgentStreamEnvironmentManager.__new__(AgentStreamEnvironmentManager)
    rows = lambda active: [{"active_masks": active}]
    total_batch_list = [rows(True), rows(False), rows(True)]
    total_infos = [
        [{"won": True, "score": 1.0, "slug": "tau2", "env_error": False}],
        [{"won": False, "score": 0.0, "slug": "tau2", "env_error": True}],   # rows already dropped
        [{"won": False, "score": 0.5, "slug": "appworld", "env_error": False, "score_error": ""}],
    ]
    out = m.success_evaluator(total_batch_list=total_batch_list, total_infos=total_infos)
    assert out["env_error_rate"].tolist() == [0.0, 1.0, 0.0]
    assert out["tau2_env_error_rate"].tolist() == [0.0, 1.0]
    assert out["success_rate"].tolist() == [1.0, 0.0]            # the error episode is absent
    assert out["tau2_score"].tolist() == [1.0]
    assert out["appworld_score"].tolist() == [0.5]


# ------------------------------------------------------------- rollout loop
def test_drop_env_error_trajectories_deactivates_every_row_once():
    total_batch_list = [
        [{"active_masks": True}, {"active_masks": True}],   # env 0: healthy
        [{"active_masks": True}, {"active_masks": False}],  # env 1: reset error, row 0 was generated
    ]
    infos = [{"env_error": False}, {"env_error": True}]
    assert drop_env_error_trajectories(total_batch_list, infos) == 1
    assert [r["active_masks"] for r in total_batch_list[0]] == [True, True]
    assert [r["active_masks"] for r in total_batch_list[1]] == [False, False]
    assert drop_env_error_trajectories(total_batch_list, infos) == 0  # idempotent on later steps


# ------------------------------------------------------------------ trainer
def test_trainer_forwards_score_and_error_rate_keys():
    from verl.trainer.ppo.metric_utils import is_episode_metric_key

    assert all(is_episode_metric_key(k) for k in (
        "success_rate", "tau2_success_rate", "appworld_score", "hle_score_error_rate",
        "env_error_rate", "bfcl_env_error_rate",
    ))
    assert not any(is_episode_metric_key(k) for k in ("traj_uid", "tool_callings", "is_action_valid", "obs_text"))
