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

from agent_system.environments.env_package.agentstream.metrics import OnlineMetricsRecorder


def _record(rec, *, slug="tau2", task, pass_idx, slot, success, step=1, env_error=False):
    rec.record_episode(
        slug=slug,
        task_id=task,
        stream_index=0,
        pass_idx=pass_idx,
        rollout_slot=slot,
        success=success,
        score=1.0 if success else 0.0,
        episode_steps=3,
        global_step=step,
        env_error=env_error,
    )


def _feed(rec):
    # 2 tasks x 2 copies per pass; success: pass 0 = 1/4, pass 1 = 2/4, pass 2 = 3/4
    plan = {0: [True, False, False, False], 1: [True, True, False, False], 2: [True, True, True, False]}
    for pass_idx, outcomes in plan.items():
        for i, ok in enumerate(outcomes):
            _record(rec, task=f"t{i // 2}", pass_idx=pass_idx, slot=i, success=ok, step=pass_idx + 1)


def test_pooled_metrics_and_first_pass_unchanged(tmp_path):
    rec = OnlineMetricsRecorder(
        str(tmp_path / "m.jsonl"), group_n=2, track_repeat_passes=True, window_episodes=4
    )
    _feed(rec)
    snap = rec.snapshot()

    assert snap["online/cumulative_success_rate"] == 0.25  # first pass only
    assert snap["online/first_pass_episodes"] == 4
    assert snap["multipass/global/episodes"] == 12
    assert snap["multipass/global/cumulative_success_rate"] == 6 / 12
    assert snap["multipass/global/single/episodes"] == 6  # slots 0 and 2 of every pass
    assert snap["multipass/global/single/cumulative_success_rate"] == 4 / 6
    # the window holds the latest 4 episodes, i.e. the last pass
    assert snap["multipass/global/window_episodes"] == 4
    assert snap["multipass/global/window_success_rate"] == 3 / 4


def test_no_per_pass_subtrees_and_window_disabled_by_default(tmp_path):
    rec = OnlineMetricsRecorder(str(tmp_path / "a.jsonl"), group_n=2, track_repeat_passes=True)
    _feed(rec)
    snap = rec.snapshot()
    assert not any(k.startswith("online/pass") for k in snap)
    assert not any("window" in k for k in snap)  # window_episodes=0 disables it


def test_per_benchmark_curves_only_with_several_benchmarks(tmp_path):
    rec = OnlineMetricsRecorder(str(tmp_path / "s.jsonl"), group_n=1, track_repeat_passes=True)
    _record(rec, slug="tau2", task="t0", pass_idx=0, slot=0, success=True)
    assert "multipass/global/cumulative_success_rate" in rec.snapshot()
    assert not any(k.startswith("multipass/tau2/") for k in rec.snapshot())

    _record(rec, slug="bfcl", task="b0", pass_idx=0, slot=0, success=False)
    snap = rec.snapshot()
    assert snap["multipass/tau2/cumulative_success_rate"] == 1.0
    assert snap["multipass/bfcl/cumulative_success_rate"] == 0.0


def test_no_multipass_family_without_tracking(tmp_path):
    rec = OnlineMetricsRecorder(str(tmp_path / "c.jsonl"), group_n=2, window_episodes=4)
    _feed(rec)
    snap = rec.snapshot()
    assert not any(k.startswith("multipass/") for k in snap)
    assert snap["online/cumulative_success_rate"] == 0.25


def test_env_errors_excluded_from_pooled(tmp_path):
    rec = OnlineMetricsRecorder(
        str(tmp_path / "d.jsonl"), group_n=1, track_repeat_passes=True, window_episodes=10
    )
    _record(rec, task="t0", pass_idx=0, slot=0, success=True)
    _record(rec, task="t1", pass_idx=1, slot=0, success=False, env_error=True)
    snap = rec.snapshot()
    assert snap["multipass/global/episodes"] == 1
    assert snap["multipass/global/window_episodes"] == 1
    assert snap["online/env_error_episodes"] == 1


def test_restore_rebuilds_pooled_and_window(tmp_path):
    path = str(tmp_path / "e.jsonl")
    rec = OnlineMetricsRecorder(path, group_n=2, track_repeat_passes=True, window_episodes=4)
    _feed(rec)
    expected = rec.snapshot()

    restored = OnlineMetricsRecorder(
        path, group_n=2, restore_up_to_step=3, track_repeat_passes=True, window_episodes=4
    )
    assert restored.snapshot() == expected

    # rows after the checkpointed step are replayed later, so they must not be restored
    early = OnlineMetricsRecorder(
        path, group_n=2, restore_up_to_step=1, track_repeat_passes=True, window_episodes=4
    )
    snap = early.snapshot()
    assert snap["multipass/global/episodes"] == 4
    assert snap["multipass/global/window_success_rate"] == 0.25
