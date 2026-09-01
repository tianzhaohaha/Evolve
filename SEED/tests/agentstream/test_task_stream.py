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

from agent_system.environments.env_package.agentstream.task_stream import (
    TaskStreamScheduler,
    order_tasks,
    select_holdout_tasks,
    select_tasks,
)

UNIVERSE = {"hle": [str(i) for i in range(50)], "bfcl": [f"b{i}" for i in range(30)]}


def test_selection_is_deterministic_and_holdout_is_disjoint():
    train = select_tasks(UNIVERSE, 10)
    assert train == select_tasks(UNIVERSE, 10)
    assert all(len(v) == 10 for v in train.values())
    val = select_holdout_tasks(UNIVERSE, 10, 4)
    for slug in UNIVERSE:
        assert len(val[slug]) == 4
        assert not set(val[slug]) & set(train[slug])


def test_interleaved_preserves_within_benchmark_order():
    per_bm = select_tasks(UNIVERSE, 6)
    sequential = order_tasks(per_bm, "sequential", seed=44)
    interleaved = order_tasks(per_bm, "interleaved", seed=44)
    assert sorted(sequential) == sorted(interleaved)
    for slug in per_bm:
        assert [t for s, t in interleaved if s == slug] == [t for s, t in sequential if s == slug]


def test_scheduler_state_round_trips_and_stops_when_exhausted():
    train = select_tasks(UNIVERSE, 4)
    sched = TaskStreamScheduler(train_tasks=train, mode="interleaved", stream_seed=44, on_exhausted="stop")
    assert sched.stream_length == 8
    first = sched.next_batch(3)
    assert first.stream_indices == [0, 1, 2] and first.passes == [0, 0, 0]

    clone = TaskStreamScheduler(train_tasks=train, mode="interleaved", stream_seed=44, on_exhausted="stop")
    clone.load_state_dict(sched.state_dict())
    assert clone.next_batch(3).refs == sched.next_batch(3).refs

    replicated = first.replicate(2)
    assert replicated.refs == [r for r in first.refs for _ in range(2)]
