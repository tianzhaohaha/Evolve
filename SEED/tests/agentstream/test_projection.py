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

from agent_system.environments.env_package.agentstream.projection import (
    agentstream_projection,
    agentstream_projection_detailed,
)

GOOD = '<think>plan</think><action>{"name": "search", "arguments": {"query": "x"}}</action>'


def test_valid_think_and_action():
    payloads, valids = agentstream_projection([GOOD])
    assert valids == [1]
    assert payloads[0] == {"name": "search", "arguments": {"query": "x"}}


def test_reasons_cover_each_failure_mode():
    cases = {
        "no_action_tag": "<think>plan</think>no action here",
        "bad_action_json": "<think>plan</think><action>{not json}</action>",
        "missing_think": '<action>{"name": "finish", "arguments": {}}</action>',
    }
    _, valids, extras = agentstream_projection_detailed(list(cases.values()))
    assert valids == [0, 0, 0]
    assert [e["reason"] for e in extras] == list(cases)


def test_require_think_can_be_relaxed_and_fences_are_tolerated():
    raw = '<action>```json\n{"name": "finish", "arguments": {"answer": "42"}}\n```</action>'
    payloads, valids, extras = agentstream_projection_detailed([raw], require_think=False)
    assert valids == [1]
    assert payloads[0]["arguments"] == {"answer": "42"}
    assert extras[0]["think_present"] is False


def test_tool_call_alias_only_with_opt_in_and_coerces_string_arguments():
    raw = '<think>t</think><tool_call>{"name": "finish", "arguments": "{\\"answer\\": \\"a\\"}"}</tool_call>'
    _, strict, _ = agentstream_projection_detailed([raw])
    payloads, relaxed, extras = agentstream_projection_detailed([raw], accept_tool_call=True)
    assert strict == [0]
    assert relaxed == [1]
    assert payloads[0]["arguments"] == {"answer": "a"}
    assert extras[0]["used_tool_call_alias"] is True


def test_batch_shape_is_preserved_for_invalid_rows():
    payloads, valids = agentstream_projection([GOOD, "garbage", GOOD])
    assert valids == [1, 0, 1]
    assert len(payloads) == 3
    assert payloads[1] == {"name": "", "arguments": {}}
