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

import pytest

from agent_system.environments.env_package.agentstream.as_config import (
    DEFAULT_BENCHMARK_KWARGS,
    DEFAULT_OBSERVATION_MAX_CHARS,
    parse_agentstream_config,
    parse_int_mapping,
    resolve_benchmark_kwargs,
)


def _config(**agentstream):
    base = {"exgentic_root": "/x", "benchmarks": ["bfcl"]}
    base.update(agentstream)
    return {"env": {"agentstream": base}}


def test_resolve_benchmark_kwargs_layers_overrides_on_defaults():
    merged = resolve_benchmark_kwargs("hle", {"judge_model": "m"})
    assert merged["text_only"] is True  # default kept
    assert merged["judge_model"] == "m"  # override applied
    assert DEFAULT_BENCHMARK_KWARGS["hle"] == {"text_only": True, "agent_timeout": 3600}
    assert resolve_benchmark_kwargs("unknown", None) == {}


def test_parse_int_mapping_accepts_json_string_mapping_and_none():
    assert parse_int_mapping('{"hle": 2}', "k") == {"hle": 2}
    assert parse_int_mapping({"a": "3"}, "k") == {"a": 3}
    assert parse_int_mapping(None, "k") == {}
    assert parse_int_mapping("  ", "k") == {}
    with pytest.raises(ValueError):
        parse_int_mapping("[1]", "k")


def test_episode_limits_fall_back_to_globals():
    cfg = parse_agentstream_config(
        _config(
            benchmarks="hle,browsecompplus",
            max_steps_per_benchmark='{"hle": 2}',
            observation_max_chars_per_benchmark={"browsecompplus": 8192},
        )
    )
    assert cfg.benchmarks == ["browsecompplus", "hle"]
    assert cfg.episode_limits("hle", 40) == {"max_steps": 2, "observation_max_chars": DEFAULT_OBSERVATION_MAX_CHARS}
    assert cfg.episode_limits("browsecompplus", 40) == {"max_steps": 40, "observation_max_chars": 8192}


def test_isolated_requires_single_benchmark():
    with pytest.raises(ValueError):
        parse_agentstream_config(_config(benchmarks=["bfcl", "hle"], stream_mode="isolated"))
