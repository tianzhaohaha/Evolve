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

"""Text-action projection for AgentStream environments.

Follows the SEED projection convention (see env_package/*/projection.py):
``projection(text_actions) -> (payloads, valids)`` where each payload is a
plain dict ``{"name": str, "arguments": dict}`` consumable by the worker-side
``SessionDriver.step``. Invalid outputs keep the batch shape and are marked
``valids[i] = 0`` so the trainer's invalid-action penalty applies unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Native tool-call alias (e.g. Qwen3-*-2507), only honored with accept_tool_call.
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# Tolerate ```json fences inside the action tag.
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _extract_json_payload(raw: str, coerce_string_arguments: bool = False) -> Dict[str, Any]:
    text = raw.strip()
    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("action payload must be a JSON object")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("action payload must contain a non-empty 'name'")
    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if coerce_string_arguments and isinstance(arguments, str):
        # Native tool-call emitters sometimes serialize arguments as a JSON string.
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("'arguments' must be a JSON object")
    return {"name": name.strip(), "arguments": arguments}


def agentstream_projection_detailed(
    text_actions: List[str],
    require_think: bool = True,
    accept_tool_call: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int], List[Dict[str, Any]]]:
    """Like :func:`agentstream_projection`, plus per-sample diagnostics.

    ``accept_tool_call`` falls back to the first ``<tool_call>`` block when no
    ``<action>`` tag is present (payload schema is identical). Each extras dict
    carries ``reason`` ("" | "no_action_tag" | "bad_action_json" |
    "missing_think"), ``think_present`` and ``used_tool_call_alias``.
    """
    payloads: List[Dict[str, Any]] = []
    valids: List[int] = []
    extras: List[Dict[str, Any]] = []

    for raw in text_actions:
        raw = raw if isinstance(raw, str) else str(raw)
        valid = 1
        reason = ""
        used_alias = False
        payload: Dict[str, Any] = {"name": "", "arguments": {}}

        match = _ACTION_RE.search(raw)
        coerce = False
        if match is None and accept_tool_call:
            match = _TOOL_CALL_RE.search(raw)
            used_alias = match is not None
            coerce = True
        if match is None:
            valid = 0
            reason = "no_action_tag"
        else:
            try:
                payload = _extract_json_payload(match.group(1), coerce_string_arguments=coerce)
            except Exception:
                valid = 0
                reason = "bad_action_json"

        think = _THINK_RE.search(raw)
        think_present = bool(think is not None and think.group(1).strip())
        if valid and require_think and not think_present:
            valid = 0
            reason = "missing_think"

        payloads.append(payload)
        valids.append(valid)
        extras.append(
            {
                "reason": reason,
                "think_present": think_present,
                "used_tool_call_alias": used_alias,
            }
        )

    return payloads, valids, extras


def agentstream_projection(
    text_actions: List[str],
    require_think: bool = True,
    accept_tool_call: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Parse policy outputs into exgentic-compatible action payloads."""
    payloads, valids, _ = agentstream_projection_detailed(
        text_actions, require_think=require_think, accept_tool_call=accept_tool_call
    )
    return payloads, valids
