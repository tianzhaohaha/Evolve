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
# Tolerate ```json fences inside the action tag.
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _extract_json_payload(raw: str) -> Dict[str, Any]:
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
    if not isinstance(arguments, dict):
        raise ValueError("'arguments' must be a JSON object")
    return {"name": name.strip(), "arguments": arguments}


def agentstream_projection(
    text_actions: List[str],
    require_think: bool = True,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Parse policy outputs into exgentic-compatible action payloads."""
    payloads: List[Dict[str, Any]] = []
    valids: List[int] = []

    for raw in text_actions:
        raw = raw if isinstance(raw, str) else str(raw)
        valid = 1
        payload: Dict[str, Any] = {"name": "", "arguments": {}}

        match = _ACTION_RE.search(raw)
        if match is None:
            valid = 0
        else:
            try:
                payload = _extract_json_payload(match.group(1))
            except Exception:
                valid = 0

        if valid and require_think:
            think = _THINK_RE.search(raw)
            if think is None or not think.group(1).strip():
                valid = 0

        payloads.append(payload)
        valids.append(valid)

    return payloads, valids
