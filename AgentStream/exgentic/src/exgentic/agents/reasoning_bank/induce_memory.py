# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def format_trajectory(think_list: list[str], action_list: list[str], observation_list: list[str] | None = None) -> str:
    trajectory = []
    obs_list = observation_list or []
    for i, (t, a) in enumerate(zip(think_list, action_list)):
        if t:
            trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{a}\n</action>")
        else:
            obs = obs_list[i] if i < len(obs_list) else ""
            if obs:
                trajectory.append(f"<observation>\n{obs}\n</observation>\n<action>\n{a}\n</action>")
            else:
                trajectory.append(f"<action>\n{a}\n</action>")
    return "\n\n".join(trajectory)


def induce_memory(
    query: str,
    think_list: list[str],
    action_list: list[str],
    status: str,
    eval_thoughts: str,
    llm_call_fn: Callable[[str, str], str],
    observation_list: list[str] | None = None,
) -> list[str]:
    from .prompts.memory_instruction import FAILED_SI, SUCCESSFUL_SI

    trajectory = format_trajectory(think_list, action_list, observation_list)
    trajectory = f"**Query:** {query}\n\n**Trajectory:**\n{trajectory}"

    if eval_thoughts:
        status_label = "succeeded" if status == "success" else "failed"
        trajectory += f"\n\nThe task {status_label} because: {eval_thoughts}"

    if status == "success":
        generated_text = llm_call_fn(trajectory, SUCCESSFUL_SI)
    else:
        generated_text = llm_call_fn(trajectory, FAILED_SI)

    memory_items = [item.strip() for item in generated_text.split("\n\n") if item.strip()]

    logger.info(
        "ReasoningBank memory induction: status=%s, generated %d items",
        status, len(memory_items),
    )

    return memory_items
