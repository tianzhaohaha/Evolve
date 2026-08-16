# Copyright 2026 The Google Research Authors.
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

logger = logging.getLogger(__name__)


class TrajectoryEvaluator:

    def __init__(self, llm_call_fn):
        self._llm_call = llm_call_fn

    def evaluate(
        self,
        intent: str,
        think_list: list[str],
        action_list: list[str],
        observation_list: list[str],
        final_response: str = "",
    ) -> dict[str, str]:
        from .prompts.eval_prompts import build_text_eval_prompt, extract_content

        action_history = ""
        for idx, act in enumerate(action_list):
            think = think_list[idx] if idx < len(think_list) else ""
            if think:
                action_history += f"{idx+1}: <think>{think}</think>\n   <action>{act}</action>\n"
            else:
                action_history += f"{idx+1}: <action>{act}</action>\n"

        last_obs = observation_list[-5:] if len(observation_list) >= 5 else observation_list
        combined_obs = "\n\n---\n\n".join(
            f"[Page state {i+1}/{len(last_obs)}]\n{c}"
            for i, c in enumerate(last_obs)
        )

        MAX_OBS_CHARS = 40000
        if len(combined_obs) > MAX_OBS_CHARS:
            combined_obs = combined_obs[:MAX_OBS_CHARS]

        prompt, sys_msg = build_text_eval_prompt(
            combined_obs, intent, final_response, action_history
        )

        msg_str = self._llm_call(prompt, sys_msg)

        thoughts = extract_content(msg_str, "Thoughts:")
        status_raw = extract_content(msg_str, "Status:").replace('"', "").strip().lower()


        if "success" in status_raw:
            status = "success"
        else:
            status = "failure"

        return {"thoughts": thoughts, "status": status}
