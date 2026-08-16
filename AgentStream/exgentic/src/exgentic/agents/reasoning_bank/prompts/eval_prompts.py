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


def extract_content(text: str, start_tag: str) -> str:
    for line in text.split("\n"):
        if line.startswith(start_tag):
            return line[len(start_tag):].strip()
    return ""


def build_text_eval_prompt(
    cap: str, intent: str, response: str, last_actions: str
) -> tuple[str, str]:
    system_msg = """You are an expert in evaluating the performance of a task-solving agent. The agent is designed to help a human user complete a task by taking actions in an environment. Given the user's intent, the agent's action history, the environment's feedback, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not.

*Strictness rules*
Before calling a task successful, verify all three:
- Completeness: every constraint in the intent is satisfied.
- Grounding: every value or result the agent reports is traceable to a specific observation from the environment; values that were inferred, guessed, or summarized without a visible source count as failures.
- Right target: when the task names a specific entity, confirm the agent acted on that exact entity and not an adjacent one.
When uncertain on any of these, mark failure. A false success is more harmful than a false failure, because memory induction amplifies it into future behavior.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>"
Status: "success" or "failure"
"""
    prompt = f"""User Intent: {intent}

Action History:
{last_actions}

Environment feedback (last observations):

```
{cap}
```

Agent response to the user: {response if response else "N/A"}."""
    return prompt, system_msg
