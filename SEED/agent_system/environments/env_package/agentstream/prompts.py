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

"""Prompt templates for AgentStream benchmarks, in SEED house style.

Kept inside the package (instead of agent_system/environments/prompts/) so the
integration stays fully self-contained. The template family mirrors
ALFWORLD_TEMPLATE / ALFWORLD_TEMPLATE_NO_HIS: <think> reasoning + <action>
answer, with a bounded action history window.

The action payload is a JSON object because AgentStream benchmarks expose
structured tool schemas rather than a flat admissible-command list.
"""

# Per-benchmark one-line intros keep prompts short while giving the policy a
# stable domain cue (useful for sequential/interleaved streams).
BENCHMARK_INTROS = {
    "bfcl": "You are an expert function-calling agent. Solve the task by calling the correct functions with correct arguments.",
    "tau2": "You are an expert customer-service agent. Help the user while strictly following the domain policy.",
    "appworld": "You are an expert digital assistant operating apps through code-based APIs on behalf of a supervisor.",
    "hle": "You are an expert examinee answering an extremely difficult exam question.",
    "browsecompplus": "You are an expert research agent. Use the search tools to locate evidence and answer the question.",
    "swebench": "You are an expert software engineer. Fix the described issue in the repository.",
}

DEFAULT_INTRO = "You are an expert agent interacting with a benchmark environment."


AGENTSTREAM_TEMPLATE_NO_HIS = """
{benchmark_intro}
Your task is: {task_description}
Task context: {task_context}
Your current observation is: {current_observation}
The actions you may take (name, description, arguments) are:
{available_actions}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one action and present it within <action> </action> tags as a JSON object of the form {{"name": "<action_name>", "arguments": {{...}}}}.
"""

AGENTSTREAM_TEMPLATE = """
{benchmark_intro}
Your task is: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
The actions you may take (name, description, arguments) are:
{available_actions}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, choose exactly one action and present it within <action> </action> tags as a JSON object of the form {{"name": "<action_name>", "arguments": {{...}}}}.
"""


def render_prompt(
    *,
    slug: str,
    task: str,
    context: str,
    actions_text: str,
    observation: str,
    step_count: int = 0,
    history_text: str = "",
    history_len: int = 0,
) -> str:
    """Single prompt renderer shared by the RL manager and the SFT pipeline."""
    intro = BENCHMARK_INTROS.get(slug, DEFAULT_INTRO)
    if step_count <= 0 or not history_text:
        return AGENTSTREAM_TEMPLATE_NO_HIS.format(
            benchmark_intro=intro,
            task_description=task,
            task_context=context or "{}",
            current_observation=observation,
            available_actions=actions_text,
        )
    return AGENTSTREAM_TEMPLATE.format(
        benchmark_intro=intro,
        task_description=task,
        step_count=step_count,
        history_length=history_len,
        action_history=history_text,
        current_step=step_count + 1,
        current_observation=observation,
        available_actions=actions_text,
    )
