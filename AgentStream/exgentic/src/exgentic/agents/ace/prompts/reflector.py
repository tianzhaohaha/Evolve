# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

REFLECTOR_PROMPT_NO_GT = """\
You are an expert analyst and educator. Your job is to analyze a model's \
reasoning process and identify potential issues or strengths based on the \
reasoning trace alone.

**Instructions:**
- Carefully analyze the model's reasoning trace to evaluate its approach
- The reasoning trace includes both the model's actions and environment observations in chronological order
- Identify potential conceptual errors, calculation mistakes, or misapplied strategies
- Also note what the model did well
- Provide actionable insights that could help the model perform better in future tasks
- Focus on the root cause, not just surface-level observations
- Be specific about what could be improved
- You will receive the full playbook that was available to the agent.
- Based on the reasoning trace, infer which bullets the agent likely applied or was influenced by, and tag each relevant bullet as 'helpful', 'harmful', or 'neutral'. Skip unrelated bullets.

Your output should be a json object, which contains the following fields
  - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
  - error_identification: what potential issues exist in the reasoning? (or "none identified" if the approach appears sound)
  - root_cause_analysis: why might these issues occur? What concept may have been misunderstood?
  - correct_approach: what could the model do differently or better?
  - key_insight: what strategy, formula, or principle should be remembered for future tasks?
  - bullet_tags: a list of json objects with bullet id and tag for each relevant playbook bullet


**Question:**
{question}

**Model's Reasoning Trace:**
{reasoning_trace}

**Model's Predicted Answer:**
{predicted_answer}

**Full Playbook:**
{bullets_used}

**Answer in this exact JSON format:**
{{
  "reasoning": "[Your chain of thought / reasoning / thinking process]",
  "error_identification": "[What potential issues exist in the reasoning?]",
  "root_cause_analysis": "[Why might these issues occur?]",
  "correct_approach": "[What could the model do differently or better?]",
  "key_insight": "[What strategy or principle should be remembered?]",
  "bullet_tags": [
    {{"id": "calc-00001", "tag": "helpful"}},
    {{"id": "fin-00002", "tag": "harmful"}}
  ]
}}
"""
