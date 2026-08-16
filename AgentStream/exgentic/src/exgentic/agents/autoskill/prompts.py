# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

QUERY_REWRITE_PROMPT = """\
You are a retrieval query rewriter. Your job is to rewrite the current user task \
into a concise, standalone search query for skill retrieval.

Core rules:
- Produce exactly ONE line of output: the rewritten query.
- Resolve references ("it", "this", "the above") using the provided context.
- Keep only retrieval-relevant constraints (format, audience, quality, domain).
- Preserve the task anchor (what the task is about).
- Do NOT include generic process words without a concrete topic anchor.

Task: {task}
Context: {context}

Rewritten query:"""


SKILL_EXTRACTION_PROMPT = """\
You are a skill extractor that turns agent interaction traces into reusable skills.

## Extraction Principles
- Treat the task description and environment observations as primary evidence.
- Extract ONLY when there are durable, reusable constraints, policies, workflows, \
or strategies that would help in FUTURE similar tasks.
- Do NOT extract one-shot task-specific facts or generic "be helpful" patterns.
- Capture HOW TO DO similar tasks, rather than this-instance facts.
- Remove case-specific entities (names, URLs, dates) and preserve only portable rules.
- Do NOT invent workflow steps unless explicitly demonstrated in the trace.
- If nothing reusable is found, return an empty skills list.

## Session Information
Task: {task}

## Session Trace (Actions & Observations)
{session_trace}

## Output Format
Return a JSON object with this schema:
{{
  "skills": [
    {{
      "name": "<concise, searchable name>",
      "description": "<what this skill does and when to use it>",
      "instructions": "<markdown body with # Goal, # Constraints & Style, # Workflow (optional)>",
      "triggers": ["<intent phrase 1>", "<intent phrase 2>", ...],
      "tags": ["<keyword1>", "<keyword2>", ...],
      "confidence": <float 0.0-1.0>
    }}
  ]
}}

If nothing reusable is detected, return: {{"skills": []}}
"""

SKILL_JUDGE_PROMPT = """\
You are a skill set manager. Given a newly extracted skill candidate and the most \
similar existing skill from the skill bank, decide the appropriate action.

## Decision Procedure
1. Check if the candidate represents the same capability as the existing skill \
(same job-to-be-done, same deliverable type, overlapping constraints).
2. Apply discard gate: reject generic, low-signal, non-portable candidates.
3. Compare on four axes: job-to-be-done, deliverable type, hard constraints/success \
criteria, and required tools/workflow.
4. Choose "merge" ONLY when they are the same capability after removing instance details.
5. Choose "add" when the candidate is a distinct durable capability.
6. Choose "discard" when the candidate is too generic or non-reusable.

## Candidate Skill
Name: {candidate_name}
Description: {candidate_description}
Instructions: {candidate_instructions}
Triggers: {candidate_triggers}
Tags: {candidate_tags}

## Most Similar Existing Skill (may be empty if no skills exist)
Name: {existing_name}
Description: {existing_description}
Instructions: {existing_instructions}
Triggers: {existing_triggers}
Tags: {existing_tags}
Similarity Score: {similarity_score}

## Output Format
Return a JSON object:
{{
  "action": "add" | "merge" | "discard",
  "target_skill_id": "<id of existing skill to merge with, or null>",
  "reason": "<brief explanation>"
}}
"""

SKILL_MERGE_PROMPT = """\
You are a skill merger. Combine an existing skill with a new candidate into one \
improved skill that preserves the best of both.

## Merge Rules
- Preserve the original capability identity (name and core goal).
- Perform semantic union rather than raw concatenation.
- Import only reusable, non-conflicting additions from the candidate.
- Avoid regressions: keep important checks from the existing skill.
- Remove case-specific entities and one-off facts.
- Do NOT invent any new standards or details not present in either skill.
- Deduplicate sections, bullets, triggers, tags.
- Keep language consistent across all fields.

## Existing Skill
Name: {existing_name}
Description: {existing_description}
Instructions: {existing_instructions}
Triggers: {existing_triggers}
Tags: {existing_tags}

## Candidate Skill (new evidence)
Name: {candidate_name}
Description: {candidate_description}
Instructions: {candidate_instructions}
Triggers: {candidate_triggers}
Tags: {candidate_tags}

## Output Format
Return a JSON object with the merged skill:
{{
  "name": "<merged name>",
  "description": "<merged description>",
  "instructions": "<merged instructions (markdown)>",
  "triggers": ["<trigger1>", ...],
  "tags": ["<tag1>", ...]
}}
"""

SKILL_CONTEXT_TEMPLATE = """\
## Retrieved Skills (from accumulated experience)
The following skills were retrieved based on relevance to the current task. \
Use a skill ONLY when it directly matches the current intent. \
Otherwise, ignore all retrieved skills and act normally. \
Never explicitly mention that skills were retrieved/injected.

{skills_block}
"""

SKILL_ENTRY_TEMPLATE = """\
### Skill: {name}
- **Description**: {description}
- **Tags**: {tags}
- **Triggers**: {triggers}

**Instructions**:
{instructions}
"""
