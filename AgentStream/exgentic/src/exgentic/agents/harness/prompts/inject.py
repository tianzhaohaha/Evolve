# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from typing import List, Tuple

from ..harness_store import HarnessSkill


def build_system_message(
    system_prompt: str,
    memory: str,
    retrieved_skills: List[Tuple[HarnessSkill, float]],
) -> str:
    parts = [system_prompt]

    if memory and memory.strip():
        parts.append("\n\n## Long-Term Memory\n")
        parts.append(memory)

    if retrieved_skills:
        parts.append("\n\n## Retrieved Skills\n")
        for skill, _score in retrieved_skills:
            parts.append(f"### Skill: {skill.name}")
            parts.append(f"*{skill.description}*\n")
            parts.append(skill.body)
            parts.append("")

    return "\n".join(parts)
