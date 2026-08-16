# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

from .prompts import SKILL_EXTRACTION_PROMPT
from .skill_store import SkillEntry


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


def _repair_json_via_llm(
    raw_text: str,
    llm_call: Callable[[str, str], str],
    model: str,
) -> Optional[Dict[str, Any]]:
    repair_prompt = (
        "The following text was supposed to be valid JSON matching the schema "
        '{"skills": [{"name": str, "description": str, "instructions": str, '
        '"triggers": [str], "tags": [str], "confidence": float}]} '
        "but it is malformed. Please fix it and return ONLY valid JSON. "
        'If extraction fails, output {"skills": []}.\n\n'
        f"Malformed text:\n{raw_text[:4000]}"
    )
    try:
        repaired = llm_call(model, repair_prompt)
        if repaired:
            return extract_json_from_text(repaired)
    except Exception:
        pass
    return None


def extract_skills_from_trace(
    task: str,
    benchmark_id: str,
    session_trace: str,
    llm_call: Callable[[str, str], str],
    model: str,
    logger: Optional[Any] = None,
) -> List[SkillEntry]:
    import logging
    _logger = logger or logging.getLogger(__name__)

    prompt = SKILL_EXTRACTION_PROMPT.format(
        task=task,
        session_trace=session_trace,
    )

    _logger.info(
        "AutoSkill extraction: task='%s', trace_len=%d chars",
        task[:80], len(session_trace),
    )

    raw = llm_call(model, prompt)
    if not raw:
        _logger.info("AutoSkill extraction: LLM returned empty response")
        return []

    _logger.debug("AutoSkill extraction: raw LLM response length=%d", len(raw))

    parsed = extract_json_from_text(raw)
    if parsed is None:
        _logger.info("AutoSkill extraction: levels 1-3 JSON parse failed, attempting LLM repair")
        parsed = _repair_json_via_llm(raw, llm_call, model)
    if not parsed or not isinstance(parsed, dict):
        _logger.warning("AutoSkill extraction: all 4 JSON recovery levels failed")
        return []

    skills_data = parsed.get("skills", [])
    if not isinstance(skills_data, list):
        _logger.warning("AutoSkill extraction: 'skills' field is not a list")
        return []

    _logger.info("AutoSkill extraction: LLM returned %d skill candidates", len(skills_data))

    results: List[SkillEntry] = []
    for item in skills_data:
        if not isinstance(item, dict):
            continue

        name = item.get("name", "").strip()
        description = item.get("description", "").strip()
        instructions = item.get("instructions", "").strip()
        confidence = float(item.get("confidence", 0.6))

        if not name or not description:
            _logger.debug("AutoSkill extraction: skipping candidate with empty name/description")
            continue
        if not instructions:
            instructions = description

        entry = SkillEntry(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            instructions=instructions,
            triggers=item.get("triggers", [])[:8],
            tags=item.get("tags", [])[:8],
            confidence=confidence,
        )
        _logger.info(
            "AutoSkill extraction: extracted skill '%s' (confidence=%.2f)",
            name, confidence,
        )
        results.append(entry)

    return results
