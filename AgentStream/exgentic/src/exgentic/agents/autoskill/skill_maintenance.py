# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Tuple

from .prompts import SKILL_JUDGE_PROMPT, SKILL_MERGE_PROMPT
from .skill_extraction import extract_json_from_text
from .skill_retrieval import hybrid_search, compute_embedding
from .skill_store import SkillEntry, SkillStore


def judge_skill(
    candidate: SkillEntry,
    existing: Optional[SkillEntry],
    similarity_score: float,
    llm_call: Callable[[str, str], str],
    model: str,
) -> Tuple[str, Optional[str], str]:
    prompt = SKILL_JUDGE_PROMPT.format(
        candidate_name=candidate.name,
        candidate_description=candidate.description,
        candidate_instructions=candidate.instructions[:2000],
        candidate_triggers=", ".join(candidate.triggers),
        candidate_tags=", ".join(candidate.tags),
        existing_name=existing.name if existing else "(none - no existing skill)",
        existing_description=existing.description if existing else "",
        existing_instructions=(existing.instructions[:2000] if existing else ""),
        existing_triggers=", ".join(existing.triggers) if existing else "",
        existing_tags=", ".join(existing.tags) if existing else "",
        similarity_score=f"{similarity_score:.3f}",
    )

    raw = llm_call(model, prompt)
    parsed = extract_json_from_text(raw)

    if parsed and isinstance(parsed, dict):
        action = parsed.get("action", "discard").lower().strip()
        target_id = parsed.get("target_skill_id")
        reason = parsed.get("reason", "")
        if action in ("add", "merge", "discard"):
            if action == "merge" and existing:
                return action, existing.id, reason
            elif action == "merge" and not existing:
                return "add", None, reason + " (no target for merge, adding instead)"
            return action, target_id, reason

    if existing and similarity_score >= 0.82:
        return "merge", existing.id, "high similarity (deterministic fallback)"
    elif not existing or similarity_score <= 0.22:
        return "add", None, "low similarity, distinct skill (deterministic fallback)"
    elif similarity_score >= 0.50:
        return "merge", existing.id, "moderate similarity (deterministic fallback)"
    else:
        return "add", None, "below merge threshold (deterministic fallback)"


def merge_skills(
    existing: SkillEntry,
    candidate: SkillEntry,
    llm_call: Callable[[str, str], str],
    model: str,
) -> SkillEntry:
    prompt = SKILL_MERGE_PROMPT.format(
        existing_name=existing.name,
        existing_description=existing.description,
        existing_instructions=existing.instructions[:3000],
        existing_triggers=json.dumps(existing.triggers, ensure_ascii=False),
        existing_tags=json.dumps(existing.tags, ensure_ascii=False),
        candidate_name=candidate.name,
        candidate_description=candidate.description,
        candidate_instructions=candidate.instructions[:3000],
        candidate_triggers=json.dumps(candidate.triggers, ensure_ascii=False),
        candidate_tags=json.dumps(candidate.tags, ensure_ascii=False),
    )

    raw = llm_call(model, prompt)
    parsed = extract_json_from_text(raw)

    merged = SkillEntry(
        id=existing.id,
        name=existing.name,
        description=existing.description,
        instructions=existing.instructions,
        triggers=list(existing.triggers),
        tags=list(existing.tags),
        examples=list(existing.examples),
        version=existing.version,
        confidence=max(existing.confidence, candidate.confidence),
        created_at=existing.created_at,
        updated_at=existing.updated_at,
    )

    if parsed and isinstance(parsed, dict):
        if parsed.get("name"):
            merged.name = parsed["name"]
        if parsed.get("description"):
            merged.description = parsed["description"]
        if parsed.get("instructions"):
            merged.instructions = parsed["instructions"]
        if parsed.get("triggers"):
            merged.triggers = list(set(existing.triggers + parsed["triggers"]))[:10]
        if parsed.get("tags"):
            merged.tags = list(set(existing.tags + parsed["tags"]))[:10]
    else:
        merged.triggers = list(set(existing.triggers + candidate.triggers))[:10]
        merged.tags = list(set(existing.tags + candidate.tags))[:10]
        if candidate.instructions and candidate.instructions not in existing.instructions:
            merged.instructions = (
                existing.instructions + "\n\n## Updated Constraints\n" + candidate.instructions
            )

    merged.bump_version()
    return merged


def maintain_skill(
    candidate: SkillEntry,
    store: SkillStore,
    llm_call: Callable[[str, str], str],
    model: str,
    embedding_model: str = "text-embedding-3-small",
    bm25_weight: float = 0.1,
    dedupe_similarity_threshold: float = 0.4,
    logger: Any = None,
) -> Tuple[str, Optional[SkillEntry]]:
    import logging
    _logger = logger or logging.getLogger(__name__)

    # Compute candidate embedding for retrieval
    candidate_text = candidate.to_search_text()
    candidate_embedding = None
    try:
        candidate_embedding = compute_embedding(candidate_text, model=embedding_model)
    except Exception as exc:
        _logger.warning(
            "AutoSkill maintenance: failed to compute candidate embedding for '%s': %s",
            candidate.name, exc,
        )
    _logger.info(
        "AutoSkill maintenance: candidate='%s', embedding_dim=%d",
        candidate.name, len(candidate_embedding) if candidate_embedding else 0,
    )

    results = hybrid_search(
        store=store,
        query=candidate_text,
        query_embedding=candidate_embedding,
        top_k=1,
        threshold=0.0,
        bm25_weight=bm25_weight,
        embedding_model=embedding_model,
    )

    existing: Optional[SkillEntry] = None
    similarity_score = 0.0
    if results:
        existing, similarity_score = results[0]
        _logger.info(
            "AutoSkill maintenance: best match='%s' (score=%.3f)",
            existing.name, similarity_score,
        )
    else:
        _logger.info("AutoSkill maintenance: no existing skills in store")

    if existing and similarity_score < dedupe_similarity_threshold:
        _logger.info(
            "AutoSkill maintenance: similarity %.3f < threshold %.3f, skipping merge consideration",
            similarity_score, dedupe_similarity_threshold,
        )
        existing = None
        similarity_score = 0.0

    action, _, reason = judge_skill(
        candidate, existing, similarity_score, llm_call, model,
    )
    _logger.info(
        "AutoSkill maintenance: judge decision='%s', reason='%s'",
        action, reason[:100],
    )

    if action == "discard":
        _logger.info("AutoSkill maintenance: discarded candidate '%s'", candidate.name)
        return "discard", None
    elif action == "merge" and existing:
        merged = merge_skills(existing, candidate, llm_call, model)
        store.update_skill(merged)
        if candidate_embedding:
            new_emb = compute_embedding(merged.to_search_text(), model=embedding_model)
            if new_emb:
                store.set_embedding(merged.id, new_emb)
        _logger.info(
            "AutoSkill maintenance: merged into '%s' (v%s → v%s)",
            merged.name, existing.version, merged.version,
        )
        return "merge", merged
    else:
        store.add_skill(candidate)
        if candidate_embedding:
            store.set_embedding(candidate.id, candidate_embedding)
        _logger.info(
            "AutoSkill maintenance: added new skill '%s' (id=%s)",
            candidate.name, candidate.id,
        )
        return "add", candidate
