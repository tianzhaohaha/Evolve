# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import logging
import math
from typing import List, Tuple

from .harness_store import HarnessSkill, HarnessStore

logger = logging.getLogger(__name__)

_st_model = None
_st_model_name = None


def _get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    global _st_model, _st_model_name
    if _st_model is None or _st_model_name != model_name:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model: %s", model_name)
        _st_model = SentenceTransformer(model_name)
        _st_model_name = model_name
    return _st_model


def compute_embedding(
    text: str, model: str = "all-MiniLM-L6-v2"
) -> List[float]:
    try:
        st = _get_st_model(model)
        vec = st.encode([text])[0]
        return vec.tolist()
    except Exception as exc:
        logger.error("Local embedding failed: %s", exc)
        return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_skills(
    task_text: str,
    store: HarnessStore,
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[Tuple[HarnessSkill, float]]:
    if store.skill_count == 0:
        return []

    query_embedding = compute_embedding(task_text, model=embedding_model)
    if not query_embedding:
        logger.warning("Failed to compute query embedding, returning no skills")
        return []

    skills = store.list_skills()
    embeddings = store.get_embeddings()
    for skill in skills:
        if skill.name not in embeddings:
            emb = compute_embedding(skill.description, model=embedding_model)
            if emb:
                store.set_embedding(skill.name, emb)

    embeddings = store.get_embeddings()
    scored: List[Tuple[HarnessSkill, float]] = []
    for skill in skills:
        emb = embeddings.get(skill.name)
        if emb:
            score = cosine_similarity(query_embedding, emb)
            scored.append((skill, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = scored[:top_k]

    if results:
        logger.info(
            "Skill retrieval: query='%s...' → retrieved %d/%d skills: %s",
            task_text[:60], len(results), len(skills),
            [(s.name, f"{score:.3f}") for s, score in results],
        )
    else:
        logger.info("Skill retrieval: no skills scored above 0")

    return results
