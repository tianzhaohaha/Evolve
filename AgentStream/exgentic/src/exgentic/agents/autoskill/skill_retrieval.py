# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .skill_store import SkillEntry, SkillStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\W\d_]+", re.UNICODE)
_STOPWORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "if", "then", "else", "when",
    "that", "this", "it", "its", "as", "so", "no", "yes",
])


def tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


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


def compute_embedding(text: str, model: str = "all-MiniLM-L6-v2") -> List[float]:
    st = _get_st_model(model)
    vec = st.encode([text])[0]
    return vec.tolist()


def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avg_doc_len: float,
    doc_count: int,
    df: Dict[str, int],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    doc_len = len(doc_tokens)
    doc_tf = Counter(doc_tokens)
    score = 0.0

    for term in query_tokens:
        if term not in doc_tf:
            continue
        tf = doc_tf[term]
        n = df.get(term, 0)
        idf = math.log((doc_count - n + 0.5) / (n + 0.5) + 1.0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1)))
        score += idf * tf_norm

    return score


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def hybrid_search(
    store: SkillStore,
    query: str,
    query_embedding: Optional[List[float]] = None,
    top_k: int = 5,
    threshold: float = 0.3,
    bm25_weight: float = 0.1,
    embedding_model: str = "text-embedding-3-small",
) -> List[Tuple[SkillEntry, float]]:
    skills = store.list_skills()
    if not skills:
        return []

    query_tokens = tokenize(query)
    doc_tokens_map: Dict[str, List[str]] = {}
    df: Dict[str, int] = Counter()

    for skill in skills:
        tokens = tokenize(skill.to_search_text())
        doc_tokens_map[skill.id] = tokens
        for t in set(tokens):
            df[t] += 1

    avg_doc_len = sum(len(t) for t in doc_tokens_map.values()) / max(len(skills), 1)

    bm25_scores: Dict[str, float] = {}
    for skill in skills:
        bm25_scores[skill.id] = bm25_score(
            query_tokens, doc_tokens_map[skill.id],
            avg_doc_len, len(skills), df,
        )

    vec_scores: Dict[str, float] = {}
    if query_embedding:
        embeddings = store.get_embeddings()
        for skill in skills:
            emb = embeddings.get(skill.id)
            if emb:
                vec_scores[skill.id] = cosine_similarity(query_embedding, emb)
            else:
                vec_scores[skill.id] = 0.0
    else:
        bm25_weight = 1.0
        for skill in skills:
            vec_scores[skill.id] = 0.0

    bm25_max = max(bm25_scores.values()) if bm25_scores else 0.0
    norm_bm25: Dict[str, float] = {}
    if bm25_max > 0:
        norm_bm25 = {k: v / bm25_max for k, v in bm25_scores.items()}
    else:
        norm_bm25 = {k: 0.0 for k in bm25_scores}

    final_scores: Dict[str, float] = {}
    for skill in skills:
        sid = skill.id
        final_scores[sid] = (
            (1 - bm25_weight) * vec_scores.get(sid, 0.0)
            + bm25_weight * norm_bm25.get(sid, 0.0)
        )

    skill_map = {s.id: s for s in skills}
    results = [
        (skill_map[sid], score)
        for sid, score in sorted(final_scores.items(), key=lambda x: -x[1])
        if score >= threshold
    ]

    return results[:top_k]
