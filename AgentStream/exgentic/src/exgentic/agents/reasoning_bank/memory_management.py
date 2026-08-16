# Copyright 2026 Google LLC
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
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from .rb_store import MemoryEntry, ReasoningBankStore

logger = logging.getLogger(__name__)

RETRIEVAL_INSTRUCTION = (
    "Given the prior web navigation queries, your task is to analyze a current "
    "query's intent and select relevant prior queries that could help resolve it."
)

_st_model = None
_st_model_name = None


def _get_st_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _st_model, _st_model_name
    if _st_model is None or _st_model_name != model_name:
        logger.info("Loading SentenceTransformer model: %s", model_name)
        _st_model = SentenceTransformer(model_name)
        _st_model_name = model_name
    return _st_model


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return x / norm


def compute_embedding(text: str, model: str = "all-MiniLM-L6-v2") -> list[float]:
    st = _get_st_model(model)
    vec = st.encode([text])[0]
    return vec.tolist()


def select_memory(
    store: "ReasoningBankStore",
    cur_query: str,
    embedding_model: str,
    top_k: int = 1,
    exclude_task_id: str | None = None,
) -> list["MemoryEntry"]:
    
    cache_emb = store.get_embeddings_array()
    if cache_emb is None or len(cache_emb) == 0:
        logger.info("ReasoningBank retrieval: no cached embeddings, returning empty.")
        return []

    entries = store.get_entries()
    entry_ids = store.get_entry_ids()

    instruction_query = get_detailed_instruct(RETRIEVAL_INSTRUCTION, cur_query)
    instruct_vec = np.array(
        compute_embedding(instruction_query, embedding_model),
        dtype=np.float32,
    ).reshape(1, -1)

    instruct_vec = l2_normalize(instruct_vec, axis=1)
    cache_emb_norm = l2_normalize(cache_emb, axis=1)

    scores = (instruct_vec @ cache_emb_norm.T).squeeze(0) * 100.0  # (N,)

    id_score_pairs = []
    for i, (eid, score) in enumerate(zip(entry_ids, scores)):
        if exclude_task_id and eid == exclude_task_id:
            continue
        id_score_pairs.append((i, float(score)))

    id_score_pairs.sort(key=lambda x: x[1], reverse=True)

    top_entries = []
    for idx, _score in id_score_pairs[:top_k]:
        top_entries.append(entries[idx])

    return top_entries


def format_memories_for_prompt(entries: list["MemoryEntry"]) -> str:
    mem_items = []
    for entry in entries:
        for item in entry.memory_items:
            if item.strip():
                mem_items.append(item.strip())
    return "\n\n".join(mem_items)
