# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("amem")


class EmbeddingRetriever:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[str]) -> None:
        if not documents:
            return

        if not self.corpus:
            self.corpus = list(documents)
            self.embeddings = self.model.encode(documents)
        else:
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def reset(self, documents: List[str]) -> None:
        self.corpus = []
        self.embeddings = None
        if documents:
            self.add_documents(documents)

    def search(self, query: str, k: int = 5) -> List[int]:
        if not self.corpus or self.embeddings is None:
            return []

        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity(
            [query_embedding], self.embeddings
        )[0]
        k = min(k, len(self.corpus))
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return top_k_indices.tolist()
