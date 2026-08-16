# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .prompts import (
    ANALYZE_CONTENT_PROMPT,
    FOCUSED_KEYWORDS_PROMPT,
    heuristic_context,
    heuristic_keywords,
    parse_analyze_content,
    validate_analysis_result,
    _parse_list_items,
)

logger = logging.getLogger("amem")


@dataclass
class MemoryNote:

    content: str
    id: str = ""
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    context: str = "General"
    links: List[int] = field(default_factory=list)
    importance_score: float = 1.0
    retrieval_count: int = 0
    timestamp: str = ""
    last_accessed: str = ""
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    category: str = "Uncategorized"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        if not self.timestamp:
            self.timestamp = current_time
        if not self.last_accessed:
            self.last_accessed = current_time
        # Ensure context is a string
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

    @staticmethod
    def analyze_content(
        content: str,
        llm_call: Callable[[str], str],
    ) -> Dict[str, Any]:
        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        try:
            response = llm_call(prompt)
            analysis = parse_analyze_content(response, content)

            # Retry focused keywords if empty
            if not analysis["keywords"]:
                logger.info(
                    "Keywords empty after initial parse -- retrying with focused prompt"
                )
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = llm_call(retry_prompt)
                analysis["keywords"] = _parse_list_items(retry_response)

            return validate_analysis_result(analysis, content)

        except Exception as e:
            logger.error("Error analyzing content: %s", e)
            return {
                "keywords": heuristic_keywords(content),
                "context": heuristic_context(content),
                "tags": heuristic_keywords(content, 3),
            }

    @classmethod
    def create_with_analysis(
        cls,
        content: str,
        llm_call: Callable[[str], str],
        timestamp: Optional[str] = None,
        importance_score: float = 1.0,
    ) -> "MemoryNote":
        analysis = cls.analyze_content(content, llm_call)
        return cls(
            content=content,
            keywords=analysis["keywords"],
            context=analysis["context"],
            tags=analysis["tags"],
            timestamp=timestamp or "",
            importance_score=importance_score,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "keywords": self.keywords,
            "tags": self.tags,
            "context": self.context,
            "links": self.links,
            "importance_score": self.importance_score,
            "retrieval_count": self.retrieval_count,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "evolution_history": self.evolution_history,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNote":
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            keywords=data.get("keywords", []),
            tags=data.get("tags", []),
            context=data.get("context", "General"),
            links=data.get("links", []),
            importance_score=data.get("importance_score", 1.0),
            retrieval_count=data.get("retrieval_count", 0),
            timestamp=data.get("timestamp", ""),
            last_accessed=data.get("last_accessed", ""),
            evolution_history=data.get("evolution_history", []),
            category=data.get("category", "Uncategorized"),
        )

    def to_retrieval_document(self) -> str:
        return (
            f"content:{self.content} "
            f"context:{self.context} "
            f"keywords: {', '.join(self.keywords)} "
            f"tags: {', '.join(self.tags)}"
        )
