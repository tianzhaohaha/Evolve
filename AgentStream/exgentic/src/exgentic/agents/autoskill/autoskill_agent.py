# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar, Optional

from pydantic import ConfigDict

from ...core.agent import Agent
from ...core.types import ModelSettings
from ...utils.settings import RunnerName


class AutoSkillAgent(Agent):

    display_name: ClassVar[str] = "AutoSkill Agent"
    slug_name: ClassVar[str] = "autoskill"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    
    model: str = "gpt-4o"
    skill_model: Optional[str] = None

    retrieve_k: int = 3
    retrieval_threshold: float = 0.4
    bm25_weight: float = 0.1
    dedupe_similarity_threshold: float = 0.4
    embedding_model: str = "all-MiniLM-L6-v2"
    enable_query_rewrite: bool = True
    max_context_chars: int = 6000

    shuffle_mode: str = "isolated"

    benchmark_id: Optional[str] = None

    enable_tool_shortlisting: bool = False
    max_selected_tools: int = 30

    runner: RunnerName | None = None
    model_settings: ModelSettings | None = None

    @classmethod
    def _get_instance_class(cls):
        from .autoskill_instance import AutoSkillAgentInstance
        return AutoSkillAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.agents.autoskill.autoskill_instance:AutoSkillAgentInstance"

    def _get_instance_kwargs(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "model": self.model,
            "skill_model": self.skill_model or self.model,
            "retrieve_k": self.retrieve_k,
            "retrieval_threshold": self.retrieval_threshold,
            "bm25_weight": self.bm25_weight,
            "dedupe_similarity_threshold": self.dedupe_similarity_threshold,
            "embedding_model": self.embedding_model,
            "enable_query_rewrite": self.enable_query_rewrite,
            "max_context_chars": self.max_context_chars,
            "shuffle_mode": self.shuffle_mode,
            "model_settings": self.model_settings,
            "benchmark_id": self.benchmark_id,
            "enable_tool_shortlisting": self.enable_tool_shortlisting,
            "max_selected_tools": self.max_selected_tools,
        }

    @property
    def model_name(self) -> str:
        return str(self.model).split("/")[-1]

    def get_models_names(self) -> list[str]:
        names = [str(self.model)]
        sm = self.skill_model or self.model
        if sm != self.model:
            names.append(str(sm))
        return names
