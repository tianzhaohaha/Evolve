# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar, Optional

from pydantic import ConfigDict

from ...core.agent import Agent
from ...core.types import ModelSettings
from ...utils.settings import RunnerName


class ReasoningBankAgent(Agent):

    display_name: ClassVar[str] = "ReasoningBank Agent"
    slug_name: ClassVar[str] = "reasoning_bank"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "gpt-4o"
    memory_model: Optional[str] = None
    eval_model: Optional[str] = None
    embedding_model: str = "all-MiniLM-L6-v2" 

    top_k_memories: int = 1
    max_memory_items: int = 3

    shuffle_mode: str = "isolated"

    benchmark_id: Optional[str] = None

    enable_tool_shortlisting: bool = False
    max_selected_tools: int = 30

    runner: RunnerName | None = None
    model_settings: ModelSettings | None = None

    @classmethod
    def _get_instance_class(cls):
        from .rb_instance import ReasoningBankAgentInstance
        return ReasoningBankAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.agents.reasoning_bank.rb_instance:ReasoningBankAgentInstance"

    def _get_instance_kwargs(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "model": self.model,
            "memory_model": self.memory_model or self.model,
            "eval_model": self.eval_model or self.model,
            "embedding_model": self.embedding_model,
            "top_k_memories": self.top_k_memories,
            "max_memory_items": self.max_memory_items,
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
        mm = self.memory_model or self.model
        if mm != self.model:
            names.append(str(mm))
        em = self.eval_model or self.model
        if em != self.model and em != mm:
            names.append(str(em))
        return names
