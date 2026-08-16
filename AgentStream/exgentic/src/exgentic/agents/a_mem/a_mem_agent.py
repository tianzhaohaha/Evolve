# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar, Optional

from pydantic import ConfigDict

from ...core.agent import Agent
from ...core.types import ModelSettings
from ...utils.settings import RunnerName


class AMemAgent(Agent):

    display_name: ClassVar[str] = "A-Mem Agent"
    slug_name: ClassVar[str] = "a_mem"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "gpt-4o"
    memory_model: Optional[str] = None

    retrieve_k: int = 10
    evo_threshold: int = 100 
    embedding_model: str = "all-MiniLM-L6-v2" 

    shuffle_mode: str = "isolated" 

    benchmark_id: Optional[str] = None 

    enable_tool_shortlisting: bool = False
    max_selected_tools: int = 30
    runner: RunnerName | None = None
    model_settings: ModelSettings | None = None

    @classmethod
    def _get_instance_class(cls):
        from .a_mem_instance import AMemAgentInstance
        return AMemAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.agents.a_mem.a_mem_instance:AMemAgentInstance"

    def _get_instance_kwargs(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "model": self.model,
            "memory_model": self.memory_model or self.model,
            "retrieve_k": self.retrieve_k,
            "evo_threshold": self.evo_threshold,
            "embedding_model": self.embedding_model,
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
        return names
