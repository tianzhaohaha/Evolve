# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from typing import Any, ClassVar, Optional

from pydantic import ConfigDict

from ...core.agent import Agent
from ...core.types import ModelSettings
from ...utils.settings import RunnerName


class ACEAgent(Agent):

    display_name: ClassVar[str] = "ACE Agent"
    slug_name: ClassVar[str] = "ace"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "gpt-4o"
    curator_model: Optional[str] = None 

    max_num_rounds: int = 3 
    curator_frequency: int = 1
    playbook_token_budget: int = 80000           

    shuffle_mode: str = "isolated"

    benchmark_id: Optional[str] = None

    initial_playbook: Optional[str] = None
    initial_playbook_path: Optional[str] = None

    use_json_mode: bool = True
    runner: RunnerName | None = None
    model_settings: ModelSettings | None = None

    enable_tool_shortlisting: bool = False
    max_selected_tools: int = 30

    use_bulletpoint_analyzer: bool = False
    bulletpoint_analyzer_threshold: float = 0.90

    @classmethod
    def _get_instance_class(cls):
        from .ace_instance import ACEAgentInstance
        return ACEAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.agents.ace.ace_instance:ACEAgentInstance"

    def _get_instance_kwargs(self, session_id: str) -> dict[str, Any]:
        pb = self.initial_playbook
        if pb is None and self.initial_playbook_path:
            with open(self.initial_playbook_path, "r", encoding="utf-8") as fh:
                pb = fh.read()

        return {
            "session_id": session_id,
            "model": self.model,
            "curator_model": self.curator_model or self.model,
            "max_num_rounds": self.max_num_rounds,
            "curator_frequency": self.curator_frequency,
            "playbook_token_budget": self.playbook_token_budget,
            "shuffle_mode": self.shuffle_mode,
            "initial_playbook": pb,
            "use_json_mode": self.use_json_mode,
            "model_settings": self.model_settings,
            "benchmark_id": self.benchmark_id,
            "use_bulletpoint_analyzer": self.use_bulletpoint_analyzer,
            "bulletpoint_analyzer_threshold": self.bulletpoint_analyzer_threshold,
            "enable_tool_shortlisting": self.enable_tool_shortlisting,
            "max_selected_tools": self.max_selected_tools,
        }

    @property
    def model_name(self) -> str:
        return str(self.model).split("/")[-1]

    def get_models_names(self) -> list[str]:
        names = [str(self.model)]
        cm = self.curator_model or self.model
        if cm != self.model:
            names.append(str(cm))
        return names
