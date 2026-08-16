# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm import (
    ChatCompletionAssistantMessage,
    ChatCompletionSystemMessage,
    ChatCompletionToolMessage,
    ChatCompletionUserMessage,
)

from ...core.agent_instance import AgentInstance
from ...core.types import (
    Action,
    ActionType,
    Message,
    MessageAction,
    MessageObservation,
    MessagePayload,
    ModelSettings,
    Observation,
)
from ...utils.cost import LiteLLMCostReport
from ...utils.settings import get_settings

from .evolver import run_evolver
from .harness_store import HarnessStore
from .prompts.inject import build_system_message
from .retriever import retrieve_skills

try:
    from ...agents.litellm_tool_calling.utils import ToolCall, ToolsActionsRegistry
except ImportError:
    ToolsActionsRegistry = None
    ToolCall = dict

from ..tool_shortlisting import shortlist_tools

settings = get_settings()


class HarnessAgentInstance(AgentInstance):

    def __init__(
        self,
        session_id: str,
        model: str = "gpt-4o",
        evolver_model: str = "gpt-4o",
        top_k_skills: int = 3,
        embedding_model: str = "all-MiniLM-L6-v2",
        shuffle_mode: str = "isolated",
        model_settings: Optional[ModelSettings] = None,
        benchmark_id: Optional[str] = None,
        enable_tool_shortlisting: bool = False,
        max_selected_tools: int = 30,
    ) -> None:
        super().__init__(session_id)

        self.model = model
        self.evolver_model = evolver_model
        self.top_k_skills = top_k_skills
        self.embedding_model = embedding_model
        self.shuffle_mode = shuffle_mode
        self.benchmark_id = benchmark_id
        self.enable_tool_shortlisting = enable_tool_shortlisting
        self.max_selected_tools = max_selected_tools

        if model_settings is None:
            self._model_settings = ModelSettings()
        elif isinstance(model_settings, ModelSettings):
            self._model_settings = model_settings
        else:
            self._model_settings = ModelSettings()

        self._cost = LiteLLMCostReport.initialize_empty(model_name=self.model)
        self._store: Optional[HarnessStore] = None

        self.messages: list[
            Union[
                ChatCompletionAssistantMessage,
                ChatCompletionToolMessage,
                ChatCompletionUserMessage,
                ChatCompletionSystemMessage,
            ]
        ] = []
        self._step_count: int = 0

        self._registry: Optional[ToolsActionsRegistry] = None
        self._all_actions: list[ActionType] = []

        self._observation_log: List[Dict[str, Any]] = []
        self._action_log: List[Dict[str, Any]] = []

        self._injected_skill_names: List[str] = []

    def _log_failure(
        self, component: str, error: Exception, context: Dict[str, Any]
    ) -> None:
        try:
            log_path = self.paths.agent_dir / "harness_failures.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "component": component,
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                **{k: str(v)[:2000] if isinstance(v, str) else v
                   for k, v in context.items()},
            }
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def start(
        self,
        task: str,
        context: Dict[str, Any],
        actions: list[ActionType],
    ) -> None:
        super().start(task, context, actions)

        self._all_actions = list(self.actions)
        if ToolsActionsRegistry is not None:
            self._registry = ToolsActionsRegistry(self._all_actions)

        task_group = str(
            context.get("task_group")
            or context.get("task_id")
            or context.get("task_name")
            or "default"
        )
        self._store = HarnessStore.get_or_create(
            shuffle_mode=self.shuffle_mode,
            task_group=task_group,
            benchmark_id=self.benchmark_id,
        )
        self._store.increment_session()

        retrieved = []
        if self._store.skill_count > 0:
            try:
                retrieved = retrieve_skills(
                    task_text=task,
                    store=self._store,
                    top_k=self.top_k_skills,
                    embedding_model=self.embedding_model,
                )
            except Exception as exc:
                self.logger.warning("Harness: skill retrieval failed: %s", exc)

        self._injected_skill_names = [s.name for s, _ in retrieved]

        # Mark retrieved skills as used (LRU)
        if self._injected_skill_names:
            self._store.touch_skills(self._injected_skill_names)

        # Build system message
        system_content = build_system_message(
            system_prompt=self._store.system_prompt,
            memory=self._store.memory,
            retrieved_skills=retrieved,
        )
        self._add_message(
            ChatCompletionSystemMessage(role="system", content=system_content)
        )
        self.logger.info(
            "Harness system message built: prompt_len=%d  memory_len=%d  "
            "skills_injected=%s  total_system_len=%d",
            len(self._store.system_prompt),
            len(self._store.memory),
            self._injected_skill_names,
            len(system_content),
        )

        content_parts: list[Any] = []
        ctx = ""
        if self.context:
            for k, v in self.context.items():
                if isinstance(v, dict) and v.get("type") == "image_url":
                    content_parts.append({"type": "image_url", "image_url": {"url": v["data"], "detail": "high"}})
                else:
                    ctx += f"\n<{k}>\n{v}\n</{k}>"

        text_content = f"{self.task}\n{ctx}"
        if content_parts:
            content_parts.insert(0, {"type": "text", "text": text_content})
            self._add_message(ChatCompletionUserMessage(role="user", content=content_parts))
        else:
            self._add_message(
                ChatCompletionUserMessage(role="user", content=text_content)
            )

        self.logger.info(
            "Harness instance started  store=%s  session_count=%d  "
            "skill_count=%d  skills_injected=%d  benchmark=%s  tools=%d",
            self._store.store_id,
            self._store.session_count,
            self._store.skill_count,
            len(self._injected_skill_names),
            self.benchmark_id or "(none)",
            len(self._all_actions),
        )

    def react(self, observation: Optional[Observation]) -> Optional[Action]:
        self._step_count += 1

        self._observe(observation)
        self._log_observation(observation)

        tools = self._assistant_tools()
        response = self._completion(
            model=self.model,
            messages=self.messages,
            tools=tools if tools else None,
        )

        if response is None:
            self.logger.error("Harness: LLM returned None response")
            return None

        if response.usage:
            self._cost.update_cost_from_tokens(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")

        if finish_reason == "tool_calls" and self._registry is not None:
            tool_calls = self._extract_tool_calls(message)
            self._add_message(
                ChatCompletionAssistantMessage(
                    role="assistant",
                    tool_calls=[
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ],
                )
            )
            actions = self._registry.tool_calls_to_action(tool_calls)

            for tc in tool_calls:
                self._action_log.append({
                    "step": self._step_count,
                    "action": tc["name"],
                    "arguments": tc["arguments"],
                })

            self.logger.info("Harness step %d: tool_calls=%s", self._step_count,
                             [tc["name"] for tc in tool_calls])
            return actions
        else:
            content = message.content if message.content else ""
            self._add_message(
                ChatCompletionAssistantMessage(
                    role="assistant", content=content
                )
            )
            self._action_log.append({
                "step": self._step_count,
                "action": "message",
                "content": content,
            })
            self.logger.info("Harness step %d: message response", self._step_count)
            return MessageAction(arguments=Message(content=content))

    def close(self) -> None:
        store = self._store
        if store is None:
            return

        ops_applied = 0
        ops_summary: list[str] = []

        self.logger.info(
            "Harness close: starting evolver  trajectory_steps=%d  "
            "actions=%d  observations=%d",
            self._step_count, len(self._action_log), len(self._observation_log),
        )

        if self._observation_log or self._action_log:
            try:
                trajectory = self._build_session_trace()
                self.logger.info(
                    "Harness close: trajectory built  length=%d chars", len(trajectory)
                )
                ops_applied, ops_summary = run_evolver(
                    store=store,
                    task=self.task if self.task else "",
                    injected_skill_names=self._injected_skill_names,
                    trajectory=trajectory,
                    llm_call=self._llm_call_simple,
                    evolver_model=self.evolver_model,
                    embedding_model=self.embedding_model,
                )
                self.logger.info(
                    "Harness close: evolver done  ops_applied=%d  ops=%s",
                    ops_applied, ops_summary,
                )
            except Exception as exc:
                self.logger.warning("Harness: evolver failed: %s", exc)
                self._log_failure("evolver", exc, {
                    "observation_count": len(self._observation_log),
                    "action_count": len(self._action_log),
                })

        if ops_applied > 0:
            version = store.commit_version(
                session_id=self.session_id,
                ops_summary=ops_summary,
            )
            self.logger.info(
                "Harness close: version committed  v=%d  skill_count=%d",
                version, store.skill_count,
            )

        store.record_learning(
            session_id=self.session_id,
            task_id=str(self.context.get("task_id", "") if self.context else ""),
            benchmark_id=self.benchmark_id or "",
            ops_applied=ops_applied,
            ops_summary=ops_summary,
        )

        try:
            cp = str(self.paths.agent_dir / "harness_checkpoint.json")
            store.save_checkpoint(cp)
            txt = str(self.paths.agent_dir / "harness_state.md")
            store.save_harness_text(txt)
        except Exception as exc:
            self.logger.warning("Harness: failed to save checkpoint: %s", exc)

    def get_cost(self) -> LiteLLMCostReport:
        return self._cost

    def _add_message(self, message: Any) -> None:
        self.messages.append(message)

    def _observe(self, observation: Optional[Observation]) -> None:
        if observation is None:
            return

        observations = observation.to_observation_list()
        if observation.is_empty():
            if not any(obs.invoking_actions for obs in observations):
                return

        for obs in observations:
            if isinstance(obs, MessageObservation) and isinstance(
                obs.result, MessagePayload
            ):
                self._add_message(
                    ChatCompletionUserMessage(
                        role="user", content=obs.result.message
                    )
                )
                continue

            if len(obs.invoking_actions) > 0:
                invoking = obs.invoking_actions[0]
                if invoking.name == "message":
                    self._add_message(
                        ChatCompletionUserMessage(
                            role="user", content=str(obs)
                        )
                    )
                    continue

                action_id = invoking.id
                tool_call_id = invoking.id
                if not (
                    isinstance(tool_call_id, str)
                    and tool_call_id.startswith("call_")
                ):
                    if self._registry is not None:
                        tool_call_id = (
                            self._registry.action_id_to_tool_call_id.get(
                                action_id, tool_call_id
                            )
                        )

                value = obs.result
                try:
                    content = json.dumps(
                        value, ensure_ascii=False, separators=(",", ":")
                    )
                except TypeError:
                    content = str(value)

                if tool_call_id is not None:
                    self._add_message(
                        ChatCompletionToolMessage(
                            role="tool",
                            tool_call_id=tool_call_id,
                            content=content,
                        )
                    )
                else:
                    self._add_message(
                        ChatCompletionUserMessage(
                            role="user",
                            content=f"Tool result: {content}",
                        )
                    )
            else:
                self._add_message(
                    ChatCompletionUserMessage(
                        role="user", content=str(obs)
                    )
                )

    def _log_observation(self, observation: Optional[Observation]) -> None:
        if observation is None or observation.is_empty():
            return

        for obs in observation.to_observation_list():
            result = obs.result
            if result is None:
                continue

            entry: Dict[str, Any] = {"step": self._step_count}
            if isinstance(result, str):
                entry["content"] = result
            elif isinstance(result, dict):
                entry["content"] = json.dumps(result, ensure_ascii=False)
            else:
                entry["content"] = str(result)

            if obs.invoking_actions:
                entry["action"] = obs.invoking_actions[0].name

            self._observation_log.append(entry)

    def _assistant_tools(self) -> list[dict[str, Any]]:
        if self._registry is None:
            return []
        tools = self._registry.openai_tools()
        if not self.enable_tool_shortlisting:
            return tools

        def _cost_cb(usage):
            if usage:
                self._cost.update_cost_from_tokens(
                    usage.prompt_tokens, usage.completion_tokens
                )

        return shortlist_tools(
            tools=tools,
            max_selected=self.max_selected_tools,
            messages=self.messages,
            completion_fn=self._completion,
            model=self.model,
            logger=self.logger,
            cost_callback=_cost_cb,
        )

    @staticmethod
    def _extract_tool_calls(message: Any) -> list[dict[str, str]]:
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return []
        tool_calls = []
        for tc in message.tool_calls:
            tool_calls.append({
                "name": tc.function.name,
                "arguments": tc.function.arguments,
                "id": tc.id,
            })
        return tool_calls

    def _completion(self, **kwargs) -> Any:
        call_kwargs = self._model_settings.model_dump(
            exclude_none=True,
            exclude={"num_retries", "retry_after", "retry_strategy"},
        )
        call_kwargs.update(kwargs)
        if call_kwargs.get("tools") is None:
            call_kwargs.pop("tools", None)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = litellm.completion(**call_kwargs)
                return response
            except Exception as exc:
                self.logger.warning(
                    "Harness LLM call attempt %d/%d failed: %s",
                    attempt + 1, max_attempts, exc,
                )
                if attempt + 1 >= max_attempts:
                    raise
                time.sleep(2 ** attempt)
        return None

    def _llm_call_simple(
        self,
        model: str,
        prompt: str,
        *,
        json_mode: bool = False,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                resp = litellm.completion(**kwargs)
                if resp.usage:
                    self._cost.update_cost_from_tokens(
                        resp.usage.prompt_tokens,
                        resp.usage.completion_tokens,
                    )
                content = resp.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned None content")
                return content
            except Exception as exc:
                self.logger.warning(
                    "Harness simple LLM call attempt %d/%d failed: %s",
                    attempt + 1, max_attempts, exc,
                )
                if attempt + 1 >= max_attempts:
                    self._log_failure("llm_call", exc, {
                        "model": model,
                        "prompt_length": len(prompt),
                        "attempts": max_attempts,
                    })
                    raise
                time.sleep(2 ** attempt)
        return ""

    def _build_session_trace(self) -> str:
        events: List[Dict[str, Any]] = []
        for entry in self._action_log:
            events.append({"type": "action", **entry})
        for entry in self._observation_log:
            events.append({"type": "observation", **entry})

        events.sort(key=lambda e: (e.get("step", 0), 0 if e["type"] == "action" else 1))

        lines: List[str] = []
        for event in events:
            step = event.get("step", "?")
            if event["type"] == "action":
                action = event.get("action", "?")
                args = event.get("arguments", event.get("content", ""))
                lines.append(f"[Step {step}] Action: {action}")
                if args:
                    lines.append(f"  Args: {str(args)[:500]}")
            else:
                action = event.get("action", "env")
                content = event.get("content", "")
                lines.append(f"[Step {step}] Observation from {action}:")
                lines.append(f"  {str(content)[:500]}")

        return "\n".join(lines) if lines else "(No session trace recorded)"
