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

from .prompts import QUERY_REWRITE_PROMPT, SKILL_CONTEXT_TEMPLATE, SKILL_ENTRY_TEMPLATE
from .skill_extraction import extract_skills_from_trace
from .skill_maintenance import maintain_skill
from .skill_retrieval import compute_embedding, hybrid_search
from .skill_store import SkillStore

try:
    from ...agents.litellm_tool_calling.utils import ToolCall, ToolsActionsRegistry
except ImportError:
    ToolsActionsRegistry = None
    ToolCall = dict

from ..tool_shortlisting import shortlist_tools

settings = get_settings()


class AutoSkillAgentInstance(AgentInstance):

    def __init__(
        self,
        session_id: str,
        model: str = "gpt-4o",
        skill_model: str = "gpt-4o",
        retrieve_k: int = 5,
        retrieval_threshold: float = 0.3,
        bm25_weight: float = 0.1,
        dedupe_similarity_threshold: float = 0.4,
        embedding_model: str = "text-embedding-3-small",
        enable_query_rewrite: bool = True,
        max_context_chars: int = 6000,
        shuffle_mode: str = "isolated",
        model_settings: Optional[ModelSettings] = None,
        benchmark_id: Optional[str] = None,
        enable_tool_shortlisting: bool = False,
        max_selected_tools: int = 30,
    ) -> None:
        super().__init__(session_id)

        self.model = model
        self.skill_model = skill_model
        self.retrieve_k = retrieve_k
        self.retrieval_threshold = retrieval_threshold
        self.bm25_weight = bm25_weight
        self.dedupe_similarity_threshold = dedupe_similarity_threshold
        self.embedding_model = embedding_model
        self.enable_query_rewrite = enable_query_rewrite
        self.max_context_chars = max_context_chars
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
        self._store: Optional[SkillStore] = None

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

    def _log_failure(
        self, component: str, error: Exception, context: Dict[str, Any]
    ) -> None:
        try:
            log_path = self.paths.agent_dir / "autoskill_failures.jsonl"
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
        self._store = SkillStore.get_or_create(
            shuffle_mode=self.shuffle_mode,
            task_group=task_group,
            benchmark_id=self.benchmark_id,
        )
        self._store.increment_session()

        skill_context = self._retrieve_skills(task, context)

        system_content = self._build_system_prompt(skill_context)
        self._add_message(
            ChatCompletionSystemMessage(role="system", content=system_content)
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
            "AutoSkill instance started  store=%s  session_count=%d  "
            "skill_count=%d  benchmark=%s  tools=%d",
            self._store.store_id,
            self._store.session_count,
            self._store.skill_count,
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
            self.logger.error("AutoSkill: LLM returned None response")
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

            self.logger.info("AutoSkill step %d: tool_calls=%s", self._step_count,
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
            self.logger.info("AutoSkill step %d: message response", self._step_count)
            return MessageAction(arguments=Message(content=content))

    def close(self) -> None:
        store = self._store
        if store is None:
            return

        action_taken = "no_extraction"
        skill_name = ""

        if self._observation_log or self._action_log:
            try:
                session_trace = self._build_session_trace()
                candidates = extract_skills_from_trace(
                    task=self.task if self.task else "",
                    benchmark_id=self.benchmark_id or "",
                    session_trace=session_trace,
                    llm_call=self._llm_call_simple,
                    model=self.skill_model,
                    logger=self.logger,
                )

                if candidates:
                    candidate = candidates[0]
                    action_taken, _ = maintain_skill(
                        candidate=candidate,
                        store=store,
                        llm_call=self._llm_call_simple,
                        model=self.skill_model,
                        embedding_model=self.embedding_model,
                        bm25_weight=self.bm25_weight,
                        dedupe_similarity_threshold=self.dedupe_similarity_threshold,
                        logger=self.logger,
                    )
                    skill_name = candidate.name
                    self.logger.info(
                        "AutoSkill close: action=%s  skill=%s",
                        action_taken, skill_name,
                    )

            except Exception as exc:
                self.logger.warning("AutoSkill: skill extraction/maintenance failed: %s", exc)
                self._log_failure("skill_evolution", exc, {
                    "observation_count": len(self._observation_log),
                    "action_count": len(self._action_log),
                })

        store.record_learning(
            session_id=self.session_id,
            task_id=str(self.context.get("task_id", "") if self.context else ""),
            benchmark_id=self.benchmark_id or "",
            action=action_taken,
            skill_name=skill_name,
        )

        try:
            cp = str(self.paths.agent_dir / "skillstore_checkpoint.json")
            store.save_checkpoint(cp)
            txt = str(self.paths.agent_dir / "skillbank.txt")
            store.save_skills_text(txt)
        except Exception as exc:
            self.logger.warning("AutoSkill: failed to save checkpoint: %s", exc)

    def get_cost(self) -> LiteLLMCostReport:
        return self._cost

    def _retrieve_skills(self, task: str, context: Dict[str, Any]) -> str:
        store = self._store
        if store is None or store.skill_count == 0:
            return ""

        query = task
        if self.enable_query_rewrite and task:
            try:
                ctx_parts = []
                for _, v in context.items():
                    if isinstance(v, str):
                        ctx_parts.append(v)
                ctx_str = " ".join(ctx_parts)
                rewritten = self._llm_call_simple(
                    self.skill_model,
                    QUERY_REWRITE_PROMPT.format(task=task, context=ctx_str),
                )
                if rewritten and len(rewritten.strip()) > 5:
                    query = rewritten.strip()
                    self.logger.info("AutoSkill retrieval: query rewritten to '%s'", query[:100])
            except Exception as exc:
                self.logger.debug("AutoSkill retrieval: query rewrite failed: %s", exc)

        query_embedding = None
        try:
            query_embedding = compute_embedding(query, model=self.embedding_model)
        except Exception as exc:
            self.logger.warning("AutoSkill retrieval: failed to compute query embedding: %s", exc)

        results = hybrid_search(
            store=store,
            query=query,
            query_embedding=query_embedding,
            top_k=self.retrieve_k,
            threshold=self.retrieval_threshold,
            bm25_weight=self.bm25_weight,
            embedding_model=self.embedding_model,
        )

        if not results:
            self.logger.info("AutoSkill retrieval: no skills above threshold %.2f", self.retrieval_threshold)
            return ""

        self.logger.info(
            "AutoSkill retrieval: %d skills retrieved (top score=%.3f)",
            len(results), results[0][1],
        )

        skills_block = ""
        char_budget = self.max_context_chars
        for skill, _ in results:
            entry_text = SKILL_ENTRY_TEMPLATE.format(
                name=skill.name,
                description=skill.description,
                tags=", ".join(skill.tags),
                triggers=", ".join(skill.triggers),
                instructions=skill.instructions,
            )
            if len(skills_block) + len(entry_text) > char_budget:
                break
            skills_block += entry_text + "\n"

        return SKILL_CONTEXT_TEMPLATE.format(skills_block=skills_block)

    def _build_system_prompt(self, skill_context: str) -> str:
        parts = [
            "You are an expert agent that completes tasks using available tools.",
            "Think step-by-step before acting.",
            "Use available tools to interact with the environment.",
            "When you are confident in your solution, use the finish/submit tool.",
        ]

        if skill_context:
            parts.extend(["", skill_context])
        else:
            parts.extend([
                "",
                "## Skills",
                "(No accumulated skills yet. This is an early session.)",
            ])

        return "\n".join(parts)

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
                    "AutoSkill LLM call attempt %d/%d failed: %s",
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
                    "AutoSkill simple LLM call attempt %d/%d failed: %s",
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
                    lines.append(f"  Args: {str(args)}")
            else:
                action = event.get("action", "env")
                content = event.get("content", "")
                lines.append(f"[Step {step}] Observation from {action}:")
                lines.append(f"  {content}")

        return "\n".join(lines) if lines else "(No session trace recorded)"
