# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import re
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

from .memory_note import MemoryNote
from .memory_store import MemoryStore
from .prompts import GENERATE_QUERY_PROMPT, parse_keywords_response
from ..tool_shortlisting import shortlist_tools

try:
    from ...agents.litellm_tool_calling.utils import ToolCall, ToolsActionsRegistry
except ImportError:
    ToolsActionsRegistry = None  
    ToolCall = dict


class AMemAgentInstance(AgentInstance):

    def __init__(
        self,
        session_id: str,
        model: str = "gpt-4o",
        memory_model: str = "gpt-4o",
        retrieve_k: int = 10,
        evo_threshold: int = 100,
        embedding_model: str = "all-MiniLM-L6-v2",
        shuffle_mode: str = "isolated",
        model_settings: Optional[ModelSettings] = None,
        benchmark_id: Optional[str] = None,
        enable_tool_shortlisting: bool = False,
        max_selected_tools: int = 30,
    ) -> None:
        super().__init__(session_id)

        self.model = model
        self.memory_model = memory_model
        self.retrieve_k = retrieve_k
        self.evo_threshold = evo_threshold
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
        self._store: Optional[MemoryStore] = None

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

        self._interaction_log: List[str] = []

        # Memory tracking for this session
        self._memories_added: int = 0
        self._evolutions_triggered: int = 0

    def _log_failure(
        self, component: str, error: Exception, context: Dict[str, Any]
    ) -> None:
        try:
            log_path = self.paths.agent_dir / "amem_failures.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "component": component,
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                **{
                    k: str(v)[:2000] if isinstance(v, str) else v
                    for k, v in context.items()
                },
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
        self._store = MemoryStore.get_or_create(
            shuffle_mode=self.shuffle_mode,
            task_group=task_group,
            benchmark_id=self.benchmark_id,
            embedding_model=self.embedding_model,
            evo_threshold=self.evo_threshold,
        )
        self._store.increment_session()

        system_content = self._build_system_prompt_with_memories()
        self._add_message(
            ChatCompletionSystemMessage(role="system", content=system_content)
        )

        content_parts: list[Any] = []
        ctx_str = ""
        if self.context:
            for k, v in self.context.items():
                if isinstance(v, dict) and v.get("type") == "image_url":
                    content_parts.append({"type": "image_url", "image_url": {"url": v["data"], "detail": "high"}})
                else:
                    ctx_str += f"\n<{k}>\n{v}\n</{k}>"

        text_content = f"{self.task}\n{ctx_str}"
        if content_parts:
            content_parts.insert(0, {"type": "text", "text": text_content})
            self._add_message(ChatCompletionUserMessage(role="user", content=content_parts))
        else:
            self._add_message(
                ChatCompletionUserMessage(role="user", content=text_content)
            )

        self._interaction_log.append(f"Task: {self.task}")

        self.logger.info(
            "A-Mem instance started  store=%s  session_count=%d  "
            "memory_count=%d  benchmark=%s  tools=%d",
            self._store.store_id,
            self._store.session_count,
            self._store.memory_count,
            self.benchmark_id or "(none)",
            len(self._all_actions),
        )

    def react(self, observation: Optional[Observation]) -> Optional[Action]:
        self._step_count += 1

        observation_text = self._observe(observation)
        if observation_text:
            self._interaction_log.append(
                f"Environment: {observation_text}"
            )

        tools = self._assistant_tools()
        response = self._completion(
            model=self.model,
            messages=self.messages,
            tools=tools if tools else None,
        )

        if response is None:
            self.logger.error("A-Mem: LLM returned None response")
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
                self._interaction_log.append(
                    f"Agent action: {tc['name']}({tc['arguments']})"
                )

            self.logger.info(
                "A-Mem step %d: tool_calls=%s",
                self._step_count,
                [tc["name"] for tc in tool_calls],
            )
            return actions
        else:
            content = message.content if message.content else ""

            if not content:
                self.logger.warning(
                    "A-Mem step %d: empty content response (finish_reason=%s), "
                    "treating as agent inability to continue",
                    self._step_count, finish_reason,
                )
                return None

            self._add_message(
                ChatCompletionAssistantMessage(
                    role="assistant", content=content
                )
            )

            self._interaction_log.append(
                f"Agent says: {content}"
            )

            self.logger.info("A-Mem step %d: message response", self._step_count)
            return MessageAction(arguments=Message(content=content))

    def close(self) -> None:

        store = self._store
        if store is None:
            return

        if self._interaction_log:
            self._store_session_memories()

        store.record_learning(
            session_id=self.session_id,
            task_id=str(
                self.context.get("task_id", "") if self.context else ""
            ),
            memories_added=self._memories_added,
            evolutions_triggered=self._evolutions_triggered,
            summary=(
                f"steps={self._step_count}  "
                f"interactions={len(self._interaction_log)}  "
                f"memories_added={self._memories_added}  "
                f"evolutions={self._evolutions_triggered}"
            ),
            benchmark_id=self.benchmark_id or "",
        )

        try:
            cp = str(self.paths.agent_dir / "memory_checkpoint.json")
            store.save_checkpoint(cp)
            mt = str(self.paths.agent_dir / "memories.txt")
            store.save_memories_text(mt)
        except Exception as exc:
            self.logger.warning("A-Mem: failed to save checkpoint: %s", exc)

        self.logger.info(
            "A-Mem session closed: %d memories added, %d evolutions",
            self._memories_added,
            self._evolutions_triggered,
        )

    def get_cost(self) -> LiteLLMCostReport:
        return self._cost

    def _build_system_prompt_with_memories(self) -> str:

        parts: List[str] = [
            "You are an expert agent that completes tasks using available tools.",
            "Think step-by-step before acting.",
            "Use available tools to interact with the environment.",
            "When you are confident in your solution, use the finish/submit tool.",
        ]

        store = self._store
        if store is not None and store.memory_count > 0 and self.task:
            query = self._generate_query_keywords(self.task)
            memory_context = self._retrieve_memory_context(query)
            if memory_context:
                parts.append("")
                parts.append(
                    "Based on the context below, complete the task. "
                    "Use the context to inform your decisions."
                )
                parts.append("")
                parts.append(f"Context:\n{memory_context}")

        return "\n".join(parts)

    def _generate_query_keywords(self, question: str) -> str:

        try:
            prompt = GENERATE_QUERY_PROMPT.format(question=question)
            response = self._memory_llm_call(prompt)
            keywords = parse_keywords_response(response)
            if keywords:
                self.logger.info(
                    "A-Mem: generated query keywords: %s", keywords
                )
                return keywords
        except Exception as exc:
            self.logger.warning(
                "A-Mem: keyword extraction failed, using raw task: %s", exc
            )
        return question

    def _retrieve_memory_context(self, query: str) -> str:

        store = self._store
        if store is None:
            return ""

        retrieved = store.find_related_with_neighbors(query, k=self.retrieve_k)
        if not retrieved:
            return ""


        model_lower = self.model.lower() if self.model else ""
        needs_budget = "gemini" in model_lower

        if needs_budget:
            budget = 30000
            used = 0
            lines: List[str] = []
            for mem in retrieved:
                content = mem.content
                if len(content) > 5000:
                    content = content[:5000] + "... [truncated]"
                entry = (
                    f"memory content: {content}  "
                    f"memory context: {mem.context}  "
                    f"memory keywords: {mem.keywords}  "
                    f"memory tags: {mem.tags}"
                )
                if used + len(entry) > budget and lines:
                    break
                lines.append(entry)
                used += len(entry)
            self.logger.info(
                "A-Mem: injected %d/%d retrieved memories (%d chars, budget=%d)",
                len(lines), len(retrieved), used, budget,
            )
        else:
            lines = []
            for mem in retrieved:
                lines.append(
                    f"memory content: {mem.content}  "
                    f"memory context: {mem.context}  "
                    f"memory keywords: {mem.keywords}  "
                    f"memory tags: {mem.tags}"
                )

        return "\n".join(lines)

    def _store_session_memories(self) -> None:
        store = self._store
        if store is None:
            return

        session_content = "\n".join(self._interaction_log)

        if len(session_content.strip()) < 10:
            return

        try:
            note = MemoryNote.create_with_analysis(
                content=session_content,
                llm_call=self._memory_llm_call,
            )

            evolved = store.add_memory(
                note=note,
                llm_call=self._memory_llm_call,
            )

            self._memories_added += 1
            if evolved:
                self._evolutions_triggered += 1

            self.logger.info(
                "A-Mem: stored session memory [%s] evolved=%s (total=%d)",
                note.id[:8],
                evolved,
                store.memory_count,
            )
        except Exception as exc:
            self.logger.warning(
                "A-Mem: failed to store session memory: %s", exc
            )
            self._log_failure(
                "session_memory_storage", exc, {
                    "content_preview": session_content[:500],
                    "interaction_count": len(self._interaction_log),
                },
            )

    def _add_message(self, message: Any) -> None:
        self.logger.debug(
            "Adding message: role=%s", getattr(message, "role", "?")
        )
        self.messages.append(message)

    def _observe(self, observation: Optional[Observation]) -> Optional[str]:

        if observation is None:
            return None

        observations = observation.to_observation_list()
        if observation.is_empty():
            if not any(obs.invoking_actions for obs in observations):
                return None

        collected_texts: List[str] = []

        for obs in observations:
            if isinstance(obs, MessageObservation) and isinstance(
                obs.result, MessagePayload
            ):
                self._add_message(
                    ChatCompletionUserMessage(
                        role="user", content=obs.result.message
                    )
                )
                collected_texts.append(obs.result.message)
                continue

            if len(obs.invoking_actions) > 0:
                invoking = obs.invoking_actions[0]
                if invoking.name == "message":
                    text = str(obs)
                    self._add_message(
                        ChatCompletionUserMessage(
                            role="user", content=text
                        )
                    )
                    collected_texts.append(text)
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
                collected_texts.append(
                    f"Result of {invoking.name}: "
                    f"{self._summarize_for_memory(content)}"
                )
            else:
                text = str(obs)
                self._add_message(
                    ChatCompletionUserMessage(
                        role="user", content=text
                    )
                )
                collected_texts.append(text)

        if collected_texts:
            return "\n".join(collected_texts)
        return None

    def _summarize_for_memory(self, content: str) -> str:

        if len(content) < 2000:
            return content

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            if len(content) > 10000:
                return content[:10000] + f"\n... [truncated, total {len(content)} chars]"
            return content

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                if len(data) > 10000:
                    return data[:10000] + f"\n... [truncated, total {len(data)} chars]"
                return data

        if isinstance(data, list) and data and isinstance(data[0], dict):
            first = data[0]
            snippet_key = None
            if "snippet" in first:
                snippet_key = "snippet"
            elif "content" in first and "docid" in first:
                snippet_key = "content"

            if snippet_key is not None:
                summaries: List[str] = []
                for item in data:
                    docid = item.get("docid", "?")
                    score = item.get("score")
                    snippet = item.get(snippet_key, "")
                    title = ""
                    if isinstance(snippet, str) and snippet.startswith("---"):
                        title_match = re.search(r"title:\s*(.+)", snippet)
                        if title_match:
                            title = title_match.group(1).strip()
                    score_str = f" score:{score:.3f}" if isinstance(score, (int, float)) else ""
                    snippet_preview = snippet[:400].replace("\n", " ") if isinstance(snippet, str) else str(snippet)[:400]
                    summaries.append(
                        f"[doc:{docid}{score_str}] {title} | {snippet_preview}"
                    )
                return "\n".join(summaries)

            max_item_chars = 200
            summaries_generic: List[str] = []
            for i, item in enumerate(data):
                item_str = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                if len(item_str) > max_item_chars:
                    item_str = item_str[:max_item_chars] + "..."
                summaries_generic.append(item_str)
            result = f"[{len(data)} items]\n" + "\n".join(summaries_generic)
            return result

        if isinstance(data, dict) and len(content) > 5000:
            compact = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            if len(compact) > 5000:
                return compact[:5000] + f"... [truncated, total {len(compact)} chars]"
            return compact

        return content

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
                choice = response["choices"][0] if response.get("choices") else None
                if choice:
                    msg = choice.get("message") or {}
                    has_content = bool(msg.get("content"))
                    has_tools = bool(msg.get("tool_calls"))
                    if not has_content and not has_tools:
                        if attempt + 1 < max_attempts:
                            self.logger.warning(
                                "A-Mem LLM call attempt %d/%d: empty response "
                                "(finish_reason=%s), retrying...",
                                attempt + 1, max_attempts,
                                choice.get("finish_reason"),
                            )
                            time.sleep(2 ** attempt)
                            continue
                return response
            except Exception as exc:
                self.logger.warning(
                    "A-Mem LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                if attempt + 1 >= max_attempts:
                    raise
                time.sleep(2 ** attempt)
        return None

    def _llm_call_simple(
        self,
        model: str,
        prompt: str,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000,
        }

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
                    "A-Mem simple LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                if attempt + 1 >= max_attempts:
                    self._log_failure(
                        "llm_call", exc, {
                            "model": model,
                            "prompt_length": len(prompt),
                            "prompt_preview": prompt[:500],
                            "attempts": max_attempts,
                        },
                    )
                    raise
                time.sleep(2 ** attempt)
        return ""

    def _memory_llm_call(self, prompt: str) -> str:
        return self._llm_call_simple(self.memory_model, prompt)
