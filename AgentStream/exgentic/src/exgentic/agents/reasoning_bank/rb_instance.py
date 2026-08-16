# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Union

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

from .rb_store import MemoryEntry, ReasoningBankStore
from .memory_management import (
    compute_embedding,
    format_memories_for_prompt,
    select_memory,
)
from .prompts.memory_instruction import MEMORY_INJECTION_INSTRUCTION
from ..tool_shortlisting import shortlist_tools

try:
    from ...agents.litellm_tool_calling.utils import ToolCall, ToolsActionsRegistry
except ImportError:
    ToolsActionsRegistry = None
    ToolCall = dict

settings = get_settings()
logger = logging.getLogger(__name__)


class ReasoningBankAgentInstance(AgentInstance):

    def __init__(
        self,
        session_id: str,
        model: str = "gpt-4o",
        memory_model: str = "gpt-4o",
        eval_model: str = "gpt-4o",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k_memories: int = 1,
        max_memory_items: int = 3,
        shuffle_mode: str = "isolated",
        model_settings: Optional[ModelSettings] = None,
        benchmark_id: Optional[str] = None,
        enable_tool_shortlisting: bool = False,
        max_selected_tools: int = 30,
    ) -> None:
        super().__init__(session_id)

        self.model = model
        self.memory_model = memory_model
        self.eval_model = eval_model
        self.embedding_model = embedding_model
        self.top_k_memories = top_k_memories
        self.max_memory_items = max_memory_items
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
        self._store: Optional[ReasoningBankStore] = None

        self.messages: list[
            Union[
                ChatCompletionAssistantMessage,
                ChatCompletionToolMessage,
                ChatCompletionUserMessage,
                ChatCompletionSystemMessage,
            ]
        ] = []

        self._registry: Optional[Any] = None
        self._all_actions: list[ActionType] = []
        self._step_count: int = 0

        self._think_list: list[str] = []
        self._action_list: list[str] = []
        self._observation_log: list[str] = []
        self._action_log: list[Dict[str, Any]] = []

        self._query_embedding: Optional[list[float]] = None
        self._task_query: str = ""

    def start(self, task: str, context: Dict[str, Any], actions: list[ActionType]) -> None:
        super().start(task, context, actions)

        self._all_actions = list(self.actions)
        if ToolsActionsRegistry is not None:
            self._registry = ToolsActionsRegistry(self._all_actions)
        else:
            self._registry = None

        self._store = ReasoningBankStore.get_or_create(
            shuffle_mode=self.shuffle_mode,
            benchmark_id=self.benchmark_id,
        )
        self._store.increment_session()

        self._task_query = task
        if context:
            context_str = ""
            for k, v in context.items():
                if isinstance(v, dict) and v.get("type") == "image_url":
                    continue
                context_str += f"\n{k}: {v}"
            self._task_query = f"{task}{context_str}"

        try:
            self._query_embedding = compute_embedding(
                self._task_query, self.embedding_model
            )
        except Exception as e:
            logger.warning("Failed to compute query embedding: %s", e)
            self._query_embedding = None

        retrieved_memories: list[MemoryEntry] = []
        if self._query_embedding is not None and self._store.entry_count > 0:
            try:
                retrieved_memories = select_memory(
                    store=self._store,
                    cur_query=self._task_query,
                    embedding_model=self.embedding_model,
                    top_k=self.top_k_memories,
                )
            except Exception as e:
                logger.warning("Memory retrieval failed: %s", e)

        sys_prompt = "You are an expert agent that completes tasks using available tools."
        if retrieved_memories:
            memory_text = format_memories_for_prompt(retrieved_memories)
            if memory_text.strip():
                sys_prompt += "\n\n" + MEMORY_INJECTION_INSTRUCTION
                sys_prompt += "\n\n" + memory_text

        self.messages = [
            ChatCompletionSystemMessage(role="system", content=sys_prompt),
        ]

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
            self.messages.append(ChatCompletionUserMessage(role="user", content=content_parts))
        else:
            self.messages.append(
                ChatCompletionUserMessage(role="user", content=text_content)
            )

        logger.info(
            "ReasoningBank start: retrieved %d memories for task (store has %d entries)",
            len(retrieved_memories), self._store.entry_count,
        )

    def react(self, observation: Optional[Observation]) -> Optional[Action]:
        self._step_count += 1

        self._observe(observation)

        tools = self._assistant_tools()

        response = self._completion(
            model=self.model,
            messages=self.messages,
            tools=tools if tools else None,
        )
        if response is None:
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
            self.messages.append(
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

            think = message.content if message.content else ""
            self._think_list.append(think)
            for tc in tool_calls:
                self._action_list.append(f"{tc['name']}({tc['arguments']})")
                self._action_log.append({
                    "step": self._step_count,
                    "action": tc["name"],
                    "arguments": tc["arguments"],
                })

            actions = self._registry.tool_calls_to_action(tool_calls)
            return actions
        else:
            content = message.content if message.content else ""

            if not content:
                logger.warning(
                    "ReasoningBank step %d: empty content response (finish_reason=%s), "
                    "treating as agent inability to continue",
                    self._step_count, finish_reason,
                )
                return None

            self.messages.append(
                ChatCompletionAssistantMessage(role="assistant", content=content)
            )

            self._think_list.append(content)
            self._action_list.append(f"send_msg_to_user('{content[:200]}')")

            return MessageAction(arguments=Message(content=content))

    def close(self) -> None:
        if not self._store:
            return

        if not self._action_list:
            logger.info("ReasoningBank close: no actions recorded, skipping memory induction.")
            self._save_session_artifacts()
            return

        if self._query_embedding is None:
            logger.info("ReasoningBank close: embedding unavailable, skipping memory induction.")
            self._save_session_artifacts()
            return

        try:
            self._run_post_session_learning()
        except Exception as e:
            logger.warning("ReasoningBank close: memory induction failed: %s", e)

        self._save_session_artifacts()

    def _save_session_artifacts(self) -> None:
        if not self._store:
            return
        try:
            cp = str(self.paths.agent_dir / "memory_checkpoint.json")
            self._store.save_checkpoint(cp)
            mt = str(self.paths.agent_dir / "memories.txt")
            self._store.save_memories_text(mt)
        except Exception as exc:
            logger.warning("ReasoningBank: failed to save session artifacts: %s", exc)

    def get_cost(self) -> LiteLLMCostReport:
        return self._cost

    def _run_post_session_learning(self) -> None:
        from .evaluator import TrajectoryEvaluator
        from .induce_memory import induce_memory

        evaluator = TrajectoryEvaluator(
            llm_call_fn=lambda prompt, sys_msg: self._llm_call_simple(
                self.eval_model, prompt, system_msg=sys_msg
            )
        )

        final_response = ""
        for act in reversed(self._action_list):
            if "send_msg_to_user" in act:
                try:
                    final_response = act[act.index("(") + 1:act.rindex(")")]
                    final_response = final_response.strip("'\"")
                except (ValueError, IndexError):
                    pass
                break

        eval_result = evaluator.evaluate(
            intent=self.task or self._task_query,
            think_list=self._think_list,
            action_list=self._action_list,
            observation_list=self._observation_log,
            final_response=final_response,
        )

        status = "success" if eval_result["status"] == "success" else "fail"
        eval_thoughts = eval_result.get("thoughts", "")

        logger.info(
            "ReasoningBank eval: status=%s, thoughts=%s",
            status, eval_thoughts[:100],
        )

        memory_items = induce_memory(
            query=self.task or self._task_query,
            think_list=self._think_list,
            action_list=self._action_list,
            status=status,
            eval_thoughts=eval_thoughts,
            llm_call_fn=lambda user_msg, sys_msg: self._llm_call_simple(
                self.memory_model, user_msg, system_msg=sys_msg
            ),
            observation_list=self._observation_log,
        )

        if memory_items and self._query_embedding is not None:
            entry = MemoryEntry(
                task_id=self.session_id,
                query=self.task or self._task_query,
                think_list=self._think_list,
                action_list=self._action_list,
                status=status,
                memory_items=memory_items,
                template_id=self.context.get("template_id") if self.context else None,
            )
            self._store.add_entry(entry, self._query_embedding)
            logger.info(
                "ReasoningBank: stored %d memory items (store now has %d entries)",
                len(memory_items), self._store.entry_count,
            )

    def _completion(self, **kwargs) -> Any:
        """Standard completion with tool support and retry logic."""
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
                            logger.warning(
                                "ReasoningBank LLM call attempt %d/%d: empty response "
                                "(finish_reason=%s), retrying...",
                                attempt + 1, max_attempts,
                                choice.get("finish_reason"),
                            )
                            time.sleep(2 ** attempt)
                            continue
                return response
            except Exception as exc:
                logger.warning(
                    "ReasoningBank LLM call attempt %d/%d failed: %s",
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
        system_msg: str = "",
        json_mode: bool = False,
    ) -> str:
        messages: list = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 1.0, 
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
                return resp.choices[0].message.content or ""
            except Exception as exc:
                if attempt + 1 >= max_attempts:
                    logger.error("Simple LLM call failed: %s", exc)
                    return ""
                time.sleep(2 ** attempt)
        return ""

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
                self.messages.append(
                    ChatCompletionUserMessage(role="user", content=obs.result.message)
                )
                self._observation_log.append(obs.result.message)
                continue

            if len(obs.invoking_actions) > 0:
                invoking = obs.invoking_actions[0]
                if invoking.name == "message":
                    text = str(obs)
                    self.messages.append(
                        ChatCompletionUserMessage(role="user", content=text)
                    )
                    self._observation_log.append(text)
                    continue

                tool_call_id = invoking.id
                if not (
                    isinstance(tool_call_id, str)
                    and tool_call_id.startswith("call_")
                ):
                    if self._registry is not None:
                        tool_call_id = (
                            self._registry.action_id_to_tool_call_id.get(
                                tool_call_id, tool_call_id
                            )
                        )

                value = obs.result
                try:
                    content = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                except TypeError:
                    content = str(value)

                if tool_call_id is not None:
                    self.messages.append(
                        ChatCompletionToolMessage(
                            role="tool",
                            tool_call_id=tool_call_id,
                            content=content,
                        )
                    )
                else:
                    self.messages.append(
                        ChatCompletionUserMessage(
                            role="user",
                            content=f"Tool result: {content}",
                        )
                    )
                self._observation_log.append(content)
            else:
                content = str(obs.result) if hasattr(obs, "result") else str(obs)
                self.messages.append(
                    ChatCompletionUserMessage(role="user", content=content)
                )
                self._observation_log.append(content)


    def _assistant_tools(self) -> list | None:
        if self._registry is None:
            return None
        tools = self._registry.openai_tools()
        if not tools:
            return None

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
            logger=logger,
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
