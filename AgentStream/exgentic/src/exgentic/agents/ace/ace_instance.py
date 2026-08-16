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
from ...utils.settings import get_settings

from .playbook_store import PlaybookStore
from .playbook_utils import (
    apply_curator_operations,
    extract_json_from_text,
    get_playbook_stats,
    update_bullet_counts,
)
from .prompts.curator import CURATOR_PROMPT_NO_GT
from .prompts.reflector import REFLECTOR_PROMPT_NO_GT
from .bulletpoint_analyzer import BulletpointAnalyzer, DEDUP_AVAILABLE
from ..tool_shortlisting import shortlist_tools

try:
    from ...agents.litellm_tool_calling.utils import ToolsActionsRegistry
except ImportError:
    ToolsActionsRegistry = None

_BULLET_ID_RE = re.compile(r"\[([a-z]{2,5}-\d{5})\]")

settings = get_settings()


class ACEAgentInstance(AgentInstance):

    def __init__(
        self,
        session_id: str,
        model: str = "gpt-4o",
        curator_model: str = "gpt-4o",
        max_num_rounds: int = 3,
        curator_frequency: int = 1,
        playbook_token_budget: int = 80000,
        shuffle_mode: str = "isolated",
        initial_playbook: Optional[str] = None,
        use_json_mode: bool = True,
        model_settings: Optional[ModelSettings] = None,
        benchmark_id: Optional[str] = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90,
        enable_tool_shortlisting: bool = False,
        max_selected_tools: int = 30,
    ) -> None:
        super().__init__(session_id)

        self.model = model
        self.curator_model = curator_model
        self.max_num_rounds = max_num_rounds
        self.curator_frequency = curator_frequency
        self.playbook_token_budget = playbook_token_budget
        self.shuffle_mode = shuffle_mode
        self.initial_playbook = initial_playbook
        self.use_json_mode = use_json_mode
        self.benchmark_id = benchmark_id
        self.use_bulletpoint_analyzer = use_bulletpoint_analyzer
        self.bulletpoint_analyzer_threshold = bulletpoint_analyzer_threshold
        self.enable_tool_shortlisting = enable_tool_shortlisting
        self.max_selected_tools = max_selected_tools

        if model_settings is None:
            self._model_settings = ModelSettings()
        elif isinstance(model_settings, ModelSettings):
            self._model_settings = model_settings
        else:
            self._model_settings = ModelSettings()

        self._cost = LiteLLMCostReport.initialize_empty(model_name=self.model)
        self._store: Optional[PlaybookStore] = None

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
            log_path = self.paths.agent_dir / "ace_failures.jsonl"
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

    def _log_bullet_usage(self, bullet_ids: List[str]) -> None:
        store = self._store
        if store is None:
            return
        try:
            log_path = self.paths.agent_dir / "bullet_usage_log.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            from .playbook_utils import extract_playbook_bullets
            bullets_text = extract_playbook_bullets(store.playbook, bullet_ids)

            entry = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "task_id": str(
                    self.context.get("task_id", "") if self.context else ""
                ),
                "benchmark_id": self.benchmark_id or "",
                "store_id": store.store_id,
                "session_count": store.session_count,
                "bullet_ids_used": bullet_ids,
                "bullet_count": len(bullet_ids),
                "bullets_detail": bullets_text,
                "total_steps": self._step_count,
                "total_observations": len(self._observation_log),
                "total_actions": len(self._action_log),
                "question_preview": (
                    self.task if self.task else ""
                ),
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
        self._store = PlaybookStore.get_or_create(
            shuffle_mode=self.shuffle_mode,
            task_group=task_group,
            initial_playbook=self.initial_playbook,
            benchmark_id=self.benchmark_id,
        )
        self._store.increment_session()

        playbook = self._store.playbook
        system_content = self._build_system_prompt(playbook)
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
            "ACE v2 instance started  store=%s  session_count=%d  "
            "playbook_bullets=%d  benchmark=%s  tools=%d",
            self._store.store_id,
            self._store.session_count,
            get_playbook_stats(playbook)["total_bullets"],
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
            self.logger.error("ACE v2: LLM returned None response")
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

            self.logger.info("ACE v2 step %d: tool_calls=%s", self._step_count,
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
            self.logger.info("ACE v2 step %d: message response", self._step_count)
            return MessageAction(arguments=Message(content=content))

    def close(self) -> None:
        store = self._store
        if store is None:
            return

        bullet_ids = _BULLET_ID_RE.findall(
            " ".join(
                str(m.get("content", "")) if isinstance(m, dict)
                else str(getattr(m, "content", ""))
                for m in self.messages
                if (isinstance(m, dict) and m.get("role") == "assistant")
                or getattr(m, "role", "") == "assistant"
            )
        )
        self._log_bullet_usage(bullet_ids)

        reflection_content = "(empty)"
        if self._observation_log:
            try:
                reflection_content = self._run_post_session_reflection()
            except Exception as exc:
                self.logger.warning(
                    "ACE v2: post-session reflection failed: %s", exc
                )
                self._log_failure("reflector", exc, {
                    "observation_count": len(self._observation_log),
                    "action_count": len(self._action_log),
                })

        if store.session_count % self.curator_frequency == 0:
            try:
                self._run_curator(reflection_content)
            except Exception as exc:
                self.logger.warning(
                    "ACE v2: curator failed: %s", exc
                )
                self._log_failure("curator", exc, {
                    "reflection_preview": reflection_content[:500],
                })

        store.record_learning(
            session_id=self.session_id,
            task_id=str(self.context.get("task_id", "") if self.context else ""),
            was_correct_before=False,
            was_correct_after=False,
            summary=(
                f"steps={self._step_count}  "
                f"observations={len(self._observation_log)}  "
                f"actions={len(self._action_log)}"
            ),
            benchmark_id=self.benchmark_id or "",
        )

        try:
            cp = str(self.paths.agent_dir / "playbook_checkpoint.json")
            store.save_checkpoint(cp)
            pb = str(self.paths.agent_dir / "playbook.txt")
            store.save_playbook_text(pb)
        except Exception as exc:
            self.logger.warning("ACE v2: failed to save checkpoint: %s", exc)

    def get_cost(self) -> LiteLLMCostReport:
        return self._cost

    def _build_system_prompt(self, playbook: str) -> str:
        stats = get_playbook_stats(playbook)
        has_content = stats["total_bullets"] > 0

        parts = [
            "You are an expert agent that completes tasks using available tools.",
            "You have access to a curated playbook of strategies and insights "
            "learned from previous tasks.  Use these to make better decisions.",
            "",
            "## Guidelines",
            "- Read the playbook carefully and apply relevant strategies",
            "- Pay attention to common mistakes listed and avoid them",
            "- Use available tools to interact with the environment",
            "- Think step-by-step before acting",
            "- When you are confident in your solution, use the finish/submit tool",
            "- When a playbook bullet influences your decision, mention its ID "
            "(e.g. [err-00001]) in your reasoning text",
        ]

        if has_content:
            parts.extend([
                "",
                "## Playbook (accumulated strategies & insights)",
                "Each line has a bullet ID and usage stats "
                "(helpful=N means it helped N times, harmful=N means it misled N times).",
                "Prefer high-helpful, low-harmful bullets.",
                "",
                playbook,
            ])
        else:
            parts.extend([
                "",
                "## Playbook",
                "(No strategies accumulated yet. This is the first session.)",
            ])

        return "\n".join(parts)

    def _add_message(self, message: Any) -> None:
        self.logger.debug("Adding message: role=%s", getattr(message, "role", "?"))
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
                    "ACE LLM call attempt %d/%d failed: %s",
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
                    "ACE simple LLM call attempt %d/%d failed: %s",
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

    def _run_post_session_reflection(self) -> str:
        store = self._store
        assert store is not None

        playbook = store.playbook
        session_trace = self._build_session_trace()

        prompt = REFLECTOR_PROMPT_NO_GT.format(
            question=self.task if self.task else "",
            reasoning_trace=session_trace,
            predicted_answer="(see session trace above)",
            bullets_used=playbook,
        )

        raw = self._llm_call_simple(
            self.model, prompt, json_mode=self.use_json_mode
        )

        bullet_tags: List[Dict] = []
        reflection_text = raw
        parsed = extract_json_from_text(raw)
        if parsed and isinstance(parsed, dict):
            bullet_tags = parsed.get("bullet_tags", [])
            reflection_text = parsed.get("reasoning", raw)
        else:
            self.logger.warning(
                "ACE Reflector: JSON parse failed, raw length=%d", len(raw)
            )
            self._log_failure(
                "reflector_parse", ValueError("JSON parse failed"), {
                    "raw_response_preview": raw[:1000],
                },
            )

        if bullet_tags:
            store.playbook = update_bullet_counts(store.playbook, bullet_tags)

        self.logger.info(
            "ACE post-session reflection: %d bullet tags updated",
            len(bullet_tags),
        )

        return reflection_text

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

    def _run_curator(self, reflection_content: str) -> None:
        store = self._store
        assert store is not None

        playbook = store.playbook
        stats = get_playbook_stats(playbook)

        question_context = self.task if self.task else ""
        if self.context:
            question_context += "".join(
                f"\n<{k}>\n{v}\n</{k}>"
                for k, v in self.context.items()
            )

        prompt = CURATOR_PROMPT_NO_GT.format(
            token_budget=self.playbook_token_budget,
            current_step=store.session_count,
            total_samples="ongoing",
            playbook_stats=json.dumps(stats, indent=2),
            recent_reflection=reflection_content,
            current_playbook=playbook,
            question_context=question_context,
        )

        raw = self._llm_call_simple(
            self.curator_model, prompt, json_mode=self.use_json_mode
        )

        if raw.startswith("INCORRECT_DUE_TO_EMPTY_RESPONSE"):
            self.logger.warning("ACE Curator: skipping due to empty response")
            self._log_failure(
                "curator_empty_response", ValueError("empty LLM response"), {
                    "session_count": store.session_count,
                },
            )
            return

        parsed = extract_json_from_text(raw)
        if parsed and isinstance(parsed, dict):
            if "operations" not in parsed or not isinstance(parsed["operations"], list):
                self.logger.warning("ACE Curator: missing or invalid 'operations' field")
                self._log_failure(
                    "curator_schema", ValueError("missing 'operations' list"), {
                        "raw_response_preview": raw[:1000],
                        "parsed_keys": list(parsed.keys()),
                    },
                )
                return

            ops = parsed["operations"]
            valid_ops = []
            for op in ops:
                if not isinstance(op, dict) or "type" not in op:
                    continue
                if op["type"] == "ADD":
                    if "section" in op and "content" in op:
                        valid_ops.append(op)
                else:
                    valid_ops.append(op)

            if valid_ops:
                new_playbook, new_id = apply_curator_operations(
                    playbook, valid_ops, store.next_global_id
                )
                store.playbook = new_playbook
                store.next_global_id = new_id
                self.logger.info("ACE Curator: applied %d operations", len(valid_ops))

            if self.use_bulletpoint_analyzer and DEDUP_AVAILABLE:
                self.logger.info(
                    "ACE BulletpointAnalyzer: running (threshold=%.2f)",
                    self.bulletpoint_analyzer_threshold,
                )
                analyzer = BulletpointAnalyzer(
                    llm_merge_fn=lambda p: self._llm_call_simple(
                        self.curator_model, p, json_mode=False
                    ),
                )
                store.playbook = analyzer.analyze(
                    playbook=store.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True,
                )
        else:
            self.logger.warning("ACE Curator: failed to parse response")
            self._log_failure(
                "curator_parse", ValueError("JSON parse failed"), {
                    "raw_response_preview": raw[:1000],
                },
            )
