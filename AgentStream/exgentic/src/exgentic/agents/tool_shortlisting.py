# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import logging
from typing import Any, Callable, List

from litellm import (
    ChatCompletionDeveloperMessage,
    ChatCompletionUserMessage,
)


def shortlist_tools(
    tools: list[dict[str, Any]],
    max_selected: int,
    messages: list[Any],
    completion_fn: Callable[..., Any],
    model: str,
    logger: logging.Logger,
    *,
    cost_callback: Callable[[Any], None] | None = None,
) -> list[dict[str, Any]]:
    if len(tools) <= max_selected:
        return tools

    logger.info("Tool shortlisting: %d available -> selecting top %d", len(tools), max_selected)

    names = [tool["function"]["name"] for tool in tools]
    names_str = ""
    for tool in tools:
        names_str += f"\n- {tool['function']['name']}: {tool['function']['description']}"

    history_text = _render_history(messages)

    dev = ChatCompletionDeveloperMessage(
        role="developer",
        content=(
            f"Please before providing your next move list the names of the top "
            f"{max_selected} tools that are somewhat relevant for the next step, "
            "ordered by relevancy (most to least). Return ONLY a JSON object with this shape: "
            '{\n  "tools": ["tool_name_1", "tool_name_2", ...]\n}.\n'
            f"Choose from these tools only: {names_str}.\n"
            f"Do not call any of those tools just return the list of the top "
            f"{max_selected} relevant tools names in the required format."
        ),
    )
    history_msg = ChatCompletionUserMessage(
        role="user",
        content=f"Conversation so far (plain text):\n{history_text}",
    )

    try:
        response = completion_fn(model=model, messages=[dev, history_msg])
    except Exception as exc:
        logger.warning("Tool shortlisting LLM call failed: %s", exc)
        return tools[:max_selected]

    if cost_callback and response and response.usage:
        cost_callback(response.usage)

    text = response.choices[0].message.content
    if text is None:
        text = str(response.choices[0].message)

    positions = []
    for name in names:
        idx = text.find(name)
        if idx != -1:
            positions.append((idx, name))

    if len(positions) == 0:
        logger.info("Tool shortlist fallback: no matches, taking first %d", max_selected)
        return tools[:max_selected]

    positions.sort(key=lambda x: x[0])
    selected_names = [name for _, name in positions][:max_selected]
    name_to_tool = {tool["function"]["name"]: tool for tool in tools}
    selected_tools = [name_to_tool[name] for name in selected_names]
    logger.info("Tool shortlist: %d -> %d", len(tools), len(selected_tools))
    return selected_tools


def _render_history(messages: list[Any]) -> str:
    parts: List[str] = []
    for message in messages:
        msg = message if isinstance(message, dict) else dict(message)
        role = msg.get("role") or "unknown"
        if role == "tool":
            content = msg.get("content", "")
            parts.append(f"tool: {content}")
            continue
        content = msg.get("content")
        if content:
            parts.append(f"{role}: {content}")
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            parts.append(f"{role} tool_call: {fn.get('name', '?')}({fn.get('arguments', '')})")
    return "\n".join(parts)[-8000:]
