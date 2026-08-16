# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import litellm

from .harness_store import HarnessStore
from .prompts.evolver import EVOLVER_SYSTEM_PROMPT, build_evolution_user_message
from .retriever import compute_embedding

logger = logging.getLogger(__name__)

EVOLVER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_prompt",
            "description": "Read the current system prompt.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Read the current long-term memory document.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List all skills with their names and descriptions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "Read the full body of a specific skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The skill name to read."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_prompt",
            "description": "Replace the entire system prompt with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "body": {"type": "string", "description": "The full new system prompt text."},
                },
                "required": ["body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_memory",
            "description": "Replace the entire long-term memory document with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "body": {"type": "string", "description": "The full new memory document text."},
                },
                "required": ["body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_skill",
            "description": "Add a new skill to the skill library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Unique skill name."},
                    "description": {"type": "string", "description": "One-line description of what the skill does."},
                    "body": {"type": "string", "description": "Full skill content/instructions."},
                },
                "required": ["name", "description", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_skill",
            "description": "Modify an existing skill's description and/or body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The skill name to edit."},
                    "description": {"type": "string", "description": "New description (optional, omit to keep current)."},
                    "body": {"type": "string", "description": "New body (optional, omit to keep current)."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_skill",
            "description": "Delete a skill from the skill library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The skill name to delete."},
                },
                "required": ["name"],
            },
        },
    },
]


# ================================================================== #
# Tool executor
# ================================================================== #

class _EvolverToolExecutor:
    """Executes evolver tools against a HarnessStore.

    Tracks changes and enforces constraints (at most 1 edit_prompt, 1 edit_memory).
    """

    def __init__(self, store: HarnessStore, embedding_model: str) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.ops_applied: List[str] = []
        self._edit_prompt_used = False
        self._edit_memory_used = False

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool call and return the result string."""
        if tool_name == "read_prompt":
            return self.store.system_prompt or "(Empty)"

        elif tool_name == "read_memory":
            return self.store.memory or "(Empty)"

        elif tool_name == "list_skills":
            index = self.store.get_skill_index()
            if not index:
                return "(No skills yet)"
            return json.dumps(index, ensure_ascii=False, indent=2)

        elif tool_name == "read_skill":
            name = args.get("name", "")
            skills = {s.name: s for s in self.store.list_skills()}
            skill = skills.get(name)
            if skill is None:
                return f"ERROR: Skill '{name}' not found."
            return f"Name: {skill.name}\nDescription: {skill.description}\n\n{skill.body}"

        elif tool_name == "edit_prompt":
            if self._edit_prompt_used:
                return "ERROR: edit_prompt already used this session (limit: 1 per task)."
            body = args.get("body", "")
            if not body:
                return "ERROR: 'body' is required."
            self.store.edit_prompt(body)
            self._edit_prompt_used = True
            self.ops_applied.append("edit_prompt")
            return "OK: System prompt updated."

        elif tool_name == "edit_memory":
            if self._edit_memory_used:
                return "ERROR: edit_memory already used this session (limit: 1 per task)."
            body = args.get("body", "")
            self.store.edit_memory(body)
            self._edit_memory_used = True
            self.ops_applied.append("edit_memory")
            return "OK: Memory updated."

        elif tool_name == "add_skill":
            name = args.get("name", "")
            description = args.get("description", "")
            body = args.get("body", "")
            if not name or not description or not body:
                return "ERROR: 'name', 'description', and 'body' are all required."
            # Check if skill already exists
            existing = {s.name for s in self.store.list_skills()}
            if name in existing:
                return f"ERROR: Skill '{name}' already exists. Use edit_skill to modify it."
            self.store.add_skill(name, description, body)
            # Compute embedding
            emb = compute_embedding(description, model=self.embedding_model)
            if emb:
                self.store.set_embedding(name, emb)
            self.ops_applied.append(f"add_skill:{name}")
            return f"OK: Skill '{name}' added."

        elif tool_name == "edit_skill":
            name = args.get("name", "")
            if not name:
                return "ERROR: 'name' is required."
            description = args.get("description")
            body = args.get("body")
            if description is None and body is None:
                return "ERROR: At least one of 'description' or 'body' must be provided."
            success = self.store.edit_skill(name, description=description, body=body)
            if not success:
                return f"ERROR: Skill '{name}' not found."
            if description:
                emb = compute_embedding(description, model=self.embedding_model)
                if emb:
                    self.store.set_embedding(name, emb)
            self.ops_applied.append(f"edit_skill:{name}")
            return f"OK: Skill '{name}' updated."

        elif tool_name == "delete_skill":
            name = args.get("name", "")
            if not name:
                return "ERROR: 'name' is required."
            success = self.store.delete_skill(name)
            if not success:
                return f"ERROR: Skill '{name}' not found."
            self.ops_applied.append(f"delete_skill:{name}")
            return f"OK: Skill '{name}' deleted."

        else:
            return f"ERROR: Unknown tool '{tool_name}'."


# ================================================================== #
# Multi-turn evolver loop
# ================================================================== #

def _run_evolver_loop(
    model: str,
    system_prompt: str,
    user_message: str,
    executor: _EvolverToolExecutor,
    max_turns: int = 20,
) -> None:
    """Run multi-turn evolver with tools.

    The LLM can read harness state, then make changes via tool calls.
    Loop ends when LLM stops calling tools (finish_reason != tool_calls).
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for turn in range(max_turns):
        # Call LLM
        max_attempts = 3
        response = None
        for attempt in range(max_attempts):
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    tools=EVOLVER_TOOLS,
                    temperature=0.0,
                )
                break
            except Exception as exc:
                logger.warning("Evolver loop turn %d attempt %d failed: %s", turn, attempt + 1, exc)
                if attempt + 1 >= max_attempts:
                    logger.error("Evolver loop: all attempts failed at turn %d", turn)
                    return
                time.sleep(2 ** attempt)

        if response is None:
            break

        choice = response.choices[0]
        message = choice.message

        # Check for tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            # Add assistant message to history
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
            messages.append(assistant_msg)

            # Execute each tool call
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                result = executor.execute(tc.function.name, args)
                # Log each tool call with args summary and result preview
                args_summary = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
                logger.info(
                    "Evolver turn %d: %s(%s) → %s",
                    turn, tc.function.name, args_summary, result[:120],
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            continue  # Next turn
        else:
            # LLM finished (no more tool calls)
            logger.info("Evolver finished after %d turns", turn + 1)
            break


# ================================================================== #
# Main evolver entry point
# ================================================================== #

def run_evolver(
    store: HarnessStore,
    task: str,
    injected_skill_names: List[str],
    trajectory: str,
    llm_call: Any,  # kept for interface compat (unused in multi-turn impl)
    evolver_model: str,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> tuple[int, List[str]]:
    """Run the evolver: multi-turn tool-calling to read and modify harness.

    Returns (number of ops applied, list of op summaries).
    """
    # Take snapshot for full rollback on catastrophic failure
    full_snapshot = store.snapshot()

    try:
        # Build user message
        user_message = build_evolution_user_message(
            task=task,
            injected_skill_names=injected_skill_names,
            trajectory=trajectory,
        )

        # Create tool executor
        executor = _EvolverToolExecutor(store, embedding_model)

        # Run multi-turn loop
        _run_evolver_loop(
            model=evolver_model,
            system_prompt=EVOLVER_SYSTEM_PROMPT,
            user_message=user_message,
            executor=executor,
            max_turns=20,
        )

        ops_applied = len(executor.ops_applied)
        logger.info("Harness evolver: %d ops applied: %s", ops_applied, executor.ops_applied)
        return ops_applied, executor.ops_applied

    except Exception as exc:
        logger.warning("Harness evolver failed, rolling back: %s", exc)
        store.rollback(full_snapshot)
        return 0, []
