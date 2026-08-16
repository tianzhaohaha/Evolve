# Copyright 2026 SEED x AgentStream integration.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge to the exgentic host-side API bundled inside AgentStream.

All exgentic imports are lazy and go through :func:`bootstrap_exgentic`, which
adds ``<exgentic_root>/src`` to ``sys.path``. Benchmark heavy dependencies stay
inside the per-benchmark uv venvs created by ``exgentic install`` (see the
AgentStream README); only the lightweight host layer (pydantic models, runner
transport) is imported into SEED processes.

Two call sites use this module:

  * the host-side :class:`BenchmarkHub` (task discovery + session kwargs),
    owned by the environment manager / factory;
  * the Ray worker-side :class:`SessionDriver` (session lifecycle + steps),
    one per parallel environment slot.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Tuple

_BOOTSTRAPPED: Dict[str, bool] = {}


def bootstrap_exgentic(exgentic_root: str) -> None:
    """Make the bundled exgentic package importable (idempotent)."""
    root = os.path.abspath(os.path.expanduser(exgentic_root))
    if _BOOTSTRAPPED.get(root):
        return
    src = os.path.join(root, "src")
    if not os.path.isdir(os.path.join(src, "exgentic")):
        raise FileNotFoundError(
            f"env.agentstream.exgentic_root='{exgentic_root}' does not contain src/exgentic. "
            "Point it at the AgentStream/exgentic checkout."
        )
    if src not in sys.path:
        sys.path.insert(0, src)
    _BOOTSTRAPPED[root] = True


def load_benchmark_class(slug: str):
    from exgentic.interfaces.registry import load_benchmark

    return load_benchmark(slug)


def make_benchmark(slug: str, bm_kwargs: Dict[str, Any], runner: Optional[str]):
    """Instantiate the host-side Benchmark config object."""
    cls = load_benchmark_class(slug)
    kwargs = dict(bm_kwargs)
    if runner and "runner" not in kwargs:
        kwargs["runner"] = runner
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Host side: task discovery + session kwargs
# ---------------------------------------------------------------------------

class BenchmarkHub:
    """Keeps one Benchmark + Evaluator per slug alive on the host process.

    ``get_session_kwargs`` results are plain serializable dicts (guaranteed by
    the exgentic Evaluator contract), so they can be shipped to Ray workers.
    """

    def __init__(
        self,
        exgentic_root: str,
        slugs: List[str],
        benchmark_kwargs: Dict[str, Dict[str, Any]],
        runner: Optional[str],
        output_dir: str,
        run_id: str,
    ) -> None:
        bootstrap_exgentic(exgentic_root)
        self._stack = ExitStack()
        self._enter_run_context(run_id, output_dir)

        self._benchmarks: Dict[str, Any] = {}
        self._evaluators: Dict[str, Any] = {}
        for slug in slugs:
            self._benchmarks[slug] = make_benchmark(slug, benchmark_kwargs.get(slug, {}), runner)

    def _enter_run_context(self, run_id: str, output_dir: str) -> None:
        try:
            from exgentic.core.context import run_scope

            os.makedirs(output_dir, exist_ok=True)
            self._stack.enter_context(run_scope(run_id=run_id, output_dir=output_dir))
        except Exception as exc:  # pragma: no cover - context is best-effort
            print(f"[BenchmarkHub] warning: could not enter exgentic run context: {exc}")

    def _evaluator(self, slug: str):
        if slug not in self._evaluators:
            self._evaluators[slug] = self._benchmarks[slug].get_evaluator()
        return self._evaluators[slug]

    def list_tasks(self, slug: str) -> List[str]:
        return [str(t) for t in self._evaluator(slug).list_tasks()]

    def list_all_tasks(self) -> Dict[str, List[str]]:
        return {slug: self.list_tasks(slug) for slug in sorted(self._benchmarks)}

    def session_kwargs(self, slug: str, task_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        from exgentic.core.types import SessionIndex

        sid = session_id or f"seed_{uuid.uuid4().hex[:12]}"
        index = SessionIndex(task_id=str(task_id), session_id=sid)
        kwargs = self._evaluator(slug).get_session_kwargs(index)
        return dict(kwargs)

    def close(self) -> None:
        for slug, evaluator in list(self._evaluators.items()):
            try:
                evaluator.close()
            except Exception:
                pass
            self._evaluators.pop(slug, None)
        for slug, bm in list(self._benchmarks.items()):
            try:
                bm.close()
            except Exception:
                pass
        try:
            self._stack.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Worker side: one live session per environment slot
# ---------------------------------------------------------------------------

def render_action_schemas(actions: List[Any]) -> str:
    """Human/LLM-readable list of available actions with JSON schemas."""
    blocks: List[str] = []
    for action_type in actions:
        try:
            schema = action_type.arguments.model_json_schema()
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            arg_lines = []
            for prop_name, prop in props.items():
                type_name = prop.get("type", prop.get("anyOf", "any"))
                req = "required" if prop_name in required else "optional"
                desc = prop.get("description", "")
                arg_lines.append(f"    - {prop_name} ({type_name}, {req}): {desc}".rstrip())
            args_text = "\n".join(arg_lines) if arg_lines else "    (no arguments)"
        except Exception:
            args_text = "    (schema unavailable)"
        blocks.append(f"- {action_type.name}: {action_type.description}\n{args_text}")
    return "\n".join(blocks)


def observation_to_text(observation: Any, max_chars: int = 4096) -> str:
    """Render an exgentic Observation into plain text for the policy prompt."""
    if observation is None:
        return ""
    try:
        parts: List[str] = []
        for single in observation.to_observation_list():
            result = getattr(single, "result", single)
            if isinstance(result, (dict, list)):
                try:
                    parts.append(json.dumps(result, ensure_ascii=False, default=str))
                except Exception:
                    parts.append(str(result))
            else:
                parts.append(str(result))
        text = "\n".join(p for p in parts if p)
    except Exception:
        text = str(observation)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[observation truncated]"
    return text


class SessionDriver:
    """Drives one exgentic Session with (name, arguments) dict actions.

    Mirrors the role of an exgentic agent instance in
    ``core/orchestrator/session.py::run_session``, but is controlled step by
    step from the outside (SEED's rollout loop).
    """

    def __init__(
        self,
        exgentic_root: str,
        runner: Optional[str],
        output_dir: str,
        run_id: str,
        max_steps: int,
    ) -> None:
        bootstrap_exgentic(exgentic_root)
        self._runner = runner
        self._max_steps = max_steps
        self._stack = ExitStack()
        try:
            from exgentic.core.context import run_scope

            os.makedirs(output_dir, exist_ok=True)
            self._stack.enter_context(run_scope(run_id=run_id, output_dir=output_dir))
        except Exception as exc:  # pragma: no cover
            print(f"[SessionDriver] warning: could not enter exgentic run context: {exc}")

        self._benchmarks: Dict[str, Any] = {}
        self._session: Any = None
        self._action_types: List[Any] = []
        self._slug: str = ""
        self._task_id: str = ""
        self._steps: int = 0
        self._finished: bool = True
        self._final_score: Optional[Dict[str, Any]] = None

    # -- lifecycle -----------------------------------------------------------

    def _benchmark(self, slug: str, bm_kwargs: Dict[str, Any]):
        if slug not in self._benchmarks:
            self._benchmarks[slug] = make_benchmark(slug, bm_kwargs, self._runner)
        return self._benchmarks[slug]

    def close_session(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def reset(
        self,
        slug: str,
        task_id: str,
        bm_kwargs: Dict[str, Any],
        session_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Start a new session for (slug, task_id); returns initial payload."""
        self.close_session()
        self._slug = slug
        self._task_id = str(task_id)
        self._steps = 0
        self._finished = False
        self._final_score = None

        benchmark = self._benchmark(slug, bm_kwargs)
        self._session = benchmark.get_session(**session_kwargs)
        observation = self._session.start()
        self._action_types = list(self._session.actions)

        return {
            "slug": slug,
            "task_id": self._task_id,
            "task": str(self._session.task),
            "context": self._safe_context(),
            "actions_text": render_action_schemas(self._action_types),
            "observation": observation_to_text(observation),
        }

    def _safe_context(self) -> str:
        try:
            context = self._session.context or {}
            if isinstance(context, dict):
                return json.dumps(context, ensure_ascii=False, default=str)[:2048]
            return str(context)[:2048]
        except Exception:
            return ""

    # -- stepping ------------------------------------------------------------

    def step(self, action_payload: Dict[str, Any]) -> Tuple[str, bool, Dict[str, Any]]:
        """Execute one parsed action dict: {'name': str, 'arguments': dict}.

        Returns (observation_text, done, info). Reward shaping happens in the
        caller (envs.py) using info['won'] / info['score'].
        """
        if self._finished or self._session is None:
            # Step after done: keep the vectorized loop shape-stable.
            return "", True, self._terminal_info(post_done=True)

        self._steps += 1
        name = str(action_payload.get("name", ""))
        arguments = action_payload.get("arguments", {})

        action = self._build_action(name, arguments)
        action_invalid = action is None or not getattr(
            getattr(action, "validation", None), "valid", True
        )

        observation = None
        step_error = ""
        if action is not None:
            try:
                observation = self._session.step(action)
            except Exception as exc:
                step_error = f"{type(exc).__name__}: {exc}"

        done = False
        try:
            done = bool(self._session.done())
        except Exception:
            pass
        if observation is None and not step_error:
            done = True
        if self._steps >= self._max_steps:
            done = True

        if done:
            return self._finalize(limit_reached=self._steps >= self._max_steps)

        if step_error:
            obs_text = f"[environment error] {step_error}. Choose a different action."
        elif action is None:
            obs_text = (
                f"[invalid action] Unknown action name '{name}'. "
                "Pick one of the available actions."
            )
        else:
            obs_text = observation_to_text(observation)

        info = {
            "won": False,
            "score": 0.0,
            "slug": self._slug,
            "task_id": self._task_id,
            "step_count": self._steps,
            "action_error": bool(step_error) or action_invalid,
        }
        return obs_text, False, info

    def _build_action(self, name: str, arguments: Any):
        matched = None
        for action_type in self._action_types:
            if action_type.name == name:
                matched = action_type
                break
        try:
            if matched is not None:
                from exgentic.core.actions import build_action

                return build_action(matched, arguments)
            return None
        except Exception:
            return None

    def _finalize(self, limit_reached: bool) -> Tuple[str, bool, Dict[str, Any]]:
        success = False
        score = 0.0
        score_error = ""
        try:
            session_score = self._session.score()
            success = bool(getattr(session_score, "success", False))
            raw_score = getattr(session_score, "score", None)
            score = float(raw_score) if raw_score is not None else float(success)
        except Exception as exc:
            score_error = f"{type(exc).__name__}: {exc}"

        self._finished = True
        self._final_score = {"won": success, "score": score}
        self.close_session()

        info = self._terminal_info()
        info["limit_reached"] = limit_reached
        if score_error:
            info["score_error"] = score_error
        return "", True, info

    def _terminal_info(self, post_done: bool = False) -> Dict[str, Any]:
        final = self._final_score or {"won": False, "score": 0.0}
        return {
            "won": bool(final["won"]),
            "score": float(final["score"]),
            "slug": self._slug,
            "task_id": self._task_id,
            "step_count": self._steps,
            "action_error": False,
            "post_done": post_done,
        }

    def close(self) -> None:
        self.close_session()
        for slug, bm in list(self._benchmarks.items()):
            try:
                bm.close()
            except Exception:
                pass
        try:
            self._stack.close()
        except Exception:
            pass
