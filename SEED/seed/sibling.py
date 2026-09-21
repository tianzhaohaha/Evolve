"""Sibling-success local teacher and sample routing for SEED (pure helpers).

``algorithm.seed.local_teacher_source=sibling_success`` replaces the analyzer-written hindsight
skill with the action skeleton of a verified successful rollout of the same task (same GRPO
group), used as the OPD teacher context only for the failed rollouts of groups that contain both
outcomes. ``algorithm.seed.route_mode=sample`` gives every row a policy-gradient weight so a
trajectory is trained by one objective: successes by PG, failed rows of mixed groups by the OPD
teacher (PG weight ``failed_weight``), rows of uniform-outcome groups by PG as before.

``build_reference_solution`` renders the same trajectory with the policy's full responses (reasoning
and action) instead of action-only skeletons; it is the context of the offline delta test
(examples/agentstream_trainer/delta_test.py) and the candidate format for a reasoning-aware sibling
teacher.

Everything here is pure (numpy / json / re) so it is unit-testable without Ray or torch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Same tag grammar as agent_system/.../agentstream/projection.py, duplicated so seed/ stays env-agnostic.
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_WS_RE = re.compile(r"\s+")

SIBLING_ANALYSIS_MODE = "sibling_success"
_OMISSION_MARKER = "... ({n} steps omitted) ..."


def _collapse(text: object) -> str:
    return _WS_RE.sub(" ", str(text or "")).strip()


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(max_chars - 3, 0)] + "..."


def extract_action_text(response: object, max_chars: int = 300) -> str:
    """Compact action of one step: the first <action> JSON, else the first <tool_call>, else ''."""
    raw = str(response or "")
    match = _ACTION_RE.search(raw) or _TOOL_CALL_RE.search(raw)
    if match is None:
        return ""
    text = match.group(1).strip()
    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    try:
        text = json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = _collapse(text)
    return _truncate(text, max_chars)


def _step_lines(steps: Sequence[Mapping[str, object]], *, obs_chars: int, render) -> List[str]:
    """``Step k | obs: ... | <render(step)>`` per step; skips ``action_valid=False`` and steps ``render`` rejects."""
    lines: List[str] = []
    for step in steps:
        if step.get("action_valid") is False:
            continue
        body = render(step)
        if not body:
            continue
        obs = _truncate(_collapse(step.get("observation", "")), obs_chars)
        lines.append(f"Step {int(step.get('step_index', len(lines))) + 1} | obs: {obs} | {body}")
    return lines


def _cap_lines(lines: List[str], max_chars: int) -> Tuple[str, int]:
    """Join step lines. Over ``max_chars`` the leading lines are kept, the rest is replaced by an
    omission marker and the final line is always kept. Returns ``(text, rendered_steps)``."""
    if not lines:
        return "", 0
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text, len(lines)

    last = lines[-1]
    budget = max_chars - len(last) - len(_OMISSION_MARKER.format(n=len(lines))) - 2
    kept: List[str] = []
    used = 0
    for line in lines[:-1]:
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    omitted = len(lines) - 1 - len(kept)
    if omitted > 0:
        kept.append(_OMISSION_MARKER.format(n=omitted))
    kept.append(last)
    return _truncate("\n".join(kept), max_chars), len(lines) - omitted


def build_action_skeleton(
    steps: Sequence[Mapping[str, object]],
    *,
    obs_chars: int = 160,
    action_chars: int = 300,
    max_chars: int = 6000,
) -> Tuple[str, int]:
    """Render a trajectory as ``Step k | obs: ... | action: ...`` lines.

    Step numbers are 1-based, matching the policy prompt's "You are now at step N".
    Steps without a parsable action or flagged ``action_valid=False`` are skipped. Over
    ``max_chars`` the leading lines are kept, the rest is replaced by an omission marker and the
    final step is always kept (it is usually the submit / finish action). Returns
    ``(text, rendered_steps)``; ``("", 0)`` when no step carries an action.
    """

    def render(step: Mapping[str, object]) -> str:
        action = extract_action_text(step.get("response", ""), max_chars=action_chars)
        return f"action: {action}" if action else ""

    return _cap_lines(_step_lines(steps, obs_chars=obs_chars, render=render), max_chars)


def build_reference_solution(
    steps: Sequence[Mapping[str, object]],
    *,
    obs_chars: int = 160,
    response_chars: int = 1200,
    max_chars: int = 6000,
) -> Tuple[str, int]:
    """Render a trajectory as ``Step k | obs: ... | response: ...`` lines with the policy's full
    response (reasoning and action tags as written, whitespace collapsed, capped at
    ``response_chars``), so a reference shows how the sibling reasoned, not only what it did.
    Skip and cap rules are those of :func:`build_action_skeleton`.
    """

    def render(step: Mapping[str, object]) -> str:
        response = _truncate(_collapse(step.get("response", "")), response_chars)
        return f"response: {response}" if response else ""

    return _cap_lines(_step_lines(steps, obs_chars=obs_chars, render=render), max_chars)


def group_outcomes(
    uids: Sequence[object],
    traj_uids: Sequence[object],
    traj_success: Mapping[object, float],
) -> Dict[object, Tuple[List[object], List[object]]]:
    """Per GRPO group ``uid``: ``(successful trajectories, failed trajectories)`` in first-seen
    order. Trajectories without a success label are left out of both lists."""
    groups: Dict[object, Tuple[List[object], List[object]]] = {}
    seen = set()
    for uid, traj_uid in zip(uids, traj_uids):
        if traj_uid in seen:
            continue
        seen.add(traj_uid)
        success = traj_success.get(traj_uid)
        if success is None:
            continue
        bucket = groups.setdefault(uid, ([], []))
        bucket[0 if float(success) >= 1.0 else 1].append(traj_uid)
    return groups


@dataclass(frozen=True)
class SiblingReference:
    traj_uid: object  # the successful sibling whose skeleton is the reference
    text: str
    rendered_steps: int
    total_steps: int


def select_sibling_references(
    episodes: Mapping[object, Sequence[Mapping[str, object]]],
    uids: Sequence[object],
    traj_uids: Sequence[object],
    traj_success: Mapping[object, float],
    traj_rewards: Optional[Mapping[object, float]] = None,
    *,
    obs_chars: int = 160,
    action_chars: int = 300,
    max_chars: int = 6000,
) -> Tuple[Dict[object, SiblingReference], Dict[str, float]]:
    """Map every failed trajectory of a mixed-outcome group to its group's reference solution.

    The reference is the shortest successful sibling (ties: higher reward, then ``traj_uid``)
    whose skeleton is non-empty. Returns ``(failed_traj_uid -> reference, metrics)``.
    """
    groups = group_outcomes(uids, traj_uids, traj_success)
    rewards = traj_rewards or {}
    references: Dict[object, SiblingReference] = {}
    group_refs: List[SiblingReference] = []
    mixed = allfail = allsuccess = 0
    for successes, failures in groups.values():
        if not successes:
            allfail += 1
            continue
        if not failures:
            allsuccess += 1
            continue
        mixed += 1
        ranked = sorted(
            successes,
            key=lambda t: (len(episodes.get(t, ())), -float(rewards.get(t, 0.0)), str(t)),
        )
        for candidate in ranked:
            steps = episodes.get(candidate, ())
            text, rendered = build_action_skeleton(
                steps, obs_chars=obs_chars, action_chars=action_chars, max_chars=max_chars
            )
            if not text:
                continue
            reference = SiblingReference(candidate, text, rendered, len(steps))
            group_refs.append(reference)
            for failed in failures:
                references[failed] = reference
            break
    total = len(groups)
    metrics = {
        "seed/sibling/groups_total": float(total),
        "seed/sibling/mixed_group_ratio": mixed / total if total else 0.0,
        "seed/sibling/allfail_group_ratio": allfail / total if total else 0.0,
        "seed/sibling/allsuccess_group_ratio": allsuccess / total if total else 0.0,
        "seed/sibling/failed_trajs_masked": float(len(references)),
        "seed/sibling/ref_chars_mean": float(np.mean([len(r.text) for r in group_refs])) if group_refs else 0.0,
        "seed/sibling/ref_steps_mean": float(np.mean([r.rendered_steps for r in group_refs])) if group_refs else 0.0,
    }
    return references, metrics


def compute_pg_row_weights(
    uids: Sequence[object],
    traj_uids: Sequence[object],
    traj_success: Mapping[object, float],
    failed_weight: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Per-row policy-gradient weight: ``failed_weight`` on rows of failed trajectories that
    belong to a mixed-outcome group, 1.0 everywhere else (successes, uniform groups, unlabeled)."""
    groups = group_outcomes(uids, traj_uids, traj_success)
    mixed_groups = [(s, f) for s, f in groups.values() if s and f]
    weighted_trajs = {t for _, failures in mixed_groups for t in failures}
    weights = np.array(
        [float(failed_weight) if t in weighted_trajs else 1.0 for t in traj_uids], dtype=np.float32
    )
    n = len(weights)
    rows_weighted = sum(1 for t in traj_uids if t in weighted_trajs)
    metrics = {
        "seed/route/rows_pg_weighted_ratio": rows_weighted / n if n else 0.0,
        "seed/route/rows_pg_full_ratio": (n - rows_weighted) / n if n else 0.0,
        "seed/route/mixed_group_ratio": len(mixed_groups) / len(groups) if groups else 0.0,
        "seed/route/failed_weight": float(failed_weight),
    }
    return weights, metrics
