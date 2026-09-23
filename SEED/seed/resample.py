"""Sibling resample pass for SEED (``algorithm.seed.sibling_resample``): pure helpers.

Context distillation from same-task siblings. After the ordinary rollout, every mixed-outcome GRPO
group nominates its shortest successful rollout; those tasks are rolled out once more with that
rollout rendered as a "Reference Solution" in every step's prompt (``obs['text']``), while
``obs['text_base']`` keeps the plain prompt. The trainer re-tokenises the new rows under the plain
prompt and trains on them like any other rows (they keep the fresh ``uid`` of the second pass, so
they form their own groups). The offline delta test measured the effect this pass exploits: the same
demonstration in context lifts the frozen policy from 0.40 to 0.73 success, whereas re-scoring the
policy's old samples with it carries no information.

Everything here is pure (numpy) so it is unit-testable without Ray or torch.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from seed.prompting import build_augmented_observation_text
from seed.sibling import SiblingReference


def augment_observations(texts: Sequence[str], references: Sequence[str]) -> List[str]:
    """Prompt texts with the per-slot reference injected as the "Reference Solution" section;
    slots with an empty reference are returned verbatim."""
    return [
        build_augmented_observation_text(observation=text, reference_solution=ref) if ref else text
        for text, ref in zip(texts, references)
    ]


def build_resample_requests(
    references: Mapping[object, SiblingReference],
    task_refs_by_uid: Mapping[object, Tuple[int, str, str]],
    *,
    max_groups: int,
) -> List[Dict[str, object]]:
    """One request per group to roll out again: the first ``max_groups`` groups (by ``sample_id``)
    that have both a reference and a task identity ``(sample_id, task_slug, task_id)``."""
    rows = [
        (task_refs_by_uid[uid][0], uid, ref)
        for uid, ref in references.items()
        if uid in task_refs_by_uid
    ]
    rows.sort(key=lambda r: r[0])
    return [
        {
            "uid": uid,
            "sample_id": int(sample_id),
            "task_slug": str(task_refs_by_uid[uid][1]),
            "task_id": str(task_refs_by_uid[uid][2]),
            "reference_solution": ref.text,
            "reference_traj_uid": ref.traj_uid,
        }
        for sample_id, uid, ref in rows[: max(int(max_groups), 0)]
    ]


def reconcile_non_tensor_keys(
    main: Mapping[str, np.ndarray],
    extra: Mapping[str, np.ndarray],
    *,
    is_broadcast_key: Callable[[str], bool],
) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    """Non-tensor columns of the resample rows aligned to the main batch's key set so that
    ``DataProto.concat`` accepts them: keys the main batch lacks are dropped; the per-episode
    metric columns (batch means broadcast to every row, ``is_broadcast_key`` =
    ``verl.trainer.ppo.metric_utils.is_episode_metric_key``) and any other key the extra rows lack
    are copied from the main batch's first row. The metric columns are overwritten on purpose:
    ``compute_data_metrics`` logs row 0 of each after ``_balance_batch`` reorders the rows, so a
    differing value on a resample row would leak into ``episode/*``.
    Returns ``(columns, dropped_keys, filled_keys)``."""
    n = len(next(iter(extra.values()))) if extra else 0
    columns: Dict[str, np.ndarray] = {}
    dropped = sorted(k for k in extra if k not in main)
    filled = []
    for key, column in main.items():
        if key in extra and not is_broadcast_key(key):
            columns[key] = extra[key]
        else:
            columns[key] = np.repeat(column[:1], n, axis=0)
            filled.append(key)
    return columns, dropped, filled


def summarize_resample_batch(
    uids: Sequence[object],
    traj_uids: Sequence[object],
    episode_rewards: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    """Outcome of the resample pass from its own columns: rows, trajectories, groups, success
    rate (``episode_rewards >= threshold``) and the share of uniform-outcome groups (no GRPO
    signal)."""
    seen: Dict[object, Tuple[object, bool]] = {}
    for uid, traj_uid, reward in zip(uids, traj_uids, episode_rewards):
        seen.setdefault(traj_uid, (uid, float(reward) >= threshold))
    by_group: Dict[object, List[bool]] = {}
    for uid, success in seen.values():
        by_group.setdefault(uid, []).append(success)
    successes = [s for outcomes in by_group.values() for s in outcomes]
    uniform = sum(1 for outcomes in by_group.values() if len(set(outcomes)) == 1)
    return {
        "rows": float(len(uids)),
        "trajs": float(len(seen)),
        "groups": float(len(by_group)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "uniform_group_ratio": uniform / len(by_group) if by_group else 0.0,
    }
