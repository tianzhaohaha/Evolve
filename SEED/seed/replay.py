"""Task-free experience replay for the SEED trainer.

Stores whole GRPO groups (all step samples of one task's rollouts, with the
advantages, old/ref/teacher log-probs already attached) after advantages are
computed, keeps a uniform reservoir sample of the stream so far, and hands a
few groups back per step to mix into the actor update. No task or domain
identity is used anywhere; the reservoir keeps every part of the stream
represented in proportion on its own.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from verl.protocol import DataProto

# Trainer-internal bookkeeping column that must not survive into stored groups.
_TRANSIENT_NON_TENSOR_KEYS = ("_batch_source_idx",)


def split_groups(batch: DataProto) -> List[DataProto]:
    """Split a batch into its GRPO groups (one per ``uid``), preserving order."""
    uids = np.asarray(batch.non_tensor_batch["uid"])
    groups = []
    for uid in dict.fromkeys(uids.tolist()):
        group = batch.select_idxs(np.flatnonzero(uids == uid))
        for key in _TRANSIENT_NON_TENSOR_KEYS:
            group.non_tensor_batch.pop(key, None)
        group.meta_info = {}  # never keep a reference to the live batch's per-step payloads
        groups.append(group)
    return groups


def has_policy_signal(group: DataProto) -> bool:
    """A group whose advantages are all zero carries no policy gradient."""
    return bool(torch.any(group.batch["advantages"] != 0))


def _live_view(live: DataProto) -> DataProto:
    return DataProto(
        batch=live.batch,
        non_tensor_batch={k: v for k, v in live.non_tensor_batch.items() if k not in _TRANSIENT_NON_TENSOR_KEYS},
        meta_info=dict(live.meta_info),
    )


def describe_mismatch(live: DataProto, replay: List[DataProto]) -> str:
    """Name what keeps ``replay`` from being concatenated behind ``live`` ('' if nothing does).

    Compared per stored group against the live batch: tensor / non-tensor key
    sets, then per-key shapes beyond the batch dimension.
    """
    live_view = _live_view(live)
    for i, group in enumerate(replay):
        for kind, a, b in (
            ("tensor", live_view.batch, group.batch),
            ("non-tensor", live_view.non_tensor_batch, group.non_tensor_batch),
        ):
            a_keys, b_keys = set(a.keys()), set(b.keys())
            if a_keys != b_keys:
                return f"group {i} {kind} keys: only in live={sorted(a_keys - b_keys)}, only in replay={sorted(b_keys - a_keys)}"
            for key in a_keys:
                a_shape, b_shape = tuple(a[key].shape[1:]), tuple(b[key].shape[1:])
                if a_shape != b_shape:
                    return f"group {i} {kind} '{key}' shape: live={a_shape}, replay={b_shape}"
    return ""


def merge_for_update(live: DataProto, replay: List[DataProto]) -> Tuple[Optional[DataProto], str]:
    """Concatenate replay groups behind the live batch for the actor update.

    Returns ``(merged, "")``, or ``(None, reason)`` when the groups cannot be
    concatenated (key sets, shapes, or a concat error); replay must never take
    the main update down with it. The live batch is not mutated: the merged
    proto carries a shallow copy of its meta_info.
    """
    reason = describe_mismatch(live, replay)
    if reason:
        return None, reason
    try:
        return DataProto.concat([_live_view(live), *replay]), ""
    except (RuntimeError, ValueError) as exc:
        return None, f"concat failed: {type(exc).__name__}: {exc}"


class ReplayBuffer:
    """Reservoir of GRPO groups; ``sample()`` before ``add_batch()`` keeps a step from replaying itself."""

    def __init__(self, capacity: int, groups_per_step: int, seed: int = 0):
        if groups_per_step < 1:
            raise ValueError("replay.groups_per_step must be >= 1")
        if capacity < groups_per_step:
            raise ValueError("replay.capacity must be >= replay.groups_per_step")
        self.capacity = int(capacity)
        self.groups_per_step = int(groups_per_step)
        self._rng = np.random.default_rng(int(seed))
        self._groups: List[DataProto] = []
        self._seen = 0

    def __len__(self) -> int:
        return len(self._groups)

    def add_batch(self, batch: DataProto) -> None:
        """Reservoir-insert every group of ``batch`` that carries policy signal."""
        for group in split_groups(batch):
            if has_policy_signal(group):
                self._add_group(group)

    def _add_group(self, group: DataProto) -> None:
        self._seen += 1
        if len(self._groups) < self.capacity:
            self._groups.append(group)
            return
        slot = int(self._rng.integers(self._seen))
        if slot < self.capacity:
            self._groups[slot] = group

    def sample(self) -> List[DataProto]:
        """Uniformly draw up to ``groups_per_step`` stored groups without replacement."""
        count = min(self.groups_per_step, len(self._groups))
        chosen = self._rng.choice(len(self._groups), size=count, replace=False) if count else []
        return [self._groups[int(i)] for i in chosen]

    def permutation(self, n: int) -> np.ndarray:
        return self._rng.permutation(n)

    def metrics(self) -> Dict[str, float]:
        return {"replay/buffer_groups": float(len(self._groups)), "replay/seen_groups": float(self._seen)}
