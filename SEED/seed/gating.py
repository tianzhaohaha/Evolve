"""GRPO-floor guards for SEED's OPD term (pure helpers, no Ray).

The OPD loss ``gate * (teacher_lp - student_lp)`` only ever pushes sampled tokens up. Three
default-off switches keep that push from dragging SEED below plain GRPO when the teacher context
carries no information:

* ``algorithm.seed.success_only`` -> :func:`should_analyze_trajectory`: only verified successes are
  analyzed and distilled (mirror of ``failed_only``), so the term never pushes a failed rollout up.
* ``actor_rollout_ref.actor.opd_norm_mode=response`` (``core_algos.compute_opd_loss``): the OPD
  numerator is divided by all response tokens instead of the masked ones, so the coefficient is a
  fixed per-token ratio to the policy gradient whatever the mask size.
* ``algorithm.seed.traj_gap_gate`` -> :func:`compute_traj_gap_gate`: a trajectory keeps its
  teacher signal only if the context makes the whole trajectory more likely (mean gap > margin).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import torch


def should_analyze_trajectory(success_value: Optional[float], *, failed_only: bool, success_only: bool) -> bool:
    """Whether a trajectory enters SEED analysis (and therefore the teacher mask).

    ``success_only``: verified successes only, unlabelled trajectories are left out. ``failed_only``:
    failures plus unlabelled trajectories (original behaviour). Both off: every trajectory.
    """
    if success_only:
        return success_value is not None and float(success_value) >= 1.0
    if not failed_only:
        return True
    return success_value is None or float(success_value) < 1.0


def compute_traj_gap_gate(
    traj_uids: Sequence[object],
    teacher_log_prob: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    signal_mask: torch.Tensor,
    *,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-trajectory information gate on the teacher signal.

    For every trajectory with at least one signalled row, ``teacher_log_prob - old_log_probs`` is
    averaged over the response tokens of its signalled rows; the trajectory keeps its signal iff
    that mean exceeds ``margin`` (nats per token). Returns ``signal_mask`` as a 1-D bool row mask
    with the rows of rejected trajectories cleared (rows without signal stay False) and the
    ``seed/traj_gate/*`` metrics.
    """
    rows = signal_mask.detach()
    if rows.dim() == 2:
        rows = rows.any(dim=-1)
    rows = rows.bool().cpu().tolist()
    resp = response_mask.detach().to(device="cpu", dtype=torch.float32)
    gap = (teacher_log_prob.detach().cpu().float() - old_log_probs.detach().cpu().float()) * resp
    row_gap, row_tokens = gap.sum(dim=-1).tolist(), resp.sum(dim=-1).tolist()

    traj_gap: Dict[object, float] = defaultdict(float)
    traj_tokens: Dict[object, float] = defaultdict(float)
    for idx, traj_uid in enumerate(traj_uids):
        if rows[idx]:
            traj_gap[traj_uid] += row_gap[idx]
            traj_tokens[traj_uid] += row_tokens[idx]
    traj_mean = {t: traj_gap[t] / traj_tokens[t] for t in traj_gap if traj_tokens[t] > 0}
    keep_traj = {t: mean > float(margin) for t, mean in traj_mean.items()}
    keep_rows = torch.tensor(
        [rows[idx] and keep_traj.get(traj_uid, False) for idx, traj_uid in enumerate(traj_uids)],
        dtype=torch.bool,
    )

    passed = [mean for t, mean in traj_mean.items() if keep_traj[t]]
    rejected = [mean for t, mean in traj_mean.items() if not keep_traj[t]]
    metrics = {
        "seed/traj_gate/margin": float(margin),
        "seed/traj_gate/trajs_scored": float(len(traj_mean)),
        "seed/traj_gate/pass_ratio": len(passed) / len(traj_mean) if traj_mean else 0.0,
        "seed/traj_gate/rows_before": float(sum(rows)),
        "seed/traj_gate/rows_after": float(keep_rows.sum().item()),
        "seed/traj_gate/gap_mean_pass": sum(passed) / len(passed) if passed else 0.0,
        "seed/traj_gate/gap_mean_fail": sum(rejected) / len(rejected) if rejected else 0.0,
    }
    return keep_rows.to(signal_mask.device), metrics
