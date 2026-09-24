"""GRPO-floor guards (seed/gating.py) and the response-normalised OPD loss through the actor."""

import pytest
import torch

from seed.gating import compute_traj_gap_gate, should_analyze_trajectory
from tests.trainer.ppo.test_opd_spec_first import assert_update_matches, run_update  # noqa: F401  (pytest fixture)


# ---------------------------------------------------------------- analysis selection


@pytest.mark.parametrize("success_value", [None, 0.0, 1.0])
def test_should_analyze_trajectory_defaults_to_every_trajectory(success_value):
    assert should_analyze_trajectory(success_value, failed_only=False, success_only=False)


def test_should_analyze_trajectory_failed_only_keeps_failures_and_unlabelled():
    assert should_analyze_trajectory(None, failed_only=True, success_only=False)
    assert should_analyze_trajectory(0.0, failed_only=True, success_only=False)
    assert not should_analyze_trajectory(1.0, failed_only=True, success_only=False)


def test_should_analyze_trajectory_success_only_keeps_verified_successes():
    assert should_analyze_trajectory(1.0, failed_only=False, success_only=True)
    assert not should_analyze_trajectory(0.0, failed_only=False, success_only=True)
    assert not should_analyze_trajectory(None, failed_only=False, success_only=True)


# ---------------------------------------------------------------- trajectory gap gate

# Trajectory a (rows 0-1): gap sum 0.8 over 7 response tokens; b (rows 2-3): -0.4 over 6 tokens;
# c (row 4): no signal, its large gap must never count. Row 0's last position is padding.
TRAJ_UIDS = ["a", "a", "b", "b", "c"]
TEACHER = torch.tensor([
    [0.4, 0.0, 0.0, 9.0],
    [0.0, 0.4, 0.0, 0.0],
    [-0.4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
])
OLD = torch.zeros(5, 4)
RESPONSE_MASK = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
SIGNAL = torch.tensor([True, True, True, True, False])
MEAN_A, MEAN_B = 0.8 / 7, -0.4 / 6


def test_traj_gap_gate_keeps_trajectories_whose_mean_gap_exceeds_the_margin():
    keep, metrics = compute_traj_gap_gate(TRAJ_UIDS, TEACHER, OLD, RESPONSE_MASK, SIGNAL, margin=0.0)

    assert keep.dtype == torch.bool
    assert keep.tolist() == [True, True, False, False, False]
    assert metrics["seed/traj_gate/trajs_scored"] == 2.0
    assert metrics["seed/traj_gate/pass_ratio"] == pytest.approx(0.5)
    assert metrics["seed/traj_gate/rows_before"] == 4.0
    assert metrics["seed/traj_gate/rows_after"] == 2.0
    assert metrics["seed/traj_gate/gap_mean_pass"] == pytest.approx(MEAN_A)
    assert metrics["seed/traj_gate/gap_mean_fail"] == pytest.approx(MEAN_B)
    assert metrics["seed/traj_gate/margin"] == 0.0


def test_traj_gap_gate_margin_is_strict_and_2d_masks_are_reduced_per_row():
    # A margin equal to the best trajectory's own mean (as the gate computed it) rejects it: strict '>'.
    best = compute_traj_gap_gate(TRAJ_UIDS, TEACHER, OLD, RESPONSE_MASK, SIGNAL)[1]["seed/traj_gate/gap_mean_pass"]
    keep, metrics = compute_traj_gap_gate(TRAJ_UIDS, TEACHER, OLD, RESPONSE_MASK, SIGNAL, margin=best)
    assert not keep.any() and metrics["seed/traj_gate/pass_ratio"] == 0.0

    keep_2d, metrics_2d = compute_traj_gap_gate(
        TRAJ_UIDS, TEACHER, OLD, RESPONSE_MASK, SIGNAL[:, None] & RESPONSE_MASK.bool(), margin=0.0
    )
    assert keep_2d.tolist() == [True, True, False, False, False]
    assert metrics_2d["seed/traj_gate/rows_after"] == 2.0

    keep_none, metrics_none = compute_traj_gap_gate(TRAJ_UIDS, TEACHER, OLD, RESPONSE_MASK, torch.zeros(5, dtype=torch.bool))
    assert not keep_none.any()
    assert metrics_none["seed/traj_gate/trajs_scored"] == 0.0 and metrics_none["seed/traj_gate/pass_ratio"] == 0.0


# ---------------------------------------------------------------- actor: response normalisation


def test_actor_response_norm_matches_reference_and_shares_the_mode_with_gen(run_update):
    result = run_update(norm_mode="response")
    assert_update_matches(result, route=True)
    assert [call["norm_mode"] for call in result.calls] == ["response", "response"]
    # teacher_signal_mask = [True, False]: 3 masked tokens out of 7 response tokens.
    assert result.metrics["actor/opd_mask_token_fraction"][0] == pytest.approx(3 / 7)


def test_actor_default_norm_mode_is_mask(run_update):
    result = run_update()
    assert [call["norm_mode"] for call in result.calls] == ["mask", "mask"]
    assert result.metrics["actor/opd_mask_token_fraction"][0] == pytest.approx(3 / 7)


# ---------------------------------------------------------------- trainer: gen channel through the gate

def _gate_batch(gen_mask):
    import numpy as np
    from verl import DataProto

    tensors = {
        "responses": torch.zeros((5, 4), dtype=torch.long), "attention_mask": RESPONSE_MASK.clone(),
        "teacher_log_prob": TEACHER.clone(), "old_log_probs": OLD.clone(), "teacher_signal_mask": SIGNAL.clone(),
        "gen_teacher_log_prob": TEACHER.clone(), "gen_skill_mask": gen_mask,
    }
    return DataProto.from_dict(tensors=tensors, non_tensors={"traj_uid": np.asarray(TRAJ_UIDS, dtype=object)})


def _gate_trainer():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer._get_seed_traj_gap_gate_margin = lambda: 0.0
    return trainer


def test_trainer_gate_leaves_an_empty_gen_mask_alone_and_reports_no_gen_metrics():
    metrics = {}
    batch = _gate_trainer()._apply_seed_traj_gap_gate(batch=_gate_batch(torch.zeros(5, dtype=torch.bool)), metrics=metrics)
    assert batch.batch["teacher_signal_mask"].tolist() == [True, True, False, False, False]
    assert not batch.batch["gen_skill_mask"].any()
    assert metrics["seed/traj_gate/pass_ratio"] == pytest.approx(0.5)
    assert not any(key.startswith("seed/traj_gate/gen_") for key in metrics)


@pytest.mark.parametrize("two_d", [False, True])
def test_trainer_gate_covers_the_gen_channel_with_its_own_metrics(two_d):
    gen_mask = (SIGNAL[:, None] & RESPONSE_MASK.bool()) if two_d else SIGNAL.clone()
    metrics = {}
    batch = _gate_trainer()._apply_seed_traj_gap_gate(batch=_gate_batch(gen_mask), metrics=metrics)
    kept = batch.batch["gen_skill_mask"]
    assert (kept.any(dim=-1) if two_d else kept).tolist() == [True, True, False, False, False]
    assert metrics["seed/traj_gate/gen_pass_ratio"] == pytest.approx(0.5)
    assert metrics["seed/traj_gate/gen_gap_mean_pass"] == pytest.approx(MEAN_A)
    assert metrics["seed/traj_gate/gen_gap_mean_fail"] == pytest.approx(MEAN_B)
    assert metrics["seed/traj_gate/gen_rows_after"] == 2.0
