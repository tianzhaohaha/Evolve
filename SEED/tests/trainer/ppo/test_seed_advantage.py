import numpy as np
import torch

from gigpo.core_gigpo import compute_seed_advantage_components


def test_seed_uses_independent_episode_and_step_skill_teacher_weights():
    response_mask = torch.ones((2, 2))
    old_log_prob = torch.zeros((2, 2))

    _, _, teacher_advantages, scores, metrics = compute_seed_advantage_components(
        token_level_rewards=torch.zeros((2, 2)),
        step_rewards=torch.zeros((2, 2)),
        response_mask=response_mask,
        anchor_obs=np.asarray(["obs-a", "obs-b"], dtype=object),
        index=np.asarray(["task", "task"], dtype=object),
        traj_index=np.asarray(["traj-a", "traj-b"], dtype=object),
        episode_teacher_log_prob=torch.ones((2, 2)),
        step_teacher_log_prob=torch.full((2, 2), 2.0),
        old_log_prob=old_log_prob,
        critical_step_mask=torch.tensor([1.0, 0.0]),
        step_skill_mask=torch.tensor([0.0, 1.0]),
        episode_skill_teacher_advantage_w=0.25,
        step_skill_teacher_advantage_w=0.5,
    )

    expected = torch.tensor([[0.25, 0.25], [1.0, 1.0]])
    torch.testing.assert_close(teacher_advantages, expected)
    torch.testing.assert_close(scores, expected)
    assert metrics["seed/adv/episode_skill_teacher_weight"] == 0.25
    assert metrics["seed/adv/step_skill_teacher_weight"] == 0.5


def test_seed_adds_episode_and_step_skill_teacher_advantages_on_same_step():
    response_mask = torch.ones((1, 2))
    old_log_prob = torch.zeros((1, 2))

    _, _, teacher_advantages, scores, _ = compute_seed_advantage_components(
        token_level_rewards=torch.zeros((1, 2)),
        step_rewards=torch.zeros((1, 2)),
        response_mask=response_mask,
        anchor_obs=np.asarray(["obs"], dtype=object),
        index=np.asarray(["task"], dtype=object),
        traj_index=np.asarray(["traj"], dtype=object),
        episode_teacher_log_prob=torch.ones((1, 2)),
        step_teacher_log_prob=torch.full((1, 2), 2.0),
        old_log_prob=old_log_prob,
        critical_step_mask=torch.tensor([1.0]),
        step_skill_mask=torch.tensor([1.0]),
        episode_skill_teacher_advantage_w=0.001,
        step_skill_teacher_advantage_w=0.001,
    )

    expected = torch.full((1, 2), 0.003)
    torch.testing.assert_close(teacher_advantages, expected)
    torch.testing.assert_close(scores, expected)


def _grouped_inputs():
    # One task, two trajectories: rewards 1 and 0 -> episode advantages +/-1 (mean_std_norm).
    token_level_rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    return dict(
        token_level_rewards=token_level_rewards,
        step_rewards=torch.zeros((2, 2)),
        response_mask=torch.ones((2, 2)),
        anchor_obs=np.asarray(["obs-a", "obs-b"], dtype=object),
        index=np.asarray(["task", "task"], dtype=object),
        traj_index=np.asarray(["traj-a", "traj-b"], dtype=object),
        old_log_prob=torch.zeros((2, 2)),
        critical_step_mask=torch.tensor([1.0, 1.0]),
        mode="mean_std_norm",
    )


def test_seed_outcome_weight_zero_leaves_pure_teacher_advantage():
    episode_adv, _, teacher_adv, scores, metrics = compute_seed_advantage_components(
        episode_teacher_log_prob=torch.tensor([[0.5, -0.5], [0.2, 0.2]]),
        episode_skill_teacher_advantage_w=1.0,
        outcome_advantage_w=0.0,
        **_grouped_inputs(),
    )
    assert episode_adv.abs().sum() == 0  # reported episode term carries the (zero) outcome weight
    torch.testing.assert_close(scores, teacher_adv)
    torch.testing.assert_close(scores, torch.tensor([[0.5, -0.5], [0.2, 0.2]]))
    assert metrics["seed/adv/outcome_advantage_weight"] == 0.0


def test_seed_multiplicative_teacher_reweights_outcome_without_flipping_sign():
    inputs = _grouped_inputs()
    # Large gaps saturate the clip; the masked-out third token (response_mask=0) carries no advantage.
    inputs["response_mask"] = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    inputs["token_level_rewards"] = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    inputs["step_rewards"] = torch.zeros((2, 3))
    inputs["old_log_prob"] = torch.zeros((2, 3))
    gap = torch.tensor([[5.0, -5.0, 9.0], [5.0, -5.0, 9.0]])
    episode_adv, _, teacher_adv, scores, metrics = compute_seed_advantage_components(
        episode_teacher_log_prob=gap,
        episode_skill_teacher_advantage_w=1.0,
        teacher_adv_mode="multiplicative",
        teacher_adv_mult_eps=0.2,
        **inputs,
    )
    a = episode_adv[:, :1]  # per-trajectory outcome advantage, sign +/-
    assert a[0, 0] > 0 and a[1, 0] < 0
    # A>0: teacher-preferred token amplified (x1.2), disfavored damped (x0.8); A<0 mirrored.
    expected = torch.stack([a[0, 0] * torch.tensor([1.2, 0.8, 0.0]), a[1, 0] * torch.tensor([0.8, 1.2, 0.0])])
    torch.testing.assert_close(scores, expected)
    assert torch.all(torch.sign(scores[:, :2]) == torch.sign(episode_adv[:, :2]))
    torch.testing.assert_close(teacher_adv, scores - episode_adv)
    assert metrics["seed/adv/teacher_adv_multiplicative"] == 1.0


def test_seed_defaults_are_unchanged_additive_path():
    inputs = _grouped_inputs()
    gap = torch.tensor([[0.3, 0.3], [-0.1, -0.1]])
    episode_adv, _, teacher_adv, scores, _ = compute_seed_advantage_components(
        episode_teacher_log_prob=gap, episode_skill_teacher_advantage_w=1.0, **inputs
    )
    torch.testing.assert_close(scores, episode_adv + gap)
    torch.testing.assert_close(teacher_adv, gap)


def test_seed_outcome_weight_scales_reported_episode_term():
    episode_adv, _, _, scores, _ = compute_seed_advantage_components(
        episode_teacher_log_prob=torch.zeros((2, 2)),
        episode_skill_teacher_advantage_w=1.0,
        outcome_advantage_w=0.5,
        **_grouped_inputs(),
    )
    torch.testing.assert_close(scores, episode_adv)  # no teacher gap -> scores are the weighted episode term
    assert episode_adv.abs().max() < 1.0  # mean_std_norm gives |adv|~1 before the 0.5 weight
