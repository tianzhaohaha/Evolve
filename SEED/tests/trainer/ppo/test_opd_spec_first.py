"""CPU regression tests through the real actor update, including its mask dtypes."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_opd_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import performance
from verl.workers.actor import dp_actor


@pytest.fixture
def run_update(monkeypatch):
    # Keep the actual loss/dispatch logic; replace only model inference and GPU diagnostics.
    monkeypatch.setattr(performance, "_get_current_mem_info", lambda: ("0",) * 4)
    monkeypatch.setattr(dp_actor, "get_torch_device", lambda: SimpleNamespace(current_device=lambda: "cpu"))

    def run(mask_dtype=torch.int64, dominance="spec_first", gate_eps=0.0, positive_only=False,
            spec_coef=0.5, gen_coef=0.3, token_spec_mask=False, has_spec=True, gen_active=True,
            extra_tensors=None, norm_mode="mask"):
        config = OmegaConf.create({
            "use_remove_padding": False, "use_torch_compile": False,
            "ulysses_sequence_parallel_size": 1, "use_dynamic_bsz": False,
            "ppo_mini_batch_size": 2, "ppo_micro_batch_size_per_gpu": 2, "ppo_epochs": 1,
            "clip_ratio": 0.2, "clip_ratio_low": None, "clip_ratio_high": None,
            "entropy_coeff": 0.0, "loss_agg_mode": "token-mean", "policy_loss": {"loss_mode": "vanilla"},
            "use_kl_loss": True, "kl_loss_coef": 0.01, "kl_loss_type": "low_var_kl",
            "opd_loss_coef": spec_coef, "opd_gen_loss_coef": gen_coef,
            "opd_gate_beta": 2.0, "opd_gen_gate_beta": 0.75,
            "opd_gate_eps": gate_eps, "opd_positive_only": positive_only, "opd_gen_dominance": dominance,
            "opd_norm_mode": norm_mode,
        })
        model = torch.nn.Module()
        model.register_parameter("log_probs", torch.nn.Parameter(torch.full((2, 4), -2.0)))
        actor = dp_actor.DataParallelPPOActor(config, model, torch.optim.SGD(model.parameters(), lr=0.1))
        monkeypatch.setattr(actor, "_maybe_start_memory_history", lambda: None)
        monkeypatch.setattr(actor, "_forward_micro_batch", lambda **kwargs: (None, model.log_probs))
        gradients, calls = [], []

        def capture_step():
            gradients.append(model.log_probs.grad.detach().clone())
            return model.log_probs.grad.norm()

        def capture_opd(**kwargs):
            calls.append(kwargs)
            return compute_opd_loss(**kwargs)

        monkeypatch.setattr(actor, "_optimizer_step", capture_step)
        monkeypatch.setattr(dp_actor, "compute_opd_loss", capture_opd)
        response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=mask_dtype)
        tensors = {
            "responses": torch.ones(2, 4, dtype=torch.long),
            "input_ids": torch.ones(2, 5, dtype=torch.long),
            "attention_mask": torch.cat([torch.ones(2, 1, dtype=mask_dtype), response_mask], dim=-1),
            "position_ids": torch.arange(5).expand(2, -1),
            "old_log_probs": torch.full((2, 4), -2.1),
            "advantages": torch.tensor([[1., -1., 0.5, 0.], [-0.5, 1., 0., 0.5]]),
            "ref_log_prob": torch.full((2, 4), -2.2),
            "gen_teacher_log_prob": torch.tensor([[-1.5, -1.25, -1.875, -2.5], [-2.5, -1.5, -1., -1.5]], requires_grad=True),
            "gen_skill_mask": torch.tensor([gen_active, gen_active]),
        }
        if has_spec:
            tensors["teacher_log_prob"] = torch.tensor([[-1., -2.5, -1.875, -2.], [-1.75, -1.5, -3., -2.]], requires_grad=True)
            tensors["teacher_signal_mask"] = (
                torch.tensor([[True, True, True, False], [True, False, True, True]])
                if token_spec_mask else torch.tensor([True, False])
            )
        tensors.update(extra_tensors or {})
        batch = DataProto.from_dict(tensors=tensors, meta_info={"temperature": 1.0})
        original_masks = {k: v.clone() for k, v in tensors.items() if "mask" in k}
        metrics = actor.update_policy(batch)
        for key, original in original_masks.items():
            torch.testing.assert_close(batch.batch[key], original, rtol=0, atol=0)
        assert tensors["gen_teacher_log_prob"].grad is None
        if has_spec:
            assert tensors["teacher_log_prob"].grad is None
        assert len(gradients) == 1
        return SimpleNamespace(config=config, batch=batch, mask=response_mask, metrics=metrics,
                               gradient=gradients[0], calls=calls)

    return run


def expected_update(result, route):
    """Reference: unchanged PPO/spec/KL plus gen weighted only where spec is effective."""
    config, data, mask = result.config, result.batch.batch, result.mask
    lp = torch.full((2, 4), -2.0, requires_grad=True)
    loss = compute_policy_loss(data["old_log_probs"], lp, data["advantages"], mask, cliprange=0.2)[0]
    kwargs = {"gate_eps": config.opd_gate_eps, "positive_only": config.opd_positive_only,
              "norm_mode": config.opd_norm_mode}
    spec_outputs = gen_outputs = tuple(torch.tensor(0.) for _ in range(5))
    if config.opd_loss_coef > 0 and "teacher_log_prob" in data:
        spec_outputs = compute_opd_loss(lp, data["teacher_log_prob"], mask, data["teacher_signal_mask"],
                                        gate_beta=config.opd_gate_beta, **kwargs)
        loss = loss + config.opd_loss_coef * spec_outputs[0]
    if config.opd_gen_loss_coef > 0:
        weights = data["gen_skill_mask"].to(lp.dtype)
        gen_mask = mask
        if route:
            gap = data["teacher_log_prob"].detach() - lp.detach()
            active = data["teacher_signal_mask"]
            if active.dim() == 1:
                active = active[:, None]
            active = active & (gap.abs() >= config.opd_gate_eps)
            if config.opd_positive_only:
                active = active & (gap > 0)
            weights = weights[:, None] * torch.where(active, 1 - torch.sigmoid(config.opd_gate_beta * gap), 1.)
            gen_mask = mask.float()
        gen_outputs = compute_opd_loss(lp, data["gen_teacher_log_prob"], gen_mask, weights,
                                       gate_beta=config.opd_gen_gate_beta, **kwargs)
        loss = loss + config.opd_gen_loss_coef * gen_outputs[0]
    kl = kl_penalty(lp, data["ref_log_prob"], config.kl_loss_type)
    loss = loss + config.kl_loss_coef * agg_loss(kl, mask, config.loss_agg_mode)
    loss.backward()
    return lp.grad, spec_outputs, gen_outputs


def assert_update_matches(result, route):
    grad, spec, gen = expected_update(result, route)
    torch.testing.assert_close(result.gradient, grad, rtol=0, atol=0)
    suffixes = ("loss", "active_token_ratio", "gate_mean", "gate_active_ratio", "teacher_gap_mean")
    for prefix, outputs in (("opd", spec), ("opd_gen", gen)):
        for suffix, expected in zip(suffixes, outputs):
            assert result.metrics[f"actor/{prefix}_{suffix}"][0] == expected.item()


@pytest.mark.parametrize("mask_dtype", [torch.int64, torch.bool, torch.float32])
@pytest.mark.parametrize("token_spec_mask", [False, True])
@pytest.mark.parametrize("gate_eps,positive_only", [(0., False), (0.25, False), (0., True), (0.25, True)])
def test_spec_first_preserves_fractional_weights_and_effective_spec_support(
    run_update, mask_dtype, token_spec_mask, gate_eps, positive_only
):
    result = run_update(mask_dtype=mask_dtype, token_spec_mask=token_spec_mask,
                        gate_eps=gate_eps, positive_only=positive_only)
    assert_update_matches(result, route=True)
    assert result.calls[0]["response_mask"].dtype == mask_dtype  # spec stays unchanged
    assert result.calls[1]["response_mask"].is_floating_point()
    weights = result.calls[1]["opd_step_mask"]
    assert torch.any((weights > 0) & (weights < 1))
    assert not weights.requires_grad


@pytest.mark.parametrize("mask_dtype", [torch.int64, torch.bool, torch.float32])
@pytest.mark.parametrize("options", [
    {"dominance": "none"},
    {"dominance": "none", "gate_eps": 0.25, "positive_only": True},
    {"spec_coef": 0.0},
    {"gen_coef": 0.0},
    {"has_spec": False},
    {"spec_coef": 0.0, "gen_coef": 0.0},
])
def test_nonrouting_paths_preserve_original_losses_gradients_and_dtypes(run_update, mask_dtype, options):
    result = run_update(mask_dtype=mask_dtype, **options)
    assert_update_matches(result, route=False)
    assert all(call["response_mask"].dtype == mask_dtype for call in result.calls)


def test_spec_first_empty_gen_mask_stays_zero(run_update):
    result = run_update(gen_active=False)
    assert_update_matches(result, route=True)
    assert result.metrics["actor/opd_gen_loss"] == [0.0]