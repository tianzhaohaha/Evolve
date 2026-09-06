from types import SimpleNamespace

import numpy as np
import pytest
import torch

from seed.replay import ReplayBuffer, has_policy_signal, merge_for_update, split_groups
from verl.protocol import DataProto


def _batch(uids, adv_scale=1.0, extra_key=False, adv_width=2):
    n = len(uids)
    tensors = {
        "input_ids": torch.arange(n * 4).view(n, 4),
        "attention_mask": torch.ones(n, 4, dtype=torch.long),
        "advantages": torch.full((n, adv_width), adv_scale),
    }
    if extra_key:
        tensors["extra"] = torch.zeros(n, 1)
    non_tensors = {
        "uid": np.array(uids, dtype=object),
        "_batch_source_idx": np.arange(n, dtype=np.int64),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={"global_token_num": [4] * n})


def test_split_groups_preserves_order_and_strips_transient_keys():
    batch = _batch(["b", "a", "b", "c", "a"])
    groups = split_groups(batch)
    assert [g.non_tensor_batch["uid"][0] for g in groups] == ["b", "a", "c"]
    assert [len(g) for g in groups] == [2, 2, 1]
    assert all("_batch_source_idx" not in g.non_tensor_batch and g.meta_info == {} for g in groups)
    torch.testing.assert_close(groups[0].batch["input_ids"], batch.batch["input_ids"][[0, 2]])


def test_zero_advantage_groups_are_not_stored():
    buffer = ReplayBuffer(capacity=4, groups_per_step=1, seed=0)
    buffer.add_batch(_batch(["a", "a"], adv_scale=0.0))
    assert len(buffer) == 0 and not has_policy_signal(split_groups(_batch(["z"], adv_scale=0.0))[0])
    buffer.add_batch(_batch(["b", "b"]))
    assert len(buffer) == 1


def test_reservoir_is_bounded_and_sampling_precedes_insertion():
    buffer = ReplayBuffer(capacity=2, groups_per_step=2, seed=123)
    assert buffer.sample() == []
    for step in range(5):
        buffer.add_batch(_batch([f"g{step}", f"g{step}"]))
    assert len(buffer) == 2
    assert buffer.metrics() == {"replay/buffer_groups": 2.0, "replay/seen_groups": 5.0}
    sampled = buffer.sample()
    assert len(sampled) == 2 and len({s.non_tensor_batch["uid"][0] for s in sampled}) == 2


def test_reservoir_keeps_every_group_with_equal_probability():
    keep = np.zeros(6)
    for seed in range(300):
        buffer = ReplayBuffer(capacity=2, groups_per_step=1, seed=seed)
        for i in range(6):
            buffer.add_batch(_batch([f"g{i}"]))
        for group in buffer._groups:
            keep[int(group.non_tensor_batch["uid"][0][1:])] += 1
    assert keep.min() > 300 * 2 / 6 * 0.6 and keep.max() < 300 * 2 / 6 * 1.4


def test_merge_concatenates_without_mutating_live_batch():
    live = _batch(["x", "x", "y"])
    merged = merge_for_update(live, split_groups(_batch(["r", "r"])))
    assert len(merged) == 5
    assert "_batch_source_idx" not in merged.non_tensor_batch
    assert "_batch_source_idx" in live.non_tensor_batch and len(live) == 3
    merged.meta_info["global_token_num"] = [1]
    assert live.meta_info["global_token_num"] == [4, 4, 4]
    assert list(merged.non_tensor_batch["uid"]) == ["x", "x", "y", "r", "r"]


def test_merge_refuses_mismatched_keys_or_shapes():
    live = _batch(["x"])
    assert merge_for_update(live, split_groups(_batch(["r"], extra_key=True))) is None
    assert merge_for_update(live, split_groups(_batch(["r"], adv_width=3))) is None


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=1, groups_per_step=2)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=1, groups_per_step=0)


def test_mix_replay_pads_permutes_and_keeps_live_batch_intact():
    from omegaconf import OmegaConf

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    config = OmegaConf.create(
        {
            "trainer": {"n_gpus_per_node": 2, "nnodes": 1, "balance_batch": True},
            "algorithm": {"use_kl_in_reward": False},
            "actor_rollout_ref": {
                "rollout": {"log_prob_micro_batch_size_per_gpu": 1},
                "ref": {"log_prob_micro_batch_size_per_gpu": 1},
                "actor": {"use_kl_loss": False, "ppo_micro_batch_size_per_gpu": 1, "ppo_mini_batch_size": 2},
            },
        }
    )

    class Stub:
        _balance_batch = RayPPOTrainer._balance_batch
        _mix_replay = RayPPOTrainer._mix_replay

    stub = Stub()
    stub.config = config
    stub.actor_rollout_wg = SimpleNamespace(world_size=2)
    stub._seed_replay = ReplayBuffer(capacity=4, groups_per_step=1, seed=0)

    first, metrics = _batch(["a", "a", "b"]), {}
    assert stub._mix_replay(first, metrics) is first  # empty buffer: nothing to mix
    assert metrics["replay/buffer_groups"] == 2.0 and metrics["replay/sampled_groups"] == 0.0

    second, metrics = _batch(["c", "c", "c"]), {}
    merged = stub._mix_replay(second, metrics)
    sampled = int(metrics["replay/sampled_samples"])
    assert len(second) == 3 and "_batch_source_idx" in second.non_tensor_batch
    assert sampled >= 1 and len(merged) % 2 == 0 and len(merged) >= 3 + sampled
    assert set(merged.batch.keys()) == set(second.batch.keys())
    assert merged.meta_info["global_token_num"] == [4] * len(merged)
    assert second.meta_info["global_token_num"] == [4, 4, 4]
    assert 0.0 < metrics["replay/frac_of_batch"] < 1.0
    uids = set(merged.non_tensor_batch["uid"])
    assert "c" in uids and (uids & {"a", "b"})
