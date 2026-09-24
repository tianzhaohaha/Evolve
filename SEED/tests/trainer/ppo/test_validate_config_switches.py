"""_validate_config on the yaml defaults plus the global-skill / resample switch combinations."""

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

BASE = [
    "algorithm.adv_estimator=seed",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.rollout.name=vllm",
    "env.rollout.n=6",
    "actor_rollout_ref.rollout.max_model_len=32768",  # policy_vllm analysis needs context + completion to fit
]
POOL_SYNC = [
    "algorithm.seed.global_pool.source=pool",
    "algorithm.seed.global_pool.admission=success",
    "algorithm.seed.global_pool.judge_backend=policy_vllm",
    "algorithm.seed.analysis_backend=policy_vllm",
]
RESAMPLE = ["algorithm.seed.sibling_resample.enable=True", "algorithm.seed.sibling_resample.max_groups=4"]


def validate(*overrides):
    config = OmegaConf.merge(OmegaConf.load("verl/trainer/config/ppo_trainer.yaml"), OmegaConf.from_dotlist([*BASE, *overrides]))
    OmegaConf.set_struct(config, False)
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config, trainer.use_critic, trainer.use_reference_policy, trainer.use_rm = config, False, False, False
    trainer._validate_config()


@pytest.mark.parametrize("overrides", [
    [],
    RESAMPLE,
    RESAMPLE + ["algorithm.seed.sibling_resample.baseline=source"],
    ["algorithm.seed.global_pool.source=pool", "actor_rollout_ref.actor.opd_gen_loss_coef=0.005"],  # original consumer
    POOL_SYNC + ["actor_rollout_ref.actor.opd_gen_loss_coef=0.005", "algorithm.seed.global_pool.rewrite=aggregate"],
    POOL_SYNC + RESAMPLE + ["algorithm.seed.sibling_resample.pool_max_groups=3"],  # resample channel only, gen off
    POOL_SYNC + RESAMPLE + ["algorithm.seed.sibling_resample.pool_max_groups=3", "algorithm.seed.global_pool.rewrite=deinstantiate"],
])
def test_accepted_switch_combinations(overrides):
    validate(*overrides)


@pytest.mark.parametrize("overrides,message", [
    (["algorithm.seed.global_pool.source=pool"], "no consumer"),
    (["algorithm.seed.sibling_resample.pool_max_groups=1"], "sibling_resample.enable=True"),
    (RESAMPLE + ["algorithm.seed.sibling_resample.pool_max_groups=1"], "global_pool.source=pool"),
    (POOL_SYNC + RESAMPLE + ["algorithm.seed.sibling_resample.pool_max_groups=1", "algorithm.seed.enable_analysis=False"], "enable_analysis=True"),
    (RESAMPLE + ["algorithm.seed.sibling_resample.pool_max_groups=-1"], "pool_max_groups must be >= 0"),
    (RESAMPLE + ["algorithm.seed.sibling_resample.baseline=group"], "baseline must be 'own' or 'source'"),
    (["algorithm.seed.global_pool.source=pool", "actor_rollout_ref.actor.opd_gen_loss_coef=0.005", "algorithm.seed.global_pool.judge_backend=policy_vllm", "algorithm.seed.global_pool.admission=success"], "analysis_backend=policy_vllm"),
    (["algorithm.seed.global_pool.source=pool", "actor_rollout_ref.actor.opd_gen_loss_coef=0.005", "algorithm.seed.global_pool.judge_backend=policy_vllm", "algorithm.seed.analysis_backend=policy_vllm"], "admission=success"),
    (["algorithm.seed.global_pool.source=pool", "actor_rollout_ref.actor.opd_gen_loss_coef=0.005", "algorithm.seed.global_pool.admission=best"], "admission must be one of"),
    (["algorithm.seed.global_pool.source=pool", "actor_rollout_ref.actor.opd_gen_loss_coef=0.005", "algorithm.seed.global_pool.judge_backend=local"], "judge_backend must be one of"),
    (["algorithm.seed.global_pool.source=pool", "actor_rollout_ref.actor.opd_gen_loss_coef=0.005", "algorithm.seed.global_pool.rewrite=merge"], "rewrite must be one of"),
])
def test_rejected_switch_combinations(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate(*overrides)
