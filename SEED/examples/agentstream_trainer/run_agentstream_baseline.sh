#!/usr/bin/env bash

# Run one SEED-paper baseline on the AgentStream stream from the shared Stage-1 SFT checkpoint.
#
# Usage:
#   bash examples/agentstream_trainer/run_agentstream_baseline.sh <baseline> <mode> [hydra overrides...]
#     baseline  vanilla | grpo | grpo_base | seed | sdar | opsd | rlsd
#     mode      random | isolated | sequential | interleaved
#
# All baselines share one recipe (agentstream_full.env): same SFT init, stream, seed, group
# size, lr, KL, step budget and evaluation. The extensions of this repo (gen OPD channel,
# global pool, EMA, replay, positive-only, failed-skill-positive) are forced off. Only the
# objective differs:
#   vanilla    frozen policy: no analysis, no actor update (critic_warmup), one validation
#              before the stream (test_freq=0) -> online curve of the SFT model along the stream
#   grpo       outcome advantage only, no analysis
#   grpo_base  same objective, but started from the raw backbone instead of the SFT checkpoint
#              (the SEED paper's GRPO row; isolates what the hindsight-skill SFT contributes)
#   seed       GRPO + gated OPD, self-evolving analyzer (policy_vllm)     paper: lambda_opd=0.01, beta=5
#   sdar       GRPO + gated OPD, fixed external analyzer (openai backend) approximates SDAR's static skill source
#   opsd       teacher gap only (outcome_advantage_w=0); env reward is kept so success labels stay correct
#   rlsd       GRPO advantage * clip(exp(sign(A) * gap), 1 -/+ eps)     multiplicative teacher reweighting
#
# Every switch is a hydra override appended after the launcher's own arguments (last one wins),
# so nothing pinned in .env or agentstream_full.env can silently change a baseline. Extra
# overrides given on the command line come last and may refine any of them. The summary the
# child launcher prints still shows the full.env defaults; the hydra command line is authoritative.

set -euo pipefail

baseline="${1:?usage: run_agentstream_baseline.sh <baseline> <mode> [hydra overrides...]}"
mode="${2:?usage: run_agentstream_baseline.sh <baseline> <mode> [hydra overrides...]}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

overrides=(
    actor_rollout_ref.actor.opd_loss_coef=0
    actor_rollout_ref.actor.opd_gen_loss_coef=0
    actor_rollout_ref.actor.opd_gate_eps=0
    actor_rollout_ref.actor.opd_positive_only=False
    actor_rollout_ref.actor.ema_mode=off
    algorithm.seed.global_pool.source=copy
    algorithm.seed.global_pool.admit_failed=False
    algorithm.seed.failed_skill_positive=False
    algorithm.seed.replay.enable=False
    algorithm.seed.local_teacher_source=skill
    algorithm.seed.route_mode=none
    algorithm.seed.success_only=False
    actor_rollout_ref.actor.opd_norm_mode=mask
    algorithm.seed.traj_gap_gate.enable=False
)
# "<baseline>_base" = the same objective started from the raw backbone instead of the SFT
# checkpoint (the paper's setting for every non-SEED row); the KL reference follows the init.
method="${baseline%_base}"
[[ "$method" != "$baseline" ]] && export AGENTSTREAM_RL_INIT_FROM_BASE=true
case "$method" in
    vanilla)   overrides+=(actor_rollout_ref.actor.optim.lr=0 algorithm.seed.enable_analysis=False
                           trainer.critic_warmup=1000000 trainer.test_freq=0 trainer.val_before_train=True) ;;
    grpo)      overrides+=(algorithm.seed.enable_analysis=False) ;;
    seed)      overrides+=(actor_rollout_ref.actor.opd_loss_coef=0.01) ;;
    sdar)      overrides+=(actor_rollout_ref.actor.opd_loss_coef=0.01 algorithm.seed.analysis_backend=openai) ;;
    opsd)      overrides+=(algorithm.seed.episode_skill_teacher_advantage_w=1.0 algorithm.seed.outcome_advantage_w=0) ;;
    rlsd)      overrides+=(algorithm.seed.episode_skill_teacher_advantage_w=1.0
                           algorithm.seed.teacher_adv_mode=multiplicative
                           algorithm.seed.teacher_adv_mult_eps=0.2) ;;
    *)
        echo "Unknown baseline '$baseline'. Use vanilla, grpo, seed, sdar, opsd, or rlsd, optionally with a _base suffix." >&2
        exit 2
        ;;
esac

export AGENTSTREAM_METHOD_TAG="$baseline"   # experiment name prefix (agentstream_full.env)
echo "AgentStream baseline '$baseline': ${overrides[*]} $*"
exec bash "$SCRIPT_DIR/run_agentstream_sft_glm_self.sh" "$mode" "${overrides[@]}" "$@"
