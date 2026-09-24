#!/usr/bin/env bash
# Switch blocks of the SEED extensions (SEED/README.md "可选改进开关" [6]-[10]), shared by
# run_ours_debug.sh (experiment arms) and run_ours_method.sh (the method). Sourced, never executed: each
# block is a list of AGENTSTREAM_* assignments for `env` / `export`; later assignments win, so an arm may
# override any earlier block. Every value here is an env-file switch of agentstream_full.env.

# The seed arm of run_agentstream_baseline.sh: single-track OPD 0.01, every extension forced off (the
# env-file defaults are NOT the off set, so a method must pin all of them).
SEED_BASE_ENV=(
    "AGENTSTREAM_SEED_SKILL_MODE=${AGENTSTREAM_SEED_SKILL_MODE:-episode_only}"
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.01
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0
    AGENTSTREAM_SEED_OPD_GEN_DOMINANCE=none
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=copy
    AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED=False
    AGENTSTREAM_SEED_GLOBAL_POOL_EVICT_POLICY=gate_ema
    AGENTSTREAM_SEED_GLOBAL_POOL_WINDOW_STEPS=48
    AGENTSTREAM_SEED_GLOBAL_POOL_ADMISSION=gap
    AGENTSTREAM_SEED_GLOBAL_POOL_JUDGE_BACKEND=openai
    AGENTSTREAM_SEED_GLOBAL_POOL_REWRITE=none
    AGENTSTREAM_SEED_FAILED_SKILL_POSITIVE=False
    AGENTSTREAM_SEED_OPD_GATE_EPS=0
    AGENTSTREAM_SEED_OPD_POSITIVE_ONLY=False
    AGENTSTREAM_SEED_EMA_MODE=off
    AGENTSTREAM_SEED_EMA_TAU=0.9
    AGENTSTREAM_SEED_REPLAY_ENABLE=False
    AGENTSTREAM_SEED_REPLAY_CAPACITY=32
    AGENTSTREAM_SEED_LOCAL_TEACHER_SOURCE=skill
    AGENTSTREAM_SEED_ROUTE_MODE=none
    AGENTSTREAM_SEED_SUCCESS_ONLY=False
    AGENTSTREAM_SEED_OPD_NORM_MODE=mask
    AGENTSTREAM_SEED_TRAJ_GAP_GATE=False
    AGENTSTREAM_SEED_TRAJ_GAP_GATE_MARGIN=0.0
    AGENTSTREAM_SEED_SIBLING_RESAMPLE=False
    AGENTSTREAM_SEED_SIBLING_RESAMPLE_MAX_GROUPS=4
    AGENTSTREAM_SEED_SIBLING_RESAMPLE_BASELINE=own
    AGENTSTREAM_SEED_SIBLING_RESAMPLE_POOL_MAX_GROUPS=0
)
# A4 base [6][7]: distil successes only with the PG denominator; the skill losses stay active.
A4_ENV=(
    AGENTSTREAM_SEED_SUCCESS_ONLY=True
    AGENTSTREAM_SEED_OPD_NORM_MODE=response
)
# A3 floor [6][7][8]: A4 plus the trajectory gate (= GRPO when the gate shuts OPD off; measured pass ratio 0-3%).
A3_ENV=("${A4_ENV[@]}" AGENTSTREAM_SEED_TRAJ_GAP_GATE=True)
# [9] sibling resample with the source-group baseline.
RESAMPLE_ENV=(
    AGENTSTREAM_SEED_SIBLING_RESAMPLE=True
    AGENTSTREAM_SEED_SIBLING_RESAMPLE_MAX_GROUPS=4
    AGENTSTREAM_SEED_SIBLING_RESAMPLE_BASELINE=source
)
# [10] pool filled by the policy's own judge (success admission, no API key), 10-step window eviction.
POOL_ENV=(
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=pool
    AGENTSTREAM_SEED_GLOBAL_POOL_ADMISSION=success
    AGENTSTREAM_SEED_GLOBAL_POOL_JUDGE_BACKEND=policy_vllm
    AGENTSTREAM_SEED_GLOBAL_POOL_EVICT_POLICY=window
    AGENTSTREAM_SEED_GLOBAL_POOL_WINDOW_STEPS=10
)
# Skill form stored in the pool, "<rewrite mode>:<tag>": the arm dimension of E4-E6 / E2 / E8.
SKILL_FORMS=(none:raw deinstantiate:deinst aggregate:agg)

# The method = arm E5 of run_ours_debug.sh: A3 floor + sibling resample (source baseline, 4 groups) +
# pool resample (3 all-fail groups per step) with de-instantiated skills. Job 71186: +4.8 over its anchor.
OURS_METHOD_ENV=(
    "${SEED_BASE_ENV[@]}" "${A3_ENV[@]}" "${RESAMPLE_ENV[@]}" "${POOL_ENV[@]}"
    AGENTSTREAM_SEED_SIBLING_RESAMPLE_POOL_MAX_GROUPS=3
    AGENTSTREAM_SEED_GLOBAL_POOL_REWRITE=deinstantiate
)
