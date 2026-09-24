#!/usr/bin/env bash
# Global-skill experiments in two families (SEED/README.md "可选改进开关" [6]-[10], seed/resample.py,
# seed/skill_rewrite.py, seed/global_pool.py):
#   Sampling family, on the A3 floor (success_only + response norm + trajectory gate: OPD is off unless
#   the teacher context makes the trajectory more likely, so the floor is GRPO -- job 70458 measured a
#   0.7% gate pass ratio and opd_loss 0.0000):
#     E1  A3 anchor                          reference inside the job (cross-job noise is 5-9 pts).
#     E3  + sibling resample, source baseline   [9]: mixed groups re-run with their shortest success as the
#                                            reference; the new rows are normalised with the main-pass
#                                            statistics of their source group (all-success second passes
#                                            still carry signal; 38% of them did not under own groups).
#     E4-E6  E3 + pool resample              [10] channel 2: all-fail groups retrieve a cross-task skill
#            (raw / de-instantiated / aggregated)   from the pool and are re-run with it; the pool is filled by
#                                            the policy's own judge (no API key), window eviction.
#     E2a-E2c  A3 + gen OPD from the pool    [10] channel 1 under the gate: the pool skill is the gen OPD
#            (raw / de-instantiated / aggregated)   teacher context (coef 0.005) only on trajectories it makes
#                                            more likely -- harmless by construction, and the gate's
#                                            seed/traj_gate/gen_pass_ratio says whether a pool skill ever is.
#   Loss family, on the A4 base (success_only + response norm, NO gate -- both skill losses stay alive:
#   job 70458 measured opd mask 10.5% of response tokens, opd_loss 0.002, teacher gap -0.008):
#     E7  A4 anchor                          local track only (self skill as teacher, coef 0.01).
#     E8a-E8c  A4 + gen OPD from the pool    [10] channel 1: the retrieved pool skill is the teacher context
#            (raw / de-instantiated / aggregated)   of the gen OPD loss (coef 0.005) on the same rows.
# Why two bases: the A3 gate also covers the gen channel, and a pool skill is further from a trajectory
# than that trajectory's own skill, so on A3 the gen loss is expected to be gated to ~0 (E2 ~= E1; the
# arm then measures the pass ratio, not a training effect). A4 keeps both losses active whatever the
# gap; its own cost is a weak success self-cloning term that landed within noise of A3.
# Why these mechanisms: the offline delta tests showed only the sampling-time use of experience carries
# information (same-task demonstration 0.40 -> 0.73; re-scoring old samples: -0.19 nats/token on successes
# and failures alike; a neighbour's raw trajectory transfers negatively, an abstract skill about zero).
# Read the paired per-step differences inside one job only: E3 - E1, E4/E5/E6 - E3, E2x - E1 (A3 family);
# E8x - E7 (loss family); node3 carries both anchors, so E7 - E1 is the A4 - A3 gap inside one job. The loss family's main readout is mechanism-level even when the score does not
# move: actor/opd_gen_teacher_gap_mean vs actor/opd_teacher_gap_mean (does a pool skill make the successful
# trajectory more likely than its own skill does), actor/opd_gen_gate_mean, seed/global_pool/retrieval_hit_ratio.
#
# Layout (as run_global_ablation.sh): COMMON_ENV = the stream settings shared with
# run_baseline_suite.sh, SEED_BASE_ENV pins every extension off (= `run_agentstream_baseline.sh seed`),
# A4_ENV / A3_ENV / RESAMPLE_ENV / POOL_ENV are the reusable blocks, SKILL_FORMS the skill-form dimension,
# and every run_exp call spells out only what it adds. Hydra overrides go after `--`. Activate Conda in the caller.
#
# Usage: bash examples/agentstream_trainer/run_ours_debug.sh [--dry-run]
#   --dry-run prints each arm's resolved setup without training.
#   DEBUG_ARMS=E1,E3 bash ...           # run a subset of arm ids (default: every arm below)
#   TOTAL_STEPS=2 bash ...              # smoke run
# Three nodes: run_ours_debug_node{1,2,3}.sh set DEBUG_ARMS so that every arm has its anchor on its node
# (node1 E1,E3,E4,E2a; node2 E1,E5,E6,E2b; node3 E7,E8a,E8b,E8c,E1,E2c); pair only inside a node.
#
# Steps and continuation: TOTAL_STEPS (default 10) caps the stream; the last step is checkpointed
# (SAVE_FREQ = TOTAL_STEPS, one checkpoint kept) and the experiment name carries no step count, so a
# promising arm can be continued later in the same directory:
#   DEBUG_RUN_ID=<original PBS_JOBID> DEBUG_ARMS=E4 TOTAL_STEPS=20 bash examples/agentstream_trainer/run_ours_debug.sh
# resume_mode=auto restores actor / optimizer / lr scheduler / dataloader / stream_state.json, the
# online recorder (agentstream_online_metrics.jsonl) and the pool (global_skill_pool.json) from
# global_step_10/ and streams tasks 101-200, saving again at step 20. Keep tag, STREAM_MODE, GPU count,
# learning rate and switches identical. To continue the same wandb curve add
# WANDB_RUN_ID=<run id> WANDB_RESUME=allow. The step-10 actor/ (~50 GB) is not removed automatically.
# The holdout is validated once at the last step, so it only matches the baselines' (step 20) after a
# continuation; the online curves (online/*) match the baselines step by step from the start.
#
# =============================================================================
# 全流程命令（在 SEED 根目录执行；PBS 作业只需在激活 Conda 后调用对应的一行）
# =============================================================================
# 共同前提：.env 里有 OPENAI_* / OPENROUTER_API_KEY / HF_TOKEN，公共配置在
# examples/agentstream_trainer/agentstream_full.env（bfcl/tau2/browsecompplus 三域、每域 64 题、
# holdout 32、batch 10、group 6、history 3、prompt 38912、response 512、episode_only）。
#
# Stage 1+2：SFT 数据 + SFT 模型（PBS: select=1:ncpus=48:ngpus=4）
#   bash examples/agentstream_trainer/run_stage12.sh --dry-run all      # 只打印解析结果
#   bash examples/agentstream_trainer/run_stage12.sh prepare            # Stage 1：rollout -> GLM 标注 -> parquet
#   bash examples/agentstream_trainer/run_stage12.sh sft                # Stage 2：SFT 3 epoch -> 导出 HF 模型
#
# Stage 3 基线套件（PBS: select=1:ncpus=48:ngpus=2，walltime 168h）
#   bash examples/agentstream_trainer/run_baseline_suite.sh --dry-run    # 预览命令
#   bash examples/agentstream_trainer/run_baseline_suite.sh
#
# Stage 3 实验 E：三节点并行（10 步，保留最后一个 checkpoint；采样族在 A3 上、loss 族在 A4 上）
#   bash examples/agentstream_trainer/run_ours_debug_node1.sh --dry-run      # E1 E3 E4 E2a
#   bash examples/agentstream_trainer/run_ours_debug_node1.sh
#   bash examples/agentstream_trainer/run_ours_debug_node2.sh                # E1 E5 E6 E2b
#   bash examples/agentstream_trainer/run_ours_debug_node3.sh                # E7 E8a E8b E8c E1 E2c
#   TOTAL_STEPS=2 DEBUG_ARMS=E4,E8a bash examples/agentstream_trainer/run_ours_debug.sh   # 冒烟
#   DEBUG_RUN_ID=<job id> DEBUG_ARMS=E4 TOTAL_STEPS=20 bash examples/agentstream_trainer/run_ours_debug.sh   # 续跑到 20 步
#
# 读数（同一节点内逐步配对，总体与分域；不要与其他作业的曲线比）
#   采样族：E3 − E1 = 来源组基线下的重采样增益（对照 B3 − B1 = +0.032±0.016）；E4/E5/E6 − E3 = 池救援价值
#           seed/resample/{adv_mean, adv_pos_frac, uniform_group_ratio, pool_groups_allfail, pool_hit_rate, pool_rescue_rate}
#   门控 gen：E2x − E1（预期 ≈ 0）；机制读数 seed/traj_gate/gen_{pass_ratio, gap_mean_fail} = 池 skill 在下限判据下有无信息
#   loss 族：E8x − E7 = 池 skill 作 gen teacher 的增益；机制读数 actor/opd_gen_teacher_gap_mean（对比 actor/opd_teacher_gap_mean）、
#           actor/opd_gen_gate_mean、seed/global_pool/{retrieval_hit_ratio, size, admission_added, judge_parse_failed, rewrite_chars_mean}
#   任一臂里 pool_hit_rate / opd_gen_loss 恒为 0 = 该臂在测空，先看冒烟
#   环境：prompt_length/max 某步低于 8000 = 检索失效（examples/agentstream_trainer/check_browsecomp_health.py）
#
# 结果
#   wandb: online/*（在线累计分，主指标）、val/<slug>_score|success_rate（holdout）、actor/opd_*、seed/*、timing_s/*
#   磁盘: ../ckpt/debug_ours_<job>_<tag>_<mode>_online_s44/（CHECKPOINTS_ROOT，在仓库根 Evolve/ 下）
#         global_step_<TOTAL_STEPS>/、agentstream_online_metrics.jsonl、global_skill_pool.json、<step>.jsonl
#   python examples/agentstream_trainer/analyze_results.py ../ckpt/*/agentstream_online_metrics.jsonl --csv all.csv

set -eo pipefail

[[ "${1:-}" == --dry-run ]] && { export DRY_RUN=true; shift; }
(( $# == 0 )) || { echo "Usage: $0 [--dry-run]   (DEBUG_ARMS / DEBUG_RUN_ID / TOTAL_STEPS via the environment)" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Machine paths/credentials once, before the overrides below (no xtrace: .env holds keys).
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
export ENV_FILE=/dev/null PYTHONUNBUFFERED=1

# ===== Stream settings: identical to run_baseline_suite.sh (benchmarks, single pass, 10 tasks per
# step; NUM_TASKS / VAL_TASKS derive from agentstream_full.env). Differences: TOTAL_STEPS caps the
# stream and the last step is checkpointed (one kept) so an arm can be continued. TEST_FREQ follows
# the step count, so the holdout is validated once at step TOTAL_STEPS. =====
STREAM_MODE="${STREAM_MODE:-interleaved}"
TOTAL_STEPS="${TOTAL_STEPS:-10}"
COMMON_ENV=(
    AGENTSTREAM_BENCHMARKS=bfcl,tau2,browsecompplus
    AGENTSTREAM_RL_STREAM_PROFILE=single_pass
    AGENTSTREAM_RL_TRAIN_DATA_SIZE=10                                     # tasks per step, as the suite
    "AGENTSTREAM_RL_EPOCHS=$TOTAL_STEPS"                                  # 20 = full single pass
    "AGENTSTREAM_RL_SAVE_FREQ=$TOTAL_STEPS"                               # checkpoint the last step only
    AGENTSTREAM_RL_MAX_CKPT_TO_KEEP=1
    "AGENTSTREAM_SEED_SKILL_MODE=${AGENTSTREAM_SEED_SKILL_MODE:-episode_only}"
    AGENTSTREAM_SEED_OPD_GEN_DOMINANCE=none                               # inert while gen is off
    AGENTSTREAM_SEED_EMA_TAU=0.9                                          # inert while EMA is off
    AGENTSTREAM_SEED_REPLAY_CAPACITY=32                                   # inert while replay is off
)
# ===== SEED baseline switches shared by every arm: the full forced-off set of the seed arm of
# run_agentstream_baseline.sh (single-track OPD 0.01, policy_vllm analyzer; every extension off). =====
SEED_BASE_ENV=(
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.01
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0
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
    AGENTSTREAM_SEED_REPLAY_ENABLE=False
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
# ===== Reusable blocks =====
# A4 base [6][7]: distil successes only with the PG denominator; both skill losses stay active.
A4_ENV=(
    AGENTSTREAM_SEED_SUCCESS_ONLY=True
    AGENTSTREAM_SEED_OPD_NORM_MODE=response
)
# A3 floor [6][7][8]: A4 plus the trajectory gate (= GRPO when the gate shuts OPD off).
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
# Skill form stored in the pool, "<rewrite mode>:<tag>": the arm dimension of E4-E6 and E8a-E8c.
SKILL_FORMS=(none:raw deinstantiate:deinst aggregate:agg)

# DEBUG_RUN_ID reuses an earlier job's experiment names (continuation); a new job gets fresh names.
RUN_ID="${DEBUG_RUN_ID:-${PBS_JOBID:-$(date +%Y%m%d_%H%M%S)}}"
DEBUG_ARMS="${DEBUG_ARMS:-all}"
LOG_DIR="$PROJECT_ROOT/logs/agentstream"
mkdir -p "$LOG_DIR"
declare -A STATUS
GROUP_NAMES=()

# run_exp <arm id> <display name> <unique tag> <environment assignments...> [-- <hydra overrides...>]
# Arms not listed in DEBUG_ARMS (comma list; "all") are skipped. Later assignments win inside `env`,
# so an arm may override any COMMON_ENV / SEED_BASE_ENV value. Hydra overrides after `--` are
# appended last to the trainer command (they beat every env-derived value); trainer.resume_mode=auto
# goes first because agentstream_full.env pins single_pass runs to disable, and an arm may still
# override it after `--`.
run_exp() {
    local id="$1" name="$2" tag="$3"
    shift 3
    if [[ "$DEBUG_ARMS" != all && ",$DEBUG_ARMS," != *",$id,"* ]]; then
        echo "[SKIP] $id $name (not in DEBUG_ARMS=$DEBUG_ARMS)"
        return 0
    fi
    local -a envs=() overrides=()
    while (( $# )); do
        if [[ "$1" == -- ]]; then shift; overrides=("$@"); break; fi
        envs+=("$1"); shift
    done
    local log="$LOG_DIR/debug_ours_${STREAM_MODE}_${tag}_${RUN_ID}_steps${TOTAL_STEPS}.log"
    GROUP_NAMES+=("$id $name")
    echo ""
    echo "======================================================================"
    echo "[Arm] $id $name   start $(date)"
    echo "Env: ${envs[*]}   Hydra: trainer.resume_mode=auto ${overrides[*]}"
    echo "Log: $log"
    echo "======================================================================"
    set +e
    # Experiment name per arm and run id (no step count, so a continuation lands in the same directory);
    # inherited run-specific names are cleared.
    env -u EXPERIMENT_NAME -u EXPERIMENT_NAME_PREFIX -u DEFAULT_LOCAL_DIR \
        "${COMMON_ENV[@]}" "${SEED_BASE_ENV[@]}" "${envs[@]}" "AGENTSTREAM_EXPERIMENT_PREFIX=debug_ours_${RUN_ID}_${tag}" \
        bash "$SCRIPT_DIR/run_agentstream_sft_glm_self.sh" "$STREAM_MODE" trainer.resume_mode=auto "${overrides[@]}" 2>&1 | tee "$log"
    STATUS["$id $name"]=${PIPESTATUS[0]}
    set -e
    echo "[$([[ ${STATUS["$id $name"]} -eq 0 ]] && echo SUCCESS || echo FAILED)] $id $name   end $(date)"
}

# ===== Experiment E (TOTAL_STEPS steps, last checkpoint kept) =====
# ---- Sampling family on the A3 floor ----
# E1: the floor itself; the in-job anchor of the sampling family.
run_exp E1 "A3 floor" "a3" \
    "${A3_ENV[@]}"

# E3: [9] with the source-group baseline (E3 - E1 vs the own-group B3 - B1 = +0.032 +- 0.016).
run_exp E3 "A3 + sibling resample (source baseline)" "a3_resample" \
    "${A3_ENV[@]}" "${RESAMPLE_ENV[@]}"

# E4-E6: [10] channel 2 -- all-fail groups re-run with a retrieved pool skill; the skill form is the arm.
POOL_RESAMPLE_ARMS=(E4 E5 E6)
for i in "${!SKILL_FORMS[@]}"; do
    rewrite="${SKILL_FORMS[$i]%%:*}" form="${SKILL_FORMS[$i]##*:}"
    run_exp "${POOL_RESAMPLE_ARMS[$i]}" "A3 + resample + pool resample ($form skills)" "a3_resample_pool_$form" \
        "${A3_ENV[@]}" "${RESAMPLE_ENV[@]}" "${POOL_ENV[@]}" \
        AGENTSTREAM_SEED_SIBLING_RESAMPLE_POOL_MAX_GROUPS=3 \
        "AGENTSTREAM_SEED_GLOBAL_POOL_REWRITE=$rewrite"
done

# E2a-E2c: [10] channel 1 under the floor -- the pool skill as the gen OPD teacher context (coef 0.005),
# gated per trajectory like the local skill. Expected to stay ~E1 (the local skill passes 0.7%); the
# readout is gen_pass_ratio / gen_gap_mean_fail per skill form. E8 measures the same term ungated.
GEN_GATED_ARMS=(E2a E2b E2c)
for i in "${!SKILL_FORMS[@]}"; do
    rewrite="${SKILL_FORMS[$i]%%:*}" form="${SKILL_FORMS[$i]##*:}"
    run_exp "${GEN_GATED_ARMS[$i]}" "A3 + gen OPD from pool ($form skills)" "a3_genopd_$form" \
        "${A3_ENV[@]}" "${POOL_ENV[@]}" \
        AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0.005 \
        "AGENTSTREAM_SEED_GLOBAL_POOL_REWRITE=$rewrite"
done

# ---- Loss family on the A4 base ----
# E7: the base itself (local track alive: self skill as teacher, coef 0.01); the anchor of the loss family.
run_exp E7 "A4 base" "a4" \
    "${A4_ENV[@]}"

# E8a-E8c: [10] channel 1 -- the retrieved pool skill as the gen OPD teacher context (coef 0.005) on the
# same successful rows; no gate, so E8 - E7 is the gen term itself. The skill form is the arm.
GEN_POOL_ARMS=(E8a E8b E8c)
for i in "${!SKILL_FORMS[@]}"; do
    rewrite="${SKILL_FORMS[$i]%%:*}" form="${SKILL_FORMS[$i]##*:}"
    run_exp "${GEN_POOL_ARMS[$i]}" "A4 + gen OPD from pool ($form skills)" "a4_genopd_$form" \
        "${A4_ENV[@]}" "${POOL_ENV[@]}" \
        AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0.005 \
        "AGENTSTREAM_SEED_GLOBAL_POOL_REWRITE=$rewrite"
done

# ---- Earlier arms (jobs 69310 / 69900 / 70458 / 70633), finished and analysed; kept for reference ----
# # GRPO reference (analysis off, no OPD): the launcher hard-codes SEED_ENABLE_ANALYSIS=True, so the
# # analysis switch goes through a hydra override, as run_agentstream_baseline.sh does.
# run_exp B1 "GRPO (analysis off, OPD 0)" "grpo" \
#     AGENTSTREAM_SEED_OPD_LOSS_COEF=0 \
#     -- algorithm.seed.enable_analysis=False actor_rollout_ref.actor.opd_loss_coef=0
# # SEED reference: the unmodified objective, OPD 0.01 (= `run_agentstream_baseline.sh seed`).
# run_exp B2 "SEED (OPD 0.01)" "seed_opd001"
# # GRPO + sibling resample, own-group baseline: B3 - B1 = +0.032 +- 0.016 (tau2), 38% uniform resample groups.
# run_exp B3 "GRPO + sibling resample (max_groups 4)" "grpo_resample" \
#     AGENTSTREAM_SEED_OPD_LOSS_COEF=0 \
#     AGENTSTREAM_SEED_SIBLING_RESAMPLE=True \
#     AGENTSTREAM_SEED_SIBLING_RESAMPLE_MAX_GROUPS=4 \
#     -- algorithm.seed.enable_analysis=False actor_rollout_ref.actor.opd_loss_coef=0
# # A4 anchor candidate: success_only + response norm, no gate (within noise of A3 / GRPO / SEED).
# run_exp A4 "SEED + success_only + response norm" "floor_success_only_respnorm" \
#     AGENTSTREAM_SEED_SUCCESS_ONLY=True \
#     AGENTSTREAM_SEED_OPD_NORM_MODE=response
# # F1-F3 / S0 (job 69310): the floor ladder on the unmodified SEED objective; see git history.

echo ""
echo "======================================================================"
echo "Suite summary (run $RUN_ID, arms=$DEBUG_ARMS, steps=$TOTAL_STEPS, end $(date))"
failed=0
for name in "${GROUP_NAMES[@]}"; do
    printf '  %-60s exit=%s\n' "$name" "${STATUS[$name]}"
    (( STATUS[$name] == 0 )) || failed=1
done
echo "======================================================================"
exit $failed
