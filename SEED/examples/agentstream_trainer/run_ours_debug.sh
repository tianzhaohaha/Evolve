#!/usr/bin/env bash
# Experiment suite for the three "GRPO floor" guards on top of the SEED baseline
# (SEED/README.md "可选改进开关" [6]-[8], seed/gating.py):
#   [6] AGENTSTREAM_SEED_SUCCESS_ONLY   only verified successes are analyzed / distilled
#   [7] AGENTSTREAM_SEED_OPD_NORM_MODE  OPD divided by all response tokens (the PG denominator)
#   [8] AGENTSTREAM_SEED_TRAJ_GAP_GATE  a trajectory keeps its teacher signal only if the context
#                                       raises its mean log-prob by more than the margin
# Experiment A: FOUR ARMS IN ONE JOB -- grpo, seed, F3, F2. Baselines from another job are not valid
# controls: F3 (algorithmically GRPO once the gate shuts OPD off) landed 6 pts below the other job's
# grpo while S0 (plain SEED) matched that job's seed, every run launched after 2026-09-19 shows a
# larger first update (ppo_kl 0.03-0.04 vs 0.02) with the response length collapsing to ~50 tokens,
# and the browsecomp retriever dropped out mid-run. Readouts: F3 - grpo (floor), seed - grpo (does
# SEED trail GRPO in this environment?), F2 - F3 (anchor value of the normalised success-only term).
# Same layout as run_global_ablation.sh: COMMON_ENV holds the stream settings shared with
# run_baseline_suite.sh, SEED_BASE_ENV pins every extension off (= `run_agentstream_baseline.sh
# seed`), and each run_exp call spells out only the settings under test; hydra overrides go after
# `--` (needed for keys the launcher hard-codes, e.g. algorithm.seed.enable_analysis). Activate
# Conda in the caller.
#
# Usage: bash examples/agentstream_trainer/run_ours_debug.sh [--dry-run]
#   --dry-run prints each group's resolved setup without training.
#   TOTAL_STEPS=4 bash examples/agentstream_trainer/run_ours_debug.sh    # short smoke run
# Stream / hyper-parameters = the baseline suite (3 domains x 64 tasks, 10 per step, no checkpoints).
# TOTAL_STEPS (default 20 = the full single pass, as the suite) caps the stream: a shorter run stops
# early, validates holdout once at its last step, and compares to the baselines' online curves at
# the same step (their holdout is measured at step 20, so use the online metrics for comparison).
# Floor claims are power-limited (paired noise floor ~1.7 pts over 15-20 steps), so prefer 20.
#
# =============================================================================
# 全流程命令（在 SEED 根目录执行；PBS 作业只需在激活 Conda 后调用对应的一行）
# =============================================================================
# 共同前提：.env 里有 OPENAI_* / OPENROUTER_API_KEY / HF_TOKEN，公共配置在
# examples/agentstream_trainer/agentstream_full.env（v5：bfcl/tau2/browsecompplus 三域、每域 64 题、
# holdout 32、batch 10 x 20 步、group 6、history 3、prompt 38912、response 512、episode_only）。
# 改 benchmark / 题数 / 步数 JSON 后先提升 AGENTSTREAM_RUN_VERSION，再从 Stage 1 重来。
#
# Stage 1+2：SFT 数据 + SFT 模型（PBS: select=1:ncpus=48:ngpus=4）
#   bash examples/agentstream_trainer/run_stage12.sh --dry-run all      # 只打印解析结果
#   bash examples/agentstream_trainer/run_stage12.sh prepare            # Stage 1：rollout -> GLM 标注 -> parquet
#   bash examples/agentstream_trainer/run_stage12.sh sft                # Stage 2：SFT 3 epoch -> 导出 HF 模型
#   # 建议 prepare / sft 分两个 PBS 作业提交；Stage 1 中断可原样重提（RESUME）
#
# Stage 3 基线套件（PBS: select=1:ncpus=48:ngpus=2，walltime 168h）
#   bash examples/agentstream_trainer/run_baseline_suite.sh --dry-run    # 预览命令
#   bash examples/agentstream_trainer/run_baseline_suite.sh              # vanilla grpo seed opsd rlsd
#   # 单跑一个基线：bash examples/agentstream_trainer/run_agentstream_baseline.sh grpo interleaved
#
# Stage 3 实验 A：同一作业内四臂对照（grpo / seed / F3 / F2，步数由 TOTAL_STEPS 控制；旧的 F1-F3、S0 已注释保留；可选 A5 超时二分）
#   bash examples/agentstream_trainer/run_ours_debug.sh --dry-run
#   bash examples/agentstream_trainer/run_ours_debug.sh
#   TOTAL_STEPS=15 bash examples/agentstream_trainer/run_ours_debug.sh   # 时间不够时
#   bash examples/agentstream_trainer/run_global_ablation.sh             # 旧的 global-pool 消融
#
# 读数（四臂在同一作业内逐步配对，总体与分域；不要再与其他作业的曲线比）
#   下限：F3 − grpo 在噪声内 = 门控关掉 OPD 后守住 GRPO 下限
#   SEED vs GRPO：seed − grpo 的符号决定"SEED 低于 GRPO"在当前环境是否成立
#   锚：F2 − F3 为正且不低于 seed = 成功轨迹上归一化后的同向项可以不带失败轨迹的模仿而保留
#   机制：seed/traj_gate/pass_ratio（F3 应接近 0）、actor/opd_mask_token_fraction、actor/opd_loss
#   环境：第一次更新后 response_length/mean 是否塌到 50；prompt_length/mean 第 1 步低于 3500 = 检索失效，该臂作废
#
# 结果
#   wandb: online/*（在线累计分，主指标）、val/<slug>_score|success_rate（holdout）、actor/opd_*、
#          seed/traj_gate/*、timing_s/*
#   磁盘: ../ckpt/<experiment>/（CHECKPOINTS_ROOT，在仓库根 Evolve/ 下，不在 SEED 内）
#         agentstream_online_metrics.jsonl（每个 episode 一行）、<step>.jsonl（rollout 原文）、
#         seed_analysis/step_*.jsonl（success_only 下只含成功轨迹）
#   python examples/agentstream_trainer/analyze_results.py ../ckpt/*/agentstream_online_metrics.jsonl --csv all.csv

set -eo pipefail

[[ "${1:-}" == --dry-run ]] && { export DRY_RUN=true; shift; }
(( $# == 0 )) || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }

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
# stream (20 = full pass, as the suite) and no checkpoints. TEST_FREQ follows the step count, so the
# holdout is validated once at step TOTAL_STEPS. =====
STREAM_MODE="${STREAM_MODE:-interleaved}"
TOTAL_STEPS="${TOTAL_STEPS:-12}"
COMMON_ENV=(
    AGENTSTREAM_BENCHMARKS=bfcl,tau2,browsecompplus
    AGENTSTREAM_RL_STREAM_PROFILE=single_pass
    AGENTSTREAM_RL_TRAIN_DATA_SIZE=10                                     # tasks per step, as the suite
    "AGENTSTREAM_RL_EPOCHS=$TOTAL_STEPS"                                  # 20 = full single pass
    AGENTSTREAM_RL_SAVE_FREQ=0                                            # never checkpoint
    "AGENTSTREAM_SEED_SKILL_MODE=${AGENTSTREAM_SEED_SKILL_MODE:-episode_only}"
    AGENTSTREAM_SEED_OPD_GEN_DOMINANCE=none                               # inert while gen is off
    AGENTSTREAM_SEED_GLOBAL_POOL_EVICT_POLICY=gate_ema                    # inert while pool=copy
    AGENTSTREAM_SEED_EMA_TAU=0.9                                          # inert while EMA is off
    AGENTSTREAM_SEED_REPLAY_CAPACITY=32                                   # inert while replay is off
)
# ===== SEED baseline switches shared by every group: the full forced-off set of the seed arm of
# run_agentstream_baseline.sh (single-track OPD 0.01, policy_vllm analyzer; every extension off). =====
SEED_BASE_ENV=(
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.01
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=copy
    AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED=False
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
)

RUN_ID="${PBS_JOBID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/logs/agentstream"
mkdir -p "$LOG_DIR"
declare -A STATUS
GROUP_NAMES=()

# run_exp <display name> <unique tag> <environment assignments...> [-- <hydra overrides...>]
# Later assignments win inside `env`, so a group may override any COMMON_ENV / SEED_BASE_ENV value.
# Hydra overrides after `--` are appended last to the trainer command (they beat every env-derived
# value, exactly like the overrides of run_agentstream_baseline.sh).
run_exp() {
    local name="$1" tag="$2"
    shift 2
    local -a envs=() overrides=()
    while (( $# )); do
        if [[ "$1" == -- ]]; then shift; overrides=("$@"); break; fi
        envs+=("$1"); shift
    done
    local log="$LOG_DIR/debug_ours_${STREAM_MODE}_steps${TOTAL_STEPS}_${tag}_${RUN_ID}.log"
    GROUP_NAMES+=("$name")
    echo ""
    echo "======================================================================"
    echo "[Group] $name   start $(date)"
    echo "Env: ${envs[*]}   Hydra: ${overrides[*]}"
    echo "Log: $log"
    echo "======================================================================"
    set +e
    # Fresh experiment name per group and run; inherited run-specific names are cleared.
    env -u EXPERIMENT_NAME -u EXPERIMENT_NAME_PREFIX -u DEFAULT_LOCAL_DIR \
        "${COMMON_ENV[@]}" "${SEED_BASE_ENV[@]}" "${envs[@]}" "AGENTSTREAM_EXPERIMENT_PREFIX=debug_ours_${RUN_ID}_steps${TOTAL_STEPS}_${tag}" \
        bash "$SCRIPT_DIR/run_agentstream_sft_glm_self.sh" "$STREAM_MODE" "${overrides[@]}" 2>&1 | tee "$log"
    STATUS[$name]=${PIPESTATUS[0]}
    set -e
    echo "[$([[ ${STATUS[$name]} -eq 0 ]] && echo SUCCESS || echo FAILED)] $name   end $(date)"
}

# ---- 2026-09-21 job 69310 (F1-F3, S0): finished, analysed, kept for reference ----
# # ===== Ladder: SEED baseline + one more floor guard per group (OPD 0.01 everywhere) =====
# # F1: direction only. Removing the push on failed rollouts (87% of rows) -- enough on its own?
# #     Expect analyzed_traj_count ~ successes (~8/60) and opd_mask_token_fraction ~ 0.1; the mask
# #     shrink raises the per-token pressure, so step-1 KL may exceed the baseline's 0.043 -- if the
# #     responses shorten like G1 did, that is the evidence that [6] and [7] must be enabled together.
# run_exp "F1 SEED + success_only" "floor_success_only" \
#     AGENTSTREAM_SEED_SUCCESS_ONLY=True
#
# # F2: direction + magnitude. Same numerator, PG denominator: the coefficient no longer scales with
# #     the mask. Expect step-1 KL near grpo's 0.020, a small and stable opd_loss, response length
# #     tracking grpo, and the paired difference to grpo within noise.
# run_exp "F2 SEED + success_only + response norm" "floor_success_only_respnorm" \
#     AGENTSTREAM_SEED_SUCCESS_ONLY=True \
#     AGENTSTREAM_SEED_OPD_NORM_MODE=response
#
# # F3: direction + magnitude + information. Can the self-written skill pass the trajectory gate?
# #     Expect a low pass_ratio (baseline gap mean is -0.02..-0.04 nats/token), i.e. OPD ~ off and
# #     F3 ~ grpo; F3 - F2 measures the residual same-direction term on successes.
# run_exp "F3 SEED + success_only + response norm + traj gate" "floor_full" \
#     AGENTSTREAM_SEED_SUCCESS_ONLY=True \
#     AGENTSTREAM_SEED_OPD_NORM_MODE=response \
#     AGENTSTREAM_SEED_TRAJ_GAP_GATE=True \
#     AGENTSTREAM_SEED_TRAJ_GAP_GATE_MARGIN=0.0
#
# # S0: the unmodified SEED objective with half the OPD coefficient (no floor guard). Pairs with the
# #     suite's seed arm (0.01): how much of the gap to grpo is recovered by pressure alone, versus
# #     by fixing direction / magnitude / information in F1-F3.
# run_exp "S0 SEED (OPD 0.005)" "seed_opd0005" \
#     AGENTSTREAM_SEED_OPD_LOSS_COEF=0.005
#
# # # Optional controls on the unmodified SEED baseline (all trajectories distilled):
# # # norm_only isolates the per-step magnitude jitter of mask normalisation (mask fraction 0.13-0.75);
# # # gate_only shows the gate's pass ratio under the original teacher = "the self-written skill
# # # carries no information" in one number.
# # run_exp "C1 SEED + response norm" "norm_only" \
# #     AGENTSTREAM_SEED_OPD_NORM_MODE=response
# # run_exp "C2 SEED + traj gate" "gate_only" \
# #     AGENTSTREAM_SEED_TRAJ_GAP_GATE=True \
# #     AGENTSTREAM_SEED_TRAJ_GAP_GATE_MARGIN=0.0

# ===== Experiment A: four arms in one job (OPD 0.01 unless stated; 12 steps by default) =====
# A1: GRPO reference -- analysis off, no OPD. The launcher hard-codes SEED_ENABLE_ANALYSIS=True, so the
#     analysis switch must go through a hydra override (after `--`), as run_agentstream_baseline.sh does.
run_exp "A1 GRPO (analysis off, OPD 0)" "grpo" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0 \
    -- algorithm.seed.enable_analysis=False actor_rollout_ref.actor.opd_loss_coef=0

# A2: SEED reference -- the unmodified objective, OPD 0.01 (= `run_agentstream_baseline.sh seed`).
run_exp "A2 SEED (OPD 0.01)" "seed_opd001"

# A3: floor candidate -- success_only + response norm + trajectory gate. The gate shuts OPD off under
#     the self-written skill (pass_ratio ~ 0), so A3 should coincide with A1; A3 - A1 is the floor readout.
run_exp "A3 SEED + success_only + response norm + traj gate" "floor_full" \
    AGENTSTREAM_SEED_SUCCESS_ONLY=True \
    AGENTSTREAM_SEED_OPD_NORM_MODE=response \
    AGENTSTREAM_SEED_TRAJ_GAP_GATE=True

# A4: anchor candidate -- success_only + response norm, no gate: a normalised self-imitation term on
#     successes only. A4 - A3 (and A4 vs A2) tells whether the anchor survives without failed-row BC.
run_exp "A4 SEED + success_only + response norm" "floor_success_only_respnorm" \
    AGENTSTREAM_SEED_SUCCESS_ONLY=True \
    AGENTSTREAM_SEED_OPD_NORM_MODE=response

# # A5 (optional bisect of the first-update difference between jobs): GRPO with the old 600 s step
# # timeout. Enable only if the walltime allows a fifth arm.
# run_exp "A5 GRPO, step timeout 600" "grpo_timeout600" \
#     AGENTSTREAM_SEED_OPD_LOSS_COEF=0 \
#     AGENTSTREAM_RL_STEP_TIMEOUT=600 \
#     -- algorithm.seed.enable_analysis=False actor_rollout_ref.actor.opd_loss_coef=0

echo ""
echo "======================================================================"
echo "Suite summary (run $RUN_ID, steps=$TOTAL_STEPS, end $(date))"
failed=0
for name in "${GROUP_NAMES[@]}"; do
    printf '  %-56s exit=%s\n' "$name" "${STATUS[$name]}"
    (( STATUS[$name] == 0 )) || failed=1
done
echo "======================================================================"
exit $failed
