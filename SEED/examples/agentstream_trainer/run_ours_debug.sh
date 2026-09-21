#!/usr/bin/env bash
# Experiment suite for the three "GRPO floor" guards on top of the SEED baseline
# (SEED/README.md "可选改进开关" [6]-[8], seed/gating.py):
#   [6] AGENTSTREAM_SEED_SUCCESS_ONLY   only verified successes are analyzed / distilled
#   [7] AGENTSTREAM_SEED_OPD_NORM_MODE  OPD divided by all response tokens (the PG denominator)
#   [8] AGENTSTREAM_SEED_TRAJ_GAP_GATE  a trajectory keeps its teacher signal only if the context
#                                       raises its mean log-prob by more than the margin
# The groups form a ladder (F1 -> F2 -> F3): each adds one switch, so the paired per-step difference
# between neighbours is that switch's marginal effect, and every group pairs with the seed / grpo
# baselines of run_baseline_suite.sh (same stream, same tasks per step). Same layout as
# run_global_ablation.sh: COMMON_ENV holds the stream settings shared with run_baseline_suite.sh,
# SEED_BASE_ENV pins every other extension off (= `run_agentstream_baseline.sh seed`), and each
# run_exp call spells out only the switches under test. Activate Conda in the caller.
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
# Stage 3 下限守卫的阶梯实验（本脚本：F1-F3 三组 + S0 系数对照，步数由 TOTAL_STEPS 控制，默认 20；两组可选对照已注释）
#   bash examples/agentstream_trainer/run_ours_debug.sh --dry-run
#   bash examples/agentstream_trainer/run_ours_debug.sh
#   TOTAL_STEPS=15 bash examples/agentstream_trainer/run_ours_debug.sh   # 时间不够时
#   bash examples/agentstream_trainer/run_global_ablation.sh             # 旧的 global-pool 消融
#
# 读数（与 seed / grpo 逐步配对，总体与分域）
#   方向 [6]：seed/analyzed_traj_count（≈ 每步成功轨迹数）、seed/analysis_mode_success_only
#   力度 [7]：actor/opd_mask_token_fraction、actor/opd_loss、第一步的 actor/kl_loss（grpo 约 0.020，seed 0.043）
#   信息 [8]：seed/traj_gate/{pass_ratio, gap_mean_pass, gap_mean_fail, rows_before, rows_after}
#   行为：response_length/mean 曲线是否与 grpo 重合、browsecomp 逐步成功率是否不再归零
#   判据：与 grpo 的配对差不低于 -1 SE 且 browsecomp 不归零 = 下限守住
#   S0：原始 SEED 只把 OPD 系数从 0.01 降到 0.005，与 seed 基线配对 = 单纯减力度能挽回多少
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

# run_exp <display name> <unique tag> <environment assignments...>
# Later assignments win inside `env`, so a group may override any COMMON_ENV / SEED_BASE_ENV value.
run_exp() {
    local name="$1" tag="$2"
    shift 2
    local log="$LOG_DIR/debug_ours_${STREAM_MODE}_steps${TOTAL_STEPS}_${tag}_${RUN_ID}.log"
    GROUP_NAMES+=("$name")
    echo ""
    echo "======================================================================"
    echo "[Group] $name   start $(date)"
    echo "Overrides: $*"
    echo "Log: $log"
    echo "======================================================================"
    set +e
    # Fresh experiment name per group and run; inherited run-specific names are cleared.
    env -u EXPERIMENT_NAME -u EXPERIMENT_NAME_PREFIX -u DEFAULT_LOCAL_DIR \
        "${COMMON_ENV[@]}" "${SEED_BASE_ENV[@]}" "$@" "AGENTSTREAM_EXPERIMENT_PREFIX=debug_ours_${RUN_ID}_steps${TOTAL_STEPS}_${tag}" \
        bash "$SCRIPT_DIR/run_agentstream_sft_glm_self.sh" "$STREAM_MODE" 2>&1 | tee "$log"
    STATUS[$name]=${PIPESTATUS[0]}
    set -e
    echo "[$([[ ${STATUS[$name]} -eq 0 ]] && echo SUCCESS || echo FAILED)] $name   end $(date)"
}

# ===== Ladder: SEED baseline + one more floor guard per group (OPD 0.01 everywhere) =====
# F1: direction only. Removing the push on failed rollouts (87% of rows) -- enough on its own?
#     Expect analyzed_traj_count ~ successes (~8/60) and opd_mask_token_fraction ~ 0.1; the mask
#     shrink raises the per-token pressure, so step-1 KL may exceed the baseline's 0.043 -- if the
#     responses shorten like G1 did, that is the evidence that [6] and [7] must be enabled together.
run_exp "F1 SEED + success_only" "floor_success_only" \
    AGENTSTREAM_SEED_SUCCESS_ONLY=True

# F2: direction + magnitude. Same numerator, PG denominator: the coefficient no longer scales with
#     the mask. Expect step-1 KL near grpo's 0.020, a small and stable opd_loss, response length
#     tracking grpo, and the paired difference to grpo within noise.
run_exp "F2 SEED + success_only + response norm" "floor_success_only_respnorm" \
    AGENTSTREAM_SEED_SUCCESS_ONLY=True \
    AGENTSTREAM_SEED_OPD_NORM_MODE=response

# F3: direction + magnitude + information. Can the self-written skill pass the trajectory gate?
#     Expect a low pass_ratio (baseline gap mean is -0.02..-0.04 nats/token), i.e. OPD ~ off and
#     F3 ~ grpo; F3 - F2 measures the residual same-direction term on successes.
run_exp "F3 SEED + success_only + response norm + traj gate" "floor_full" \
    AGENTSTREAM_SEED_SUCCESS_ONLY=True \
    AGENTSTREAM_SEED_OPD_NORM_MODE=response \
    AGENTSTREAM_SEED_TRAJ_GAP_GATE=True \
    AGENTSTREAM_SEED_TRAJ_GAP_GATE_MARGIN=0.0

# S0: the unmodified SEED objective with half the OPD coefficient (no floor guard). Pairs with the
#     suite's seed arm (0.01): how much of the gap to grpo is recovered by pressure alone, versus
#     by fixing direction / magnitude / information in F1-F3.
run_exp "S0 SEED (OPD 0.005)" "seed_opd0005" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.005

# # Optional controls on the unmodified SEED baseline (all trajectories distilled):
# # norm_only isolates the per-step magnitude jitter of mask normalisation (mask fraction 0.13-0.75);
# # gate_only shows the gate's pass ratio under the original teacher = "the self-written skill
# # carries no information" in one number.
# run_exp "C1 SEED + response norm" "norm_only" \
#     AGENTSTREAM_SEED_OPD_NORM_MODE=response
# run_exp "C2 SEED + traj gate" "gate_only" \
#     AGENTSTREAM_SEED_TRAJ_GAP_GATE=True \
#     AGENTSTREAM_SEED_TRAJ_GAP_GATE_MARGIN=0.0

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
