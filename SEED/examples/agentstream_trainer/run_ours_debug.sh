#!/usr/bin/env bash
# Debug suite for OUR method (SEED + dual-track OPD + global pool + gate eps + positive-only
# + EMA KL reference + replay): several parameter groups, one after another, on the formal
# five-benchmark setting but only a few steps and without checkpoints. Same layout as
# run_global_ablation.sh: COMMON_ENV holds the shared knobs; every run_exp call spells out the
# full switch set of our method, so any value can be changed per group in place.
# Activate Conda in the caller.
#
# Usage: bash examples/agentstream_trainer/run_ours_debug.sh [--dry-run]
#   --dry-run prints each group's resolved setup without training.
#   AGENTSTREAM_RL_EPOCHS=8 bash examples/agentstream_trainer/run_ours_debug.sh   # longer debug run
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
#   # 换 skill 模式而复用 rollout（可选，非正式基线）：
#   AGENTSTREAM_SEED_SKILL_MODE=episode_step bash examples/agentstream_trainer/run_stage12.sh \
#       --reuse-rollouts outputs/agentstream_episode_skill_pipeline_qwen3_4b_2507_v5 all
#
# Stage 3 基线套件（PBS: select=1:ncpus=48:ngpus=2，walltime 168h）
#   bash examples/agentstream_trainer/run_baseline_suite.sh --dry-run    # 预览各 baseline 的命令
#   STREAM_MODE=isolated bash examples/agentstream_trainer/run_baseline_suite.sh   # 每域独立 run（<实验名>_<域>）
#   bash examples/agentstream_trainer/run_baseline_suite.sh              # vanilla grpo grpo_base seed sdar opsd rlsd
#   # 幂等：已到 25 步的基线自动跳过，未完成的从最近 checkpoint 续跑，原样重提即可
#   # 单跑一个基线：bash examples/agentstream_trainer/run_agentstream_baseline.sh grpo interleaved
#
# Stage 3 我们的方法
#   bash examples/agentstream_trainer/run_ours_debug.sh --dry-run        # 调试套件：4 步 x 多组开关
#   bash examples/agentstream_trainer/run_ours_debug.sh
#   bash examples/agentstream_trainer/run_global_ablation.sh             # 正式消融：与基线同 batch / 步数
#
# Stage 3 使用 episode_step（episode skill + 关键步 skill；正式基线不用，作为我们方法的消融）
#   前提：已用上面的 --reuse-rollouts 命令生成 ..._v5-episode_step 数据并导出 ...-sft-v5-episode_step 模型。
#   只需在调用任何 Stage-3 脚本前导出同一个变量（PBS 里放在 conda activate 之后）：
#     export AGENTSTREAM_SEED_SKILL_MODE=episode_step
#     bash examples/agentstream_trainer/run_ours_debug.sh --dry-run      # 检查 "skill mode"/"model" 两行
#     bash examples/agentstream_trainer/run_ours_debug.sh
#     bash examples/agentstream_trainer/run_global_ablation.sh
#   full.env 据此自动切换：起点模型 -> -episode_step 的 SFT 目录；分析器 skill_mode=episode_step、
#   输出上限 4096、每条轨迹最多 5 条 step skill；实验名与 checkpoint 目录带 -episode_step 后缀，
#   与 episode_only 的 run 互不覆盖。生效标志：wandb seed/step_skill_teacher/step_skill_step_ratio > 0。
#   注意：该变量同时换了 SFT 起点，与 episode_only 的基线不再同起点；要做同起点对比，
#   至少把 seed 基线（建议整套）也在 export 之后重跑：bash examples/agentstream_trainer/run_baseline_suite.sh
#
# 结果
#   wandb: online/*（在线累计分，主指标）、val/<slug>_score|success_rate（holdout）、actor/opd_*、timing_s/*
#   磁盘: ../ckpt/<experiment>/（CHECKPOINTS_ROOT，在仓库根 Evolve/ 下，不在 SEED 内）
#         agentstream_online_metrics.jsonl（每个 episode 一行）、<step>.jsonl（rollout 原文）、global_step_N/
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

# ===== Shared settings: identical to the formal runs (agentstream_full.env / run_baseline_suite.sh)
# except the step count and checkpointing. NUM_TASKS, VAL_TASKS, VAL_REPEATS and TEST_FREQ come
# from the config; a 4-step run therefore validates once, at its last step. =====
STREAM_MODE="${STREAM_MODE:-interleaved}"
COMMON_ENV=(
    AGENTSTREAM_BENCHMARKS=bfcl,tau2,browsecompplus
    AGENTSTREAM_RL_STREAM_PROFILE=single_pass
    AGENTSTREAM_RL_TRAIN_DATA_SIZE=10                                     # tasks per step, as formal
    "AGENTSTREAM_RL_EPOCHS=${AGENTSTREAM_RL_EPOCHS:-4}"                   # debug: first steps of the stream
    AGENTSTREAM_RL_SAVE_FREQ=0                                            # never checkpoint
    "AGENTSTREAM_SEED_SKILL_MODE=${AGENTSTREAM_SEED_SKILL_MODE:-episode_only}"   # episode_step: see header
    AGENTSTREAM_SEED_OPD_GEN_DOMINANCE=none
    AGENTSTREAM_SEED_GLOBAL_POOL_EVICT_POLICY=gate_ema
    AGENTSTREAM_SEED_EMA_TAU=0.9
    AGENTSTREAM_SEED_REPLAY_CAPACITY=32
)

RUN_ID="${PBS_JOBID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/logs/agentstream"
mkdir -p "$LOG_DIR"
declare -A STATUS
GROUP_NAMES=()

# run_exp <display name> <unique tag> <environment assignments...>
run_exp() {
    local name="$1" tag="$2"
    shift 2
    local log="$LOG_DIR/debug_ours_${STREAM_MODE}_${tag}_${RUN_ID}.log"
    GROUP_NAMES+=("$name")
    echo ""
    echo "======================================================================"
    echo "[Debug group] $name   start $(date)"
    echo "Overrides: $*"
    echo "Log: $log"
    echo "======================================================================"
    set +e
    # Fresh experiment name per group and run; inherited run-specific names are cleared.
    env -u EXPERIMENT_NAME -u EXPERIMENT_NAME_PREFIX -u DEFAULT_LOCAL_DIR \
        "${COMMON_ENV[@]}" "$@" "AGENTSTREAM_EXPERIMENT_PREFIX=debug_ours_${RUN_ID}_${tag}" \
        bash "$SCRIPT_DIR/run_agentstream_sft_glm_self.sh" "$STREAM_MODE" 2>&1 | tee "$log"
    STATUS[$name]=${PIPESTATUS[0]}
    set -e
    echo "[$([[ ${STATUS[$name]} -eq 0 ]] && echo SUCCESS || echo FAILED)] $name   end $(date)"
}

# ===== Debug groups: every switch of our method is spelled out per run, edit any value =====
run_exp "Ours (all on)" "full" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=pool \
    AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED=True \
    AGENTSTREAM_SEED_FAILED_SKILL_POSITIVE=False \
    AGENTSTREAM_SEED_OPD_GATE_EPS=0.05 \
    AGENTSTREAM_SEED_OPD_POSITIVE_ONLY=True \
    AGENTSTREAM_SEED_EMA_MODE=ref \
    AGENTSTREAM_SEED_REPLAY_ENABLE=True \
    AGENTSTREAM_SEED_REPLAY_GROUPS_PER_STEP=2

run_exp "Ours, no positive-only, eps=0" "no_posonly" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=pool \
    AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED=True \
    AGENTSTREAM_SEED_FAILED_SKILL_POSITIVE=False \
    AGENTSTREAM_SEED_OPD_GATE_EPS=0 \
    AGENTSTREAM_SEED_OPD_POSITIVE_ONLY=False \
    AGENTSTREAM_SEED_EMA_MODE=ref \
    AGENTSTREAM_SEED_REPLAY_ENABLE=True \
    AGENTSTREAM_SEED_REPLAY_GROUPS_PER_STEP=2

run_exp "Ours, pool=copy (no cross-task skills)" "pool_copy" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=copy \
    AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED=True \
    AGENTSTREAM_SEED_FAILED_SKILL_POSITIVE=False \
    AGENTSTREAM_SEED_OPD_GATE_EPS=0.05 \
    AGENTSTREAM_SEED_OPD_POSITIVE_ONLY=True \
    AGENTSTREAM_SEED_EMA_MODE=ref \
    AGENTSTREAM_SEED_REPLAY_ENABLE=True \
    AGENTSTREAM_SEED_REPLAY_GROUPS_PER_STEP=2

run_exp "Ours, no EMA ref, no replay" "no_ema_replay" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=pool \
    AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED=True \
    AGENTSTREAM_SEED_FAILED_SKILL_POSITIVE=False \
    AGENTSTREAM_SEED_OPD_GATE_EPS=0.05 \
    AGENTSTREAM_SEED_OPD_POSITIVE_ONLY=True \
    AGENTSTREAM_SEED_EMA_MODE=off \
    AGENTSTREAM_SEED_REPLAY_ENABLE=False \
    AGENTSTREAM_SEED_REPLAY_GROUPS_PER_STEP=2

echo ""
echo "======================================================================"
echo "Debug suite summary (run $RUN_ID, end $(date))"
failed=0
for name in "${GROUP_NAMES[@]}"; do
    printf '  %-40s exit=%s\n' "$name" "${STATUS[$name]}"
    (( STATUS[$name] == 0 )) || failed=1
done
echo "======================================================================"
exit $failed
