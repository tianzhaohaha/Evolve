#!/usr/bin/env bash
# Experiment suite for OUR method on top of the SEED baseline: sibling-success local teacher
# (AGENTSTREAM_SEED_LOCAL_TEACHER_SOURCE=sibling_success) and sample routing
# (AGENTSTREAM_SEED_ROUTE_MODE=sample), see SEED/README.md "可选改进开关" [4][5]. Same layout as
# run_global_ablation.sh: COMMON_ENV holds the stream settings shared with run_baseline_suite.sh,
# SEED_BASE_ENV pins every other extension off (= `run_agentstream_baseline.sh seed`), and each
# run_exp call spells out only the switches under test. Activate Conda in the caller.
#
# Usage: bash examples/agentstream_trainer/run_ours_debug.sh [--dry-run]
#   --dry-run prints each group's resolved setup without training.
#   TOTAL_STEPS=4 bash examples/agentstream_trainer/run_ours_debug.sh    # short smoke run
# Stream / hyper-parameters = the baseline suite (3 domains x 64 tasks, 10 per step, no checkpoints).
# TOTAL_STEPS (default 15, below) caps the stream: the full single pass is 20 steps; a shorter run
# stops early, validates holdout once at its last step, and compares to the baselines' online curves
# at the same step (their holdout is measured at step 20, so use the online metrics for comparison).
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
#   bash examples/agentstream_trainer/run_baseline_suite.sh --dry-run    # 预览命令
#   bash examples/agentstream_trainer/run_baseline_suite.sh              # vanilla grpo seed opsd rlsd
#   # 幂等：已到 20 步的基线自动跳过，未完成的从最近 checkpoint 续跑，原样重提即可
#   # 单跑一个基线：bash examples/agentstream_trainer/run_agentstream_baseline.sh grpo interleaved
#
# Stage 3 我们的方法（本脚本：当前启用 G1-G3 三组，步数由 TOTAL_STEPS 控制，默认 15，20 = 完整单遍；G4/G5 已注释）
#   bash examples/agentstream_trainer/run_ours_debug.sh --dry-run
#   bash examples/agentstream_trainer/run_ours_debug.sh
#   bash examples/agentstream_trainer/run_global_ablation.sh             # 旧的 global-pool 消融
#
# Stage 3 使用 episode_step（episode skill + 关键步 skill；正式基线不用，作为我们方法的消融）
#   前提：已用上面的 --reuse-rollouts 命令生成 ..._v5-episode_step 数据并导出 ...-sft-v5-episode_step 模型。
#   只需在调用任何 Stage-3 脚本前导出同一个变量（PBS 里放在 conda activate 之后）：
#     export AGENTSTREAM_SEED_SKILL_MODE=episode_step
#   注意：sibling_success 只依赖成功兄弟轨迹，与 skill 模式无关（step_only 除外）；该变量同时换了
#   SFT 起点，与 episode_only 的基线不再同起点，要做同起点对比须把基线套件也在 export 之后重跑。
#
# 结果
#   wandb: online/*（在线累计分，主指标）、val/<slug>_score|success_rate（holdout）、actor/opd_*、
#          seed/sibling/*、seed/route/*、actor/pg_row_weight_mean、timing_s/*
#   磁盘: ../ckpt/<experiment>/（CHECKPOINTS_ROOT，在仓库根 Evolve/ 下，不在 SEED 内）
#         agentstream_online_metrics.jsonl（每个 episode 一行）、<step>.jsonl（rollout 原文）、
#         seed_analysis/step_*.jsonl（sibling 模式下 episode_skill 字段 = 参照骨架，带 reference_traj_uid）
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
TOTAL_STEPS="${TOTAL_STEPS:-15}"
COMMON_ENV=(
    AGENTSTREAM_BENCHMARKS=bfcl,tau2,browsecompplus
    AGENTSTREAM_RL_STREAM_PROFILE=single_pass
    AGENTSTREAM_RL_TRAIN_DATA_SIZE=10                                     # tasks per step, as the suite
    "AGENTSTREAM_RL_EPOCHS=$TOTAL_STEPS"                                  # 20 = full single pass
    AGENTSTREAM_RL_SAVE_FREQ=0                                            # never checkpoint
    "AGENTSTREAM_SEED_SKILL_MODE=${AGENTSTREAM_SEED_SKILL_MODE:-episode_only}"   # episode_step: see header
    AGENTSTREAM_SEED_OPD_GEN_DOMINANCE=none                               # inert while gen is off
    AGENTSTREAM_SEED_GLOBAL_POOL_EVICT_POLICY=gate_ema                    # inert while pool=copy
    AGENTSTREAM_SEED_EMA_TAU=0.9                                          # inert while EMA is off
    AGENTSTREAM_SEED_REPLAY_CAPACITY=32                                   # inert while replay is off
)
# ===== SEED baseline switches shared by every group: the seed arm of run_agentstream_baseline.sh
# (single-track OPD, policy_vllm analyzer; gen / pool / eps / positive-only / EMA / replay off). =====
SEED_BASE_ENV=(
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=copy
    AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED=False
    AGENTSTREAM_SEED_FAILED_SKILL_POSITIVE=False
    AGENTSTREAM_SEED_OPD_GATE_EPS=0
    AGENTSTREAM_SEED_OPD_POSITIVE_ONLY=False
    AGENTSTREAM_SEED_EMA_MODE=off
    AGENTSTREAM_SEED_REPLAY_ENABLE=False
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

# ===== Groups: sibling-success teacher and sample routing on top of SEED (OPD 0.01 unless noted) =====
# Time budget: G1-G3 run now; G4 (soft routing) and G5 (control) are commented out below and can be
# re-enabled or moved to a separate PBS job later.
# G1: does a verified sibling solution give the failed rollouts a real teacher? (spec gap should turn positive)
run_exp "G1 SEED + sibling teacher (OPD 0.01)" "sibling_opd001" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.01 \
    AGENTSTREAM_SEED_LOCAL_TEACHER_SOURCE=sibling_success \
    AGENTSTREAM_SEED_ROUTE_MODE=none \
    AGENTSTREAM_SEED_ROUTE_PG_FAILED_WEIGHT=0

# G2: coefficient scale once the teacher carries information
run_exp "G2 SEED + sibling teacher (OPD 0.05)" "sibling_opd005" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.05 \
    AGENTSTREAM_SEED_LOCAL_TEACHER_SOURCE=sibling_success \
    AGENTSTREAM_SEED_ROUTE_MODE=none \
    AGENTSTREAM_SEED_ROUTE_PG_FAILED_WEIGHT=0

# G3: hard routing: successes -> GRPO only, failed rows of mixed groups -> sibling OPD only
run_exp "G3 sibling teacher + hard routing" "sibling_route_hard" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.01 \
    AGENTSTREAM_SEED_LOCAL_TEACHER_SOURCE=sibling_success \
    AGENTSTREAM_SEED_ROUTE_MODE=sample \
    AGENTSTREAM_SEED_ROUTE_PG_FAILED_WEIGHT=0

# # G4: soft routing: failed rows keep half of their policy gradient (entropy guard)
# run_exp "G4 sibling teacher + soft routing (0.5)" "sibling_route_soft" \
#     AGENTSTREAM_SEED_OPD_LOSS_COEF=0.01 \
#     AGENTSTREAM_SEED_LOCAL_TEACHER_SOURCE=sibling_success \
#     AGENTSTREAM_SEED_ROUTE_MODE=sample \
#     AGENTSTREAM_SEED_ROUTE_PG_FAILED_WEIGHT=0.5

# # G5 control: same routing but the original hindsight-skill teacher restricted to failed
# # trajectories (SEED_FAILED_ONLY is the upstream knob read by _common/agentstream.sh). If G3 beats
# # G5, the teacher is what matters, not the routing. Note: G5 also teaches all-fail groups.
# run_exp "G5 control: hindsight teacher + failed_only + hard routing" "skill_failedonly_route_hard" \
#     AGENTSTREAM_SEED_OPD_LOSS_COEF=0.01 \
#     AGENTSTREAM_SEED_LOCAL_TEACHER_SOURCE=skill \
#     SEED_FAILED_ONLY=True \
#     AGENTSTREAM_SEED_ROUTE_MODE=sample \
#     AGENTSTREAM_SEED_ROUTE_PG_FAILED_WEIGHT=0

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
