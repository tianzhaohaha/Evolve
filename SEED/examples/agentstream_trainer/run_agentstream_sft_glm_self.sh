#!/usr/bin/env bash

# Run one AgentStream SEED RL_OPD mode from the Stage-1 SFT checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
AGENTSTREAM_CONFIG="${AGENTSTREAM_CONFIG:-$SCRIPT_DIR/agentstream_full.env}"
_caller_wandb_mode="${WANDB_MODE:-}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
if [[ -n "$_caller_wandb_mode" ]]; then
    export WANDB_MODE="$_caller_wandb_mode"
fi
if [[ ! -f "$AGENTSTREAM_CONFIG" ]]; then
    echo "AgentStream config not found: $AGENTSTREAM_CONFIG" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "$AGENTSTREAM_CONFIG"
set +a
# Stage-3-only switch: start from the raw backbone instead of the SFT checkpoint (the SEED
# paper's GRPO row; set by run_agentstream_baseline.sh grpo_base). Kept out of full.env so a
# leaked variable can never redirect Stage 2's export onto the backbone directory.
if [[ "${AGENTSTREAM_RL_INIT_FROM_BASE:-false}" == "true" ]]; then
    AGENTSTREAM_SFT_MODEL_DIR="$AGENTSTREAM_BASE_MODEL_PATH"
fi


mode="${1:-${AS_STREAM_MODE:-sequential}}"
if (( $# > 0 )); then
    shift
fi

case "$mode" in
    random|isolated|sequential|interleaved) ;;
    *)
        echo "Unsupported stream mode '$mode'. Use random, isolated, sequential, or interleaved." >&2
        exit 2
        ;;
esac

if [[ ! -f "$AGENTSTREAM_SFT_MODEL_DIR/config.json" ]]; then
    echo "AgentStream SFT model not found: $AGENTSTREAM_SFT_MODEL_DIR" >&2
    echo "Run scripts/sft/agentstream/run_all.sh first, or set AGENTSTREAM_SFT_MODEL_DIR." >&2
    exit 1
fi
if [[ ! -d "$AGENTSTREAM_EXGENTIC_ROOT/src/exgentic" ]]; then
    echo "Exgentic checkout not found: $AGENTSTREAM_EXGENTIC_ROOT" >&2
    exit 1
fi

export MODEL_PATH="$AGENTSTREAM_SFT_MODEL_DIR"
export CUDA_VISIBLE_DEVICES="$AGENTSTREAM_RL_GPUS"
export AS_BENCHMARKS="$AGENTSTREAM_BENCHMARKS"
export AS_BENCHMARK_KWARGS_JSON="$AGENTSTREAM_BENCHMARK_KWARGS_JSON"
export AS_STREAM_MODE="$mode"
export AS_PROTOCOL="$AGENTSTREAM_RL_PROTOCOL"
export AS_STREAM_SEED="$AGENTSTREAM_STREAM_SEED"
export AS_NUM_TASKS="$AGENTSTREAM_NUM_TASKS"
export AS_VAL_TASKS="$AGENTSTREAM_VAL_TASKS"
export AS_VAL_SOURCE="$AGENTSTREAM_RL_VAL_SOURCE"
export AS_BLOCK_PASSES="$AGENTSTREAM_RL_BLOCK_PASSES"
export AS_MAX_STEPS="$AGENTSTREAM_MAX_STEPS"
# Stream profile knobs come from agentstream_full.env (AGENTSTREAM_RL_STREAM_PROFILE):
#   cycle -> multi-pass RL (Run B); stop -> strict single-pass online RL (Run A).
export AS_ON_EXHAUSTED="${AGENTSTREAM_RL_ON_EXHAUSTED:-cycle}"
# true -> also log per-pass cumulative curves online/pass<K>/* on repeat passes.
export AS_TRACK_REPEAT_PASSES="${AGENTSTREAM_RL_TRACK_REPEAT_PASSES:-false}"
export AS_VAL_REPEATS="${AGENTSTREAM_RL_VAL_REPEATS:-1}"
export VAL_BEFORE_TRAIN="${AGENTSTREAM_RL_VAL_BEFORE_TRAIN:-False}"
export ACTOR_LR="${AGENTSTREAM_RL_ACTOR_LR:-1e-6}"
export ENTROPY_COEFF="${AGENTSTREAM_RL_ENTROPY_COEFF:-0.001}"
# Switch from agentstream_full.env: true -> skip full-vocab entropy in the
# old-log-prob/teacher compute_log_prob passes (memory spike fix, appworld OOM).
if [[ "${AGENTSTREAM_RL_SKIP_ENTROPY_IN_LOG_PROB:-false}" == "true" ]]; then
    export LOG_PROB_CALCULATE_ENTROPY=False
fi
# NOTE: do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here — vLLM's
# sleep-mode CuMemAllocator asserts against it and rollout init fails.
export AS_RESET_TIMEOUT="$AGENTSTREAM_RL_RESET_TIMEOUT"
export AS_STEP_TIMEOUT="$AGENTSTREAM_RL_STEP_TIMEOUT"
export MAX_ACTOR_CKPT_TO_KEEP="$AGENTSTREAM_RL_MAX_CKPT_TO_KEEP"
export TRAIN_DATA_SIZE="$AGENTSTREAM_RL_TRAIN_DATA_SIZE"
# Empty -> agentstream.sh derives benchmarks x val_tasks x repeats (isolated-safe).
export VAL_DATA_SIZE="${AGENTSTREAM_RL_VAL_DATA_SIZE:-}"
export GROUP_SIZE="$AGENTSTREAM_RL_GROUP_SIZE"
export POLICY_ROLLOUT_N=1
export N_GPUS_PER_NODE="$AGENTSTREAM_RL_N_GPUS"
export PPO_MINI_BATCH_SIZE="$AGENTSTREAM_RL_PPO_MINI_BATCH_SIZE"
export PPO_MICRO_BATCH_SIZE_PER_GPU="$AGENTSTREAM_RL_PPO_MICRO_BATCH_SIZE"
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="$AGENTSTREAM_RL_LOG_PROB_MICRO_BATCH_SIZE"
if [[ "$mode" == "isolated" ]]; then
    export TOTAL_EPOCHS="$AGENTSTREAM_RL_ISOLATED_EPOCHS"
else
    export TOTAL_EPOCHS="$AGENTSTREAM_RL_EPOCHS"
fi
export SAVE_FREQ="$AGENTSTREAM_RL_SAVE_FREQ"
export TEST_FREQ="$AGENTSTREAM_RL_TEST_FREQ"
export RL_RESUME_MODE="$AGENTSTREAM_RL_RESUME_MODE"
export RL_RESUME_FROM_PATH=null
export SEED_ENABLE_ANALYSIS=True
export SEED_SKILL_MODE="$AGENTSTREAM_SEED_SKILL_MODE"
export SEED_OPD_LOSS_COEF="$AGENTSTREAM_SEED_OPD_LOSS_COEF"
export SEED_OPD_GEN_LOSS_COEF="$AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF"
export SEED_OPD_GATE_EPS="$AGENTSTREAM_SEED_OPD_GATE_EPS"
export SEED_OPD_GEN_DOMINANCE="$AGENTSTREAM_SEED_OPD_GEN_DOMINANCE"
export SEED_OPD_POSITIVE_ONLY="$AGENTSTREAM_SEED_OPD_POSITIVE_ONLY"
export SEED_FAILED_SKILL_POSITIVE="$AGENTSTREAM_SEED_FAILED_SKILL_POSITIVE"
export SEED_LOCAL_TEACHER_SOURCE="$AGENTSTREAM_SEED_LOCAL_TEACHER_SOURCE"
export SEED_ROUTE_MODE="$AGENTSTREAM_SEED_ROUTE_MODE"
export SEED_ROUTE_PG_FAILED_WEIGHT="$AGENTSTREAM_SEED_ROUTE_PG_FAILED_WEIGHT"
export SEED_SUCCESS_ONLY="$AGENTSTREAM_SEED_SUCCESS_ONLY"
export SEED_OPD_NORM_MODE="$AGENTSTREAM_SEED_OPD_NORM_MODE"
export SEED_TRAJ_GAP_GATE="$AGENTSTREAM_SEED_TRAJ_GAP_GATE"
export SEED_TRAJ_GAP_GATE_MARGIN="$AGENTSTREAM_SEED_TRAJ_GAP_GATE_MARGIN"
export SEED_SIBLING_RESAMPLE="$AGENTSTREAM_SEED_SIBLING_RESAMPLE"
export SEED_SIBLING_RESAMPLE_MAX_GROUPS="$AGENTSTREAM_SEED_SIBLING_RESAMPLE_MAX_GROUPS"
export SEED_SIBLING_RESAMPLE_BASELINE="$AGENTSTREAM_SEED_SIBLING_RESAMPLE_BASELINE"
export SEED_SIBLING_RESAMPLE_POOL_MAX_GROUPS="$AGENTSTREAM_SEED_SIBLING_RESAMPLE_POOL_MAX_GROUPS"
export SEED_EMA_MODE="$AGENTSTREAM_SEED_EMA_MODE"
export SEED_EMA_TAU="$AGENTSTREAM_SEED_EMA_TAU"
export SEED_REPLAY_ENABLE="$AGENTSTREAM_SEED_REPLAY_ENABLE"
export SEED_REPLAY_GROUPS_PER_STEP="$AGENTSTREAM_SEED_REPLAY_GROUPS_PER_STEP"
export SEED_REPLAY_CAPACITY="$AGENTSTREAM_SEED_REPLAY_CAPACITY"
export SEED_GLOBAL_POOL_SOURCE="$AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE"
export SEED_GLOBAL_POOL_MIN_SIM="$AGENTSTREAM_SEED_GLOBAL_POOL_MIN_SIM"
export SEED_GLOBAL_POOL_SCORE_THRESHOLD="$AGENTSTREAM_SEED_GLOBAL_POOL_SCORE_THRESHOLD"
export SEED_GLOBAL_POOL_ADMIT_FAILED="$AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED"
export SEED_GLOBAL_POOL_EVICT_POLICY="$AGENTSTREAM_SEED_GLOBAL_POOL_EVICT_POLICY"
export SEED_GLOBAL_POOL_WINDOW_STEPS="$AGENTSTREAM_SEED_GLOBAL_POOL_WINDOW_STEPS"
export SEED_GLOBAL_POOL_ADMISSION="$AGENTSTREAM_SEED_GLOBAL_POOL_ADMISSION"
export SEED_GLOBAL_POOL_JUDGE_BACKEND="$AGENTSTREAM_SEED_GLOBAL_POOL_JUDGE_BACKEND"
export SEED_GLOBAL_POOL_REWRITE="$AGENTSTREAM_SEED_GLOBAL_POOL_REWRITE"
# 池准入的 judge 读的是训练进程环境里的 OPENROUTER_API_KEY（exgentic env worker 的
# dotenv 不会传给 trainer 进程），缺失时训练照跑但池永远为空，这里提前示警。
if [[ "$SEED_GLOBAL_POOL_SOURCE" == "pool" && "$SEED_GLOBAL_POOL_JUDGE_BACKEND" == "openai" && -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "WARNING: SEED_GLOBAL_POOL_SOURCE=pool but OPENROUTER_API_KEY is not set in this shell;" >&2
    echo "         judge admission will be silently disabled and the pool will stay empty." >&2
fi
export SEED_ANALYSIS_BACKEND="$AGENTSTREAM_SEED_ANALYSIS_BACKEND"
export SEED_ANALYSIS_PROMPT_VERSION="$AGENTSTREAM_SEED_ANALYSIS_PROMPT_VERSION"
export SEED_ANALYSIS_INCLUDE_EPISODE_SUMMARY=True
export SEED_ANALYSIS_MAX_COMPLETION_TOKENS="$AGENTSTREAM_SEED_ANALYSIS_MAX_COMPLETION_TOKENS"
export SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ="$AGENTSTREAM_SEED_ANALYSIS_MAX_STEP_SKILLS"
export PROJECT_NAME="${PROJECT_NAME:-agentic_agentstream}"

experiment_prefix="${AGENTSTREAM_EXPERIMENT_PREFIX:-seed_qwen2.5_3b_agentstream_sft_glm_self}"
export EXPERIMENT_NAME_PREFIX="$experiment_prefix"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${experiment_prefix}_${mode}_${AS_PROTOCOL}_s${AS_STREAM_SEED}}"

echo "Running AgentStream RL_OPD"
echo "  mode:          $mode"
echo "  benchmarks:    $AS_BENCHMARKS"
echo "  model:         $MODEL_PATH"
echo "  epochs:        $TOTAL_EPOCHS"
echo "  stream:        profile=${AGENTSTREAM_RL_STREAM_PROFILE:-multipass} on_exhausted=$AS_ON_EXHAUSTED track_repeat_passes=$AS_TRACK_REPEAT_PASSES"
echo "  tasks:         num_tasks=$AS_NUM_TASKS val_tasks=$AS_VAL_TASKS train/step=$TRAIN_DATA_SIZE x$GROUP_SIZE val_slots=${VAL_DATA_SIZE:-auto} val_repeats=$AS_VAL_REPEATS"
echo "  actor lr:      $ACTOR_LR (0 = frozen-policy control run)"
echo "  entropy coef:  $ENTROPY_COEFF (log_prob entropy calc: ${LOG_PROB_CALCULATE_ENTROPY:-True})"
echo "  OPD coef:      $SEED_OPD_LOSS_COEF"
echo "  OPD gen coef:  $SEED_OPD_GEN_LOSS_COEF"
echo "  Global pool:   $SEED_GLOBAL_POOL_SOURCE (admission=$SEED_GLOBAL_POOL_ADMISSION judge=$SEED_GLOBAL_POOL_JUDGE_BACKEND rewrite=$SEED_GLOBAL_POOL_REWRITE evict=$SEED_GLOBAL_POOL_EVICT_POLICY)"
echo "  EMA mode:      $SEED_EMA_MODE (tau=$SEED_EMA_TAU)"
echo "  Replay:        $SEED_REPLAY_ENABLE (groups/step=$SEED_REPLAY_GROUPS_PER_STEP, capacity=$SEED_REPLAY_CAPACITY)"
echo "  Local teacher: $SEED_LOCAL_TEACHER_SOURCE"
echo "  Routing:       $SEED_ROUTE_MODE (failed-row PG weight=$SEED_ROUTE_PG_FAILED_WEIGHT)"
echo "  Floor guards:  success_only=$SEED_SUCCESS_ONLY opd_norm=$SEED_OPD_NORM_MODE traj_gate=$SEED_TRAJ_GAP_GATE (margin=$SEED_TRAJ_GAP_GATE_MARGIN)"
echo "  Resample:      sibling_resample=$SEED_SIBLING_RESAMPLE (max_groups=$SEED_SIBLING_RESAMPLE_MAX_GROUPS baseline=$SEED_SIBLING_RESAMPLE_BASELINE pool_max_groups=$SEED_SIBLING_RESAMPLE_POOL_MAX_GROUPS)"
echo "  visible GPUs:  $CUDA_VISIBLE_DEVICES"
echo "  resume mode:   $RL_RESUME_MODE"
echo "  checkpoints:   save_freq=$SAVE_FREQ keep=$MAX_ACTOR_CKPT_TO_KEEP"
echo "  experiment:    $EXPERIMENT_NAME"
echo "  overrides:     $*"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo "Dry run complete; RL training was not started."
    exit 0
fi

# browsecompplus 在列时确保共享检索服务已就绪（本机自动拉起；DRY_RUN 不触发）。
bash "$SCRIPT_DIR/ensure_browsecomp_retriever.sh"

case "$mode" in
    random) launcher="$SCRIPT_DIR/run_agentstream_random.sh" ;;
    isolated) launcher="$SCRIPT_DIR/run_agentstream_isolated.sh" ;;
    sequential) launcher="$SCRIPT_DIR/run_agentstream_sequential.sh" ;;
    interleaved) launcher="$SCRIPT_DIR/run_agentstream_interleaved.sh" ;;
esac

export ENV_FILE=/dev/null
exec bash "$launcher" "$@"
