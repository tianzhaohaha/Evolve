#!/usr/bin/env bash

# Run one AgentStream SEED RL_OPD mode from the Stage-1 SFT checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
AGENTSTREAM_CONFIG="${AGENTSTREAM_CONFIG:-$SCRIPT_DIR/agentstream_full.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
if [[ ! -f "$AGENTSTREAM_CONFIG" ]]; then
    echo "AgentStream config not found: $AGENTSTREAM_CONFIG" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "$AGENTSTREAM_CONFIG"
set +a

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

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
export AS_ON_EXHAUSTED=cycle
export AS_RESET_TIMEOUT="${AGENTSTREAM_RL_RESET_TIMEOUT:-600}"
export AS_STEP_TIMEOUT="${AGENTSTREAM_RL_STEP_TIMEOUT:-600}"
export TRAIN_DATA_SIZE="$AGENTSTREAM_RL_TRAIN_DATA_SIZE"
export VAL_DATA_SIZE="$AGENTSTREAM_RL_VAL_DATA_SIZE"
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
echo "  OPD coef:      $SEED_OPD_LOSS_COEF"
echo "  visible GPUs:  $CUDA_VISIBLE_DEVICES"
echo "  resume mode:   $RL_RESUME_MODE"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo "Dry run complete; RL training was not started."
    exit 0
fi

case "$mode" in
    random) launcher="$SCRIPT_DIR/run_agentstream_random.sh" ;;
    isolated) launcher="$SCRIPT_DIR/run_agentstream_isolated.sh" ;;
    sequential) launcher="$SCRIPT_DIR/run_agentstream_sequential.sh" ;;
    interleaved) launcher="$SCRIPT_DIR/run_agentstream_interleaved.sh" ;;
esac

export ENV_FILE=/dev/null
exec bash "$launcher" "$@"
