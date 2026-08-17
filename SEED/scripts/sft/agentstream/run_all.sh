#!/usr/bin/env bash

# End-to-end AgentStream workflow:
#   Stage 1: policy rollouts -> hindsight skills -> SFT parquet
#   Stage 2: SFT training -> exported Hugging Face checkpoint
#   Stage 3: independent SEED RL_OPD runs for configured stream modes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
AGENTSTREAM_CONFIG="${AGENTSTREAM_CONFIG:-$PROJECT_ROOT/examples/agentstream_trainer/agentstream_full.env}"

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

: "${CONDA_ENV:=seed}"
: "${VLLM_CONDA_ENV:=$CONDA_ENV}"
: "${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE or the environment.}"

if [[ ! -f "$AGENTSTREAM_BASE_MODEL_PATH/config.json" ]]; then
    echo "Base model not found: $AGENTSTREAM_BASE_MODEL_PATH" >&2
    exit 1
fi
if [[ ! -d "$AGENTSTREAM_EXGENTIC_ROOT/src/exgentic" ]]; then
    echo "Exgentic checkout not found: $AGENTSTREAM_EXGENTIC_ROOT" >&2
    exit 1
fi

mkdir -p "$AGENTSTREAM_SFT_DATA_DIR/logs"
policy_base_url="${POLICY_BASE_URL:-http://${AGENTSTREAM_POLICY_HOST}:${AGENTSTREAM_POLICY_PORT}/v1}"
policy_model="${POLICY_MODEL:-$(basename "$AGENTSTREAM_BASE_MODEL_PATH")}"
vllm_log="$AGENTSTREAM_SFT_DATA_DIR/logs/policy_vllm.log"
server_pid=""

policy_ready() {
    curl -fsS "$policy_base_url/models" >/dev/null 2>&1
}

stop_policy_server() {
    if [[ -n "$server_pid" ]]; then
        echo "Stopping AgentStream Stage-1 policy server pid=$server_pid"
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
        server_pid=""
    fi
}
trap stop_policy_server EXIT

start_policy_server() {
    if policy_ready; then
        echo "Using existing policy endpoint at $policy_base_url"
        return
    fi

    echo "Starting Stage-1 policy vLLM at $policy_base_url"
    CUDA_VISIBLE_DEVICES="$AGENTSTREAM_POLICY_GPU" \
        conda run -n "$VLLM_CONDA_ENV" --no-capture-output \
        vllm serve "$AGENTSTREAM_BASE_MODEL_PATH" \
        --host "$AGENTSTREAM_POLICY_HOST" \
        --port "$AGENTSTREAM_POLICY_PORT" \
        --served-model-name "$policy_model" \
        --tensor-parallel-size "$AGENTSTREAM_POLICY_TP" \
        --gpu-memory-utilization "$AGENTSTREAM_POLICY_GPU_MEMORY_UTILIZATION" \
        --max-model-len "$AGENTSTREAM_POLICY_MAX_MODEL_LEN" \
        >"$vllm_log" 2>&1 &
    server_pid=$!

    deadline=$((SECONDS + AGENTSTREAM_VLLM_STARTUP_TIMEOUT))
    until policy_ready; do
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
            echo "Stage-1 vLLM exited before becoming ready. See $vllm_log" >&2
            exit 1
        fi
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for Stage-1 vLLM. See $vllm_log" >&2
            exit 1
        fi
        sleep 2
    done
}

if [[ "$AGENTSTREAM_RUN_PREPARE" == "true" ]]; then
    echo "Stage 1/3: preparing AgentStream hindsight-skill SFT data"
    start_policy_server
    ENV_FILE=/dev/null \
    RUN_MODE=full \
    OUTPUT_DIR="$AGENTSTREAM_SFT_DATA_DIR" \
    MODEL_PATH="$AGENTSTREAM_BASE_MODEL_PATH" \
    POLICY_BASE_URL="$policy_base_url" \
    POLICY_MODEL="$policy_model" \
    AS_BENCHMARKS="$AGENTSTREAM_BENCHMARKS" \
    AS_BENCHMARK_KWARGS_JSON="$AGENTSTREAM_BENCHMARK_KWARGS_JSON" \
    NUM_TASKS="$AGENTSTREAM_NUM_TASKS" \
    ROLLOUTS_PER_TASK="$AGENTSTREAM_SFT_ROLLOUTS_PER_TASK" \
    PARALLEL_SESSIONS="$AGENTSTREAM_SFT_PARALLEL_SESSIONS" \
    MAX_STEPS="$AGENTSTREAM_MAX_STEPS" \
    SKILL_GEN_WORKERS="$AGENTSTREAM_SFT_SKILL_GEN_WORKERS" \
    SFT_VAL_RATIO="$AGENTSTREAM_SFT_VAL_RATIO" \
    OVERWRITE="${AGENTSTREAM_PREPARE_OVERWRITE:-false}" \
    RESUME="${AGENTSTREAM_PREPARE_RESUME:-true}" \
        bash "$SCRIPT_DIR/prepare_data.sh"
    stop_policy_server
fi

if [[ "$AGENTSTREAM_RUN_SFT" == "true" ]]; then
    echo "Stage 2/3: training AgentStream hindsight-skill SFT model"
    ENV_FILE=/dev/null \
    DATA_DIR="$AGENTSTREAM_SFT_DATA_DIR" \
    MODEL_PATH="$AGENTSTREAM_BASE_MODEL_PATH" \
    EXPORT_MODEL_DIR="$AGENTSTREAM_SFT_MODEL_DIR" \
    TOTAL_EPOCHS="$AGENTSTREAM_SFT_EPOCHS" \
    SFT_CUDA_VISIBLE_DEVICES="$AGENTSTREAM_SFT_GPUS" \
    NPROC_PER_NODE="$AGENTSTREAM_SFT_NPROC" \
    TRAIN_BATCH_SIZE="$AGENTSTREAM_SFT_TRAIN_BATCH_SIZE" \
    MICRO_BATCH_SIZE_PER_GPU="$AGENTSTREAM_SFT_MICRO_BATCH_SIZE" \
        bash "$SCRIPT_DIR/train_sft.sh"
fi

if [[ "$AGENTSTREAM_RUN_RL" == "true" ]]; then
    if [[ ! -f "$AGENTSTREAM_SFT_MODEL_DIR/config.json" ]]; then
        echo "SFT model not found: $AGENTSTREAM_SFT_MODEL_DIR" >&2
        exit 1
    fi

    echo "Stage 3/3: running AgentStream SEED RL_OPD modes"
    IFS=',' read -ra modes <<< "$AGENTSTREAM_RL_MODES"
    for mode in "${modes[@]}"; do
        mode="${mode//[[:space:]]/}"
        [[ -z "$mode" ]] && continue
        echo "=== AgentStream RL_OPD mode: $mode ==="
        AGENTSTREAM_CONFIG="$AGENTSTREAM_CONFIG" \
            bash "$PROJECT_ROOT/examples/agentstream_trainer/run_agentstream_sft_glm_self.sh" "$mode"
    done
fi

echo "AgentStream full pipeline finished."