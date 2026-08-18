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

# Data-parallel policy serving: AGENTSTREAM_POLICY_GPU may be a comma list
# (e.g. "2,3,4,5"). One vllm replica is started per AGENTSTREAM_POLICY_TP
# GPUs on consecutive ports, and the comma-joined URL list is handed to the
# pipeline client, which round-robins requests across replicas.
IFS=',' read -ra _policy_gpus <<< "$AGENTSTREAM_POLICY_GPU"
policy_tp="${AGENTSTREAM_POLICY_TP:-1}"
policy_replicas=$(( ${#_policy_gpus[@]} / policy_tp ))
if (( policy_replicas < 1 )); then
    policy_replicas=1
fi

if [[ -n "${POLICY_BASE_URL:-}" ]]; then
    policy_base_url="$POLICY_BASE_URL"
else
    _policy_urls=()
    for ((i = 0; i < policy_replicas; i++)); do
        _policy_urls+=("http://${AGENTSTREAM_POLICY_HOST}:$((AGENTSTREAM_POLICY_PORT + i))/v1")
    done
    policy_base_url=$(IFS=','; echo "${_policy_urls[*]}")
fi
policy_model="${POLICY_MODEL:-$(basename "$AGENTSTREAM_BASE_MODEL_PATH")}"
server_pids=()

policy_ready() {
    local url
    local -a check_urls
    IFS=',' read -ra check_urls <<< "$policy_base_url"
    for url in "${check_urls[@]}"; do
        curl -fsS "$url/models" >/dev/null 2>&1 || return 1
    done
    return 0
}

stop_policy_server() {
    if (( ${#server_pids[@]} > 0 )); then
        echo "Stopping AgentStream Stage-1 policy server(s): ${server_pids[*]}"
        local pid
        for pid in "${server_pids[@]}"; do
            kill "$pid" >/dev/null 2>&1 || true
        done
        for pid in "${server_pids[@]}"; do
            wait "$pid" >/dev/null 2>&1 || true
        done
        server_pids=()
    fi
}
trap stop_policy_server EXIT

start_policy_server() {
    if policy_ready; then
        echo "Using existing policy endpoint(s) at $policy_base_url"
        return
    fi

    echo "Starting $policy_replicas Stage-1 policy vLLM replica(s) at $policy_base_url"
    local i
    for ((i = 0; i < policy_replicas; i++)); do
        local -a gpu_slice=("${_policy_gpus[@]:$((i * policy_tp)):$policy_tp}")
        local replica_gpus
        replica_gpus=$(IFS=','; echo "${gpu_slice[*]}")
        local replica_log="$AGENTSTREAM_SFT_DATA_DIR/logs/policy_vllm_${i}.log"
        CUDA_VISIBLE_DEVICES="$replica_gpus" \
            conda run -n "$VLLM_CONDA_ENV" --no-capture-output \
            vllm serve "$AGENTSTREAM_BASE_MODEL_PATH" \
            --host "$AGENTSTREAM_POLICY_HOST" \
            --port "$((AGENTSTREAM_POLICY_PORT + i))" \
            --served-model-name "$policy_model" \
            --tensor-parallel-size "$policy_tp" \
            --gpu-memory-utilization "$AGENTSTREAM_POLICY_GPU_MEMORY_UTILIZATION" \
            --max-model-len "$AGENTSTREAM_POLICY_MAX_MODEL_LEN" \
            >"$replica_log" 2>&1 &
        server_pids+=($!)
    done

    deadline=$((SECONDS + AGENTSTREAM_VLLM_STARTUP_TIMEOUT))
    until policy_ready; do
        local pid
        for pid in "${server_pids[@]}"; do
            if ! kill -0 "$pid" >/dev/null 2>&1; then
                echo "A Stage-1 vLLM replica exited before becoming ready. See $AGENTSTREAM_SFT_DATA_DIR/logs/policy_vllm_*.log" >&2
                exit 1
            fi
        done
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for Stage-1 vLLM replica(s). See $AGENTSTREAM_SFT_DATA_DIR/logs/policy_vllm_*.log" >&2
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