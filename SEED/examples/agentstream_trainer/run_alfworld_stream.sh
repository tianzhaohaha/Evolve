#!/usr/bin/env bash

# Stream-controlled ALFWorld: apply random / isolated / sequential / interleaved
# task-type streams to SEED's original ALFWorld benchmark (zero new deps).
# Wraps examples/seed_trainer/_common/alfworld.sh — identical SEED algorithm
# surface; only the environment block changes (ALFWORLD_ENV_NAME + stream keys).
#
#   AS_STREAM_MODE=sequential bash run_alfworld_stream.sh
#   AS_STREAM_MODE=isolated  bash run_alfworld_stream.sh   # one run per task type

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ALFWORLD_COMMON="$PROJECT_ROOT/examples/seed_trainer/_common/alfworld.sh"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CONDA_ENV="${CONDA_ENV:-seed}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
fi

export ALFWORLD_ENV_NAME="alfworld_stream/AlfredTWEnv"
AS_STREAM_MODE="${AS_STREAM_MODE:-sequential}"
AS_STREAM_SEED="${AS_STREAM_SEED:-44}"
AS_NUM_TASKS_PER_TYPE="${AS_NUM_TASKS_PER_TYPE:-30}"
AS_VAL_TASKS_PER_TYPE="${AS_VAL_TASKS_PER_TYPE:-8}"
AS_BLOCK_PASSES="${AS_BLOCK_PASSES:-1}"

stream_args=(
    "env.alfworld_stream.stream_seed=$AS_STREAM_SEED"
    "env.alfworld_stream.num_tasks_per_type=$AS_NUM_TASKS_PER_TYPE"
    "env.alfworld_stream.val_tasks_per_type=$AS_VAL_TASKS_PER_TYPE"
    "env.alfworld_stream.block_passes=$AS_BLOCK_PASSES"
)

cd "$PROJECT_ROOT"

if [[ "$AS_STREAM_MODE" == "isolated" ]]; then
    TASK_TYPES="${AS_TASK_TYPES:-pick_and_place,pick_two_obj_and_place,look_at_obj_in_light,pick_heat_then_place_in_recep,pick_cool_then_place_in_recep,pick_clean_then_place_in_recep}"
    IFS=',' read -ra TYPES <<< "$TASK_TYPES"
    for TYPE in "${TYPES[@]}"; do
        TYPE="$(echo "$TYPE" | xargs)"
        echo "=== Running isolated alfworld_stream: ${TYPE} ==="
        EXPERIMENT_NAME="${EXPERIMENT_NAME_PREFIX:-seed_alfstream}_isolated_s${AS_STREAM_SEED}_${TYPE}" \
            bash "$ALFWORLD_COMMON" \
                "env.alfworld_stream.stream_mode=isolated" \
                "env.alfworld_stream.task_types=[$TYPE]" \
                "${stream_args[@]}" "$@"
    done
else
    export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_alfstream_${AS_STREAM_MODE}_s${AS_STREAM_SEED}}"
    exec bash "$ALFWORLD_COMMON" \
        "env.alfworld_stream.stream_mode=$AS_STREAM_MODE" \
        "${stream_args[@]}" "$@"
fi
