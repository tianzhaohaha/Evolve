#!/usr/bin/env bash

# Stage-1 data preparation for AgentStream: baseline rollouts -> episode skills
# -> SFT parquet. Mirrors scripts/sft/alfworld/prepare_data.sh, with one
# simplification: this script does NOT manage a vLLM server. Point
# POLICY_BASE_URL at a running OpenAI-compatible policy endpoint, e.g.:
#   vllm serve $MODELS_ROOT/Qwen2.5-3B-Instruct --port 60001
#
# Requires AGENTSTREAM_EXGENTIC_ROOT and (via .env) OPENAI_* for the skill
# teacher, matching the repo-level .env conventions in the SEED README.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
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

: "${AGENTSTREAM_EXGENTIC_ROOT:?Please set AGENTSTREAM_EXGENTIC_ROOT to the AgentStream/exgentic checkout}"

RUN_MODE="${RUN_MODE:-full}"  # full or smoke
AS_BENCHMARKS="${AS_BENCHMARKS:-bfcl,tau2,appworld}"
if [[ -z "${AS_BENCHMARK_KWARGS_JSON:-}" ]]; then
    AS_BENCHMARK_KWARGS_JSON='{}'
fi
AS_RUNNER="${AS_RUNNER:-venv}"
MAX_STEPS="${MAX_STEPS:-30}"
HISTORY_LENGTH="${HISTORY_LENGTH:-5}"
PARALLEL_SESSIONS="${PARALLEL_SESSIONS:-8}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"
SEED="${SEED:-2026}"
INSPECTION_SAMPLES="${INSPECTION_SAMPLES:-3}"
INSPECTION_MAX_CHARS="${INSPECTION_MAX_CHARS:-4000}"

# Policy endpoint (must already be running).
MODEL_PATH="${MODEL_PATH:-${POLICY_MODEL_PATH:-${MODELS_ROOT:-Qwen2.5-3B-Instruct}}}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://127.0.0.1:60001/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_MODEL="${POLICY_MODEL:-$MODEL_NAME}"

# Skill teacher endpoint (defaults follow the repo .env: OPENAI_*).
SKILL_BASE_URL="${SKILL_BASE_URL:-${OPENAI_BASE_URL:?Please set OPENAI_BASE_URL in .env or SKILL_BASE_URL.}}"
SKILL_API_KEY="${SKILL_API_KEY:-${OPENAI_API_KEY:?Please set OPENAI_API_KEY in .env or SKILL_API_KEY.}}"
SKILL_MODEL="${SKILL_MODEL:-${OPENAI_MODEL:?Please set OPENAI_MODEL in .env or SKILL_MODEL.}}"
SKILL_GEN_WORKERS="${SKILL_GEN_WORKERS:-64}"

# shellcheck source=../_common/teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft/_common/teacher_naming.sh"

if [[ "$RUN_MODE" == "smoke" ]]; then
    NUM_TASKS="${NUM_TASKS:-2}"
    ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-1}"
    MAX_CANDIDATES="${MAX_CANDIDATES:-4}"
    OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/agentstream_episode_skill_pipeline_smoke_${SFT_SELF_DIR_SUFFIX}}"
elif [[ "$RUN_MODE" == "full" ]]; then
    NUM_TASKS="${NUM_TASKS:-50}"
    ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-8}"
    MAX_CANDIDATES="${MAX_CANDIDATES:-}"
    OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/agentstream_episode_skill_pipeline_qwen25_3b_${SFT_SELF_DIR_SUFFIX}}"
else
    echo "Unsupported RUN_MODE='$RUN_MODE'. Use RUN_MODE=full or RUN_MODE=smoke." >&2
    exit 2
fi

if ! curl -fsS "${POLICY_BASE_URL}/models" >/dev/null 2>&1; then
    echo "Policy endpoint not reachable at $POLICY_BASE_URL. Start one first, e.g.:" >&2
    echo "  vllm serve $MODEL_PATH --port ${POLICY_BASE_URL##*:} --gpu-memory-utilization 0.6" >&2
    exit 1
fi

args=(
    "$SCRIPT_DIR/pipeline.py"
    --env-file "$ENV_FILE"
    --output-dir "$OUTPUT_DIR"
    --exgentic-root "$AGENTSTREAM_EXGENTIC_ROOT"
    --benchmarks "$AS_BENCHMARKS"
    --benchmark-kwargs-json "$AS_BENCHMARK_KWARGS_JSON"
    --runner "$AS_RUNNER"
    --num-tasks-per-benchmark "$NUM_TASKS"
    --rollouts-per-task "$ROLLOUTS_PER_TASK"
    --parallel-sessions "$PARALLEL_SESSIONS"
    --max-steps "$MAX_STEPS"
    --history-length "$HISTORY_LENGTH"
    --skill-gen-workers "$SKILL_GEN_WORKERS"
    --sft-val-ratio "$SFT_VAL_RATIO"
    --seed "$SEED"
    --inspection-samples "$INSPECTION_SAMPLES"
    --inspection-max-chars "$INSPECTION_MAX_CHARS"
    --policy-base-url "$POLICY_BASE_URL"
    --policy-api-key "$POLICY_API_KEY"
    --policy-model "$POLICY_MODEL"
    --skill-base-url "$SKILL_BASE_URL"
    --skill-api-key "$SKILL_API_KEY"
    --skill-model "$SKILL_MODEL"
)
[[ -n "$MAX_CANDIDATES" ]] && args+=(--max-candidates "$MAX_CANDIDATES")
[[ "${RESUME:-false}" == "true" ]] && args+=(--resume)
[[ "${OVERWRITE:-false}" == "true" ]] && args+=(--overwrite)
[[ "${STOP_AFTER_BASELINE_ROLLOUTS:-false}" == "true" ]] && args+=(--stop-after-baseline-rollouts)

cd "$PROJECT_ROOT"
exec python3 "${args[@]}"
