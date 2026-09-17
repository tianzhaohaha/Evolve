#!/usr/bin/env bash
# PBS/local shared Stage-3 experiment list. Activate Conda in the caller.
# Usage: bash examples/agentstream_trainer/run_global_ablation.sh [--dry-run]
# Edit the shared settings and four run_exp calls below, then sync with Git.

set -eo pipefail

DRY_RUN_SUITE=false
case "${1:-}" in
    --dry-run) DRY_RUN_SUITE=true ;;
    "") ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Load machine paths/credentials once, BEFORE the experiment overrides. Do not
# enable xtrace here: the private environment file may contain API keys.
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
export ENV_FILE=/dev/null
export PYTHONUNBUFFERED=1

# ===== Shared experiment settings: edit here =====
BATCH_SIZE=10
TOTAL_STEPS=20
STREAM_MODE=interleaved
COMMON_ENV=(
    AGENTSTREAM_RL_STREAM_PROFILE=single_pass
    "AGENTSTREAM_RL_TRAIN_DATA_SIZE=$BATCH_SIZE"
    "AGENTSTREAM_RL_EPOCHS=$TOTAL_STEPS"
    AGENTSTREAM_SEED_SKILL_MODE=episode_only
    AGENTSTREAM_SEED_OPD_GEN_DOMINANCE=none
    AGENTSTREAM_SEED_GLOBAL_POOL_EVICT_POLICY=gate_ema
    AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED=True
    AGENTSTREAM_SEED_FAILED_SKILL_POSITIVE=False
    AGENTSTREAM_SEED_EMA_TAU=0.9
    AGENTSTREAM_SEED_REPLAY_GROUPS_PER_STEP=2
    AGENTSTREAM_SEED_REPLAY_CAPACITY=32
)
export "${COMMON_ENV[@]}"

# Resolve model/version/task naming from the same public config as Stage 3.
export AGENTSTREAM_CONFIG="${AGENTSTREAM_CONFIG:-$SCRIPT_DIR/agentstream_full.env}"
if [[ ! -f "$AGENTSTREAM_CONFIG" ]]; then
    echo "AgentStream config not found: $AGENTSTREAM_CONFIG" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1090
source "$AGENTSTREAM_CONFIG"
set +a

SUITE_PREFIX="${AGENTSTREAM_ABLATION_PREFIX:-seed_${AGENTSTREAM_MODEL_TAG}_agentstream_${AGENTSTREAM_RUN_VERSION}_n${AGENTSTREAM_NUM_TASKS}_${AGENTSTREAM_RL_STREAM_PROFILE}_b${BATCH_SIZE}_steps${TOTAL_STEPS}}"
RUN_ID="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
LOG_DIR="$PROJECT_ROOT/logs/agentstream"
if [[ "$DRY_RUN_SUITE" != true ]]; then
    mkdir -p "$LOG_DIR"
fi

FAILED_EXPERIMENTS=()
EXPERIMENTS=()
declare -A EXP_STATUS EXP_TEE_STATUS EXP_LOG

# run_exp <display name> <unique tag> <environment assignments...>
run_exp() {
    local name="$1" tag="$2"
    shift 2
    local log="$LOG_DIR/stage3_${STREAM_MODE}_b${BATCH_SIZE}_steps${TOTAL_STEPS}_${tag}_${RUN_ID}.log"
    EXPERIMENTS+=("$name")
    EXP_LOG["$name"]="$log"
    # Clear inherited low-level output overrides so each run stays independent.
    local -a command=(
        env -u EXPERIMENT_NAME -u EXPERIMENT_NAME_PREFIX -u DEFAULT_LOCAL_DIR
        "${COMMON_ENV[@]}" "$@"
        "AGENTSTREAM_EXPERIMENT_PREFIX=${SUITE_PREFIX}_${tag}"
        bash "$SCRIPT_DIR/run_agentstream_sft_glm_self.sh" "$STREAM_MODE"
    )

    echo ""
    echo "======================================================================"
    echo "[Experiment] $name"
    echo "Start time : $(date)"
    echo "Log        : $log"
    echo "======================================================================"
    if [[ "$DRY_RUN_SUITE" == true ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        EXP_STATUS["$name"]=preview
        EXP_TEE_STATUS["$name"]=preview
        return 0
    fi

    local -a ps
    set +e
    "${command[@]}" 2>&1 | tee "$log"
    ps=("${PIPESTATUS[@]}")
    set -e
    EXP_STATUS["$name"]=${ps[0]}
    EXP_TEE_STATUS["$name"]=${ps[1]}
    if [[ ${ps[0]} -eq 0 && ${ps[1]} -eq 0 ]]; then
        echo "[SUCCESS] $name"
    else
        echo "[FAILED] $name (training exit ${ps[0]}, tee exit ${ps[1]})"
        FAILED_EXPERIMENTS+=("$name")
    fi
    echo "End time: $(date)"
}

# ===== Experiment list: edit/add/remove run_exp calls here =====
# 1/4 Original single-track SEED; no global channel, EMA, or replay.
run_exp "SEED" "seed_only" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.01 \
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0 \
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=copy \
    AGENTSTREAM_SEED_OPD_GATE_EPS=0 \
    AGENTSTREAM_SEED_OPD_POSITIVE_ONLY=False \
    AGENTSTREAM_SEED_EMA_MODE=off \
    AGENTSTREAM_SEED_REPLAY_ENABLE=False

# 2/4 Dual-track OPD + global pool; no extra optimisations.
run_exp "SEED+Global" "global_clean" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=pool \
    AGENTSTREAM_SEED_OPD_GATE_EPS=0 \
    AGENTSTREAM_SEED_OPD_POSITIVE_ONLY=False \
    AGENTSTREAM_SEED_EMA_MODE=off \
    AGENTSTREAM_SEED_REPLAY_ENABLE=False

# 3/4 Global + replay only; gate eps=0, positive-only off, EMA off.
run_exp "SEED+Global+Replay" "global_replay" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=pool \
    AGENTSTREAM_SEED_OPD_GATE_EPS=0 \
    AGENTSTREAM_SEED_OPD_POSITIVE_ONLY=False \
    AGENTSTREAM_SEED_EMA_MODE=off \
    AGENTSTREAM_SEED_REPLAY_ENABLE=True

# 4/4 Global + gate eps + positive-only + EMA KL reference (NOT teacher) + replay.
run_exp "SEED+Global+Opt" "global_opt" \
    AGENTSTREAM_SEED_OPD_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=0.005 \
    AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=pool \
    AGENTSTREAM_SEED_OPD_GATE_EPS=0.05 \
    AGENTSTREAM_SEED_OPD_POSITIVE_ONLY=True \
    AGENTSTREAM_SEED_EMA_MODE=ref \
    AGENTSTREAM_SEED_REPLAY_ENABLE=True

echo ""
echo "======================================================================"
echo "Stage-3 Experiment Summary (run $RUN_ID, end $(date))"
for name in "${EXPERIMENTS[@]}"; do
    printf '  %-20s training=%s tee=%s log=%s\n' \
        "$name" "${EXP_STATUS[$name]}" "${EXP_TEE_STATUS[$name]}" "${EXP_LOG[$name]}"
done
echo "======================================================================"
if [[ "$DRY_RUN_SUITE" == true ]]; then
    echo "Preview only; no launcher, retriever, or training was started."
elif (( ${#FAILED_EXPERIMENTS[@]} > 0 )); then
    printf 'Failed experiment: %s\n' "${FAILED_EXPERIMENTS[@]}"
    exit 1
else
    echo "All Stage-3 experiments finished successfully."
fi