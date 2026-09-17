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
    AGENTSTREAM_BENCHMARKS=bfcl,appworld,tau2,hle,browsecompplus
    AGENTSTREAM_RL_STREAM_PROFILE=single_pass
    AGENTSTREAM_RL_TRAIN_DATA_SIZE=10                                     # tasks per step, as formal
    "AGENTSTREAM_RL_EPOCHS=${AGENTSTREAM_RL_EPOCHS:-4}"                   # debug: first steps of the stream
    AGENTSTREAM_RL_SAVE_FREQ=0                                            # never checkpoint
    AGENTSTREAM_SEED_SKILL_MODE=episode_only
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
