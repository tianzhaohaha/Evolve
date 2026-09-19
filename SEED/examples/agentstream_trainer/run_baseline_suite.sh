#!/usr/bin/env bash
# PBS/local Stage-3 baseline suite: runs the SEED-paper baselines one after another on the
# three-benchmark AgentStream stream (bfcl, tau2, browsecompplus; appworld and hle carry no
# signal for a 4B policy) from the shared SFT checkpoint. Activate Conda in the caller (the PBS
# wrapper does).
#
# Usage: [STREAM_MODE=interleaved|isolated|sequential] bash examples/agentstream_trainer/run_baseline_suite.sh [--dry-run]
#
# isolated runs one independent training run per benchmark (weights never shared); the
# launcher names them <experiment>_<benchmark> and a baseline counts as complete only when
# every benchmark run has checkpointed its last step.
#
# Every baseline uses identical shared settings (below) and differs only in the objective
# (see run_agentstream_baseline.sh). Runs are resumable and idempotent: a baseline whose
# checkpoint directory already reports TOTAL_STEPS is skipped, a partial one resumes from
# its last checkpoint, so re-submitting the same PBS job after a walltime kill continues
# where it stopped. Edit BASELINES / the shared settings, commit, pull on the cluster, qsub.

set -eo pipefail

DRY_RUN_SUITE=false
case "${1:-}" in
    --dry-run) DRY_RUN_SUITE=true ;;
    "") ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Machine paths/credentials once, before the suite overrides (no xtrace: .env holds keys).
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
export ENV_FILE=/dev/null
export PYTHONUNBUFFERED=1

# ===== Shared experiment settings: edit here (must match the runs of our own method) =====
BATCH_SIZE=10          # tasks per RL step (x GROUP_SIZE rollouts); 3 x 64 tasks / 10 -> 20 steps (last one padded from the tail)
STREAM_MODE="${STREAM_MODE:-interleaved}"
BASELINES=(vanilla grpo seed opsd rlsd)
COMMON_ENV=(
    AGENTSTREAM_BENCHMARKS=bfcl,tau2,browsecompplus
    AGENTSTREAM_RL_STREAM_PROFILE=single_pass
    "AGENTSTREAM_RL_TRAIN_DATA_SIZE=$BATCH_SIZE"
    # No checkpoints by default (as run_ours_debug.sh). Set AGENTSTREAM_RL_SAVE_FREQ>0 to bring
    # back checkpointing plus the suite's skip-if-complete / resume-from-last behaviour.
    "AGENTSTREAM_RL_SAVE_FREQ=${AGENTSTREAM_RL_SAVE_FREQ:-0}"
)
export "${COMMON_ENV[@]}"
# Hydra override shared by every baseline (appended last, so it wins over the launcher):
# full.env pins single_pass runs to resume_mode=disable; the suite needs auto so a re-submitted
# job continues from the last checkpoint (only effective when SAVE_FREQ>0, see COMMON_ENV).
COMMON_OVERRIDES=(trainer.resume_mode=auto)
# Per-baseline extra overrides (hydra key=value, space separated).
declare -A EXTRA_OVERRIDES=(
    [sdar]="algorithm.seed.analysis_num_workers=32"  # external analyzer: parallel API calls (OpenRouter probed fine at 32)
)

# Resolve naming (model tag, run version, task count, profile) from the public config.
export AGENTSTREAM_CONFIG="${AGENTSTREAM_CONFIG:-$SCRIPT_DIR/agentstream_full.env}"
set -a
# shellcheck disable=SC1090
source "$AGENTSTREAM_CONFIG"
set +a
# One single pass over the stream, derived by the config: ceil(benchmarks x NUM_TASKS / BATCH_SIZE)
# for a mixed stream, ceil(NUM_TASKS / BATCH_SIZE) per benchmark run when isolated.
if [[ "$STREAM_MODE" == isolated ]]; then
    STEPS_KEY=AGENTSTREAM_RL_ISOLATED_EPOCHS; TOTAL_STEPS="$AGENTSTREAM_RL_ISOLATED_EPOCHS"
    IFS=',' read -ra RUN_SUFFIXES <<< "_${AGENTSTREAM_BENCHMARKS//,/,_}"   # one run per benchmark
else
    STEPS_KEY=AGENTSTREAM_RL_EPOCHS; TOTAL_STEPS="$AGENTSTREAM_RL_EPOCHS"
    RUN_SUFFIXES=("")
fi

SUITE_TAG="${AGENTSTREAM_MODEL_TAG}_agentstream_${AGENTSTREAM_RUN_VERSION}${_as_skill_suffix}_n${AGENTSTREAM_NUM_TASKS}_${AGENTSTREAM_RL_STREAM_PROFILE}_b${BATCH_SIZE}_steps${TOTAL_STEPS}"
RUN_ID="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
LOG_DIR="$PROJECT_ROOT/logs/agentstream"
CKPT_ROOT="${CHECKPOINTS_ROOT:-$MODELS_ROOT/ckpt}"
[[ "$DRY_RUN_SUITE" == true ]] || mkdir -p "$LOG_DIR"

declare -A STATUS LOG
FAILED=()

completed_steps() {  # <experiment name> -> steps checkpointed by every run of it (0 if any is missing)
    local suffix f n min=""
    for suffix in "${RUN_SUFFIXES[@]}"; do
        f="$CKPT_ROOT/$1$suffix/latest_checkpointed_iteration.txt"
        n=$([[ -f "$f" ]] && cat "$f" || echo 0)
        [[ -z "$min" || n -lt min ]] && min=$n
    done
    echo "$min"
}

run_baseline() {  # <baseline>
    local baseline="$1"
    local prefix="${baseline}_${SUITE_TAG}"
    local exp_name="${prefix}_${STREAM_MODE}_${AGENTSTREAM_RL_PROTOCOL}_s${AGENTSTREAM_STREAM_SEED}"
    local log="$LOG_DIR/stage3_baseline_${STREAM_MODE}_b${BATCH_SIZE}_steps${TOTAL_STEPS}_${baseline}_${RUN_ID}.log"
    local done_steps
    done_steps="$(completed_steps "$exp_name")"
    LOG["$baseline"]="$log"

    # shellcheck disable=SC2206
    local -a extra=(${EXTRA_OVERRIDES[$baseline]:-})
    local -a command=(
        env -u EXPERIMENT_NAME -u EXPERIMENT_NAME_PREFIX -u DEFAULT_LOCAL_DIR
        "${COMMON_ENV[@]}" "$STEPS_KEY=$TOTAL_STEPS" "AGENTSTREAM_EXPERIMENT_PREFIX=$prefix"
        bash "$SCRIPT_DIR/run_agentstream_baseline.sh" "$baseline" "$STREAM_MODE"
        "${COMMON_OVERRIDES[@]}" "${extra[@]}"
    )

    echo ""
    echo "======================================================================"
    echo "[Baseline] $baseline   experiment=$exp_name"
    echo "Start time : $(date)   checkpointed steps: $done_steps / $TOTAL_STEPS"
    echo "Log        : $log"
    echo "======================================================================"
    if (( done_steps >= TOTAL_STEPS )); then
        echo "[SKIP] $baseline already complete"
        STATUS["$baseline"]=skipped
        return 0
    fi
    if [[ "$DRY_RUN_SUITE" == true ]]; then
        printf '%q ' "${command[@]}"; printf '\n'
        STATUS["$baseline"]=preview
        return 0
    fi

    local -a ps
    set +e
    "${command[@]}" 2>&1 | tee "$log"
    ps=("${PIPESTATUS[@]}")
    set -e
    STATUS["$baseline"]="exit ${ps[0]}"
    if [[ ${ps[0]} -eq 0 && ${ps[1]} -eq 0 ]]; then
        echo "[SUCCESS] $baseline"
    else
        echo "[FAILED] $baseline (training exit ${ps[0]}, tee exit ${ps[1]})"
        FAILED+=("$baseline")
    fi
    echo "End time: $(date)"
}

for b in "${BASELINES[@]}"; do
    run_baseline "$b"
done

echo ""
echo "======================================================================"
echo "Baseline suite summary (run $RUN_ID, end $(date))"
for b in "${BASELINES[@]}"; do
    printf '  %-10s %-9s log=%s\n' "$b" "${STATUS[$b]}" "${LOG[$b]}"
done
echo "======================================================================"
if [[ "$DRY_RUN_SUITE" == true ]]; then
    echo "Preview only; no launcher, retriever, or training was started."
elif (( ${#FAILED[@]} > 0 )); then
    printf 'Failed baseline: %s\n' "${FAILED[@]}"
    exit 1
else
    echo "All baselines finished (or were already complete)."
fi
