#!/usr/bin/env bash
# Our method (run_ours_method.sh = arm E5 of run_ours_debug.sh) on the other model / stream settings of the
# baseline suite: one setting per entry, the stream parameters and experiment names of run_baseline_suite.sh,
# so every run lines up with its baselines on wandb (ours_<model>_..._<mode>_online_s44[_<benchmark>]).
#   S1  Qwen3-4B-Instruct-2507, isolated    -> 3 runs of 7 steps (one per benchmark)
#   S2  Qwen2.5-7B-Instruct,    interleaved -> 1 run of 20 steps
#   S3  Qwen2.5-7B-Instruct,    isolated    -> 3 runs of 7 steps
# The 4B interleaved run of the method is job 71186's E5, continued to 20 steps by run_ours_final.sh (F1).
#
# Usage: bash examples/agentstream_trainer/run_ours_suite.sh [--dry-run]
#   OURS_JOBS=S2 bash ...          # subset (default S1,S2,S3; the settings are independent)
# Two nodes: run_ours_suite_node1.sh = S1,S2 (~27-31 h) and run_ours_suite_node2.sh = S3 (~17-21 h).
#   OURS_SAVE_FREQ=10 bash ...     # checkpoint every N steps (default 0 = none; a 7B checkpoint is ~91 GB, a 4B
#                                  # one ~48 GB, and an isolated setting keeps one per benchmark run)
# Model path and step counts derive from the model name and agentstream_full.env (interleaved 20, isolated
# 7 per benchmark; inherited AGENTSTREAM_SFT_MODEL_DIR / BASE_MODEL_PATH / RL_EPOCHS / RL_ISOLATED_EPOCHS /
# MODEL_TAG are cleared so they cannot redirect a setting), and the holdout is validated at the last step,
# exactly as for the baselines. trainer.resume_mode=auto only acts when a checkpoint exists. An interrupted
# isolated setting is completed by rerunning it with AGENTSTREAM_BENCHMARKS=<missing benchmark> (its runs
# keep their names: the isolated step count does not depend on the benchmark list); do not narrow the
# benchmarks of an interleaved setting, that changes the stream. Time: S1 ~11 h (31 min/step at 4B),
# S2 ~16-20 h, S3 ~17-21 h -> walltime >= 24 h (S3: 30 h). Activate Conda in the caller.

set -eo pipefail

[[ "${1:-}" == --dry-run ]] && { export DRY_RUN=true; shift; }
(( $# == 0 )) || { echo "Usage: $0 [--dry-run]   (OURS_JOBS / OURS_SAVE_FREQ via the environment)" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Machine paths/credentials once (no xtrace: .env holds keys).
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
export ENV_FILE=/dev/null PYTHONUNBUFFERED=1

# Stream settings of run_baseline_suite.sh; the step count per mode derives from agentstream_full.env.
COMMON_ENV=(
    "AGENTSTREAM_BENCHMARKS=${AGENTSTREAM_BENCHMARKS:-bfcl,tau2,browsecompplus}"  # env override = complete an isolated setting
    AGENTSTREAM_RL_STREAM_PROFILE=single_pass
    AGENTSTREAM_RL_TRAIN_DATA_SIZE=10
    "AGENTSTREAM_RL_SAVE_FREQ=${OURS_SAVE_FREQ:-0}"
)
# Everything a setting must derive itself (from AGENTSTREAM_BASE_MODEL_NAME and agentstream_full.env).
DERIVED=(AGENTSTREAM_MODEL_TAG AGENTSTREAM_SFT_MODEL_DIR AGENTSTREAM_BASE_MODEL_PATH AGENTSTREAM_RL_EPOCHS AGENTSTREAM_RL_ISOLATED_EPOCHS
         AGENTSTREAM_EXPERIMENT_PREFIX EXPERIMENT_NAME EXPERIMENT_NAME_PREFIX DEFAULT_LOCAL_DIR)
UNSET_DERIVED=("${DERIVED[@]/#/-u }")  # "-u NAME" pairs for env
# id | AGENTSTREAM_BASE_MODEL_NAME | stream mode
JOBS=(
    "S1|Qwen3-4B-Instruct-2507|isolated"
    "S2|Qwen2.5-7B-Instruct|interleaved"
    "S3|Qwen2.5-7B-Instruct|isolated"
)
OURS_JOBS="${OURS_JOBS:-S1,S2,S3}"
RUN_ID="${PBS_JOBID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/logs/agentstream"
mkdir -p "$LOG_DIR"
declare -A STATUS
NAMES=()

experiment_prefix() {  # <model> <mode> -> the baseline suite's naming (run_baseline_suite.sh SUITE_TAG) for this setting
    # shellcheck disable=SC2068  # UNSET_DERIVED holds "-u NAME" pairs on purpose
    env ${UNSET_DERIVED[@]} "${COMMON_ENV[@]}" "AGENTSTREAM_BASE_MODEL_NAME=$1" bash -c '
        set -eu
        set -a; source "$1" >/dev/null; set +a
        steps=$AGENTSTREAM_RL_EPOCHS; [[ "$2" == isolated ]] && steps=$AGENTSTREAM_RL_ISOLATED_EPOCHS
        echo "ours_${AGENTSTREAM_MODEL_TAG}_agentstream_${AGENTSTREAM_RUN_VERSION}${_as_skill_suffix:-}_n${AGENTSTREAM_NUM_TASKS}_${AGENTSTREAM_RL_STREAM_PROFILE}_b${AGENTSTREAM_RL_TRAIN_DATA_SIZE}_steps${steps}"
    ' _ "$SCRIPT_DIR/agentstream_full.env" "$2"
}

run_setting() {  # <job spec>
    local id model mode prefix
    IFS='|' read -r id model mode <<< "$1"
    if [[ ",$OURS_JOBS," != *",$id,"* ]]; then
        echo "[SKIP] $id $model $mode (not in OURS_JOBS=$OURS_JOBS)"
        return 0
    fi
    prefix="$(experiment_prefix "$model" "$mode")" || { echo "Cannot derive the experiment name for $model / $mode from agentstream_full.env." >&2; exit 1; }
    [[ "$prefix" == ours_*_agentstream_*_steps[0-9]* && "$prefix" != *__* ]] || { echo "Malformed experiment prefix: '$prefix'" >&2; exit 1; }
    local log="$LOG_DIR/ours_${mode}_${model}_${RUN_ID}.log"
    NAMES+=("$id $model $mode")
    echo ""
    echo "======================================================================"
    echo "[Setting] $id $model / $mode   experiment=${prefix}_${mode}_online_s44   start $(date)"
    echo "Log: $log"
    echo "======================================================================"
    set +e
    # shellcheck disable=SC2068
    env ${UNSET_DERIVED[@]} "${COMMON_ENV[@]}" "AGENTSTREAM_BASE_MODEL_NAME=$model" "AGENTSTREAM_EXPERIMENT_PREFIX=$prefix" \
        bash "$SCRIPT_DIR/run_ours_method.sh" "$mode" trainer.resume_mode=auto 2>&1 | tee "$log"
    STATUS["$id $model $mode"]=${PIPESTATUS[0]}
    set -e
    echo "[$([[ ${STATUS["$id $model $mode"]} -eq 0 ]] && echo SUCCESS || echo FAILED)] $id   end $(date)"
}

for spec in "${JOBS[@]}"; do
    run_setting "$spec"
done

echo ""
echo "======================================================================"
echo "Method suite summary (run $RUN_ID, jobs=$OURS_JOBS, end $(date))"
failed=0
for name in "${NAMES[@]}"; do
    printf '  %-45s exit=%s\n' "$name" "${STATUS[$name]}"
    (( STATUS[$name] == 0 )) || failed=1
done
echo "======================================================================"
exit $failed
