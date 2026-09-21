#!/usr/bin/env bash
# Experiment B: offline delta test (examples/agentstream_trainer/delta_test.py) as one PBS line.
# Does an in-context demonstration raise the frozen SFT policy's success rate? C0 = plain prompt;
# C2 = the task's own shortest successful C0 rollout as a "Reference Solution" (same-task upper bound);
# C4 = the most similar other task's demonstration, final step dropped (transfer, read the c0_all_fail
# row); delta = success(Cx) - success(C0), paired over tasks. Then the token-level companion
# (delta_gap.py): does the C2 reference raise the likelihood of the policy's own C0 trajectories, with
# structure around the divergence step -- the question SEED's OPD loss actually depends on.
# No training, no Ray: one vLLM endpoint for the SFT checkpoint + exgentic sessions for the rollouts
# (same layout as scripts/sft/agentstream/run_all.sh); the gap phase loads the checkpoint with
# transformers on one GPU after the endpoint is stopped.
#
# Usage: bash examples/agentstream_trainer/run_delta_test.sh [--dry-run]
#   --dry-run prints the resolved settings and the commands without starting anything.
# Overrides (env, all optional):
#   DELTA_PHASE=all|c0|materials|conditions|report|gap   (all = rollout phases, report, then gap)
#   DELTA_CONDITIONS=C2,C4                  DELTA_RUN_GAP=true   DELTA_GAP_GPU=<first of DELTA_GPUS>
#   DELTA_NUM_TASKS=16  DELTA_ROLLOUTS=8  DELTA_PARALLEL_SESSIONS=8
#   DELTA_OUTPUT_DIR=<dir>                  DELTA_MODEL_DIR=$AGENTSTREAM_SFT_MODEL_DIR (the RL start checkpoint)
#   DELTA_GPUS=$AGENTSTREAM_POLICY_GPU      DELTA_TP=$AGENTSTREAM_POLICY_TP   (vLLM replicas = GPUs / TP)
#   POLICY_BASE_URL=http://host:port/v1     reuse a running endpoint (nothing is started; POLICY_MODEL=sft)
# Sampling mirrors Stage-3 RL (agentstream_full.env): temperature 1.0, MAX_RESPONSE_LENGTH tokens,
# history_length, AGENTSTREAM_MAX_STEPS, the RL benchmark kwargs (tau2 user simulator, browsecomp
# retriever URL), think optional / tool_call accepted. The Stage-1 per-benchmark caps
# (AGENTSTREAM_MAX_STEPS_JSON / _OBS_MAX_CHARS_JSON) are NOT applied: RL does not use them.
# Phases resume from their outputs; re-submit the same job to continue. Activate Conda in the caller.
# Results: <DELTA_OUTPUT_DIR>/delta_report.md, delta_per_task_<C>.csv, materials_<C>.jsonl, C0/ C2/ C4/
# rollouts, gap_report.md, gap_per_trajectory.csv.

set -eo pipefail

DRY_RUN=false
case "${1:-}" in
    --dry-run) DRY_RUN=true; shift ;;
    "") ;;
    *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
(( $# == 0 )) || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Machine paths / keys, then the public config (no xtrace: .env holds keys).
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
AGENTSTREAM_CONFIG="${AGENTSTREAM_CONFIG:-$SCRIPT_DIR/agentstream_full.env}"
set -a
# shellcheck disable=SC1090
source "$AGENTSTREAM_CONFIG"
set +a
# exgentic cache settings shared by every entry point that starts exgentic processes.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common/exgentic_env.sh"
export PYTHONUNBUFFERED=1

# ===== Settings =====
: "${CONDA_ENV:=seed}"
: "${VLLM_CONDA_ENV:=$CONDA_ENV}"
DELTA_PHASE="${DELTA_PHASE:-all}"
DELTA_CONDITIONS="${DELTA_CONDITIONS:-C2,C4}"
DELTA_RUN_GAP="${DELTA_RUN_GAP:-true}"
DELTA_NUM_TASKS="${DELTA_NUM_TASKS:-16}"
DELTA_ROLLOUTS="${DELTA_ROLLOUTS:-8}"
DELTA_PARALLEL_SESSIONS="${DELTA_PARALLEL_SESSIONS:-${AGENTSTREAM_SFT_PARALLEL_SESSIONS:-8}}"
DELTA_OUTPUT_DIR="${DELTA_OUTPUT_DIR:-$AGENTSTREAM_SEED_ROOT/outputs/delta_test_${AGENTSTREAM_MODEL_TAG}_${AGENTSTREAM_RUN_VERSION}}"
DELTA_MODEL_DIR="${DELTA_MODEL_DIR:-$AGENTSTREAM_SFT_MODEL_DIR}"
DELTA_GPUS="${DELTA_GPUS:-$AGENTSTREAM_POLICY_GPU}"
DELTA_TP="${DELTA_TP:-${AGENTSTREAM_POLICY_TP:-1}}"
POLICY_MODEL="${POLICY_MODEL:-sft}"
RUN_ID="${PBS_JOBID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$PROJECT_ROOT/logs/agentstream"
DELTA_GAP_GPU="${DELTA_GAP_GPU:-${DELTA_GPUS%%,*}}"

# Policy endpoint: reuse POLICY_BASE_URL when given, else one vLLM replica per DELTA_TP GPUs.
IFS=',' read -ra _gpus <<< "$DELTA_GPUS"
replicas=$(( ${#_gpus[@]} / DELTA_TP )); (( replicas >= 1 )) || replicas=1
if [[ -n "${POLICY_BASE_URL:-}" ]]; then
    policy_base_url="$POLICY_BASE_URL"; start_server=false
else
    _urls=()
    for ((i = 0; i < replicas; i++)); do _urls+=("http://${AGENTSTREAM_POLICY_HOST}:$((AGENTSTREAM_POLICY_PORT + i))/v1"); done
    policy_base_url=$(IFS=','; echo "${_urls[*]}"); start_server=true
fi

rollout_phase="$DELTA_PHASE"; [[ "$DELTA_PHASE" == gap ]] && rollout_phase=""
command=(
    python examples/agentstream_trainer/delta_test.py
    --env-file /dev/null --phase "${rollout_phase:-all}" --conditions "$DELTA_CONDITIONS" --output-dir "$DELTA_OUTPUT_DIR"
    --exgentic-root "$AGENTSTREAM_EXGENTIC_ROOT" --runner "${AGENTSTREAM_RUNNER:-venv}"
    --benchmarks "$AGENTSTREAM_BENCHMARKS" --benchmark-kwargs-json "$AGENTSTREAM_BENCHMARK_KWARGS_JSON"
    --num-tasks-per-benchmark "$DELTA_NUM_TASKS"
    --holdout-after-tasks "$AGENTSTREAM_NUM_TASKS" --holdout-tasks-per-benchmark "$AGENTSTREAM_VAL_TASKS"
    --rollouts-per-task "$DELTA_ROLLOUTS" --parallel-sessions "$DELTA_PARALLEL_SESSIONS"
    --max-steps "$AGENTSTREAM_MAX_STEPS" --history-length "$history_length"
    --policy-base-url "$policy_base_url" --policy-model "$POLICY_MODEL"
    --policy-temperature 1.0 --policy-max-completion-tokens "$MAX_RESPONSE_LENGTH"
    ${POLICY_EXTRA_BODY_JSON:+--policy-extra-body-json "$POLICY_EXTRA_BODY_JSON"}
)
gap_command=(python examples/agentstream_trainer/delta_gap.py --output-dir "$DELTA_OUTPUT_DIR" --model "$DELTA_MODEL_DIR" --condition C2)
run_gap() {
    echo "Gap check on GPU $DELTA_GAP_GPU: ${gap_command[*]}"
    CUDA_VISIBLE_DEVICES="$DELTA_GAP_GPU" "${gap_command[@]}" 2>&1 | tee -a "$log"
}

echo "Delta test (experiment B)"
echo "  phase / conditions       : $DELTA_PHASE / $DELTA_CONDITIONS (gap after rollouts: $DELTA_RUN_GAP)"
echo "  tasks / rollouts         : $DELTA_NUM_TASKS per benchmark / $DELTA_ROLLOUTS per task"
echo "  benchmarks               : $AGENTSTREAM_BENCHMARKS (RL train tasks minus holdout ids [$AGENTSTREAM_NUM_TASKS, $((AGENTSTREAM_NUM_TASKS + AGENTSTREAM_VAL_TASKS))))"
echo "  model                    : $DELTA_MODEL_DIR"
echo "  endpoint                 : $policy_base_url ($([[ $start_server == true ]] && echo "start $replicas vLLM replica(s) on GPUs $DELTA_GPUS, TP $DELTA_TP" || echo reuse))"
echo "  sampling                 : temperature 1.0, $MAX_RESPONSE_LENGTH tokens, history $history_length, max_steps $AGENTSTREAM_MAX_STEPS, vLLM max_model_len $SEED_ANALYSIS_MAX_MODEL_LEN, extra body ${POLICY_EXTRA_BODY_JSON:-none}"
echo "  output                   : $DELTA_OUTPUT_DIR"
if [[ "$DRY_RUN" == true ]]; then
    [[ "$DELTA_PHASE" == gap ]] || { printf 'Command:'; printf ' %q' "${command[@]}"; printf '\n'; }
    [[ "$DELTA_PHASE" == gap || ( "$DELTA_PHASE" == all && "$DELTA_RUN_GAP" == true ) ]] && { printf 'Gap command (GPU %s):' "$DELTA_GAP_GPU"; printf ' %q' "${gap_command[@]}"; printf '\n'; }
    exit 0
fi

mkdir -p "$LOG_DIR" "$DELTA_OUTPUT_DIR/logs"
log="$LOG_DIR/delta_test_${RUN_ID}.log"
echo "Log: $log"
if [[ "$DELTA_PHASE" == gap ]]; then   # scoring only: no endpoint, no exgentic
    [[ -f "$DELTA_MODEL_DIR/config.json" ]] || { echo "SFT model not found: $DELTA_MODEL_DIR" >&2; exit 1; }
    run_gap
    exit 0
fi
[[ -d "$AGENTSTREAM_EXGENTIC_ROOT/src/exgentic" ]] || { echo "Exgentic checkout not found: $AGENTSTREAM_EXGENTIC_ROOT" >&2; exit 1; }

# ===== vLLM endpoint (same flags as scripts/sft/agentstream/run_all.sh, model = the RL start checkpoint) =====
server_pids=()
policy_ready() {
    local url; IFS=',' read -ra _check <<< "$policy_base_url"
    for url in "${_check[@]}"; do curl -fsS "$url/models" >/dev/null 2>&1 || return 1; done
}
stop_policy_server() {
    # Replicas run under setsid, so each pid is a process group: killing the group reaches vllm and its
    # engine workers (a plain kill only stops the `conda run` wrapper and leaves the GPU occupied).
    (( ${#server_pids[@]} )) || return 0
    echo "Stopping delta-test vLLM replica(s): ${server_pids[*]}"
    local pid deadline=$((SECONDS + 60))
    for pid in "${server_pids[@]}"; do kill -TERM -- "-$pid" "$pid" >/dev/null 2>&1 || true; done
    for pid in "${server_pids[@]}"; do
        while kill -0 -- "-$pid" >/dev/null 2>&1 && (( SECONDS < deadline )); do sleep 1; done
        kill -KILL -- "-$pid" >/dev/null 2>&1 || true
        wait "$pid" >/dev/null 2>&1 || true
    done
    server_pids=()
}
trap stop_policy_server EXIT
if [[ "$start_server" == true ]] && ! policy_ready; then
    [[ -f "$DELTA_MODEL_DIR/config.json" ]] || { echo "SFT model not found: $DELTA_MODEL_DIR" >&2; exit 1; }
    echo "Starting $replicas vLLM replica(s) for $DELTA_MODEL_DIR"
    for ((i = 0; i < replicas; i++)); do
        replica_gpus=$(IFS=','; echo "${_gpus[*]:$((i * DELTA_TP)):$DELTA_TP}")
        CUDA_VISIBLE_DEVICES="$replica_gpus" setsid conda run -n "$VLLM_CONDA_ENV" --no-capture-output \
            vllm serve "$DELTA_MODEL_DIR" --host "$AGENTSTREAM_POLICY_HOST" --port "$((AGENTSTREAM_POLICY_PORT + i))" \
            --served-model-name "$POLICY_MODEL" --tensor-parallel-size "$DELTA_TP" \
            --gpu-memory-utilization "$AGENTSTREAM_POLICY_GPU_MEMORY_UTILIZATION" --max-model-len "$SEED_ANALYSIS_MAX_MODEL_LEN" \
            >"$DELTA_OUTPUT_DIR/logs/policy_vllm_${i}.log" 2>&1 &
        server_pids+=($!)
    done
    deadline=$((SECONDS + AGENTSTREAM_VLLM_STARTUP_TIMEOUT))
    until policy_ready; do
        for pid in "${server_pids[@]}"; do
            kill -0 "$pid" >/dev/null 2>&1 || { echo "A vLLM replica exited early; see $DELTA_OUTPUT_DIR/logs/policy_vllm_*.log" >&2; exit 1; }
        done
        (( SECONDS < deadline )) || { echo "Timed out waiting for vLLM; see $DELTA_OUTPUT_DIR/logs/policy_vllm_*.log" >&2; exit 1; }
        sleep 2
    done
elif ! policy_ready; then
    echo "Policy endpoint not reachable: $policy_base_url" >&2; exit 1
fi
echo "Policy endpoint ready: $policy_base_url"

# browsecomp retriever (no-op unless browsecompplus is listed; reuses a healthy server).
bash "$SCRIPT_DIR/ensure_browsecomp_retriever.sh"

"${command[@]}" 2>&1 | tee "$log"
if [[ "$DELTA_PHASE" == all && "$DELTA_RUN_GAP" == true ]]; then
    stop_policy_server   # frees the GPU (waits for the whole vLLM process group) before the transformers forward
    run_gap
fi
