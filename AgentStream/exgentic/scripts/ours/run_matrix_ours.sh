#!/usr/bin/env bash
# Launch the AgentStream-baseline matrix on the SEED task stream: AGENTS x MODELS x MODES, each run =
# online pass (192 tasks, learning on) followed by the frozen holdout pass (96 tasks); everything is
# logged to wandb in real time with the SEED metric names (online/*, val/*).
#
# Usage (from AgentStream/exgentic, with OPENROUTER_API_KEY / WANDB_API_KEY in the environment):
#   bash scripts/ours/run_matrix_ours.sh                 # full matrix, PARALLEL runs at a time
#   ONLY=reasoning_bank:qwen:interleaved bash ...        # one run (agent:model-key:mode)
#   DRY_RUN=1 bash ...                                   # print the commands
# Environment:
#   MODELS      model keys, comma list (default: qwen,luna,dsflash); keys map to litellm ids below
#   AGENTS      tool_calling,reasoning_bank,ace
#   MODES       interleaved,isolated
#   PARALLEL    concurrent runs (default 6; keep <= 8 for the shared browsecomp retriever / GLM rate limit)
#   QWEN_API_BASE   the local vLLM endpoint serving Qwen3-4B-Instruct-2507 (serve_policy_vllm.sh)
#   JUDGE_MODEL / RETRIEVER_URL   forwarded to the benchmark registry (defaults = the SEED runs')
#   EXGENTIC_PYTHON interpreter (default: "uv run python")
#   EXTRA_ARGS_<KEY>  replace a model's default extra args (e.g. EXTRA_ARGS_DSFLASH="" if the provider rejects reasoning_effort)
#   RUN_NAME_PREFIX prefix for the wandb run names (e.g. smoke_), default none
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODELS="${MODELS:-qwen,luna,dsflash}"
AGENTS="${AGENTS:-tool_calling,reasoning_bank,ace}"
MODES="${MODES:-interleaved,isolated}"
PARALLEL="${PARALLEL:-6}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs_ours}"
PY="${EXGENTIC_PYTHON:-uv run python}"
QWEN_API_BASE="${QWEN_API_BASE:-http://127.0.0.1:8000/v1}"

model_id() {  # key -> litellm model id
    case "$1" in
        qwen)    echo "openai/Qwen3-4B-Instruct-2507" ;;
        luna)    echo "openrouter/openai/gpt-6-luna" ;;
        dsflash) echo "openrouter/deepseek/deepseek-v4.1-flash" ;;
        *)       echo "$1" ;;
    esac
}
model_args() {  # key -> extra runner args; EXTRA_ARGS_<KEY> (upper-case) overrides the default
    local override="EXTRA_ARGS_${1^^}"
    if [[ -n "${!override:-}" ]]; then echo "${!override}"; return; fi
    case "$1" in
        qwen) echo "--api-base $QWEN_API_BASE" ;;
        *)    echo "--reasoning-effort low" ;;
    esac
}

common=()
[[ -n "${JUDGE_MODEL:-}" ]] && common+=(--judge-model "$JUDGE_MODEL")
[[ -n "${RETRIEVER_URL:-}" ]] && common+=(--retriever-url "$RETRIEVER_URL")
[[ -n "${NUM_TASKS:-}" ]] && common+=(--num-tasks "$NUM_TASKS")
[[ -n "${NUM_HOLDOUT:-}" ]] && common+=(--num-holdout "$NUM_HOLDOUT")
[[ "${NO_WANDB:-0}" == 1 ]] && common+=(--no-wandb)

mkdir -p "$OUTPUT_BASE/logs"
cmds=()
IFS=, read -ra agents <<< "$AGENTS"; IFS=, read -ra models <<< "$MODELS"; IFS=, read -ra modes <<< "$MODES"
for agent in "${agents[@]}"; do for key in "${models[@]}"; do for mode in "${modes[@]}"; do
    run="$agent:$key:$mode"
    [[ -n "${ONLY:-}" && ",$ONLY," != *",$run,"* ]] && continue
    out="$OUTPUT_BASE/${agent}_${key}_${mode}_s44"
    mid="$(model_id "$key")"
    name_args=""
    [[ -n "${RUN_NAME_PREFIX:-}" ]] && name_args="--run-name ${RUN_NAME_PREFIX}as_${agent}_${mid##*/}_${mode}_s44"
    base="$PY scripts/ours/stream_ours.py PASS --agent $agent --model $mid --mode $mode --output-dir $out $(model_args "$key") $name_args ${common[*]}"
    cmds+=("{ ${base/PASS/online} && ${base/PASS/holdout}; } > $OUTPUT_BASE/logs/${agent}_${key}_${mode}.log 2>&1")
done; done; done

printf '%s\n' "${cmds[@]}"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
echo "Launching ${#cmds[@]} runs, $PARALLEL at a time (logs: $OUTPUT_BASE/logs)"
printf '%s\0' "${cmds[@]}" | xargs -0 -P "$PARALLEL" -I{} bash -c '{}'
