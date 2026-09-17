#!/usr/bin/env bash
# Formal Stage 1 (hindsight-skill SFT data) and Stage 2 (SFT training + HF export) on the
# five-benchmark AgentStream stream. The exported checkpoint is the shared starting point of
# every Stage-3 run (baselines and our method). Activate Conda in the caller (the PBS wrapper does).
#
# Usage: bash examples/agentstream_trainer/run_stage12.sh [--dry-run] [--reuse-rollouts DIR] [prepare|sft|all]
#   prepare  Stage 1 only: local vLLM rollouts -> GLM skill annotation -> parquet (resumable)
#   sft      Stage 2 only: SFT on the Stage-1 parquet, export to AGENTSTREAM_SFT_MODEL_DIR
#   all      both (default); Stage 2 starts only if the Stage-1 metrics check passes
#   --reuse-rollouts DIR  seed a fresh Stage-1 data dir with DIR's task list and rollouts, so only
#            the skill annotation, parquet export and SFT are redone. Typical use: a second
#            skill schema on the same rollouts, e.g.
#            AGENTSTREAM_SEED_SKILL_MODE=episode_step bash run_stage12.sh --reuse-rollouts outputs/<episode_only dir> all
# Stage 1 needs the judge/user-simulator API keys and HF_TOKEN from .env; with browsecompplus
# in the list the shared retriever is started here (CPU by default, see agentstream_full.env).
# Submit with four GPUs (select=1:ncpus=48:ngpus=4): four vLLM replicas for Stage 1, four SFT ranks.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN_STAGE=false
REUSE_ROLLOUTS=""
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --dry-run) DRY_RUN_STAGE=true; shift ;;
        --reuse-rollouts) REUSE_ROLLOUTS="${2:?--reuse-rollouts needs a directory}"; shift 2 ;;
        *) echo "Unknown option $1" >&2; exit 2 ;;
    esac
done
stage="${1:-all}"
case "$stage" in
    prepare) RUN_PREPARE=true;  RUN_SFT=false ;;
    sft)     RUN_PREPARE=false; RUN_SFT=true ;;
    all)     RUN_PREPARE=true;  RUN_SFT=true ;;
    *) echo "Usage: $0 [--dry-run] [prepare|sft|all]" >&2; exit 2 ;;
esac

# Machine paths/credentials once (no xtrace: .env holds keys), then the public config.
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
export ENV_FILE=/dev/null PYTHONUNBUFFERED=1

# ===== Formal settings: must match run_baseline_suite.sh / run_global_ablation.sh =====
export AGENTSTREAM_BENCHMARKS=bfcl,appworld,tau2,hle,browsecompplus
# Stage 1/2 run on four GPUs (PBS: ngpus=4); Stage 3 keeps its own two-GPU setting.
export AGENTSTREAM_POLICY_GPU=0,1,2,3 AGENTSTREAM_SFT_GPUS=0,1,2,3 AGENTSTREAM_SFT_NPROC=4
export CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-seed}}"   # vLLM + SFT run in the active env
export AGENTSTREAM_RUN_PREPARE=$RUN_PREPARE AGENTSTREAM_RUN_SFT=$RUN_SFT AGENTSTREAM_RUN_RL=false

export AGENTSTREAM_CONFIG="${AGENTSTREAM_CONFIG:-$SCRIPT_DIR/agentstream_full.env}"
set -a
# shellcheck disable=SC1090
source "$AGENTSTREAM_CONFIG"
set +a

if [[ -n "$REUSE_ROLLOUTS" ]]; then
    for f in sampled_tasks.jsonl baseline_rollouts.jsonl; do
        [[ -s "$REUSE_ROLLOUTS/$f" ]] || { echo "--reuse-rollouts: $REUSE_ROLLOUTS/$f missing or empty" >&2; exit 1; }
    done
    [[ "$(cd "$REUSE_ROLLOUTS" && pwd)" == "$AGENTSTREAM_SFT_DATA_DIR" ]] && { echo "--reuse-rollouts must point at a different data dir" >&2; exit 1; }
fi
RUN_ID="${PBS_JOBID:-local_$(date +%Y%m%d_%H%M%S)_$$}"
LOG="$PROJECT_ROOT/logs/agentstream/stage12_${stage}_${RUN_ID}.log"

echo "======================================================================"
echo "AgentStream Stage 1/2 ($stage)   start $(date)"
echo "  benchmarks : $AGENTSTREAM_BENCHMARKS  (SFT tasks/domain=$AGENTSTREAM_SFT_NUM_TASKS minus RL holdout ids[$AGENTSTREAM_NUM_TASKS:$((AGENTSTREAM_NUM_TASKS + AGENTSTREAM_VAL_TASKS))], rollouts/task=$AGENTSTREAM_SFT_ROLLOUTS_PER_TASK)"
echo "  base model : $AGENTSTREAM_BASE_MODEL_PATH"
echo "  skill mode : $AGENTSTREAM_SEED_SKILL_MODE  (step skills <= $AGENTSTREAM_SEED_ANALYSIS_MAX_STEP_SKILLS, analysis tokens $AGENTSTREAM_SEED_ANALYSIS_MAX_COMPLETION_TOKENS)"
echo "  SFT data   : $AGENTSTREAM_SFT_DATA_DIR${REUSE_ROLLOUTS:+  (rollouts reused from $REUSE_ROLLOUTS)}"
echo "  SFT export : $AGENTSTREAM_SFT_MODEL_DIR  (epochs=$AGENTSTREAM_SFT_EPOCHS)"
echo "  conda env  : $CONDA_ENV    policy GPUs: $AGENTSTREAM_POLICY_GPU    SFT GPUs: $AGENTSTREAM_SFT_GPUS (nproc=$AGENTSTREAM_SFT_NPROC, batch=$AGENTSTREAM_SFT_TRAIN_BATCH_SIZE)"
echo "  log        : $LOG"
echo "======================================================================"
[[ "$DRY_RUN_STAGE" == true ]] && { echo "Dry run complete; nothing was started."; exit 0; }
mkdir -p "$(dirname "$LOG")"

check_stage1() {  # Stage-1 acceptance: non-empty rollouts, parsed skills and SFT records.
    python - "$AGENTSTREAM_SFT_DATA_DIR" <<'PY'
import json, sys
from pathlib import Path
m = json.loads((Path(sys.argv[1]) / "metrics.json").read_text())
print(json.dumps(m, indent=2, ensure_ascii=False))
assert m["baseline_rollouts"] > 0 and m["parse_ok_skills"] > 0 and m["sft_records"] > 1
print("Stage 1: PASS")
PY
}

{
    if [[ "$RUN_PREPARE" == true ]]; then
        if [[ -n "$REUSE_ROLLOUTS" ]]; then
            # Task list + rollouts are the expensive, schema-independent part of Stage 1; with
            # RESUME=true the pipeline skips rollouts it already has and re-annotates from scratch.
            mkdir -p "$AGENTSTREAM_SFT_DATA_DIR"
            for f in sampled_tasks.jsonl baseline_rollouts.jsonl; do
                [[ -e "$AGENTSTREAM_SFT_DATA_DIR/$f" ]] || cp -a "$REUSE_ROLLOUTS/$f" "$AGENTSTREAM_SFT_DATA_DIR/$f"
            done
        fi
        bash "$SCRIPT_DIR/ensure_browsecomp_retriever.sh"   # no-op unless browsecompplus is listed
        AGENTSTREAM_RUN_SFT=false bash scripts/sft/agentstream/run_all.sh
        check_stage1
    fi
    if [[ "$RUN_SFT" == true ]]; then
        # SFT only: 60k-token samples fragment the CUDA allocator (tens of GB reserved but unusable).
        # Kept off the Stage-1 vLLM servers, whose allocator rejects expandable segments.
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            AGENTSTREAM_RUN_PREPARE=false bash scripts/sft/agentstream/run_all.sh
        test -f "$AGENTSTREAM_SFT_MODEL_DIR/config.json" || { echo "Stage 2: export missing at $AGENTSTREAM_SFT_MODEL_DIR" >&2; exit 1; }
        echo "Stage 2: exported $AGENTSTREAM_SFT_MODEL_DIR"
    fi
    echo "Stage 1/2 ($stage) finished $(date)"
} 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
