#!/usr/bin/env bash

# Reproduce the original SEED ALFWorld experiment (conda env `seed`, not `seed-as`).
# Thin wrapper: all algorithm / training settings come from SEED's own launchers
# (scripts/sft/alfworld/*, examples/seed_trainer/run_alfworld_sft_glm_self.sh); nothing from
# the AgentStream configuration (agentstream_full.env, AGENTSTREAM_*) is used.
#
# Usage:
#   [STAGE=all|sft|rl] [N_GPUS=2] bash examples/seed_trainer/run_alfworld_seed_repro.sh [hydra overrides...]
#
#   STAGE=sft  Stage 1: hindsight-skill SFT data (scripts/sft/alfworld/prepare_data.sh) + SFT
#              (train_sft.sh); exports $MODELS_ROOT/Qwen2.5-3B-Instruct-alfworld-episode-skill-sft-glm-self.
#              Needs the base model $MODELS_ROOT/Qwen2.5-3B-Instruct and OPENAI_* (teacher API) in .env.
#   STAGE=rl   Stage 2: self-evolving OPD RL (run_alfworld_sft_glm_self.sh).
#              Needs the Stage-1 checkpoint (or ALFWORLD_RL_MODEL_PATH).
#   STAGE=all  sft, then rl (default). A finished SFT export is skipped.
#
# The only deviations from the original scripts: the GPU count (their default is 8) and
# trainer.max_actor_ckpt_to_keep=1 (keep only the latest checkpoint). Extra arguments are hydra
# overrides forwarded to the RL launcher. RL resumes from its latest checkpoint, so re-submitting
# after a walltime kill continues. DRY_RUN=true prints the resolved settings without training.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# SEED's launchers source $ENV_FILE themselves. The shared .env is AgentStream-oriented (it pins
# CONDA_ENV=seed-as and many AGENTSTREAM_* switches), so hand them a filtered copy carrying only
# the machine paths and credentials the original scripts read.
SRC_ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
SEED_ENV_FILE="$(mktemp)"
trap 'rm -f "$SEED_ENV_FILE"' EXIT
if [[ -f "$SRC_ENV_FILE" ]]; then
    (
        set -a
        # shellcheck disable=SC1090
        source "$SRC_ENV_FILE"
        set +a
        for v in MODELS_ROOT CHECKPOINTS_ROOT TMPDIR RAY_TMPDIR \
                 OPENAI_BASE_URL OPENAI_API_KEY OPENAI_MODEL OPENAI_API_RETRIES OPENAI_API_RETRY_DELAY \
                 WANDB_API_KEY WANDB_MODE HF_TOKEN ALFWORLD_DATA; do
            if [[ -n "${!v:-}" ]]; then printf '%s=%q\n' "$v" "${!v}"; fi
        done
    ) > "$SEED_ENV_FILE"
fi
chmod 600 "$SEED_ENV_FILE"
export ENV_FILE="$SEED_ENV_FILE"
# shellcheck disable=SC1090
set -a; source "$SEED_ENV_FILE"; set +a

# ALFWorld game files: `alfworld-download -f` puts them in ~/.cache/alfworld, which is also what the
# configs (config_tw.yaml: $ALFWORLD_DATA/json_2.1.1/...) and prepare_data.sh expect.
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
if [[ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ]]; then
    echo "ALFWorld data not found: $ALFWORLD_DATA/json_2.1.1/train (run 'alfworld-download -f' or set ALFWORLD_DATA)" >&2
    exit 1
fi

export CONDA_ENV="${SEED_CONDA_ENV:-seed}"
export PYTHONUNBUFFERED=1

STAGE="${STAGE:-all}"
N_GPUS="${N_GPUS:-2}"
: "${MODELS_ROOT:?Please set MODELS_ROOT in .env}"

# The scripts assume 8 GPUs by default; scale every GPU-count knob to what the node has.
export N_GPUS_PER_NODE="$N_GPUS"
export NPROC_PER_NODE="$N_GPUS"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-$N_GPUS}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N_GPUS - 1)))}"

if (( $(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | wc -l) != N_GPUS )); then
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES does not list N_GPUS=$N_GPUS devices." >&2
    exit 1
fi

BASE_MODEL="${BASE_MODEL:-$MODELS_ROOT/Qwen2.5-3B-Instruct}"
SFT_MODEL="${ALFWORLD_RL_MODEL_PATH:-${HF_MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct-alfworld-episode-skill-sft-glm-self}}"

echo "SEED ALFWorld reproduction"
echo "  stage:      $STAGE"
echo "  conda env:  $CONDA_ENV"
echo "  GPUs:       $N_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  base model: $BASE_MODEL"
echo "  SFT model:  $SFT_MODEL"
echo "  alfworld:   $ALFWORLD_DATA"

case "$STAGE" in all|sft|rl) ;; *) echo "Unknown STAGE '$STAGE' (use all | sft | rl)" >&2; exit 2 ;; esac

run_sft() {
    if [[ -f "$SFT_MODEL/config.json" ]]; then
        echo "[SKIP] SFT checkpoint already exists: $SFT_MODEL"
        return 0
    fi
    if [[ ! -f "$BASE_MODEL/config.json" ]]; then
        echo "Base model not found: $BASE_MODEL (set BASE_MODEL)" >&2
        exit 1
    fi
    export MODEL_PATH="$BASE_MODEL"
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        echo "Dry run: would run prepare_data.sh and train_sft.sh from $BASE_MODEL"
        return 0
    fi
    bash scripts/sft/alfworld/prepare_data.sh
    bash scripts/sft/alfworld/train_sft.sh
}

run_rl() {
    if [[ "${DRY_RUN:-false}" != "true" && ! -f "$SFT_MODEL/config.json" ]]; then
        echo "SFT checkpoint not found: $SFT_MODEL (run STAGE=sft first or set ALFWORLD_RL_MODEL_PATH)" >&2
        exit 1
    fi
    # MODEL_PATH was the SFT base model in stage 1; RL starts from the SFT export.
    unset MODEL_PATH
    export ALFWORLD_RL_MODEL_PATH="$SFT_MODEL"
    bash examples/seed_trainer/run_alfworld_sft_glm_self.sh trainer.max_actor_ckpt_to_keep=1 "$@"
}

if [[ "$STAGE" == all || "$STAGE" == sft ]]; then run_sft; fi
if [[ "$STAGE" == all || "$STAGE" == rl ]]; then run_rl "$@"; fi
echo "Done."
