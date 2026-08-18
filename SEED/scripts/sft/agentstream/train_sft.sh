#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET_NAME="agentstream"
DATASET_LABEL="AgentStream"
MODEL_BASENAME="${MODEL_BASENAME:-Qwen2.5-3B-Instruct}"
MODEL_TAG="${MODEL_TAG:-qwen25_3b}"
DEFAULT_SFT_CONDA_ENV="seed"
DEFAULT_LR="5e-6"
DEFAULT_MAX_LENGTH="12288"
MULTIMODAL="false"
SFT_LOG_DATA_SAMPLES="${SFT_LOG_DATA_SAMPLES:-3}"
SFT_LOG_DATA_SAMPLES_FREQ="${SFT_LOG_DATA_SAMPLES_FREQ:-10}"
SFT_LOG_DATA_SAMPLES_MAX_CHARS="${SFT_LOG_DATA_SAMPLES_MAX_CHARS:-4000}"

# shellcheck source=../_common/trainer.sh
source "$PROJECT_ROOT/scripts/sft/_common/trainer.sh"
