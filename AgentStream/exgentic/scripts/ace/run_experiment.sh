#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

set -e

export OPENAI_API_BASE=""
export OPENAI_API_KEY=""

cd "$(dirname "$0")"

# ============================================================
# Configuration
# ============================================================
SEED=44
NUM_TASKS=50
MODEL="openai/gpt-5.4"
MAX_TOKENS="default"
REASONING_EFFORT="default"
MODE="sequential"  # isolated | sequential | interleaved


OUTPUT_BASE="./outputs"
MODEL_SHORT=$(echo $MODEL | sed 's|openai/||; s|azure/||; s|/|_|g')
RUN_TAG="ace_${MODE}_s${SEED}_${MODEL_SHORT}_${MAX_TOKENS}_${REASONING_EFFORT}"

SETTINGS_ARGS=""
[ "$MAX_TOKENS" != "default" ] && SETTINGS_ARGS="$SETTINGS_ARGS --max-tokens $MAX_TOKENS"
[ "$REASONING_EFFORT" != "default" ] && SETTINGS_ARGS="$SETTINGS_ARGS --reasoning-effort $REASONING_EFFORT"
echo "Mode: ${MODE}"
echo "ModelSettings args: ${SETTINGS_ARGS:-default (no overrides)}"

mkdir -p "$OUTPUT_BASE"

ALL_BENCHMARKS="hle,bfcl,browsecompplus,appworld,swebench,tau2"

if [ "$MODE" = "isolated" ]; then
    for BENCH in swebench tau2 browsecompplus appworld hle bfcl; do
        echo "=== Running isolated ${BENCH} ==="
        uv run python run_experiment.py \
            --mode isolated --seed $SEED --num-tasks $NUM_TASKS \
            --model $MODEL $SETTINGS_ARGS \
            --benchmarks $BENCH \
            --output-dir ${OUTPUT_BASE}/${RUN_TAG}_${BENCH} \
            2>&1 | tee ${OUTPUT_BASE}/${RUN_TAG}_${BENCH}.log
    done
else
    echo "=== Running ${MODE} (all benchmarks) ==="
    uv run python run_experiment.py \
        --mode $MODE --seed $SEED --num-tasks $NUM_TASKS \
        --model $MODEL $SETTINGS_ARGS \
        --benchmarks $ALL_BENCHMARKS \
        --output-dir ${OUTPUT_BASE}/${RUN_TAG}_all \
        2>&1 | tee ${OUTPUT_BASE}/${RUN_TAG}_all.log
fi

echo ""
echo "=== All benchmarks complete ==="
echo "Run tag: ${RUN_TAG}"
echo "Outputs in: ${OUTPUT_BASE}/${RUN_TAG}_*"
