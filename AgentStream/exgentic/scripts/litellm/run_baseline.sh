#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

set -e

export OPENAI_API_BASE=""
export OPENAI_API_KEY=""

cd "$(dirname "$0")"

SEED=42
NUM_TASKS=50
MODEL="openai/gpt-5.4"
MAX_TOKENS="default"
REASONING_EFFORT="default"

OUTPUT_BASE="./outputs"
MODEL_SHORT=$(echo $MODEL | sed 's|openai/||; s|azure/||; s|/|_|g')
RUN_TAG="baseline_s${SEED}_${MODEL_SHORT}_${MAX_TOKENS}_${REASONING_EFFORT}"

SETTINGS_ARGS=""
[ "$MAX_TOKENS" != "default" ] && SETTINGS_ARGS="$SETTINGS_ARGS --max-tokens $MAX_TOKENS"
[ "$REASONING_EFFORT" != "default" ] && SETTINGS_ARGS="$SETTINGS_ARGS --reasoning-effort $REASONING_EFFORT"
echo "ModelSettings args: ${SETTINGS_ARGS:-default (no overrides)}"

# HLE
echo "=== Running HLE ==="
uv run python run_baseline.py \
    --seed $SEED --num-tasks $NUM_TASKS \
    --model $MODEL $SETTINGS_ARGS \
    --benchmarks hle \
    --output-dir ${OUTPUT_BASE}/${RUN_TAG}_hle \
    2>&1 | tee ${OUTPUT_BASE}/${RUN_TAG}_hle.log

# BFCL
echo "=== Running BFCL ==="
uv run python run_baseline.py \
    --seed $SEED --num-tasks $NUM_TASKS \
    --model $MODEL $SETTINGS_ARGS \
    --benchmarks bfcl \
    --output-dir ${OUTPUT_BASE}/${RUN_TAG}_bfcl \
    2>&1 | tee ${OUTPUT_BASE}/${RUN_TAG}_bfcl.log

# Tau2
echo "=== Running Tau2 ==="
uv run python run_baseline.py \
    --seed $SEED --num-tasks $NUM_TASKS \
    --model $MODEL $SETTINGS_ARGS \
    --benchmarks tau2 \
    --output-dir ${OUTPUT_BASE}/${RUN_TAG}_tau2 \
    2>&1 | tee ${OUTPUT_BASE}/${RUN_TAG}_tau2.log

# BrowseCompPlus
echo "=== Running BrowseCompPlus ==="
uv run python run_baseline.py \
    --seed $SEED --num-tasks $NUM_TASKS \
    --model $MODEL $SETTINGS_ARGS \
    --benchmarks browsecompplus \
    --output-dir ${OUTPUT_BASE}/${RUN_TAG}_browsecompplus \
    2>&1 | tee ${OUTPUT_BASE}/${RUN_TAG}_browsecompplus.log

# AppWorld
echo "=== Running AppWorld ==="
uv run python run_baseline.py \
    --seed $SEED --num-tasks $NUM_TASKS \
    --model $MODEL $SETTINGS_ARGS \
    --benchmarks appworld \
    --output-dir ${OUTPUT_BASE}/${RUN_TAG}_appworld \
    2>&1 | tee ${OUTPUT_BASE}/${RUN_TAG}_appworld.log

# SWE-bench
echo "=== Running SWE-bench ==="
uv run python run_baseline.py \
    --seed $SEED --num-tasks $NUM_TASKS \
    --model $MODEL $SETTINGS_ARGS \
    --benchmarks swebench \
    --output-dir ${OUTPUT_BASE}/${RUN_TAG}_swebench \
    2>&1 | tee ${OUTPUT_BASE}/${RUN_TAG}_swebench.log

echo ""
echo "=== All benchmarks complete ==="
echo "Run tag: ${RUN_TAG}"
echo "Outputs in: ${OUTPUT_BASE}/${RUN_TAG}_*"
