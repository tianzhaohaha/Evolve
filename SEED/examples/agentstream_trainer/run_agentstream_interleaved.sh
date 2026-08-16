#!/usr/bin/env bash

# Interleaved streaming scenario: a single SEED training run whose batches mix
# tasks from all benchmarks according to AgentStream's order-preserving
# interleaving (same task sets and within-benchmark order as sequential).
# GRPO group advantages are computed within each task group, so mixed-benchmark
# batches are well-defined without cross-benchmark reward rescaling.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AS_STREAM_MODE=interleaved
export AS_PROTOCOL="${AS_PROTOCOL:-online}"
export AS_BENCHMARKS="${AS_BENCHMARKS:-bfcl,tau2,appworld}"
export AS_STREAM_SEED="${AS_STREAM_SEED:-44}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_agentstream_interleaved_${AS_PROTOCOL}_s${AS_STREAM_SEED}}"

exec bash "$SCRIPT_DIR/_common/agentstream.sh" "$@"
