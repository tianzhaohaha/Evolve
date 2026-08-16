#!/usr/bin/env bash

# Baseline: SEED-original data-processing logic on AgentStream benchmarks.
# Tasks are sampled uniformly at random per RL step (exactly like the existing
# ALFWorld/AppWorld envs), with a SEED-style train/eval split. Use this run as
# the control when comparing against isolated / sequential / interleaved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AS_STREAM_MODE=random
export AS_PROTOCOL="${AS_PROTOCOL:-split}"
export AS_BENCHMARKS="${AS_BENCHMARKS:-bfcl,tau2,appworld}"
export AS_STREAM_SEED="${AS_STREAM_SEED:-44}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_agentstream_random_${AS_PROTOCOL}_s${AS_STREAM_SEED}}"

exec bash "$SCRIPT_DIR/_common/agentstream.sh" "$@"
