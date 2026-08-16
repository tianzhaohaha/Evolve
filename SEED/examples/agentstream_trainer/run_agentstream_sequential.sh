#!/usr/bin/env bash

# Sequential streaming scenario: a single SEED training run that walks the
# benchmarks block by block (alphabetical slug order, matching AgentStream's
# get_unified_task_order). Set AS_BLOCK_PASSES > 1 to give RL multiple passes
# over each block before moving on. Validation covers all benchmarks at every
# trainer.test_freq, yielding per-benchmark forgetting / transfer curves.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AS_STREAM_MODE=sequential
export AS_PROTOCOL="${AS_PROTOCOL:-online}"
export AS_BENCHMARKS="${AS_BENCHMARKS:-bfcl,tau2,appworld}"
export AS_STREAM_SEED="${AS_STREAM_SEED:-44}"
export AS_BLOCK_PASSES="${AS_BLOCK_PASSES:-1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_agentstream_sequential_${AS_PROTOCOL}_s${AS_STREAM_SEED}}"

exec bash "$SCRIPT_DIR/_common/agentstream.sh" "$@"
