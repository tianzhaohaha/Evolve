#!/usr/bin/env bash

# Isolated streaming scenario: one independent SEED training run per benchmark
# (memory/weights never shared across benchmarks), mirroring the isolated
# branch of AgentStream/exgentic/scripts/*/run_experiment.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHMARK_LIST="${AS_BENCHMARKS:-bfcl,tau2,appworld}"
export AS_STREAM_MODE=isolated
export AS_PROTOCOL="${AS_PROTOCOL:-online}"
export AS_STREAM_SEED="${AS_STREAM_SEED:-44}"

IFS=',' read -ra BENCHES <<< "$BENCHMARK_LIST"
for BENCH in "${BENCHES[@]}"; do
    BENCH="$(echo "$BENCH" | xargs)"
    echo "=== Running isolated ${BENCH} ==="
    AS_BENCHMARKS="$BENCH" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME_PREFIX:-seed_agentstream}_isolated_${AS_PROTOCOL}_s${AS_STREAM_SEED}_${BENCH}" \
        bash "$SCRIPT_DIR/_common/agentstream.sh" "$@"
done
