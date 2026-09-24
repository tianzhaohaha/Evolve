#!/usr/bin/env bash
# Our method with the interface of run_agentstream_baseline.sh:
#   bash examples/agentstream_trainer/run_ours_method.sh <interleaved|isolated|sequential|random> [hydra overrides...]
# The method is arm E5 of run_ours_debug.sh (_common/ours_method.sh OURS_METHOD_ENV: A3 floor + sibling
# resample with the source-group baseline + pool resample with de-instantiated skills), applied to whatever
# model / stream the caller sets (AGENTSTREAM_BASE_MODEL_NAME, AGENTSTREAM_BENCHMARKS, step counts ...
# from agentstream_full.env). Experiment names follow the baseline suite with the method tag "ours":
# ours_<model tag>_agentstream_<version>_n64_single_pass_b10_steps<N>_<mode>_online_s44[_<benchmark>].
set -euo pipefail
mode="${1:?usage: run_ours_method.sh <mode> [hydra overrides...]}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common/ours_method.sh
source "$SCRIPT_DIR/_common/ours_method.sh"
export AGENTSTREAM_METHOD_TAG="${AGENTSTREAM_METHOD_TAG:-ours}"   # experiment name prefix (agentstream_full.env)
export "${OURS_METHOD_ENV[@]}"
echo "AgentStream method '$AGENTSTREAM_METHOD_TAG' (${#OURS_METHOD_ENV[@]} switches from _common/ours_method.sh): $mode $*"
exec bash "$SCRIPT_DIR/run_agentstream_sft_glm_self.sh" "$mode" "$@"
