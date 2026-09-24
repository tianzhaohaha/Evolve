#!/usr/bin/env bash
# Node 2 of the three-node split of run_ours_debug.sh: E1 (A3 anchor), E5 / E6 (pool resample, de-instantiated / aggregated skills), E2b (gated gen OPD, de-instantiated skills).
# Every arm has the anchor of its family on its own node because cross-job noise (5-9 pts) exceeds the
# arm differences: pair only inside a node. Same usage as run_ours_debug.sh (--dry-run, TOTAL_STEPS,
# DEBUG_RUN_ID); DEBUG_ARMS may be overridden from the environment.
export DEBUG_ARMS="${DEBUG_ARMS:-E1,E5,E6,E2b}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ours_debug.sh" "$@"
