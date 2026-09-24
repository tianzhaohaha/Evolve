#!/usr/bin/env bash
# Node 1 of the three-node split of run_ours_debug.sh: E1 (A3 anchor), E3 (sibling resample, source baseline), E4 (pool resample, raw skills), E2a (gated gen OPD, raw skills).
# Every arm has the anchor of its family on its own node because cross-job noise (5-9 pts) exceeds the
# arm differences: pair only inside a node. Same usage as run_ours_debug.sh (--dry-run, TOTAL_STEPS,
# DEBUG_RUN_ID); DEBUG_ARMS may be overridden from the environment.
export DEBUG_ARMS="${DEBUG_ARMS:-E1,E3,E4,E2a}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ours_debug.sh" "$@"
