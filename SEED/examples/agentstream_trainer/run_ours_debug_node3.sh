#!/usr/bin/env bash
# Node 3 of the three-node split of run_ours_debug.sh: E7 (A4 anchor), E8a-E8c (gen OPD from the pool: raw / de-instantiated / aggregated skills), plus E1 (A3 anchor) with E2c (gated gen OPD, aggregated skills).
# Every arm has the anchor of its family on its own node because cross-job noise (5-9 pts) exceeds the
# arm differences: pair only inside a node. Same usage as run_ours_debug.sh (--dry-run, TOTAL_STEPS,
# DEBUG_RUN_ID); DEBUG_ARMS may be overridden from the environment.
export DEBUG_ARMS="${DEBUG_ARMS:-E7,E8a,E8b,E8c,E1,E2c}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ours_debug.sh" "$@"
