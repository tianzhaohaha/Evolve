#!/usr/bin/env bash
# Node 1 of the two-node split of run_ours_suite.sh: S1 (4B isolated, ~11 h) then S2 (7B interleaved, ~16-20 h); walltime >= 32 h.
# Same usage as run_ours_suite.sh (--dry-run, OURS_SAVE_FREQ); OURS_JOBS may be overridden from the environment.
export OURS_JOBS="${OURS_JOBS:-S1,S2}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ours_suite.sh" "$@"
