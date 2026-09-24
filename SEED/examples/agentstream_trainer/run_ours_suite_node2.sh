#!/usr/bin/env bash
# Node 2 of the two-node split of run_ours_suite.sh: S3 (7B isolated, ~17-21 h); walltime >= 24 h.
# Same usage as run_ours_suite.sh (--dry-run, OURS_SAVE_FREQ); OURS_JOBS may be overridden from the environment.
export OURS_JOBS="${OURS_JOBS:-S3}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ours_suite.sh" "$@"
