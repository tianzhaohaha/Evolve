#!/usr/bin/env bash
# Final-method runs built on the E-arm results of run_ours_debug.sh (SEED/README.md "可选改进开关" [9][10]).
# Each entry is one arm of run_ours_debug.sh, either continued from its step-10 checkpoint or rerun from
# scratch, so this script only sets the per-run variables and calls run_ours_debug.sh:
#   F1  continue E5 of job 71186 (A3 + sibling resample + pool resample, de-instantiated skills) from
#       step 10 to 20 -- the final method over the whole single pass; its holdout at step 20 matches the
#       baseline suite's.
#   F2  continue E3 of job 71185 (A3 + sibling resample, no pool) from step 10 to 20 -- the global-pool
#       ablation on the existing lineage. That run collapsed in steps 1-4 (KL 3x, response length 141 -> 60),
#       so read F2 as "does a collapsed run recover", not as the clean ablation.
#   F3  fresh E3 for 20 steps, checkpoints at steps 10 and 20 -- the clean global-pool ablation (F1 - F3)
#       and a second sample of E3 at step 10 (vs job 71185's 0.129 and its anchor's 0.225).
# Continuation mechanics: DEBUG_RUN_ID + the arm's tag rebuild the original experiment name, and
# trainer.resume_mode=auto restores global_step_10/ (weights, optimizer, lr scheduler, dataloader,
# stream_state.json), the online recorder and the pool, then streams tasks 101-200 and saves step 20.
# The step-10 checkpoint of a continued run is kept (rotation only covers saves of one process); each run
# adds ~48 GB (F3: two checkpoints).
# wandb: WANDB_RUN_ID / WANDB_RESUME=allow append steps 11-20 to the original run (verified with a probe
# run: earlier points are kept untouched, a write to an older step is ignored, the config is updated to
# the new total_epochs). Without them wandb opens a new run with the same name; nothing is overwritten
# either way, and the online cumulative score is continuous because it is restored from the recorder.
# F3's experiment name is fixed by FINAL_RUN_ID (default "final") so an interrupted run resumes when
# resubmitted; a *second* F3 (another seed) needs a new id, e.g. FINAL_RUN_ID=final2, or auto-resume
# finds the finished global_step_20/ and exits at once.
# Precedence: run_ours_debug.sh sources <repo>/.env after receiving these variables, so .env must not
# define DEBUG_RUN_ID / DEBUG_ARMS / TOTAL_STEPS / AGENTSTREAM_RL_SAVE_FREQ / AGENTSTREAM_RL_MAX_CKPT_TO_KEEP.
#
# Usage: bash examples/agentstream_trainer/run_ours_final.sh [--dry-run]
#   FINAL_JOBS=F1,F3 bash ...     # subset (default F1,F2,F3, run in that order; independent, so they may
#                                 # also be submitted as three jobs, one id each)
#   FINAL_RUN_ID=final2 bash ...  # F3 under a fresh experiment name (a second sample instead of a resume)
# Walltime: F1 ~5.5 h (31 min/step), F2 ~3 h, F3 ~6 h; sequential ~15 h.
# Read-out: F1 at step 20 vs the v7 baselines (online curves step by step, holdout at step 20); F1 - F3 =
# value of the global pool (cross-job, same-config noise ~2-3 pts); F3 at step 10 vs 71185's E3 = was the
# -9.6 a mechanism or one unstable run.

set -eo pipefail

[[ "${1:-}" == --dry-run ]] && { export DRY_RUN=true; shift; }
(( $# == 0 )) || { echo "Usage: $0 [--dry-run]   (FINAL_JOBS via the environment)" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_RUN_ID="${FINAL_RUN_ID:-final}"  # experiment id of the fresh run (F3)

# id | DEBUG_RUN_ID | DEBUG_ARMS | TOTAL_STEPS | SAVE_FREQ | MAX_CKPT_TO_KEEP | WANDB_RUN_ID | description
JOBS=(
    "F1|71186.gaas|E5|20|20|1|bjnan8uy|continue E5 (A3 + sibling resample + pool resample deinst) to step 20 -- final method"
    "F2|71185.gaas|E3|20|20|1|dx7l9c78|continue E3 (A3 + sibling resample) to step 20 -- pool ablation on the existing lineage"
    "F3|$FINAL_RUN_ID|E3|20|10|2||fresh E3, 20 steps, checkpoints at 10 and 20 -- clean pool ablation"
)
FINAL_JOBS="${FINAL_JOBS:-F1,F2,F3}"
declare -A STATUS
NAMES=()

run_job() {  # <job spec>
    local id run_id arms steps save_freq keep wandb_id desc
    IFS='|' read -r id run_id arms steps save_freq keep wandb_id desc <<< "$1"
    if [[ ",$FINAL_JOBS," != *",$id,"* ]]; then
        echo "[SKIP] $id $desc (not in FINAL_JOBS=$FINAL_JOBS)"
        return 0
    fi
    local -a wandb=()
    [[ -n "$wandb_id" ]] && wandb=("WANDB_RUN_ID=$wandb_id" WANDB_RESUME=allow)
    NAMES+=("$id")
    echo ""
    echo "######################################################################"
    echo "[Final] $id: $desc   start $(date)"
    echo "DEBUG_RUN_ID=$run_id DEBUG_ARMS=$arms TOTAL_STEPS=$steps SAVE_FREQ=$save_freq KEEP=$keep ${wandb[*]}"
    echo "######################################################################"
    set +e
    env "DEBUG_RUN_ID=$run_id" "DEBUG_ARMS=$arms" "TOTAL_STEPS=$steps" \
        "AGENTSTREAM_RL_SAVE_FREQ=$save_freq" "AGENTSTREAM_RL_MAX_CKPT_TO_KEEP=$keep" "${wandb[@]}" \
        bash "$SCRIPT_DIR/run_ours_debug.sh" ${DRY_RUN:+--dry-run}
    STATUS[$id]=$?
    set -e
    echo "[$([[ ${STATUS[$id]} -eq 0 ]] && echo SUCCESS || echo FAILED)] $id   end $(date)"
}

for spec in "${JOBS[@]}"; do
    run_job "$spec"
done

echo ""
echo "######################################################################"
echo "Final runs summary (jobs=$FINAL_JOBS, end $(date))"
failed=0
for id in "${NAMES[@]}"; do
    printf '  %-4s exit=%s\n' "$id" "${STATUS[$id]}"
    (( STATUS[$id] == 0 )) || failed=1
done
echo "######################################################################"
exit $failed
