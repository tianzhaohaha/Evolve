#!/usr/bin/env bash
# Shared exgentic process settings for every SEED entry point that starts an exgentic
# process (Stage-1 data construction, Stage-3 RL, the browsecompplus retriever service).
# Source it before the process starts; PROJECT_ROOT is derived from this file's location
# when the caller has not set it.
#
# exgentic keeps two SQLite-backed diskcaches (WAL mode). SQLite WAL is unsafe on
# network filesystems and across nodes, and both files are opened concurrently by the
# trainer driver and by every benchmark runner subprocess (80+ per step). On the cluster
# home this corrupted mid-campaign ("database disk image is malformed" at trainer start).
#
# 1. LLM response cache (~/.cache/exgentic/litellm/cache.db, opened at settings init):
#    off by default. Opt back in with EXGENTIC_LITELLM_CACHING=true, then also point
#    EXGENTIC_LITELLM_CACHE_DIR at node-local storage.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export EXGENTIC_LITELLM_CACHING="${EXGENTIC_LITELLM_CACHING:-false}"

# 2. Benchmark session caches (browsecompplus searcher results) live at a cwd-relative
#    path, i.e. inside this repo. Replace that directory with a symlink to node-local
#    storage; a corrupt searcher cache would not crash training but turns every
#    browsecompplus search into an error (silent zero reward).
#    A pre-existing real directory is renamed, never deleted: another job on another
#    node may still hold its SQLite files open (rename keeps open handles valid).
#    Several processes run this block concurrently (retriever service, launcher, a
#    second job on another node), so the rename tolerates losing the race and the
#    link is replaced atomically (rename over the old link) instead of unlink + create.
#    Temporary names carry host + BASHPID: $$ is shared by subshells and PIDs repeat
#    across nodes.
_exgentic_session_cache="${AGENTSTREAM_TMP_ROOT:-${TMPDIR:-/tmp}}/exgentic_session_cache"
_exgentic_session_link="$PROJECT_ROOT/exgentic_session_cache"
_exgentic_session_tag="${HOSTNAME:-host}.$BASHPID"
mkdir -p "$_exgentic_session_cache"
if [[ -d "$_exgentic_session_link" && ! -L "$_exgentic_session_link" ]]; then
    mv "$_exgentic_session_link" "$_exgentic_session_link.moved.$(date +%Y%m%d_%H%M%S).$_exgentic_session_tag" 2>/dev/null || true
fi
ln -s "$_exgentic_session_cache" "$_exgentic_session_link.$_exgentic_session_tag" \
    && mv -Tf "$_exgentic_session_link.$_exgentic_session_tag" "$_exgentic_session_link"
