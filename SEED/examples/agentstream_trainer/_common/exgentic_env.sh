#!/usr/bin/env bash
# Shared exgentic process settings for every SEED entry point that starts an exgentic
# process (Stage-1 data construction, Stage-3 RL, the browsecompplus retriever service).
# Source it before the process starts.
#
# exgentic keeps two SQLite-backed diskcaches (WAL mode). SQLite WAL is unsafe on
# network filesystems and across nodes, and both files are opened concurrently by the
# trainer driver and by every benchmark runner subprocess (80+ per step). On the cluster
# home this corrupted mid-campaign ("database disk image is malformed" at trainer start).
#
# 1. LLM response cache (~/.cache/exgentic/litellm/cache.db, opened at settings init):
#    off by default. Opt back in with EXGENTIC_LITELLM_CACHING=true, then also point
#    EXGENTIC_LITELLM_CACHE_DIR at node-local storage.
export EXGENTIC_LITELLM_CACHING="${EXGENTIC_LITELLM_CACHING:-false}"

# 2. The browsecompplus search-result cache (a second SQLite at a cwd-relative path inside
#    this checkout) is disabled through the benchmark kwargs instead (as_config.py
#    DEFAULT_BENCHMARK_KWARGS -> use_cache=False): a shared file, or a shared symlink to
#    node-local storage, broke every concurrent job's searches.
#
# 3. Retriever RPC timeout (exgentic RetrieverClient). A timed-out search scores 0 silently
#    ("Action 'search' failed" observation); raise this if a run logs such observations.
export EXGENTIC_RETRIEVER_TIMEOUT="${EXGENTIC_RETRIEVER_TIMEOUT:-300}"
