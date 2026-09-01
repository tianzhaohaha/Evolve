#!/usr/bin/env bash
# Shared BrowseComp-Plus retriever for SEED x AgentStream runs.
#
# One process holds the search index (+ the Qwen3-Embedding-8B query encoder
# for faiss); every SEED env worker / Stage-1 session only sends queries to it
# through env.agentstream.benchmark_kwargs.browsecompplus.retriever_url
# (AGENTSTREAM_BROWSECOMP_RETRIEVER_URL in agentstream_full.env). Without it
# each worker would load its own copy of the index and the encoder.
#
# Prerequisite (once): exgentic install --benchmark browsecompplus
# Usage:  GPU=0 PORT=60100 bash examples/agentstream_trainer/serve_browsecomp_retriever.sh
#         SEARCHER_TYPE=bm25 ...   # no GPU, needs Java 21+ at install time

set -euo pipefail

GPU=${GPU:-0}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-60100}
SEARCHER_TYPE=${SEARCHER_TYPE:-faiss}                 # faiss | bm25
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}   # faiss only
EXGENTIC_HOME=${EXGENTIC_HOME:-$HOME/.exgentic}
ASSETS="$EXGENTIC_HOME/benchmarks/browsecompplus"
EXGENTIC_BIN="$ASSETS/venv/bin/exgentic"

if [[ ! -x "$EXGENTIC_BIN" ]]; then
    echo "browsecompplus venv not found at $ASSETS/venv; run 'exgentic install --benchmark browsecompplus' first." >&2
    exit 1
fi

if [[ "$SEARCHER_TYPE" == "bm25" ]]; then
    KWARGS=$(printf '{"searcher_type":"bm25","index_path":"%s/indexes/bm25"}' "$ASSETS")
else
    model_dir=$(tr '[:upper:]' '[:lower:]' <<< "${EMBED_MODEL##*/}")
    KWARGS=$(printf '{"searcher_type":"faiss","index_path":"%s/indexes/%s/corpus.shard*_of_4.pkl","model_name":"%s","normalize":true}' \
        "$ASSETS" "$model_dir" "$EMBED_MODEL")
fi

echo "Serving BrowseComp-Plus retriever at http://$HOST:$PORT on GPU $GPU"
echo "  kwargs: $KWARGS"
CUDA_VISIBLE_DEVICES="$GPU" exec "$EXGENTIC_BIN" serve \
    --cls exgentic.benchmarks.browsecompplus.retriever:Retriever \
    --kwargs "$KWARGS" --host "$HOST" --port "$PORT"
