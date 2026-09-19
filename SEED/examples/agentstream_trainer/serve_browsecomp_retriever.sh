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
# Usage:  DEVICE=0 PORT=60100 bash examples/agentstream_trainer/serve_browsecomp_retriever.sh
#         DEVICE=cpu ...           # same index, encoder in bf16 on CPU (AMX on Sapphire
#                                 # Rapids); threads = cores this job owns
#         SEARCHER_TYPE=bm25 ...   # no torch model at all, needs Java 21+ at install time
# Concurrent queries (one per SEED env slot) are embedded in one encoder forward
# (MAX_BATCH, BATCH_WINDOW_S), which is what makes CPU serving fast enough.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common/exgentic_env.sh"   # the retriever is an exgentic process too
DEVICE=${DEVICE:-0}                                   # cpu | GPU index
if [[ "$DEVICE" == cpu ]]; then
    CUDA_VISIBLE_DEVICES=""
    TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
else
    CUDA_VISIBLE_DEVICES=$DEVICE
    TORCH_DTYPE=${TORCH_DTYPE:-float16}
fi
MAX_BATCH=${MAX_BATCH:-64}                            # 1 = one forward per query; 64 = an isolated
                                                      # browsecompplus batch (10 tasks x 6) in one forward
BATCH_WINDOW_S=${BATCH_WINDOW_S:-0.02}
# Encoder threads: two thirds of the cores this job owns (PBS exports NCPUS; nproc
# honours cgroup limits elsewhere). The rest stays free for vLLM, FSDP and the
# benchmark runner processes that run alongside the retriever during rollout;
# taking every core starves them and the env step stalls behind the encoder.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( ${NCPUS:-$(nproc)} * 2 / 3 ))}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-60100}
SEARCHER_TYPE=${SEARCHER_TYPE:-faiss}                 # faiss | bm25
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}   # faiss only
EMBED_MODEL_PATH=${EMBED_MODEL_PATH:-$EMBED_MODEL}
EXGENTIC_HOME=${EXGENTIC_HOME:-$HOME/.exgentic}
EXGENTIC_SOURCE=${EXGENTIC_SOURCE:-$SCRIPT_DIR/../../../AgentStream/exgentic/src}
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
    KWARGS=$(printf '{"searcher_type":"faiss","index_path":"%s/indexes/%s/corpus.shard*_of_4.pkl","model_name":"%s","normalize":true,"torch_dtype":"%s","max_batch":%s,"batch_window_s":%s}' \
    "$ASSETS" "$model_dir" "$EMBED_MODEL_PATH" "$TORCH_DTYPE" "$MAX_BATCH" "$BATCH_WINDOW_S")
fi

echo "Serving BrowseComp-Plus retriever at http://$HOST:$PORT on device $DEVICE ($TORCH_DTYPE, $OMP_NUM_THREADS threads)"
echo "  kwargs: $KWARGS"
if [[ -d "$EXGENTIC_SOURCE/exgentic" ]]; then
    export PYTHONPATH="$EXGENTIC_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
fi
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" exec "$EXGENTIC_BIN" serve \
    --cls exgentic.benchmarks.browsecompplus.retriever:Retriever \
    --kwargs "$KWARGS" --host "$HOST" --port "$PORT"
