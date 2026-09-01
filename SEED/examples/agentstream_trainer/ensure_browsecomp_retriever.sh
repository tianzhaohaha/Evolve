#!/usr/bin/env bash
# Ensure the shared BrowseComp-Plus retriever is reachable before training.
# No-op unless browsecompplus is in AGENTSTREAM_BENCHMARKS. Reuses a healthy
# server; auto-starts serve_browsecomp_retriever.sh (nohup, keep-alive) when
# the URL points at this machine. The server outlives training; stop it with
# `kill $(cat logs/agentstream/browsecomp_retriever_<port>.pid)`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

case ",${AGENTSTREAM_BENCHMARKS:-}," in
    *,browsecompplus,*) ;;
    *) exit 0 ;;
esac

url="${AGENTSTREAM_BROWSECOMP_RETRIEVER_URL:-http://127.0.0.1:60100}"
retriever_ready() { curl -fsS "$url/health" >/dev/null 2>&1; }

if retriever_ready; then
    echo "Reusing existing BrowseComp-Plus retriever at $url"
    exit 0
fi

hostport="${url#*://}"; hostport="${hostport%%/*}"
host="${hostport%%:*}"
port="${hostport##*:}"; [[ "$port" == "$host" ]] && port=80
gpu="${AGENTSTREAM_RETRIEVER_GPU:-0}"

if [[ "${AGENTSTREAM_RETRIEVER_AUTO_START:-1}" != "1" ]] \
   || [[ "$host" != "127.0.0.1" && "$host" != "localhost" && "$host" != "0.0.0.0" ]]; then
    echo "BrowseComp-Plus retriever not reachable at $url (auto-start disabled or remote host)." >&2
    echo "Start it manually: GPU=$gpu PORT=$port bash $SCRIPT_DIR/serve_browsecomp_retriever.sh" >&2
    exit 1
fi

log_dir="$PROJECT_ROOT/logs/agentstream"
mkdir -p "$log_dir"
pid_file="$log_dir/browsecomp_retriever_${port}.pid"
log_file="$log_dir/browsecomp_retriever_${port}.log"

if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    # Already launched (e.g. still loading the index, or a concurrent run) -> just wait.
    pid="$(cat "$pid_file")"
    echo "Found starting retriever (pid $pid); waiting for $url/health"
else
    rm -f "$pid_file"
    echo "Starting BrowseComp-Plus retriever on GPU $gpu at $url (log: $log_file)"
    GPU="$gpu" HOST="$host" PORT="$port" \
        nohup bash "$SCRIPT_DIR/serve_browsecomp_retriever.sh" >"$log_file" 2>&1 &
    pid=$!
    echo "$pid" >"$pid_file"
fi

deadline=$((SECONDS + ${AGENTSTREAM_RETRIEVER_STARTUP_TIMEOUT:-600}))
until retriever_ready; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Retriever (pid $pid) exited before becoming ready. See $log_file" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for retriever at $url. See $log_file" >&2
        exit 1
    fi
    sleep 5
done
echo "Retriever ready at $url (pid $pid; stays up after training — stop with: kill \$(cat $pid_file))"
