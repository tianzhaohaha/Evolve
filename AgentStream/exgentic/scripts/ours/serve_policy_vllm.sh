#!/usr/bin/env bash
# OpenAI-compatible vLLM server for the base policy used by the AgentStream baselines
# (native tool calling, hermes parser for Qwen3; context 131k so 40-step episodes are not truncated).
#   MODEL_PATH=/path/Qwen3-4B-Instruct-2507 GPU=0 PORT=8000 bash scripts/ours/serve_policy_vllm.sh
set -eo pipefail
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Qwen3-4B-Instruct-2507 directory}"
SERVED_NAME="${SERVED_NAME:-Qwen3-4B-Instruct-2507}"   # must match the --model suffix used by the runner
PORT="${PORT:-8000}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
exec vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len "${MAX_MODEL_LEN:-131072}" \
    --gpu-memory-utilization "${GPU_MEM:-0.90}" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-num-seqs "${MAX_NUM_SEQS:-64}"
