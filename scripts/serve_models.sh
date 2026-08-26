#!/usr/bin/env bash
set -euo pipefail

CHAT_MODEL="${CHAT_MODEL:-/path/to/models/Qwen/Qwen3-8B}"
EMBED_MODEL="${EMBED_MODEL:-/path/to/models/Qwen/Qwen3-VL-Embedding-2B}"
RERANK_MODEL="${RERANK_MODEL:-/path/to/models/Qwen/Qwen3-VL-Reranker-2B}"
CHAT_PORT="${CHAT_PORT:-8101}"
EMBED_PORT="${EMBED_PORT:-8102}"
RERANK_PORT="${RERANK_PORT:-8103}"
CHAT_GPU="${CHAT_GPU:-0}"
EMBED_GPU="${EMBED_GPU:-1}"
RERANK_GPU="${RERANK_GPU:-2}"
PIXI="${PIXI:-pixi}"
MANIFEST="${MANIFEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../spike" && pwd)/pixi.toml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../spike" && pwd)/out}"
mkdir -p "${OUT_DIR}"

health() {
  local port="$1"
  local logfile="$2"
  for _ in $(seq 1 120); do
    if curl -sf -m 3 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "  ok: model serving on ${port} (log: ${logfile})"
      return 0
    fi
    sleep 5
  done
  echo "  FAILED: no model on port ${port} after 10m (log: ${logfile})"
  return 1
}

CHAT_LOG="${OUT_DIR}/vllm_chat.log"
echo "== chat (Qwen3-8B) on :${CHAT_PORT} GPU ${CHAT_GPU} (log: ${CHAT_LOG})"
CUDA_VISIBLE_DEVICES="${CHAT_GPU}" "${PIXI}" run -m "${MANIFEST}" vllm serve \
  "${CHAT_MODEL}" \
  --port "${CHAT_PORT}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.77 >"${CHAT_LOG}" 2>&1 &
CHAT_PID=$!
health "${CHAT_PORT}" "${CHAT_LOG}" || true

EMBED_LOG="${OUT_DIR}/vllm_embed.log"
echo "== embed (Qwen3-VL-Embedding-2B) on :${EMBED_PORT} GPU ${EMBED_GPU} (log: ${EMBED_LOG})"
CUDA_VISIBLE_DEVICES="${EMBED_GPU}" "${PIXI}" run -m "${MANIFEST}" vllm serve \
  "${EMBED_MODEL}" \
  --port "${EMBED_PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --convert embed \
  --pooler-config '{"task":"embed","pooling_type":"MEAN","use_activation":true}' >"${EMBED_LOG}" 2>&1 &
EMBED_PID=$!
health "${EMBED_PORT}" "${EMBED_LOG}" || true

RERANK_LOG="${OUT_DIR}/vllm_rerank.log"
echo "== rerank (Qwen3-VL-Reranker-2B) on :${RERANK_PORT} GPU ${RERANK_GPU} (log: ${RERANK_LOG})"
CUDA_VISIBLE_DEVICES="${RERANK_GPU}" "${PIXI}" run -m "${MANIFEST}" vllm serve \
  "${RERANK_MODEL}" \
  --port "${RERANK_PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --convert classify \
  --pooler-config '{"task":"classify","pooling_type":"LAST","use_activation":true}' \
  --hf-overrides '{"classifier_from_token":["no","yes"],"method":"from_2_way_softmax","num_labels":1}' \
  --chat-template "${SCRIPT_DIR}/score_template.jinja" >"${RERANK_LOG}" 2>&1 &
RERANK_PID=$!
health "${RERANK_PORT}" "${RERANK_LOG}" || true

echo
echo "all model servers launched (PIDs ${CHAT_PID} ${EMBED_PID} ${RERANK_PID}). logs: ${OUT_DIR}/vllm_{chat,embed,rerank}.log"
wait "${CHAT_PID}" "${EMBED_PID}" "${RERANK_PID}"