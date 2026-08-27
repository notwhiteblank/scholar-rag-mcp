#!/usr/bin/env bash
set -euo pipefail

_required=()
for _var in SCHOLAR_RAG_CHAT_MODEL SCHOLAR_RAG_EMBED_MODEL SCHOLAR_RAG_RERANK_MODEL; do
  if [ -z "${!_var:-}" ]; then
    _required+=("${_var}")
  fi
done
if [ "${#_required[@]}" -gt 0 ]; then
  echo "error: missing required model path variables: ${_required[*]}" >&2
  echo "each must be the absolute path to a local HuggingFace model directory;" >&2
  echo "the directory basename is the short name the served model is exposed under," >&2
  echo "so the matching SCHOLAR_RAG_*_MODEL client settings must use that short name." >&2
  echo "Example:" >&2
  echo "  SCHOLAR_RAG_CHAT_MODEL=/path/to/models/Qwen/Qwen3.5-0.8B \\" >&2
  echo "  SCHOLAR_RAG_EMBED_MODEL=/path/to/models/jinaai/jina-embeddings-v5-text-small \\" >&2
  echo "  SCHOLAR_RAG_RERANK_MODEL=/path/to/models/jinaai/jina-reranker-v3.5 \\" >&2
  echo "  bash scripts/serve_models.sh" >&2
  exit 1
fi

CHAT_MODEL="${SCHOLAR_RAG_CHAT_MODEL}"
EMBED_MODEL="${SCHOLAR_RAG_EMBED_MODEL}"
RERANK_MODEL="${SCHOLAR_RAG_RERANK_MODEL}"
CHAT_PORT="${CHAT_PORT:-8101}"
EMBED_PORT="${EMBED_PORT:-8102}"
RERANK_PORT="${RERANK_PORT:-8103}"
CHAT_GPU="${CHAT_GPU:-0}"
EMBED_GPU="${EMBED_GPU:-1}"
RERANK_GPU="${RERANK_GPU:-2}"
PIXI="${PIXI:-pixi}"
MANIFEST="${MANIFEST:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../spike" && pwd)/pixi.toml}"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../spike" && pwd)/out}"
mkdir -p "${OUT_DIR}"

CHAT_NAME="$(basename "${CHAT_MODEL}")"
EMBED_NAME="$(basename "${EMBED_MODEL}")"
RERANK_NAME="$(basename "${RERANK_MODEL}")"

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
echo "== chat (${CHAT_NAME}) on :${CHAT_PORT} GPU ${CHAT_GPU} (log: ${CHAT_LOG})"
CUDA_VISIBLE_DEVICES="${CHAT_GPU}" "${PIXI}" run -m "${MANIFEST}" vllm serve \
  "${CHAT_MODEL}" \
  --port "${CHAT_PORT}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.77 \
  --served-model-name "${CHAT_NAME}" >"${CHAT_LOG}" 2>&1 &
CHAT_PID=$!
health "${CHAT_PORT}" "${CHAT_LOG}" || true

EMBED_LOG="${OUT_DIR}/vllm_embed.log"
echo "== embed (${EMBED_NAME}) on :${EMBED_PORT} GPU ${EMBED_GPU} (log: ${EMBED_LOG})"
CUDA_VISIBLE_DEVICES="${EMBED_GPU}" "${PIXI}" run -m "${MANIFEST}" vllm serve \
  "${EMBED_MODEL}" \
  --port "${EMBED_PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --served-model-name "${EMBED_NAME}" >"${EMBED_LOG}" 2>&1 &
EMBED_PID=$!
health "${EMBED_PORT}" "${EMBED_LOG}" || true

RERANK_LOG="${OUT_DIR}/vllm_rerank.log"
echo "== rerank (${RERANK_NAME}) on :${RERANK_PORT} GPU ${RERANK_GPU} (log: ${RERANK_LOG})"
CUDA_VISIBLE_DEVICES="${RERANK_GPU}" "${PIXI}" run -m "${MANIFEST}" vllm serve \
  "${RERANK_MODEL}" \
  --port "${RERANK_PORT}" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --served-model-name "${RERANK_NAME}" >"${RERANK_LOG}" 2>&1 &
RERANK_PID=$!
health "${RERANK_PORT}" "${RERANK_LOG}" || true

echo
echo "all model servers launched (PIDs ${CHAT_PID} ${EMBED_PID} ${RERANK_PID}). logs: ${OUT_DIR}/vllm_{chat,embed,rerank}.log"
wait "${CHAT_PID}" "${EMBED_PID}" "${RERANK_PID}"