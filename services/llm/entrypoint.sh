#!/usr/bin/env bash
set -euo pipefail

MODEL="${LLM_MODEL:-/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"

if [[ ! -f "${MODEL}" ]]; then
    echo "=============================================================" >&2
    echo " GGUF model not found: ${MODEL}" >&2
    echo " Put it under ./models/llm/ on the host. See scripts/download_models.sh" >&2
    echo "=============================================================" >&2
    exit 1
fi

ARGS=(
    --model "${MODEL}"
    --host 0.0.0.0
    --port 8001
    --ctx-size "${LLM_CTX:-8192}"
    --n-gpu-layers "${LLM_NGL:-99}"
    --parallel "${LLM_PARALLEL:-1}"
    --batch-size "${LLM_BATCH:-512}"
    --ubatch-size "${LLM_UBATCH:-128}"
    --threads "${LLM_THREADS:-4}"
    --cache-type-k "${LLM_CACHE_TYPE:-q8_0}"
    --cache-type-v "${LLM_CACHE_TYPE:-q8_0}"
    --flash-attn on
    # --jinja makes llama.cpp use the GGUF's own chat template, which is what
    # gives us Qwen3's native <tool_call> parsing on /v1/chat/completions.
    --jinja
    --mlock
    --alias assistant
)

# shellcheck disable=SC2206
EXTRA=(${LLM_EXTRA_ARGS:-})

echo "[llm] starting llama-server with: ${ARGS[*]} ${EXTRA[*]:-}"
exec /opt/llama/llama-server "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
