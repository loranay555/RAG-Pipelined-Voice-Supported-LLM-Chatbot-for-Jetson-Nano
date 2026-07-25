#!/usr/bin/env bash
# Downloads every model the stack needs into ./models/.
# Safe to re-run: existing, non-empty files are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
[[ -f .env ]] && source .env

LLM_MODEL_FILE="${LLM_MODEL_FILE:-Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
WHISPER_MODEL_FILE="${WHISPER_MODEL_FILE:-ggml-small-q5_1.bin}"

LLM_REPO="${LLM_REPO:-unsloth/Qwen3-4B-Instruct-2507-GGUF}"
WHISPER_REPO="${WHISPER_REPO:-ggerganov/whisper.cpp}"
EMBED_REPO="${EMBED_REPO:-Xenova/multilingual-e5-small}"

HF="${HF_ENDPOINT:-https://huggingface.co}"

fetch() {
    local url="$1" dest="$2"
    if [[ -s "${dest}" ]]; then
        echo "  ✓ $(basename "${dest}") already present ($(du -h "${dest}" | cut -f1))"
        return
    fi
    echo "  ↓ $(basename "${dest}")"
    mkdir -p "$(dirname "${dest}")"
    curl -fL --retry 5 --retry-delay 3 --progress-bar -o "${dest}.part" "${url}"
    mv "${dest}.part" "${dest}"
}

echo "== LLM (llama.cpp) =="
fetch "${HF}/${LLM_REPO}/resolve/main/${LLM_MODEL_FILE}" "models/llm/${LLM_MODEL_FILE}"

echo "== STT (whisper.cpp) =="
fetch "${HF}/${WHISPER_REPO}/resolve/main/${WHISPER_MODEL_FILE}" "models/whisper/${WHISPER_MODEL_FILE}"

echo "== Embeddings (onnxruntime) =="
fetch "${HF}/${EMBED_REPO}/resolve/main/onnx/model.onnx" \
      "models/embedding/multilingual-e5-small/model.onnx"
fetch "${HF}/${EMBED_REPO}/resolve/main/tokenizer.json" \
      "models/embedding/multilingual-e5-small/tokenizer.json"

echo
echo "Done. Contents of ./models:"
du -h -d 3 models | sort -k2
