#!/usr/bin/env bash
set -euo pipefail

MODEL="${WHISPER_MODEL:-/models/ggml-small-q5_1.bin}"

if [[ ! -f "${MODEL}" ]]; then
    echo "=============================================================" >&2
    echo " Whisper model not found: ${MODEL}" >&2
    echo " Put it under ./models/whisper/ on the host. See scripts/download_models.sh" >&2
    echo "=============================================================" >&2
    exit 1
fi

echo "[stt] starting whisper-server with ${MODEL} (lang=${WHISPER_LANGUAGE:-auto})"
exec /opt/whisper/whisper-server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port 8003 \
    --threads "${WHISPER_THREADS:-4}" \
    --language "${WHISPER_LANGUAGE:-auto}" \
    --inference-path /inference
