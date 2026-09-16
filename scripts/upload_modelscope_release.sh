#!/usr/bin/env bash
# Resumable maintainer upload. The token is read from the environment or hidden stdin.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
RELEASE_ROOT="${GLCLAP_RELEASE_ROOT:-$WORKSPACE/modelscope_data/staging/releases/glclap_hotword_v1}"
PYTHON_BIN="${GLCLAP_PYTHON_BIN:-python3}"
LOG_FILE="${GLCLAP_UPLOAD_LOG:-$WORKSPACE/modelscope_upload.log}"
STATE_FILE="${GLCLAP_UPLOAD_STATE:-$WORKSPACE/modelscope_upload.completed}"

if [ -z "${MODELSCOPE_API_TOKEN:-}" ]; then
  read -r -s -p "ModelScope token: " MODELSCOPE_API_TOKEN
  printf '\n'
  export MODELSCOPE_API_TOKEN
fi

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
chmod 600 "$LOG_FILE"

{
  printf '[%s] starting resumable GLCLAP-Hotword audio upload\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON_BIN" -u "$ROOT/tools/release/upload_modelscope.py" \
    --release-root "$RELEASE_ROOT" \
    --audio-only \
    --retries 3 \
    --state-file "$STATE_FILE" \
    --no-remote-scan \
    --assume-repo-exists \
    "$@"
  printf '[%s] upload process completed successfully\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >>"$LOG_FILE" 2>&1
