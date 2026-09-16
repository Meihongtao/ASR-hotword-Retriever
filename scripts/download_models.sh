#!/usr/bin/env bash
# Download the frozen Qwen3-ASR base and verify the bundled 20 MiB adapter.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_ROOT="${GLCLAP_MODEL_DIR:-$ROOT/models}"
BASE_ID="${GLCLAP_BASE_MODEL_ID:-Qwen/Qwen3-ASR-1.7B}"
BASE_REV="${GLCLAP_BASE_MODEL_REVISION:-master}"

python -m modelscope.cli.cli download \
  --model "$BASE_ID" --revision "$BASE_REV" \
  --local_dir "$MODEL_ROOT/Qwen3-ASR-1.7B"

# ModelScope's downloader currently accepts branch/tag names but rejects the
# commit SHA returned in file metadata. Download master, then pin the exact
# GLCLAP-Hotword base content by hashing every runtime file (large shards included).
(
  cd "$MODEL_ROOT/Qwen3-ASR-1.7B"
  sha256sum --check "$ROOT/checksums/qwen3_asr_1.7b.sha256"
)

ADAPTER="$ROOT/weights/glclap_hotword_adapter.safetensors"
echo "c20f5b9869b27bc13e092fbbdd9ebab453970011172dccf2c47486b06a4ec22f  $ADAPTER" \
  | sha256sum --check --status -

echo "Base model: $MODEL_ROOT/Qwen3-ASR-1.7B (all runtime SHA256 OK)"
echo "Bundled adapter: $ADAPTER (SHA256 OK)"
