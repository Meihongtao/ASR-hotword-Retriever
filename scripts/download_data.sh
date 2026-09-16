#!/usr/bin/env bash
# Download the versioned GLCLAP-Hotword indexes/artifacts and, optionally, audio archives.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_ID="${GLCLAP_DATASET_ID:-Meiht0702/HotwordSpeech}"
REVISION="${GLCLAP_DATASET_REVISION:-master}"
DEST="${GLCLAP_MS_DATA_DIR:-${GLCLAP_DATA_DIR:-$ROOT/data}/modelscope}"
WITH_AUDIO=0
if [ "${1:-}" = "--with-audio" ]; then
  WITH_AUDIO=1
elif [ -n "${1:-}" ]; then
  echo "usage: $0 [--with-audio]" >&2
  exit 2
fi

python -m modelscope.cli.cli download \
  --dataset "$DATASET_ID" \
  --revision "$REVISION" \
  --local_dir "$DEST" \
  --include \
    'releases/glclap_hotword_v1/*.json' \
    'releases/glclap_hotword_v1/*.jsonl' \
    'releases/glclap_hotword_v1/*.sha256' \
    'releases/glclap_hotword_v1/LICENSES.md' \
    'releases/glclap_hotword_v1/artifacts/*'

(
  cd "$DEST/releases/glclap_hotword_v1"
  sha256sum --check checksums.sha256
)

if [ "$WITH_AUDIO" -eq 1 ]; then
  python -m modelscope.cli.cli download \
    --dataset "$DATASET_ID" \
    --revision "$REVISION" \
    --local_dir "$DEST" \
    --include 'releases/glclap_hotword_v1/archives/*'
  python "$ROOT/scripts/restore_archives.py" \
    --release-root "$DEST/releases/glclap_hotword_v1"
fi

echo "Downloaded $DATASET_ID@$REVISION to $DEST"
