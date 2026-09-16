#!/usr/bin/env bash
# Two-step single-GPU integration test using a tiny subset of materialized GLCLAP-Hotword.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${GLCLAP_DATA_DIR:-$ROOT/data}/glclap_hotword"
SMOKE_ROOT="${GLCLAP_SMOKE_DATA_DIR:-${GLCLAP_DATA_DIR:-$ROOT/data}/glclap_hotword-smoke}"
SMOKE_RUN="${GLCLAP_SMOKE_RUN_DIR:-${GLCLAP_RUNS_DIR:-$ROOT/runs}/smoke-glclap-hotword}"

for path in "$DATA_ROOT/train.jsonl" "$DATA_ROOT/valid.jsonl"; do
  if [ ! -f "$path" ]; then
    echo "ERROR: $path not found; prepare the GLCLAP-Hotword data first" >&2
    exit 1
  fi
done
mkdir -p "$SMOKE_ROOT"
head -n 64 "$DATA_ROOT/train.jsonl" > "$SMOKE_ROOT/train.jsonl"
head -n 16 "$DATA_ROOT/valid.jsonl" > "$SMOKE_ROOT/valid.jsonl"
ln -sfn "$DATA_ROOT/artifacts" "$SMOKE_ROOT/artifacts"

GLCLAP_NPROC_PER_NODE=1 \
GLCLAP_TRAIN_MANIFEST="$SMOKE_ROOT/train.jsonl" \
GLCLAP_VALID_MANIFEST="$SMOKE_ROOT/valid.jsonl" \
GLCLAP_OUTPUT_DIR="$SMOKE_RUN" \
bash "$ROOT/scripts/train_glclap_hotword.sh" \
  --epochs 1 --max-steps 2 --grad-accum 1 \
  --pool-neg 16 --hardneg-topk 2 \
  --eval-every-steps 0 --save-every-steps 1 \
  --valid-subsample 8 --num-workers 1 --keep-checkpoints 2

echo "Smoke training completed: $SMOKE_RUN"
