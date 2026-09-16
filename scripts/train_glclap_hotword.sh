#!/usr/bin/env bash
# Reproduce the published GLCLAP-Hotword experiment.
# Paths may be overridden without editing this file.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${GLCLAP_DATA_DIR:-$ROOT/data}/glclap_hotword"
MODEL_ROOT="${GLCLAP_MODEL_DIR:-$ROOT/models}"
RUN_ROOT="${GLCLAP_RUNS_DIR:-$ROOT/runs}"
NPROC="${GLCLAP_NPROC_PER_NODE:-2}"

TRAIN_MANIFEST="${GLCLAP_TRAIN_MANIFEST:-$DATA_ROOT/train.jsonl}"
VALID_MANIFEST="${GLCLAP_VALID_MANIFEST:-$DATA_ROOT/valid.jsonl}"
QWEN_CHECKPOINT="${GLCLAP_QWEN3ASR_CHECKPOINT:-$MODEL_ROOT/Qwen3-ASR-1.7B}"
OUTPUT_DIR="${GLCLAP_OUTPUT_DIR:-$RUN_ROOT/glclap_hotword_retriever}"

for path in \
  "$TRAIN_MANIFEST" "$VALID_MANIFEST" "$QWEN_CHECKPOINT" \
  "$DATA_ROOT/artifacts/pool_zh_q3.pt" "$DATA_ROOT/artifacts/pool_en_q3.pt" \
  "$DATA_ROOT/artifacts/pool_zh.txt" "$DATA_ROOT/artifacts/pool_en.txt" \
  "$DATA_ROOT/artifacts/hardneg.zh.json" "$DATA_ROOT/artifacts/hardneg.en.json"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: required GLCLAP-Hotword input not found: $path" >&2
    echo "Run scripts/download_data.sh and scripts/prepare_data.py first." >&2
    exit 1
  fi
done

cd "$ROOT"
torchrun --nproc_per_node="$NPROC" -m glclap.train_amphion_ddp \
  --train-manifest "$TRAIN_MANIFEST" \
  --valid-manifest "$VALID_MANIFEST" \
  --qwen3asr-checkpoint "$QWEN_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 10 \
  --batch-size 4 \
  --grad-accum 16 \
  --max-audio-seconds 14 \
  --lr 3e-4 \
  --num-workers "${GLCLAP_NUM_WORKERS:-8}" \
  --prefetch-factor 2 \
  --seed 7 \
  --adapter-hidden 1024 \
  --sampler entity \
  --pool-zh-raw "$DATA_ROOT/artifacts/pool_zh_q3.pt" \
  --pool-en-raw "$DATA_ROOT/artifacts/pool_en_q3.pt" \
  --pool-zh-words "$DATA_ROOT/artifacts/pool_zh.txt" \
  --pool-en-words "$DATA_ROOT/artifacts/pool_en.txt" \
  --pool-neg 4095 \
  --hardneg-zh "$DATA_ROOT/artifacts/hardneg.zh.json" \
  --hardneg-en "$DATA_ROOT/artifacts/hardneg.en.json" \
  --hardneg-topk 10 \
  --warmup-ratio 0.05 \
  --patience 5 \
  --eval-every-steps 9095 \
  --valid-subsample 8000 \
  --eval-pool 1000 \
  --save-every-steps 2000 \
  --keep-checkpoints 5 \
  --keep-best 5 \
  --tb-every-steps 100 \
  "$@"
