#!/usr/bin/env python3
"""Validate materialized GLCLAP-Hotword manifests before training."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


EXPECTED_TRAIN_ROWS = 1_164_214
EXPECTED_VALID_ROWS = 10_339


def verify(path: Path, check_audio: bool) -> Counter:
    counts: Counter = Counter()
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{number}: {exc}") from exc
            for field in ("audio", "text", "language", "entities", "source", "audio_id"):
                if field not in row:
                    raise ValueError(f"missing {field!r} at {path}:{number}")
            if not row["entities"]:
                raise ValueError(f"empty entities in effective manifest at {path}:{number}")
            if not isinstance(row["entities"], list):
                raise ValueError(f"entities is not a list at {path}:{number}")
            if check_audio and not Path(row["audio"]).is_file():
                raise FileNotFoundError(f"audio missing at {path}:{number}: {row['audio']}")
            row_id = str(row.get("id") or "")
            if row_id in seen_ids:
                counts["duplicate_ids"] += 1
            seen_ids.add(row_id)
            source = str(row["source"])
            counts["rows"] += 1
            counts[f"source_rows.{source}"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--allow-noncanonical-train-size", action="store_true")
    args = parser.parse_args()
    results = {}
    for split in ("train", "valid"):
        path = args.data_dir / f"{split}.jsonl"
        if not path.is_file():
            parser.error(f"manifest not found: {path}")
        results[split] = dict(verify(path, not args.skip_audio))
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    train_rows = results["train"].get("rows", 0)
    valid_rows = results["valid"].get("rows", 0)
    if not args.allow_noncanonical_train_size:
        if train_rows != EXPECTED_TRAIN_ROWS:
            raise SystemExit(
                f"expected {EXPECTED_TRAIN_ROWS} GLCLAP-Hotword train rows, "
                f"found {train_rows}"
            )
        if valid_rows != EXPECTED_VALID_ROWS:
            raise SystemExit(
                f"expected {EXPECTED_VALID_ROWS} GLCLAP-Hotword valid rows, "
                f"found {valid_rows}"
            )
    duplicates = {
        split: counts.get("duplicate_ids", 0)
        for split, counts in results.items()
        if counts.get("duplicate_ids", 0)
    }
    if duplicates:
        raise SystemExit(f"duplicate manifest ids detected: {duplicates}")


if __name__ == "__main__":
    main()
