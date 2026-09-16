#!/usr/bin/env python3
"""Materialize portable GLCLAP-Hotword indexes into manifests with local audio paths."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def resolve_public(root: Path, audio_id: str, source: str) -> Path:
    candidates = [root / audio_id]
    if source == "commonvoice" and audio_id.startswith("en/"):
        candidates.append(root / audio_id.removeprefix("en/"))
    if source == "magicdata" and audio_id.startswith("train/"):
        candidates.append(root / audio_id.removeprefix("train/"))
    matches = [path for path in candidates if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and len({path.resolve() for path in matches}) == 1:
        return matches[0]
    return candidates[0]


def materialize(
    index: Path,
    output: Path,
    release_root: Path,
    commonvoice_root: Path,
    magicdata_root: Path,
) -> Counter:
    counts: Counter = Counter()
    missing: list[dict] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with index.open(encoding="utf-8") as source_handle, output.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in source_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row["source"])
            audio_id = str(row["audio_id"])
            if source == "commonvoice":
                path = resolve_public(commonvoice_root, audio_id, source)
            elif source == "magicdata":
                path = resolve_public(magicdata_root, audio_id, source)
            else:
                path = release_root / "audio" / audio_id
            counts["rows"] += 1
            counts[f"source_rows.{source}"] += 1
            if not path.is_file():
                counts["missing"] += 1
                counts[f"source_missing.{source}"] += 1
                if len(missing) < 1000:
                    missing.append({"source": source, "audio_id": audio_id, "expected": str(path)})
                continue
            row["audio"] = str(path.resolve())
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["written"] += 1

    missing_path = output.with_suffix(output.suffix + ".missing.json")
    if missing:
        missing_path.write_text(
            json.dumps(missing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    elif missing_path.exists():
        missing_path.unlink()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--commonvoice-root", type=Path, required=True)
    parser.add_argument("--magicdata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/glclap_hotword"))
    args = parser.parse_args()
    index_root = args.release_root / "releases" / "glclap_hotword_v1"
    if not index_root.is_dir():
        index_root = args.release_root
    all_counts = {}
    for split in ("train", "valid"):
        index = index_root / f"{split}_effective.jsonl"
        if not index.is_file():
            parser.error(f"release index not found: {index}")
        all_counts[split] = dict(
            materialize(
                index,
                args.output_dir / f"{split}.jsonl",
                index_root,
                args.commonvoice_root,
                args.magicdata_root,
            )
        )
    artifact_source = index_root / "artifacts"
    artifact_link = args.output_dir / "artifacts"
    if not artifact_source.is_dir():
        parser.error(f"release artifacts not found: {artifact_source}")
    if artifact_link.is_symlink():
        if artifact_link.resolve() != artifact_source.resolve():
            parser.error(f"existing artifact link points elsewhere: {artifact_link}")
    elif artifact_link.exists():
        if artifact_link.resolve() != artifact_source.resolve():
            parser.error(
                f"refusing to replace existing artifact directory: {artifact_link}"
            )
    else:
        relative_target = os.path.relpath(artifact_source, args.output_dir)
        artifact_link.symlink_to(relative_target, target_is_directory=True)
    all_counts["artifacts"] = {"path": str(artifact_link.resolve())}
    print(json.dumps(all_counts, ensure_ascii=False, indent=2, sort_keys=True))
    if any(item.get("missing") for item in all_counts.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
