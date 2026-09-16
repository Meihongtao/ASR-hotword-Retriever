#!/usr/bin/env python3
"""Build the canonical, path-safe GLCLAP-Hotword release indexes from training manifests.

Run this on the machine that still has the original manifests and audio trees.
No audio is copied.  The public indexes contain stable ``audio_id`` values;
``audio_assets.private.jsonl`` additionally contains local source paths for the
separate staging step and MUST NOT be published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def source_and_id(audio: str) -> tuple[str, str]:
    path = audio.replace("\\", "/")
    rules = (
        ("/common-voice-en/", "commonvoice"),
        ("/magicdata_mandarin_read/", "magicdata"),
        ("/stage2_hotword_datasets/ContextASR-Bench/", "contextasr"),
        ("/stage2_hotword_datasets/domain_TTS/TTS/", "domain_tts_full_meeting"),
        ("/datas/李沐学AI_dataset/", "limu_video"),
        ("/datas/极客湾_dataset/", "jikewan_video"),
    )
    for marker, source in rules:
        if marker in path:
            suffix = path.split(marker, 1)[1].lstrip("/")
            if source == "commonvoice":
                suffix = f"en/{suffix}" if not suffix.startswith("en/") else suffix
            elif source == "magicdata":
                # SLR68 train_set.tar.gz extracts with train/ at its root.
                suffix = suffix
            elif source == "contextasr":
                suffix = f"ContextASR-Bench/{suffix}"
            elif source == "domain_tts_full_meeting":
                suffix = f"domain_TTS/TTS/{suffix}"
            else:
                suffix = f"{marker.strip('/').split('/')[-1]}/{suffix}"
            return source, suffix

    for dirname in ("hungyi", "valley101", "mark", "idiode"):
        marker = f"/datas/{dirname}_dataset/"
        if marker in path:
            return "youtube_video", f"{dirname}_dataset/{path.split(marker, 1)[1].lstrip('/')}"

    for subset in ("ZH-B", "ZH-B1", "EN-B", "EN-B1"):
        marker = f"/datas/{subset}/"
        if marker in path:
            return "bilibili_video", f"{subset}/{path.split(marker, 1)[1].lstrip('/')}"

    return "other", Path(path).name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_entities(value) -> list[str]:
    return [x.strip() for x in (value or []) if isinstance(x, str) and x.strip()]


def build_split(
    split: str,
    source_manifest: Path,
    output_dir: Path,
    private_assets: dict[str, dict],
    include_full_index: bool,
) -> dict:
    public_path = output_dir / f"{split}_index.jsonl"
    effective_path = output_dir / f"{split}_effective.jsonl"
    excluded_path = output_dir / f"{split}_excluded.jsonl"
    counts: Counter = Counter()
    checked_dirs: dict[str, bool] = {}

    with (
        source_manifest.open(encoding="utf-8") as source_handle,
        public_path.open("w", encoding="utf-8") if include_full_index else open(os.devnull, "w") as public_handle,
        effective_path.open("w", encoding="utf-8") as effective_handle,
        excluded_path.open("w", encoding="utf-8") as excluded_handle,
    ):
        for line_number, line in enumerate(source_handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            audio = str(row.get("audio") or "")
            source, audio_id = source_and_id(audio)
            entities = clean_entities(row.get("entities"))
            item = {
                "id": f"{source}:{audio_id}",
                "audio_id": audio_id,
                "audio": None,
                "text": str(row.get("text") or ""),
                "language": str(row.get("language") or ""),
                "entities": entities,
                "source": source,
            }
            if include_full_index:
                public_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            counts["rows"] += 1
            counts[f"source_rows.{source}"] += 1

            reason = None
            if not entities:
                reason = "no_entities"
            elif not audio:
                reason = "missing_audio_path"
            else:
                directory = os.path.dirname(audio)
                if directory not in checked_dirs:
                    checked_dirs[directory] = os.path.isdir(directory)
                directory_ok = checked_dirs[directory]
                if not directory_ok:
                    reason = "missing_audio_directory"

            if reason:
                # Empty-entity rows were never eligible for the local loss and
                # would make this audit file >1M lines. Only preserve actual
                # release failures (the 425 missing-directory rows in GLCLAP-Hotword).
                if reason != "no_entities":
                    excluded_handle.write(
                        json.dumps({**item, "reason": reason}, ensure_ascii=False) + "\n"
                    )
                counts[f"excluded.{reason}"] += 1
                counts[f"source_excluded.{source}"] += 1
                continue

            effective_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            counts["effective_rows"] += 1
            counts[f"source_effective.{source}"] += 1
            if source not in {"commonvoice", "magicdata"}:
                asset = private_assets.setdefault(
                    audio_id,
                    {
                        "audio_id": audio_id,
                        "release_path": f"audio/{audio_id}",
                        "source": source,
                        "source_audio": audio,
                        "splits": [],
                    },
                )
                if split not in asset["splits"]:
                    asset["splits"].append(split)

    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument(
        "--include-full-index",
        action="store_true",
        help="also write all rows, including rows without entities (large and not needed for training)",
    )
    args = parser.parse_args()
    for path in (args.train, args.valid):
        if not path.is_file():
            parser.error(f"manifest not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, dict] = {}
    splits = {
        "train": build_split(
            "train", args.train, args.output_dir, assets, args.include_full_index
        ),
        "valid": build_split(
            "valid", args.valid, args.output_dir, assets, args.include_full_index
        ),
    }

    private_path = args.output_dir / "audio_assets.private.jsonl"
    public_path = args.output_dir / "audio_assets.jsonl"
    with (
        private_path.open("w", encoding="utf-8") as private_handle,
        public_path.open("w", encoding="utf-8") as public_handle,
    ):
        for audio_id in sorted(assets):
            item = assets[audio_id]
            private_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            public_handle.write(
                json.dumps(
                    {key: value for key, value in item.items() if key != "source_audio"},
                    ensure_ascii=False,
                )
                + "\n"
            )

    release = {
        "name": "GLCLAP-Hotword",
        "release": "glclap_hotword_v1",
        "format_version": 1,
        "public_audio_policy": {
            "commonvoice": "index_only; obtain Common Voice 26.0 English from Mozilla",
            "magicdata": "index_only; obtain MagicData SLR68 train_set.tar.gz",
            "other_sources": "self-collected public audio, stored under audio/ archives; research-use intent declaration only, not a license to third-party media",
        },
        "source_provenance": {
            "commonvoice": {
                "version": "Common Voice 26.0 English",
                "license": "CC0-1.0; Mozilla terms request no third-party mirrors",
                "upstream": "https://commonvoice.mozilla.org/en/datasets",
            },
            "magicdata": {
                "version": "OpenSLR SLR68 train_set.tar.gz",
                "license": "CC-BY-NC-ND-4.0",
                "upstream": "https://www.openslr.org/68/",
            },
            "contextasr": {
                "version": "ContextASR-Bench",
                "license": "MIT",
                "upstream": "https://huggingface.co/datasets/MrSupW/ContextASR-Bench",
            },
            "domain_tts_full_meeting": {
                "redistribution": "research use only; not to be redistributed commercially"
            },
            "video_sources": {
                "redistribution": "research use only; not to be redistributed commercially"
            },
        },
        "splits": splits,
        "unique_redistributable_audio_candidates": len(assets),
        "entity_extraction": {
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": "fcb1a040bb418b0b8add6f6f6c475386abc2cb97",
            "temperature": 0.0,
            "max_tokens": 300,
            "thinking": False,
            "prompt_sha256": sha256(args.prompt) if args.prompt else None,
        },
    }
    (args.output_dir / "release.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    names = [
        "train_effective.jsonl",
        "train_excluded.jsonl",
        "valid_effective.jsonl",
        "valid_excluded.jsonl",
        "audio_assets.jsonl",
        "release.json",
    ]
    if args.include_full_index:
        names.extend(["train_index.jsonl", "valid_index.jsonl"])
    if (args.output_dir / "LICENSES.md").is_file():
        names.append("LICENSES.md")
    # These files are produced independently but are part of the same public
    # release. Include them when present so one checksum file verifies every
    # small/medium artifact needed for training and inference. Large audio tar
    # volumes keep their own inventory in archives/archives.json.
    for dirname in ("artifacts", "model"):
        root = args.output_dir / dirname
        if root.is_dir():
            names.extend(
                str(path.relative_to(args.output_dir))
                for path in sorted(root.rglob("*"))
                if path.is_file()
            )
    checksums = []
    for name in names:
        checksums.append(f"{sha256(args.output_dir / name)}  {name}")
    (args.output_dir / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    print(json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"PRIVATE (do not upload): {private_path}")


if __name__ == "__main__":
    main()
