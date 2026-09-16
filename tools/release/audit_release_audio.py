#!/usr/bin/env python3
"""Audit whether the staged ModelScope assets reproduce GLCLAP-Hotword audio inputs.

This intentionally distinguishes an exact training input from a derived clip.
For example, GLCLAP-Hotword points at a full ``meeting.wav`` for several TTS rows, while
the first ModelScope staging pass contains per-utterance slices.  The latter is
useful data but is not byte-equivalent input for exact release reproduction.

The script is read-only unless ``--output`` is supplied.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


PUBLIC_SOURCES = {"commonvoice", "magicdata"}
STAGED_MANIFESTS = {
    "tts_slices": "tts_meetings/manifest.jsonl",
    "youtube_slices": "youtube_slices/manifest.jsonl",
    "bilibili_slices": "bilibili_slices/manifest.jsonl",
}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{number}: {exc}") from exc


def subtype(row: dict) -> str:
    source = str(row.get("source") or "other")
    audio_id = str(row.get("audio_id") or "")
    if source == "tts_meeting":
        if audio_id.startswith("ContextASR-Bench/"):
            return "contextasr"
        if audio_id.startswith("domain_TTS/"):
            return "domain_tts_full_meeting"
    return source


def staged_key(row: dict, group: str) -> str | None:
    """Return a release-style asset id for byte-identical staged files."""
    source_audio = str(row.get("source_audio") or "").replace("\\", "/")
    if group == "youtube_slices":
        marker = "/datas/"
        return source_audio.split(marker, 1)[1] if marker in source_audio else None
    if group == "bilibili_slices":
        subset = {
            "bilibili-zh-b": "ZH-B",
            "bilibili-zh-b1": "ZH-B1",
            "bilibili-en-b": "EN-B",
            "bilibili-en-b1": "EN-B1",
        }.get(str(row.get("subset") or ""))
        name = Path(source_audio).name
        return f"{subset}/{name}" if subset and name else None
    # tts_meetings contains utterance slices, while GLCLAP-Hotword used whole meetings or
    # ContextASR wavs. Never claim those derived slices are exact coverage.
    return None


def load_staged(staging: Path, check_local_files: bool = False) -> tuple[set[str], dict[str, dict]]:
    exact_keys: set[str] = set()
    summary: dict[str, dict] = {}
    for group, rel in STAGED_MANIFESTS.items():
        manifest = staging / rel
        rows = with_audio = existing = 0
        keys: set[str] = set()
        if manifest.is_file():
            for row in iter_jsonl(manifest):
                rows += 1
                audio = row.get("audio")
                if audio:
                    with_audio += 1
                    if check_local_files and (manifest.parent / str(audio)).is_file():
                        existing += 1
                key = staged_key(row, group)
                if key:
                    keys.add(key)
        exact_keys.update(keys)
        summary[group] = {
            "manifest": str(manifest),
            "rows": rows,
            "rows_with_audio_path": with_audio,
            "local_audio_files_found": existing if check_local_files else None,
            "unique_exact_release_keys": len(keys),
        }
    return exact_keys, summary


def audit(exact_dir: Path, staging: Path, check_local_files: bool = False) -> dict:
    staged_keys, staged_summary = load_staged(staging, check_local_files)
    row_counts: Counter = Counter()
    unique: dict[str, set[str]] = defaultdict(set)
    covered_rows: Counter = Counter()
    covered_unique: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[str]] = defaultdict(list)

    for split in ("train", "valid"):
        path = exact_dir / f"{split}_effective.jsonl"
        if not path.is_file():
            # Backward-compatible name used by the pre-release audit staging.
            path = exact_dir / f"{split}_effective_entities.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            source = str(row.get("source") or "other")
            if source in PUBLIC_SOURCES:
                continue
            kind = subtype(row)
            audio_id = str(row.get("audio_id") or "")
            key = f"{split}:{kind}"
            row_counts[key] += 1
            unique[key].add(audio_id)
            if audio_id in staged_keys:
                covered_rows[key] += 1
                covered_unique[key].add(audio_id)
            elif len(examples[key]) < 5:
                examples[key].append(audio_id)

    coverage = {}
    for key in sorted(row_counts):
        total_unique = len(unique[key])
        hit_unique = len(covered_unique[key])
        coverage[key] = {
            "rows": row_counts[key],
            "unique_audio": total_unique,
            "covered_rows_by_exact_staged_audio": covered_rows[key],
            "covered_unique_audio": hit_unique,
            "unique_coverage_ratio": round(hit_unique / total_unique, 6)
            if total_unique else 1.0,
            "missing_examples": examples[key],
        }

    return {
        "definition": (
            "Coverage means the staged file is the same logical audio input "
            "used by GLCLAP-Hotword; derived TTS utterance slices do not cover a GLCLAP-Hotword "
            "full-meeting input."
        ),
        "exact_index_dir": str(exact_dir),
        "staging_dir": str(staging),
        "staged_manifests": staged_summary,
        "coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-local-files",
        action="store_true",
        help="stat every staged audio file (slow on network filesystems)",
    )
    args = parser.parse_args()
    report = audit(args.exact_dir, args.staging_dir, args.check_local_files)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
