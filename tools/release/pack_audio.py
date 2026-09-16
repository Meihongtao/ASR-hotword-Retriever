#!/usr/bin/env python3
"""Create deterministic, independent tar volumes from exact release assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_assets(path: Path, allowed: set[str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["source"] in allowed:
                groups[item["source"]].append(item)
    return groups


def volumes(items: list[dict], max_bytes: int):
    current, current_bytes = [], 0
    for item in sorted(items, key=lambda row: row["audio_id"]):
        size = os.path.getsize(item["source_audio"])
        if current and current_bytes + size > max_bytes:
            yield current
            current, current_bytes = [], 0
        current.append(item)
        current_bytes += size
    if current:
        yield current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="explicitly approved sources to package; never defaults to copyrighted video",
    )
    parser.add_argument("--max-volume-gib", type=float, default=7.5)
    parser.add_argument(
        "--volume-index",
        type=int,
        action="append",
        help="package only selected zero-based volumes (requires exactly one source)",
    )
    args = parser.parse_args()
    if args.volume_index and len(args.sources) != 1:
        parser.error("--volume-index requires exactly one --sources value")
    selected_volumes = set(args.volume_index or [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = load_assets(args.asset_manifest, set(args.sources))
    missing_sources = set(args.sources) - set(grouped)
    if missing_sources:
        parser.error(f"no assets found for sources: {sorted(missing_sources)}")

    inventory_path = args.output_dir / "archives.json"
    if inventory_path.is_file():
        existing = json.loads(inventory_path.read_text(encoding="utf-8"))
        by_name = {item["name"]: item for item in existing.get("archives", [])}
    else:
        by_name = {}
    max_bytes = int(args.max_volume_gib * 1024**3)
    for source in sorted(grouped):
        for number, batch in enumerate(volumes(grouped[source], max_bytes)):
            if selected_volumes and number not in selected_volumes:
                continue
            name = f"glclap_hotword_{source}_part_{number:03d}.tar"
            final_path = args.output_dir / name
            partial_path = final_path.with_suffix(".tar.partial")
            if final_path.is_file():
                print(f"[skip-existing] {final_path}", flush=True)
            else:
                with tarfile.open(partial_path, "w", format=tarfile.PAX_FORMAT) as archive:
                    for item in batch:
                        info = archive.gettarinfo(
                            item["source_audio"], arcname=item["release_path"]
                        )
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        with open(item["source_audio"], "rb") as audio:
                            archive.addfile(info, audio)
                partial_path.replace(final_path)
            item = {
                "name": name,
                "source": source,
                "files": len(batch),
                "bytes": final_path.stat().st_size,
                "sha256": sha256(final_path),
            }
            by_name[name] = item
            print(json.dumps(item, sort_keys=True), flush=True)

    inventory = {"format_version": 1, "archives": [by_name[name] for name in sorted(by_name)]}
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
