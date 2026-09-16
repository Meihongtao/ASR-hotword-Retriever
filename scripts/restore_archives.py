#!/usr/bin/env python3
"""Verify and extract independently downloadable GLCLAP-Hotword tar volumes."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, root: Path) -> None:
    root_abs = root.resolve()
    with tarfile.open(archive, "r") as tar:
        for member in tar:
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(
                    f"unsupported archive member in {archive}: {member.name}"
                )
            target = (root / member.name).resolve()
            if root_abs != target and root_abs not in target.parents:
                raise ValueError(f"unsafe archive member in {archive}: {member.name}")
            tar.extract(member, root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    inventory_path = args.release_root / "archives" / "archives.json"
    if not inventory_path.is_file():
        parser.error(f"archive inventory not found: {inventory_path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for item in inventory["archives"]:
        archive = args.release_root / "archives" / item["name"]
        if not archive.is_file():
            raise FileNotFoundError(archive)
        actual = sha256(archive)
        if actual != item["sha256"]:
            raise ValueError(f"checksum mismatch for {archive}: {actual}")
        print(f"[extract] {archive.name}", flush=True)
        safe_extract(archive, args.release_root)


if __name__ == "__main__":
    main()
