#!/usr/bin/env python3
"""Upload the curated GLCLAP-Hotword release without exposing credentials or local paths."""
from __future__ import annotations

import argparse
import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


CORE_NAMES = {
    "audio_assets.jsonl",
    "checksums.sha256",
    "LICENSES.md",
    "release.json",
    "train_effective.jsonl",
    "train_excluded.jsonl",
    "valid_effective.jsonl",
    "valid_excluded.jsonl",
}


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_hashes(root: Path) -> dict[str, str]:
    prefix = "releases/glclap_hotword_v1/"
    hashes: dict[str, str] = {}
    checksum_file = root / "checksums.sha256"
    if checksum_file.is_file():
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split(maxsplit=1)
            hashes[prefix + relative.lstrip("* ")] = digest
    inventory_file = root / "archives" / "archives.json"
    if inventory_file.is_file():
        import json

        inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
        for item in inventory["archives"]:
            hashes[prefix + "archives/" + item["name"]] = item["sha256"]
        hashes[prefix + "archives/archives.json"] = sha256(inventory_file)
    return hashes


def remote_files(api, dataset_id: str) -> dict[str, dict]:
    namespace, dataset_name = dataset_id.split("/", 1)
    root = "releases/glclap_hotword_v1"
    response = api.list_repo_tree(
        dataset_name, namespace, "master", root, True, 1, 500
    )
    files = response.get("Data", {}).get("Files", [])
    result = {}
    for item in files:
        if str(item.get("Type", item.get("type", ""))).lower() in {"tree", "dir"}:
            continue
        path = item.get("Path") or item.get("Name") or item.get("path") or item.get("name")
        if not path:
            continue
        path = str(path).lstrip("/")
        if not path.startswith(root + "/"):
            path = root + "/" + path
        result[path] = item
    return result


def remote_value(item: dict, *names: str):
    for name in names:
        if name in item:
            return item[name]
    return None


def release_files(root: Path, with_audio: bool, audio_only: bool = False) -> list[Path]:
    files = [] if audio_only else [root / name for name in sorted(CORE_NAMES)]
    if not audio_only:
        for dirname in ("artifacts",):
            files.extend(path for path in sorted((root / dirname).rglob("*")) if path.is_file())
    if with_audio or audio_only:
        archive_root = root / "archives"
        files.extend(path for path in sorted(archive_root.glob("*.tar")) if path.is_file())
        files.append(archive_root / "archives.json")
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing release files: " + ", ".join(map(str, missing)))
    forbidden = [path for path in files if "private" in path.name or path.suffix == ".partial"]
    if forbidden:
        raise ValueError("refusing to upload private/partial files: " + ", ".join(map(str, forbidden)))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default="Meiht0702/HotwordSpeech")
    parser.add_argument("--dataset-card", type=Path)
    parser.add_argument("--with-audio", action="store_true")
    parser.add_argument(
        "--audio-only", action="store_true", help="upload only archives and their inventory"
    )
    parser.add_argument(
        "--card-only", action="store_true", help="upload only --dataset-card as README.md"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--no-remote-scan", action="store_true")
    parser.add_argument(
        "--assume-repo-exists",
        action="store_true",
        help="skip the SDK create/repo-exists request for an already-created repository",
    )
    args = parser.parse_args()

    modes = sum((args.with_audio, args.audio_only, args.card_only))
    if modes > 1:
        parser.error("--with-audio, --audio-only and --card-only are mutually exclusive")
    if args.card_only and not args.dataset_card:
        parser.error("--card-only requires --dataset-card")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require --shard-count >= 1 and 0 <= --shard-index < --shard-count")
    files = [] if args.card_only else release_files(
        args.release_root, args.with_audio, args.audio_only
    )
    uploads: list[tuple[Path, str]] = [
        (path, f"releases/glclap_hotword_v1/{path.relative_to(args.release_root).as_posix()}")
        for path in files
    ]
    if args.dataset_card and not args.audio_only:
        if not args.dataset_card.is_file():
            parser.error(f"dataset card not found: {args.dataset_card}")
        uploads.insert(0, (args.dataset_card, "README.md"))

    uploads = [
        upload
        for number, upload in enumerate(uploads)
        if number % args.shard_count == args.shard_index
    ]

    total = sum(path.stat().st_size for path, _ in uploads)
    log(
        f"dataset={args.dataset_id} shard={args.shard_index}/{args.shard_count} "
        f"files={len(uploads)} bytes={total}"
    )
    for path, remote in uploads:
        print(f"{path.stat().st_size:>12}  {remote}")
    if args.dry_run:
        return

    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        raise SystemExit(
            "MODELSCOPE_API_TOKEN is not set; export a newly created token in your shell"
        )
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)
    hashes = expected_hashes(args.release_root)
    if args.assume_repo_exists:
        api.create_repo = lambda *unused_args, **unused_kwargs: None
        log("SDK repository existence check disabled for known existing dataset")
    if args.no_remote_scan:
        existing = {}
        log("remote scan disabled; using the durable local completion state")
    else:
        try:
            existing = remote_files(api, args.dataset_id)
            log(f"remote scan complete: {len(existing)} files under release root")
        except Exception as exc:
            log(f"remote scan failed ({exc!r}); continuing without skip optimization")
            existing = {}

    completed: set[str] = set()
    if args.state_file and args.state_file.is_file():
        completed = {
            line.strip()
            for line in args.state_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if args.state_file:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        log(f"loaded {len(completed)} completed files from {args.state_file}")

    def mark_completed(remote: str) -> None:
        if not args.state_file or remote in completed:
            return
        with args.state_file.open("a", encoding="utf-8") as handle:
            handle.write(remote + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        completed.add(remote)

    completed_bytes = 0
    for path, remote in uploads:
        size = path.stat().st_size
        if remote in completed:
            completed_bytes += size
            log(
                f"skip durable local completion {remote} "
                f"({completed_bytes}/{total} bytes complete)"
            )
            continue
        expected = hashes.get(remote)
        remote_item = existing.get(remote, {})
        remote_size = remote_value(remote_item, "Size", "size")
        remote_sha = remote_value(remote_item, "Sha256", "sha256", "BlobId", "blob_id")
        if expected and str(remote_sha).lower() == expected and int(remote_size or -1) == size:
            completed_bytes += size
            mark_completed(remote)
            log(
                f"skip verified remote file {remote} "
                f"({completed_bytes}/{total} bytes complete)"
            )
            continue

        for attempt in range(1, args.retries + 1):
            started = time.monotonic()
            stop_heartbeat = threading.Event()

            def heartbeat() -> None:
                while not stop_heartbeat.wait(60):
                    elapsed = int(time.monotonic() - started)
                    log(f"upload heartbeat file={remote} elapsed={elapsed}s")

            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            log(f"upload start attempt={attempt}/{args.retries} file={remote} bytes={size}")
            try:
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=remote,
                    repo_id=args.dataset_id,
                    repo_type="dataset",
                    token=token,
                    commit_message=f"Publish GLCLAP-Hotword: {remote}",
                    buffer_size_mb=8,
                    disable_tqdm=False,
                )
            except Exception as exc:
                log(f"upload failed attempt={attempt} file={remote}: {exc!r}")
                if attempt == args.retries:
                    raise
                time.sleep(min(30, attempt * 5))
            else:
                completed_bytes += size
                mark_completed(remote)
                elapsed = int(time.monotonic() - started)
                log(
                    f"upload done file={remote} elapsed={elapsed}s "
                    f"({completed_bytes}/{total} bytes complete)"
                )
                break
            finally:
                stop_heartbeat.set()
                thread.join(timeout=1)

    log("all requested files uploaded successfully")


if __name__ == "__main__":
    main()
