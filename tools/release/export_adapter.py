#!/usr/bin/env python3
"""Export the small trainable GLCLAP-Hotword state from a full training checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from glclap.checkpoint import load_adapter_state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", help="reuse a previously computed source hash")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.output.suffix != ".safetensors":
        parser.error("--output must end in .safetensors")

    legacy = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    metrics = legacy.get("metrics") or {}
    state = load_adapter_state(args.checkpoint)
    metadata = {
        "format": "glclap-adapter-v1",
        "experiment": "glclap_hotword_retriever",
        "base_model": "Qwen/Qwen3-ASR-1.7B",
        "base_model_revision": "d69410f1c275f2b0fa60cbb9960edfcdb0ae0aec",
        "source_checkpoint": args.checkpoint.name,
        "source_checkpoint_sha256": args.source_sha256 or sha256(args.checkpoint),
        "projection_dim": "512",
        "adapter_hidden": "1024",
        "score": "max_t cosine(audio_frame_t, hotword)",
        "metrics": json.dumps(metrics, sort_keys=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in state.items()}, str(args.output), metadata)

    # Strong round-trip check without instantiating the frozen base components.
    loaded = load_adapter_state(args.output)
    if loaded.keys() != state.keys():
        raise RuntimeError("safetensors round-trip changed tensor keys")
    for key in state:
        if not torch.equal(state[key], loaded[key]):
            raise RuntimeError(f"safetensors round-trip changed tensor {key}")
    with safe_open(str(args.output), framework="pt", device="cpu") as handle:
        saved_metadata = handle.metadata()
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                **metadata,
                "metrics": metrics,
                "tensor_count": len(state),
                "parameter_count": sum(tensor.numel() for tensor in state.values()),
                "sha256": sha256(args.output),
                "bytes": args.output.stat().st_size,
                "embedded_metadata": saved_metadata,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(sidecar.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
