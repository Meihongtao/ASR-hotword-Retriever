"""Repository path configuration.

Every absolute path in the repo resolves through this module so the whole
project is relocatable and reproducible on a new machine. Nothing here may
hard-code a machine-specific location.

Default layout (all relative to the repo root):

    GLCLAP_HOME                  repo root (auto-detected, or $GLCLAP_HOME)
    |-- data/                    datasets (download from ModelScope)
    |   `-- glclap_hotword/     materialized train/valid + artifacts
    |-- models/                  base models (Qwen3-ASR-1.7B, ...)
    |-- checkpoints/             trained GLCLAP checkpoints
    |-- runs/                    training outputs
    `-- third_party/             optional local FunASR source override

Environment overrides (all optional):
    GLCLAP_HOME         repo root
    GLCLAP_DATA_DIR     datasets dir
    GLCLAP_MODEL_DIR    base models dir
    GLCLAP_CKPT_DIR     checkpoints dir
    GLCLAP_RUNS_DIR     training output dir
    GLCLAP_FUNASR_ROOT  directory that contains the `funasr` package
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("GLCLAP_HOME")
    if env:
        return Path(env)
    # glclap/env.py -> repo root
    return Path(__file__).resolve().parent.parent


def _pick(env_name: str, *rel: str) -> Path:
    env = os.environ.get(env_name)
    if env:
        return Path(env)
    return repo_root().joinpath(*rel)


def data_dir() -> Path:
    return _pick("GLCLAP_DATA_DIR", "data")


def model_dir() -> Path:
    return _pick("GLCLAP_MODEL_DIR", "models")


def ckpt_dir() -> Path:
    return _pick("GLCLAP_CKPT_DIR", "checkpoints")


def runs_dir() -> Path:
    return _pick("GLCLAP_RUNS_DIR", "runs")


def funasr_candidates() -> list[str]:
    """Directories that contain a top-level `funasr` package (in priority
    order). Used to make the Qwen3ASR modeling code importable; only the
    first existing candidate is actually added to sys.path."""
    cands: list[str] = []
    env = os.environ.get("GLCLAP_FUNASR_ROOT")
    if env:
        cands.append(env)
    cands.append(str(repo_root() / "third_party" / "funasr"))
    return cands
