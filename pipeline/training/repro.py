
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def fix_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git(cmd: list[str], cwd: Path) -> str | None:
    try:
        res = subprocess.run(["git", *cmd], cwd=cwd, capture_output=True, text=True, timeout=4)
        if res.returncode == 0:
            return res.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return None


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def _dataset_signature(manifest_csv: Path | None) -> str | None:
    if manifest_csv is None or not manifest_csv.is_file():
        return None
    h = hashlib.sha256()
    with manifest_csv.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_repro_snapshot(
    repo_root: Path | None = None,
    manifest_csv: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    repo = (repo_root or Path.cwd()).resolve()
    snap: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torchvision": _package_version("torchvision"),
        "transformers": _package_version("transformers"),
        "timm": _package_version("timm"),
        "peft": _package_version("peft"),
        "wandb": _package_version("wandb"),
        "numpy": _package_version("numpy"),
        "opencv": _package_version("opencv-python") or _package_version("opencv-python-headless"),
        "git_commit": _git(["rev-parse", "--short", "HEAD"], repo),
        "git_dirty": (
            _git(["status", "--porcelain"], repo) != ""
            if _git(["status", "--porcelain"], repo) is not None else None
        ),
        "manifest_sha16": _dataset_signature(manifest_csv),
        "env_seed": os.environ.get("PYTHONHASHSEED"),
    }
    if extra:
        snap["extra"] = extra
    return snap


def write_repro(run_dir: Path, snapshot: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "repro.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
