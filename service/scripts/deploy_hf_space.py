from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

README_TEMPLATE = """---
title: Archstyle 55 Backend
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: cpu-basic
---

# Archstyle 55 — FastAPI backend

55-class architectural style classifier. See `/docs` for the API.
"""


def _copy(src: Path, dst: Path, *, ignore: tuple[str, ...] = ()) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    for entry in src.iterdir():
        if entry.name in ignore:
            continue
        if any(entry.match(pat) for pat in ("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")):
            continue
        _copy(entry, dst / entry.name, ignore=ignore)


def build_bundle(repo_root: Path, out: Path) -> None:
    backend = repo_root / "service" / "backend"
    pipeline = repo_root / "pipeline"
    splits = repo_root / "pipeline" / "results" / "splits"

    shutil.copy2(backend / "Dockerfile.space", out / "Dockerfile")
    shutil.copy2(backend / "requirements.space.txt", out / "requirements.txt")
    shutil.copy2(backend / "alembic.ini", out / "alembic.ini")

    _copy(backend / "app", out / "app", ignore=("__pycache__",))
    _copy(repo_root / "service" / "scripts", out / "scripts", ignore=("__pycache__",))

    pipe_keep = ("models", "segmentation", "xai", "data", "evaluation",
                 "training", "utils", "config.py", "__init__.py")
    out_pipe = out / "pipeline"
    out_pipe.mkdir(parents=True, exist_ok=True)
    for entry in pipeline.iterdir():
        if entry.name not in pipe_keep:
            continue
        if entry.is_dir():
            _copy(entry, out_pipe / entry.name, ignore=("__pycache__",))
        else:
            shutil.copy2(entry, out_pipe / entry.name)

    out_splits = out / "results" / "splits"
    out_splits.mkdir(parents=True, exist_ok=True)
    for fn in ("idx_to_class.json", "class_to_idx.json"):
        src = splits / fn
        if src.is_file():
            shutil.copy2(src, out_splits / fn)

    hybrid_src = repo_root / "results" / "hybrid_histgbm.pkl"
    if hybrid_src.is_file():
        shutil.copy2(hybrid_src, out / "results" / "hybrid_histgbm.pkl")

    (out / "README.md").write_text(README_TEMPLATE, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="hf user/space-name")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    assert args.token, "HF_TOKEN required"

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo, repo_type="space",
                    space_sdk="docker", exist_ok=True, private=False)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        build_bundle(Path(args.root), out)
        sizes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"bundle size: {sizes / 1024 / 1024:.1f} MB")

        api.upload_folder(
            folder_path=str(out),
            repo_id=args.repo,
            repo_type="space",
            commit_message="deploy: backend bundle",
        )
        print(f"https://huggingface.co/spaces/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
