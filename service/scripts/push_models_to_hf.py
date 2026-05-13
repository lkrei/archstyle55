from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTR_CSV = REPO_ROOT / "pipeline" / "results" / "segmentation" / "attributes.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="directory with run_*/<model>/best.pt")
    parser.add_argument("--repo-id", default=os.environ.get("HF_MODEL_REPO", "kkkaredaw/archstyle55-backbones"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--include-attributes", action="store_true",
                        help="also upload SegFormer attributes csv")
    parser.add_argument("--attributes-csv", type=Path, default=DEFAULT_ATTR_CSV)
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("HF_TOKEN env var or --token is required")

    create_repo(args.repo_id, token=args.token, exist_ok=True, private=args.private,
                repo_type="model")
    api = HfApi(token=args.token)

    if not args.source.is_dir():
        raise SystemExit(f"source dir not found: {args.source}")

    pushed = 0
    for run_dir in sorted(args.source.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        for sub in run_dir.iterdir():
            if not sub.is_dir():
                continue
            ckpt = sub / "best.pt"
            if not ckpt.is_file():
                continue
            target = f"{sub.name}/best.pt"
            print(f"upload {ckpt} -> {args.repo_id}:{target}")
            upload_file(
                path_or_fileobj=str(ckpt),
                path_in_repo=target,
                repo_id=args.repo_id,
                token=args.token,
                repo_type="model",
            )
            pushed += 1

    if args.include_attributes:
        attr = args.attributes_csv
        if attr.is_file():
            upload_file(
                path_or_fileobj=str(attr),
                path_in_repo="attributes/attributes.csv",
                repo_id=args.repo_id,
                token=args.token,
                repo_type="model",
            )
            pushed += 1

    print(f"pushed {pushed} files to {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
