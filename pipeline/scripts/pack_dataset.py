
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from ..config import DATA_DIR, SPLITS_DIR


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--out", type=Path, default=Path("archstyle55.zip"))
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"data dir missing: {args.data_dir}")

    target = args.out.with_suffix("")
    print(f"creating {target}.zip from {args.data_dir} + {args.splits_dir}", flush=True)
    work = target.parent / (target.name + "_pack")
    work.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.data_dir, work / "images", dirs_exist_ok=True)
    shutil.copytree(args.splits_dir, work / "splits", dirs_exist_ok=True)
    archive = shutil.make_archive(str(target), "zip", root_dir=str(work))
    shutil.rmtree(work)

    digest = sha256(Path(archive))
    print(f"archive: {archive}")
    print(f"sha256:  {digest}")
    Path(str(target) + ".sha256").write_text(digest, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
