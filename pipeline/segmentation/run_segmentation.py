"""Прогон сегментации по всему датасету или по подвыборке."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import DATA_DIR, IMAGE_EXTENSIONS, SEGMENT_DIR
from .segmentor import FacadeSegmentor


def discover_images(root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        for f in sorted(class_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith("."):
                items.append((class_dir.name, f))
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=SEGMENT_DIR / "segformer")
    parser.add_argument("--backend", choices=("segformer", "mask2former"), default="segformer")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="ограничить количество кадров (для абляции).")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seg = FacadeSegmentor(backend=args.backend, model_id=args.model_id, device=args.device)
    items = discover_images(args.data_dir)
    if args.limit is not None:
        items = items[: args.limit]

    errors = []
    t0 = time.time()
    for class_name, path in tqdm(items, desc=f"segment[{args.backend}]"):
        target_dir = out_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / (path.stem + ".png")
        legacy = target_dir / (path.stem + ".npy")
        if target.is_file() or legacy.is_file():
            continue
        try:
            res = seg.segment(str(path))
            mask_u8 = res.mask.astype(np.uint8)
            Image.fromarray(mask_u8, mode="L").save(target, optimize=True)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path), "error": str(exc)})

    elapsed = time.time() - t0
    print(f"done in {elapsed/60:.1f} min, errors: {len(errors)}")
    if errors:
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
