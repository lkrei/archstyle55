"""
полный прогон Mask2Former на всём корпусе не делаем.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import DATA_DIR, IMAGE_EXTENSIONS, SEGMENT_DIR
from .segmentor import FACADE_CATEGORIES, FacadeSegmentor


def per_class_iou(a: np.ndarray, b: np.ndarray, num_classes: int) -> np.ndarray:
    iou = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        ac = a == c; bc = b == c
        union = (ac | bc).sum()
        if union == 0:
            iou[c] = np.nan
        else:
            iou[c] = (ac & bc).sum() / union
    return iou


def sample_images(root: Path, n: int, seed: int = 42) -> list[Path]:
    files = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(f)
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=SEGMENT_DIR / "compare")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sample_images(args.data_dir, args.n, args.seed)

    sf = FacadeSegmentor("segformer", device=args.device)
    m2f = FacadeSegmentor("mask2former", device=args.device)

    ious = []
    for path in tqdm(files, desc="compare"):
        try:
            mask_a = sf.segment(str(path)).mask
            mask_b = m2f.segment(str(path)).mask
        except Exception:
            continue
        ious.append(per_class_iou(mask_a, mask_b, len(FACADE_CATEGORIES)))

    if not ious:
        print("no images compared")
        return 1

    arr = np.stack(ious)  # (n, classes)
    mean = np.nanmean(arr, axis=0)
    overall = float(np.nanmean(mean))
    payload = {
        "n_images": len(ious),
        "categories": list(FACADE_CATEGORIES),
        "agreement_iou_per_class": [
            None if np.isnan(v) else float(v) for v in mean
        ],
        "agreement_iou_mean": overall,
    }
    (args.out_dir / "compare.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
