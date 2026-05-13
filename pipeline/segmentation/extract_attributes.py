"""Табличные фасадные атрибуты по сегментационной маске.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..config import SEGMENT_DIR, SPLITS_DIR
from .segmentor import FACADE_CATEGORIES

WALL = 0; WINDOW = 1; DOOR = 2; ROOF = 3; BALCONY = 4
COLUMN = 5; SKY = 6; VEGETATION = 7; GROUND = 8


def feature_names() -> list[str]:
    base = [f"share_{c}" for c in FACADE_CATEGORIES]
    extras = [
        "window_to_wall", "balcony_to_wall", "column_to_wall",
        "roof_to_wall", "vegetation_to_facade", "sky_to_facade",
        "facade_compactness", "facade_aspect",
        "window_density", "horizontal_symmetry", "vertical_symmetry",
        "top_share_facade", "bottom_share_ground", "left_share_facade",
        "right_share_facade", "centre_share_facade",
    ]
    return base + extras


def _facade_mask(mask: np.ndarray) -> np.ndarray:
    return np.isin(mask, np.asarray([WALL, WINDOW, DOOR, ROOF, BALCONY, COLUMN]))


def extract(mask: np.ndarray) -> dict:
    h, w = mask.shape
    n = float(h * w)
    feats: dict[str, float] = {}
    for i, cat in enumerate(FACADE_CATEGORIES):
        feats[f"share_{cat}"] = float((mask == i).sum() / n)

    wall = max(1.0, (mask == WALL).sum())
    feats["window_to_wall"] = float((mask == WINDOW).sum() / wall)
    feats["balcony_to_wall"] = float((mask == BALCONY).sum() / wall)
    feats["column_to_wall"] = float((mask == COLUMN).sum() / wall)
    feats["roof_to_wall"] = float((mask == ROOF).sum() / wall)

    facade = _facade_mask(mask)
    facade_n = max(1.0, facade.sum())
    feats["vegetation_to_facade"] = float((mask == VEGETATION).sum() / facade_n)
    feats["sky_to_facade"] = float((mask == SKY).sum() / facade_n)

    if facade.any():
        ys, xs = np.where(facade)
        bw = xs.max() - xs.min() + 1
        bh = ys.max() - ys.min() + 1
        feats["facade_compactness"] = float(facade.sum() / max(1.0, bw * bh))
        feats["facade_aspect"] = float(bw / max(1.0, bh))
    else:
        feats["facade_compactness"] = 0.0
        feats["facade_aspect"] = 0.0

    windows = (mask == WINDOW)
    feats["window_density"] = float(windows.sum() / facade_n)

    half = w // 2
    feats["horizontal_symmetry"] = float(
        np.mean(facade[:, :half] == np.flip(facade[:, half:half * 2], axis=1))
        if half > 0 else 0.0
    )
    half_h = h // 2
    feats["vertical_symmetry"] = float(
        np.mean(facade[:half_h] == np.flip(facade[half_h:half_h * 2], axis=0))
        if half_h > 0 else 0.0
    )

    third = h // 3
    third_w = w // 3
    feats["top_share_facade"] = float(facade[:third].mean()) if third > 0 else 0.0
    feats["bottom_share_ground"] = float((mask[-third:] == GROUND).mean()) if third > 0 else 0.0
    feats["left_share_facade"] = float(facade[:, :third_w].mean()) if third_w > 0 else 0.0
    feats["right_share_facade"] = float(facade[:, -third_w:].mean()) if third_w > 0 else 0.0
    feats["centre_share_facade"] = float(
        facade[third:2 * third, third_w:2 * third_w].mean()
        if third > 0 and third_w > 0 else 0.0
    )

    return feats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-dir", type=Path, default=SEGMENT_DIR / "segformer")
    parser.add_argument("--splits", type=Path, default=SPLITS_DIR / "data_splits.json")
    parser.add_argument("--out", type=Path, default=SEGMENT_DIR / "attributes.csv")
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text())
    fieldnames = ["path", "split", "label", *feature_names()]

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, samples in splits.items():
            for s in tqdm(samples, desc=split_name):
                p = Path(s["path"])
                base = args.masks_dir / p.parent.name / p.stem
                png_path = base.with_suffix(".png")
                npy_path = base.with_suffix(".npy")
                if png_path.is_file():
                    mp = png_path
                elif npy_path.is_file():
                    mp = npy_path
                else:
                    continue
                try:
                    if mp.suffix == ".png":
                        mask = np.asarray(Image.open(mp), dtype=np.int64)
                    else:
                        mask = np.load(mp)
                except (OSError, ValueError):
                    continue
                row = {"path": str(p), "split": split_name, "label": int(s["label"])}
                row.update(extract(mask))
                writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
