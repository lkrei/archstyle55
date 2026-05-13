
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..segmentation.segmentor import FACADE_CATEGORIES

Image.MAX_IMAGE_PIXELS = None

DEFAULT_STYLES = [
    "Stalinist architecture",
    "Tudor Revival architecture",
    "Bauhaus architecture",
    "Naryshkin Baroque architecture",
    "Brutalist architecture",
    "Japanese traditional architecture",
]


def _build_palette() -> mcolors.ListedColormap:
    base = np.array([
        [200, 200, 200],
        [200,  70,  60],
        [220, 180,  60],
        [120, 100,  80],
        [120, 200, 220],
        [200, 130, 200],
        [120, 180, 240],
        [ 90, 170,  90],
        [200, 180, 140],
        [120, 120, 160],
    ], dtype=np.uint8)
    return mcolors.ListedColormap(base / 255.0)


def _overlay(rgb: np.ndarray, mask: np.ndarray, palette: np.ndarray,
             alpha: float = 0.5) -> np.ndarray:
    m_rgb = palette[mask.clip(0, len(palette) - 1)]
    blend = (alpha * m_rgb + (1.0 - alpha) * rgb).clip(0, 255).astype(np.uint8)
    return blend


def _facade_roi(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    facade = np.isin(mask, [0, 1, 2, 3, 4, 5])
    if facade.sum() < 16:
        return rgb
    ys, xs = np.where(facade)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return rgb[y0:y1, x0:x1]


def _pick_one(images_dir: Path, masks_dir: Path, style: str,
              rng: random.Random) -> tuple[Path, Path] | None:
    img_dir  = images_dir / style
    mask_dir = masks_dir  / style
    if not img_dir.is_dir() or not mask_dir.is_dir():
        return None
    candidates = []
    for img in img_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        mask = mask_dir / (img.stem + ".png")
        if mask.is_file():
            candidates.append((img, mask))
    if not candidates:
        return None
    return rng.choice(candidates)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--styles", nargs="+", default=DEFAULT_STYLES)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    palette_cmap = _build_palette()
    palette_arr = (palette_cmap.colors * 255.0).astype(np.uint8)

    picks: list[tuple[str, Path, Path]] = []
    for style in args.styles:
        chosen = _pick_one(args.images_dir, args.masks_dir, style, rng)
        if chosen is None:
            print(f"  skip {style}: no image/mask")
            continue
        picks.append((style, *chosen))

    if not picks:
        raise SystemExit("nothing to plot")

    rows = len(picks)
    fig, axes = plt.subplots(rows, 4, figsize=(13, rows * 3.4))
    if rows == 1:
        axes = axes[None, :]

    for r, (style, img_p, mask_p) in enumerate(picks):
        with Image.open(img_p).convert("RGB") as im:
            rgb = np.asarray(im)
        with Image.open(mask_p) as mim:
            mask = np.asarray(mim, dtype=np.int64)
        if mask.shape != rgb.shape[:2]:
            mim_resized = Image.fromarray(mask.astype(np.uint8)).resize(
                (rgb.shape[1], rgb.shape[0]), resample=Image.NEAREST)
            mask = np.asarray(mim_resized, dtype=np.int64)

        overlay = _overlay(rgb, mask, palette_arr, alpha=0.45)
        roi = _facade_roi(rgb, mask)

        for col, (im_arr, title) in enumerate([
            (rgb, "input"),
            (None, "mask"),
            (overlay, "overlay"),
            (roi, "facade ROI"),
        ]):
            ax = axes[r, col]
            if title == "mask":
                ax.imshow(mask, cmap=palette_cmap, vmin=0, vmax=9)
            else:
                ax.imshow(im_arr)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title)
        axes[r, 0].set_ylabel(style, rotation=0, ha="right", va="center",
                              labelpad=6, fontsize=10)

    legend_handles = [
        plt.matplotlib.patches.Patch(color=palette_cmap.colors[i], label=name)
        for i, name in enumerate(FACADE_CATEGORIES)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)

    fig.suptitle("SegFormer-B2 — facade-aware segmentation showcase", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
