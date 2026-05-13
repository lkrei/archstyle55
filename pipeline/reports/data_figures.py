"""Dataset figures: ``class-sizes``, ``split-stack``, ``image-dims``."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .class_aliases import apply_aliases

Image.MAX_IMAGE_PIXELS = None


def _load(splits_path: Path, classes_path: Path):
    splits = json.loads(splits_path.read_text())
    idx_to_class = json.loads(classes_path.read_text())
    n = len(idx_to_class)
    names = apply_aliases([idx_to_class[str(i)] for i in range(n)])
    return splits, names, n


def cmd_class_sizes(args) -> int:
    splits, names, n = _load(args.splits, args.classes)
    counts = np.zeros(n, dtype=np.int64)
    for split in ("train", "val", "test"):
        for s in splits[split]:
            counts[int(s["label"])] += 1

    order = np.argsort(counts)
    lab = [names[i] for i in order]
    val = counts[order]

    median = int(np.median(val))
    minc, maxc = int(val.min()), int(val.max())

    fig, ax = plt.subplots(figsize=(8.5, 11.5))
    bar_colors = ["#c44" if "Late 20th century Moscow" in s else "#3070b0" for s in lab]
    bars = ax.barh(np.arange(n), val, color=bar_colors)
    for r, c in zip(bars, val):
        ax.text(r.get_width() + 4, r.get_y() + r.get_height() / 2,
                str(c), va="center", fontsize=6)

    ax.axvline(median, color="grey", lw=0.8, ls=":", label=f"median = {median}")
    ax.axvline(minc,   color="#c44",  lw=0.8, ls="--", label=f"min = {minc}")
    ax.axvline(maxc,   color="#0a3",  lw=0.8, ls="--", label=f"max = {maxc}")
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(lab, fontsize=6.5)
    ax.set_xlabel("# images")
    ax.set_title(f"Размер классов в датасете ({int(val.sum())} кадров, 55 классов)\n"
                 f"min/max ratio = {minc/maxc:.3f}; "
                 f"красным — Late 20th century Moscow (шумная категория, см. 5.4)",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")
    return 0


def cmd_split_stack(args) -> int:
    splits, names, n = _load(args.splits, args.classes)
    counts = {sp: np.zeros(n, dtype=np.int64) for sp in ("train", "val", "test")}
    for sp in ("train", "val", "test"):
        for s in splits[sp]:
            counts[sp][int(s["label"])] += 1

    total = counts["train"] + counts["val"] + counts["test"]
    order = np.argsort(total)
    lab = [names[i] for i in order]
    tr = counts["train"][order]; va = counts["val"][order]; te = counts["test"][order]

    fig, ax = plt.subplots(figsize=(8.5, 11.5))
    y = np.arange(n)
    ax.barh(y, tr,                color="#3070b0", label="train")
    ax.barh(y, va, left=tr,       color="#90a040", label="val")
    ax.barh(y, te, left=tr + va,  color="#c54",    label="test")

    for i, (a, b, c, t) in enumerate(zip(tr, va, te, total)):
        if t > 0:
            ax.text(t + 4, i, f"{a}/{b}/{c}", va="center", fontsize=6)

    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.5)
    ax.set_xlabel("# images")
    ax.set_title("Stratified 70/15/15 split по 55 классам "
                 "(сверху подписаны train/val/test)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")
    return 0


def _read_size(p: str) -> tuple[int, int] | None:
    try:
        with Image.open(p) as im:
            return im.size
    except Exception:
        return None


def cmd_image_dims(args) -> int:
    splits = json.loads(args.splits.read_text())
    paths = []
    for sp in ("train", "val", "test"):
        for s in splits[sp]:
            paths.append(s["path"])
    rng = random.Random(args.seed)
    if args.sample > 0 and args.sample < len(paths):
        paths = rng.sample(paths, args.sample)

    print(f"  reading {len(paths)} headers ...")
    widths, heights = [], []
    for i, p in enumerate(paths):
        sz = _read_size(p)
        if sz is None:
            continue
        widths.append(sz[0]); heights.append(sz[1])
        if (i + 1) % 1000 == 0:
            print(f"   {i+1}/{len(paths)}")
    widths = np.array(widths); heights = np.array(heights)
    aspect = widths / heights

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    h = ax.hexbin(widths, heights, gridsize=50, cmap="viridis", bins="log",
                  mincnt=1)
    ax.plot([0, 4000], [0, 4000], "--", color="#c54", lw=0.7, label="square")
    ax.set_xlabel("width, px"); ax.set_ylabel("height, px")
    ax.set_xlim(0, np.percentile(widths, 99) + 50)
    ax.set_ylim(0, np.percentile(heights, 99) + 50)
    ax.set_title(f"Image dimensions ({len(widths)} sample)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.colorbar(h, ax=ax, label="log(count)")

    ax = axes[1]
    bins = np.linspace(0.4, 2.5, 70)
    ax.hist(aspect, bins=bins, color="#3070b0", edgecolor="white")
    ax.axvline(1.0, color="#c54", lw=0.7, ls="--", label="square")
    ax.axvline(np.median(aspect), color="#0a3", lw=0.8, ls=":",
               label=f"median = {np.median(aspect):.2f}")
    ax.set_xlabel("aspect ratio (W / H)")
    ax.set_ylabel("# images")
    ax.set_title(f"Распределение aspect ratio "
                 f"(landscape: {(aspect > 1.05).mean()*100:.1f}%, "
                 f"portrait: {(aspect < 0.95).mean()*100:.1f}%)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n": int(len(widths)),
        "width": {"min": int(widths.min()), "max": int(widths.max()),
                  "median": float(np.median(widths)),
                  "p95": float(np.percentile(widths, 95))},
        "height": {"min": int(heights.min()), "max": int(heights.max()),
                   "median": float(np.median(heights)),
                   "p95": float(np.percentile(heights, 95))},
        "aspect": {"min": float(aspect.min()), "max": float(aspect.max()),
                   "median": float(np.median(aspect)),
                   "landscape_pct": float((aspect > 1.05).mean()),
                   "portrait_pct":  float((aspect < 0.95).mean())},
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("class-sizes")
    p1.add_argument("--splits", type=Path, required=True)
    p1.add_argument("--classes", type=Path, required=True)
    p1.add_argument("--out", type=Path, required=True)
    p1.set_defaults(fn=cmd_class_sizes)

    p2 = sub.add_parser("split-stack")
    p2.add_argument("--splits", type=Path, required=True)
    p2.add_argument("--classes", type=Path, required=True)
    p2.add_argument("--out", type=Path, required=True)
    p2.set_defaults(fn=cmd_split_stack)

    p3 = sub.add_parser("image-dims")
    p3.add_argument("--splits", type=Path, required=True)
    p3.add_argument("--out", type=Path, required=True)
    p3.add_argument("--sample", type=int, default=4000)
    p3.add_argument("--seed", type=int, default=42)
    p3.set_defaults(fn=cmd_image_dims)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
