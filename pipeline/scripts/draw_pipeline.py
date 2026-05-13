
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ..config import RESULTS_DIR


def _box(ax, x, y, w, h, text, fc, ec="black", text_size=9):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor=ec, facecolor=fc,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=text_size)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "figures" / "pipeline.png")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")

    pal = {
        "data": "#E0F0FF", "prep": "#E5F3DD", "model": "#FFF2CC",
        "xai": "#F4E1F4", "eval": "#FBE5D6",
    }

    _box(ax, 0.4, 7.6, 4.4, 1.0,
         "Licensed image collection\n(CC-BY / CC-BY-SA / Public Domain)",
         fc=pal["data"])
    _box(ax, 5.2, 7.6, 3.6, 1.0,
         "Manual cleaning & relabel\nDataset_1 = 55 cls · 20 878 images",
         fc=pal["data"])
    _box(ax, 9.2, 7.6, 4.4, 1.0,
         "Stratified 70/15/15 splits\nmanifest.csv + repro.json",
         fc=pal["data"])

    _box(ax, 0.4, 5.4, 3.0, 1.5,
         "SegFormer-B2\n(ADE20K → 10 facade categories)\n(Mask2Former for IoU ablation)",
         fc=pal["prep"])
    _box(ax, 3.7, 5.4, 3.0, 1.5,
         "Facade ROI cropping\n+ optional rectify\n(vanishing points / mask quad)",
         fc=pal["prep"])
    _box(ax, 7.0, 5.4, 3.0, 1.5,
         "Color features\n(Lab/HSV + dominant + sky/veg share)",
         fc=pal["prep"])
    _box(ax, 10.3, 5.4, 3.3, 1.5,
         "Aug: RandAugment, RandomErasing,\nMixUp / CutMix, label smoothing",
         fc=pal["prep"])

    _box(ax, 0.4, 3.0, 3.0, 1.7,
         "CNNs\nResNet-50, EfficientNet B0–B3,\nEfficientNet V2-S, ConvNeXt(-V2)",
         fc=pal["model"])
    _box(ax, 3.7, 3.0, 3.0, 1.7,
         "Transformers\nViT-B/16, Swin V2-T",
         fc=pal["model"])
    _box(ax, 7.0, 3.0, 3.0, 1.7,
         "Foundation\nDINOv2 ViT-B/14 (linear / LoRA)\nCLIP / SigLIP zero-shot",
         fc=pal["model"])
    _box(ax, 10.3, 3.0, 3.3, 1.7,
         "Ensemble & hybrid\nweighted prob avg +\nimage probs ⊕ tabular attrs",
         fc=pal["model"])

    _box(ax, 0.4, 0.6, 4.4, 1.7,
         "XAI\nGrad-CAM++/Score-CAM/Eigen-CAM (CNN)\n"
         "Chefer LRP / Attention Rollout (ViT)\n"
         "ProtoPNet (concept prototypes)",
         fc=pal["xai"])
    _box(ax, 5.2, 0.6, 4.0, 1.7,
         "Evaluation\naccuracy, macro-F1, balanced acc,\n"
         "ECE/MCE + temperature scaling,\nbootstrap CI, McNemar pairwise",
         fc=pal["eval"])
    _box(ax, 9.4, 0.6, 4.2, 1.7,
         "Reports\ntraining curves, confusion + per-class F1,\n"
         "embedding UMAP, error gallery,\ncompute-cost (params/GFLOPs/ms)",
         fc=pal["eval"])

    for x in (4.8, 8.0, 11.4):
        _arrow(ax, x, 7.6, x, 7.0)
    for x in (1.9, 5.2, 8.5, 11.8):
        _arrow(ax, x, 5.4, x, 4.7)
    for x in (1.9, 5.2, 8.5, 11.8):
        _arrow(ax, x, 3.0, x, 2.3)

    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"pipeline diagram saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
