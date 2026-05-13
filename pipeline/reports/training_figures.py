
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cmd_lr_schedule(args) -> int:
    total = args.epochs * args.steps_per_epoch
    warmup = max(1, int(total * args.warmup_ratio))
    steps = np.arange(total)

    def lr_factor(s):
        if s < warmup:
            return (s + 1) / warmup
        progress = (s - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    head    = np.array([lr_factor(s) * args.base_lr for s in steps])
    backbone = head * args.backbone_mult

    epoch_axis = steps / args.steps_per_epoch

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epoch_axis, head,    color="#3070b0", label=f"head (peak {args.base_lr:.1e})")
    ax.plot(epoch_axis, backbone, color="#c54",   label=f"backbone (×{args.backbone_mult})")
    ax.axvspan(0, warmup / args.steps_per_epoch, color="#bbb", alpha=0.25,
               label=f"warmup ({args.warmup_ratio:.0%})")
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title("Cosine schedule с linear warmup (head vs backbone)")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")
    return 0


_RUN_PATH = {
    "resnet50":              "_unpacked/run_resnet50_seed42/resnet50_seed42",
    "efficientnet_b0":       "_unpacked/run_efficientnet_b0_seed42/efficientnet_b0_seed42",
    "efficientnet_b2":       "_unpacked/run_efficientnet_b2_seed42/efficientnet_b2_seed42",
    "efficientnet_b3":       "_unpacked/run_efficientnet_b3_seed42/efficientnet_b3_seed42",
    "efficientnet_v2_s":     "_unpacked/run_efficientnet_v2_s_seed42/efficientnet_v2_s_seed42",
    "convnext_small":        "_unpacked/run_convnext_small_seed42/convnext_small_seed42",
    "vit_b16":               "_unpacked/run_vit_b16_seed42/vit_b16_seed42",
    "swin_v2_t":             "_unpacked/run_swin_v2_t_seed42/swin_v2_t_seed42",
    "dinov2_vitb14_linear":  "_unpacked/run_dinov2_vitb14_linear_seed42/dinov2_vitb14_linear_seed42",
}

_SHORT = {
    "resnet50":              "ResNet-50",
    "efficientnet_b0":       "EfficientNet-B0",
    "efficientnet_b2":       "EfficientNet-B2",
    "efficientnet_b3":       "EfficientNet-B3",
    "efficientnet_v2_s":     "EfficientNet-V2-S",
    "convnext_small":        "ConvNeXt-Small",
    "vit_b16":               "ViT-B/16",
    "swin_v2_t":             "Swin-V2-T",
    "dinov2_vitb14_linear":  "DINOv2-linear",
}

_COLORS = {
    "efficientnet_v2_s":     "#3070b0",
    "efficientnet_b3":       "#0a3",
    "convnext_small":        "#c54",
    "dinov2_vitb14_linear":  "#9333a0",
    "vit_b16":               "#a07020",
    "efficientnet_b2":       "#608",
    "efficientnet_b0":       "#888",
    "resnet50":              "#226",
    "swin_v2_t":             "#a30",
}


def cmd_curves(args) -> int:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_loss, ax_vl, ax_acc, ax_lr = axes.flatten()

    summary_lines = []
    for m in args.models:
        if m not in _RUN_PATH:
            print(f"  skip {m}: unknown")
            continue
        csv_path = args.runs_dir / _RUN_PATH[m] / "metrics.csv"
        if not csv_path.is_file():
            print(f"  skip {m}: no metrics.csv at {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        col = _COLORS.get(m, "#444")
        label = _SHORT.get(m, m)

        ax_loss.plot(df["epoch"], df["train_loss"], color=col, label=label, lw=1.2)
        ax_vl.plot(df["epoch"],   df["val_loss"],   color=col, label=label, lw=1.2)
        ax_acc.plot(df["epoch"],  df["val_acc"],    color=col, label=label, lw=1.2)
        if "lr_head" in df.columns:
            ax_lr.plot(df["epoch"], df["lr_head"], color=col, label=label, lw=1.2)

        best_epoch = int(df["val_acc"].idxmax() + 1)
        best_acc = float(df["val_acc"].max())
        last_epoch = int(df["epoch"].max())
        summary_lines.append(
            f"  {label:24s} epochs={last_epoch:3d}  best epoch={best_epoch:3d}  "
            f"best val acc={best_acc:.4f}"
        )

    ax_loss.set_title("train loss"); ax_loss.set_xlabel("epoch")
    ax_loss.grid(alpha=0.3); ax_loss.legend(fontsize=8, loc="upper right")
    ax_vl.set_title("val loss"); ax_vl.set_xlabel("epoch")
    ax_vl.grid(alpha=0.3); ax_vl.legend(fontsize=8, loc="upper right")
    ax_acc.set_title("val accuracy"); ax_acc.set_xlabel("epoch")
    ax_acc.grid(alpha=0.3); ax_acc.legend(fontsize=8, loc="lower right")
    ax_lr.set_title("learning rate (head)"); ax_lr.set_xlabel("epoch")
    ax_lr.set_yscale("log"); ax_lr.grid(alpha=0.3, which="both")
    ax_lr.legend(fontsize=8, loc="lower left")

    fig.suptitle("Training curves для топ-моделей", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("\n".join(summary_lines))
    print(f"\nsaved {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("lr-schedule")
    p1.add_argument("--epochs", type=int, default=30)
    p1.add_argument("--steps-per-epoch", type=int, default=458)
    p1.add_argument("--base-lr", type=float, default=6e-4)
    p1.add_argument("--backbone-mult", type=float, default=0.1)
    p1.add_argument("--warmup-ratio", type=float, default=0.1)
    p1.add_argument("--out", type=Path, required=True)
    p1.set_defaults(fn=cmd_lr_schedule)

    p2 = sub.add_parser("curves")
    p2.add_argument("--runs-dir", type=Path, required=True,
                    help="каталог *над* _unpacked (= runs_res)")
    p2.add_argument("--models", nargs="+", required=True)
    p2.add_argument("--out", type=Path, required=True)
    p2.set_defaults(fn=cmd_curves)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
