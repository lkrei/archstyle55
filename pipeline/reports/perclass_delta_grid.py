
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .class_aliases import apply_aliases


def _recall_per_class(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    rec = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        rec[c] = np.nan if mask.sum() == 0 else (preds[mask] == c).mean()
    return rec


def _full_vs_ablation(logits: np.ndarray, labels: np.ndarray,
                      num_classes: int, drop_idx: int):
    preds_full = logits.argmax(axis=1)
    rec_full = _recall_per_class(preds_full, labels, num_classes)

    keep_mask = np.ones(num_classes, dtype=bool)
    keep_mask[drop_idx] = False
    keep_idx = np.where(keep_mask)[0]
    old_to_new = -np.ones(num_classes, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(keep_idx.size)

    sample_keep = labels != drop_idx
    L_a = logits[sample_keep][:, keep_mask]
    Y_a = old_to_new[labels[sample_keep]]
    P_a = L_a.argmax(axis=1)
    rec_a_new = _recall_per_class(P_a, Y_a, keep_idx.size)
    rec_a = np.full(num_classes, np.nan, dtype=np.float64)
    rec_a[keep_idx] = rec_a_new
    return rec_full, rec_a


_RUN_PATH = {
    "efficientnet_v2_s": "_unpacked/run_efficientnet_v2_s_seed42/efficientnet_v2_s_seed42",
    "efficientnet_b3":   "_unpacked/run_efficientnet_b3_seed42/efficientnet_b3_seed42",
    "convnext_small":    "_unpacked/run_convnext_small_seed42/convnext_small_seed42",
    "dinov2_vitb14_linear": "_unpacked/run_dinov2_vitb14_linear_seed42/dinov2_vitb14_linear_seed42",
}


def _short(name: str) -> str:
    return {
        "efficientnet_v2_s": "EfficientNet-V2-S",
        "efficientnet_b3":   "EfficientNet-B3",
        "convnext_small":    "ConvNeXt-Small",
        "dinov2_vitb14_linear": "DINOv2-linear",
        "ensemble_top3_uniform": "Ensemble top-3 (uniform)",
    }.get(name, name)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--classes", type=Path, required=True)
    p.add_argument("--drop", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs-dir", type=Path, required=True,
                   help="root containing per-run unpacked artifacts")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--ensemble", type=Path, default=None,
                   help="path к test_logits.npz ансамбля (опц.)")
    p.add_argument("--ensemble-name", default="ensemble_top3_uniform")
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args()

    idx_to_class = json.loads(args.classes.read_text(encoding="utf-8"))
    num_classes = len(idx_to_class)
    name_to_idx = {v: int(k) for k, v in idx_to_class.items()}
    if args.drop not in name_to_idx:
        raise SystemExit(f"class not found: {args.drop!r}")
    drop_idx = name_to_idx[args.drop]
    names_full = [idx_to_class[str(i)] for i in range(num_classes)]
    names_disp = apply_aliases(names_full)

    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    for m in args.models:
        if m not in _RUN_PATH:
            raise SystemExit(f"unknown model {m}; add to _RUN_PATH")
        logits_path = args.runs_dir / _RUN_PATH[m] / "test_logits.npz"
        d = np.load(logits_path, allow_pickle=False)
        L = np.asarray(d["logits"], dtype=np.float64)
        Y = np.asarray(d["labels"], dtype=np.int64)
        rec_f, rec_a = _full_vs_ablation(L, Y, num_classes, drop_idx)
        panels.append((m, rec_f, rec_a))

    if args.ensemble:
        d = np.load(args.ensemble, allow_pickle=False)
        if "logits" in d.files:
            L = np.asarray(d["logits"], dtype=np.float64)
        else:
            L = np.asarray(d["probs"], dtype=np.float64)
        Y = np.asarray(d["labels"], dtype=np.int64)
        rec_f, rec_a = _full_vs_ablation(L, Y, num_classes, drop_idx)
        panels.append((args.ensemble_name, rec_f, rec_a))

    n = len(panels)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 3.2))
    if rows == 1:
        axes = axes[None, :]

    json_dump: dict = {}
    for k, (model, rec_f, rec_a) in enumerate(panels):
        delta = rec_a - rec_f
        delta_clean = np.where(np.isfinite(delta), delta, -np.inf)
        order = np.argsort(-delta_clean)[:args.top_n]
        order = order[np.isfinite(delta[order])]

        ax = axes[k // cols, k % cols]
        y = np.arange(len(order))
        ax.barh(y - 0.2, [rec_f[i] for i in order], 0.4,
                label="full (55 cls)", color="#888")
        ax.barh(y + 0.2, [rec_a[i] for i in order], 0.4,
                label="ablation", color="#3f8")
        for j, i in enumerate(order):
            ax.text(rec_a[i] + 0.005, y[j] + 0.2, f"+{delta[i]:.2f}",
                    va="center", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels([names_disp[i] for i in order],
                                              fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_title(_short(model), fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        if k == 0:
            ax.legend(loc="lower right", fontsize=7)
        json_dump[model] = [
            {"class": names_disp[int(i)],
             "full": float(rec_f[i]), "ablation": float(rec_a[i]),
             "delta": float(delta[i])}
            for i in order
        ]

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    drop_display = apply_aliases(args.drop)
    fig.suptitle(f"Per-class recall: full 55-cls vs ablation (−{drop_display})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    args.out.with_suffix(".json").write_text(
        json.dumps(json_dump, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
