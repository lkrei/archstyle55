
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
        if mask.sum() == 0:
            rec[c] = np.nan
        else:
            rec[c] = (preds[mask] == c).mean()
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--logits", type=Path, required=True)
    p.add_argument("--classes", type=Path, required=True)
    p.add_argument("--drop", required=True, help="имя класса для удаления")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=12)
    p.add_argument("--model-name", default="model")
    args = p.parse_args()

    idx_to_class = json.loads(args.classes.read_text(encoding="utf-8"))
    num_classes = len(idx_to_class)
    name_to_idx = {v: int(k) for k, v in idx_to_class.items()}
    if args.drop not in name_to_idx:
        raise SystemExit(f"class not found: {args.drop!r}")
    drop_idx = name_to_idx[args.drop]

    data = np.load(args.logits, allow_pickle=False)
    logits = np.asarray(data["logits"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)

    preds_full = logits.argmax(axis=1)
    rec_full = _recall_per_class(preds_full, labels, num_classes)

    keep_mask = np.ones(num_classes, dtype=bool)
    keep_mask[drop_idx] = False
    keep_idx = np.where(keep_mask)[0]
    old_to_new = -np.ones(num_classes, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(keep_idx.size)

    sample_keep = labels != drop_idx
    logits_abl = logits[sample_keep][:, keep_mask]
    labels_abl_old = labels[sample_keep]
    labels_abl = old_to_new[labels_abl_old]
    preds_abl_new = logits_abl.argmax(axis=1)
    rec_abl_new = _recall_per_class(preds_abl_new, labels_abl, keep_idx.size)
    rec_abl = np.full(num_classes, np.nan, dtype=np.float64)
    rec_abl[keep_idx] = rec_abl_new

    delta = rec_abl - rec_full
    delta[~np.isfinite(delta)] = -np.inf
    order = np.argsort(-delta)[:args.top_n]
    order = order[np.isfinite(delta[order])]

    names_full = [idx_to_class[str(i)] for i in range(num_classes)]
    names_disp = apply_aliases(names_full)

    rows = []
    for i in order:
        rows.append({
            "class": names_disp[i],
            "recall_full": float(rec_full[i]),
            "recall_ablation": float(rec_abl[i]),
            "delta": float(delta[i]),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.with_suffix(".json")).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    drop_display = apply_aliases(args.drop)
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(order))
    ax.barh(y - 0.2, [rec_full[i] for i in order], 0.4, label="full (55 cls)",
            color="#888")
    ax.barh(y + 0.2, [rec_abl[i] for i in order],  0.4,
            label=f"ablation (−{drop_display[:34]})",
            color="#3f8")
    for k, i in enumerate(order):
        ax.text(rec_abl[i] + 0.005, y[k] + 0.2, f"+{delta[i]:.2f}",
                va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([names_disp[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05); ax.set_xlabel("recall")
    ax.set_title(f"{args.model_name}: top-{len(order)} классов с приростом recall после ablation")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
