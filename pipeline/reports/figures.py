
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.calibration import CalibrationReport


def plot_training_curves(history_csv: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    df = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[1].plot(df["epoch"], df["val_acc"], label="acc")
    axes[1].plot(df["epoch"], df["val_macro_f1"], label="macro-F1")
    axes[1].plot(df["epoch"], df["val_balanced_acc"], label="balanced acc")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("metric"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path,
                          title: str = "confusion matrix", normalize: bool = True) -> None:
    import matplotlib.pyplot as plt

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_sum
    else:
        cm_norm = cm
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_class_f1(report_json: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for name, stats in report_json.items():
        if not isinstance(stats, dict) or "f1-score" not in stats:
            continue
        if name in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append((name, stats["f1-score"], stats["support"]))
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.18)))
    ax.barh(range(len(names)), scores)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("F1-score")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_reliability(pre: CalibrationReport, post: CalibrationReport | None,
                     out_path: Path, title: str = "calibration") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    bins = np.linspace(0, 1, pre.n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(centers, pre.bin_acc, width=1.0 / pre.n_bins, alpha=0.6, label="pre",
           edgecolor="black")
    if post is not None:
        ax.bar(centers, post.bin_acc, width=1.0 / pre.n_bins, alpha=0.6,
               label="post", edgecolor="black")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_class_distribution(class_balance_csv: Path, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(class_balance_csv).sort_values("total")
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.2)))
    ax.barh(df["class"], df["total"])
    ax.set_xlabel("images")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_compute_cost(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in rows if "test_macro_f1" in r and r.get("gflops") is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in rows:
        ax.scatter(r["gflops"], r["test_macro_f1"], s=80)
        ax.annotate(r["model"], (r["gflops"], r["test_macro_f1"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("GFLOPs (log)"); ax.set_ylabel("test macro-F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
