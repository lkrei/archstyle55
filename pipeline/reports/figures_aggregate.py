
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .class_aliases import apply_aliases


def _read_summary(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: v for k, v in r.items()})
    return rows


def _as_float(s: str | None) -> float:
    if s is None or s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _short(name: str) -> str:
    return name.removesuffix("_seed42")


def plot_models_bar(summary_rows: list[dict], out_path: Path) -> None:
    rows = sorted(
        summary_rows,
        key=lambda r: _as_float(r.get("accuracy")),
    )
    names = [_short(r["model"]) for r in rows]
    acc = np.array([_as_float(r.get("accuracy")) for r in rows])
    acc_lo = np.array([_as_float(r.get("accuracy_ci_lo")) for r in rows])
    acc_hi = np.array([_as_float(r.get("accuracy_ci_hi")) for r in rows])
    f1 = np.array([_as_float(r.get("macro_f1")) for r in rows])
    f1_lo = np.array([_as_float(r.get("macro_f1_ci_lo")) for r in rows])
    f1_hi = np.array([_as_float(r.get("macro_f1_ci_hi")) for r in rows])

    err_acc = np.stack([acc - np.where(np.isnan(acc_lo), acc, acc_lo),
                        np.where(np.isnan(acc_hi), acc, acc_hi) - acc])
    err_f1 = np.stack([f1 - np.where(np.isnan(f1_lo), f1, f1_lo),
                       np.where(np.isnan(f1_hi), f1, f1_hi) - f1])

    n = len(names)
    y = np.arange(n)
    h = 0.4

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.55)))
    ax.barh(y - h / 2, acc, height=h, xerr=err_acc, label="accuracy",
            capsize=3, color="#3a76b3", alpha=0.85)
    ax.barh(y + h / 2, f1, height=h, xerr=err_f1, label="macro F1",
            capsize=3, color="#d97c2c", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("score")
    ax.set_xlim(0, max(1.0, np.nanmax(acc) + 0.05))
    ax.axvline(1 / 55, color="grey", linestyle=":", linewidth=1,
               label="random (1/55)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _load_run(unpacked_root: Path, model_dir_name: str) -> Path | None:
    candidates = [
        unpacked_root / f"run_{model_dir_name}" / model_dir_name,
        unpacked_root / model_dir_name,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    parent = unpacked_root / f"run_{model_dir_name}"
    if parent.is_dir():
        if (parent / "test_metrics.json").is_file():
            return parent
        kids = [p for p in parent.iterdir() if p.is_dir()]
        if len(kids) == 1:
            return kids[0]
    return None


def plot_confusion(model_name: str, run_dir: Path, class_names: list[str],
                   out_path: Path, top_k_errors: int = 3) -> None:
    cm_path = run_dir / "test_confusion.npy"
    if not cm_path.is_file():
        return
    cm = np.load(cm_path)
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sum

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{_short(model_name)} — normalized confusion matrix (test)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_class_recall_heatmap(pcr_path: Path, top_models: list[str],
                                  out_path: Path) -> None:
    with pcr_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return
    models_in_csv = [k for k in rows[0].keys() if k != "class"]
    use = [m for m in top_models if m in models_in_csv]
    if not use:
        return
    classes = [r["class"] for r in rows]
    matrix = np.array([[_as_float(r[m]) for m in use] for r in rows])
    order = np.argsort(-np.nanmean(matrix, axis=1))
    matrix = matrix[order]
    classes = [classes[i] for i in order]

    fig, ax = plt.subplots(figsize=(0.8 * len(use) + 4, max(8, len(classes) * 0.22)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(use)))
    ax.set_xticklabels([_short(m) for m in use], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_title("Per-class recall (test) — top models")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_worst_classes(pcr_path: Path, top_models: list[str],
                       out_path: Path, n_worst: int = 10) -> None:
    with pcr_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return
    use = [m for m in top_models if m in (rows[0].keys() - {"class"})]
    if not use:
        return
    means = []
    for r in rows:
        vals = [_as_float(r[m]) for m in use]
        means.append((r["class"], float(np.nanmean(vals))))
    means.sort(key=lambda x: x[1])
    worst = means[:n_worst]
    names = [w[0] for w in worst][::-1]
    scores = [w[1] for w in worst][::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.35)))
    ax.barh(range(len(names)), scores, color="#c0392b", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("recall (mean across top models)")
    ax.set_xlim(0, 1)
    ax.axvline(np.mean(scores), linestyle=":", color="grey")
    ax.set_title(f"{n_worst} hardest classes (mean of top-{len(use)})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_res"))
    parser.add_argument("--aggregate-dir", type=Path,
                        default=Path("runs_res/aggregate"))
    parser.add_argument("--classes", type=Path, required=True,
                        help="idx_to_class.json")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3,
                        help="number of top models for confusion / heatmap")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.aggregate_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_summary(args.aggregate_dir / "summary_table.csv")
    if not summary:
        raise SystemExit(f"empty summary at {args.aggregate_dir / 'summary_table.csv'}")

    summary.sort(key=lambda r: _as_float(r.get("accuracy")), reverse=True)
    top_models = [r["model"] for r in summary[: args.top_k]]
    print("top models:", [_short(m) for m in top_models])

    idx_to_class = json.loads(args.classes.read_text())
    class_names_raw = [idx_to_class[str(i)] for i in range(len(idx_to_class))]
    class_names = apply_aliases(class_names_raw)

    plot_models_bar(summary, out_dir / "bar_models_acc_f1.png")
    print(f"  wrote {out_dir / 'bar_models_acc_f1.png'}")

    unpacked = args.runs_dir / "_unpacked"
    for m in top_models:
        run_dir = _load_run(unpacked, m)
        if run_dir is None:
            print(f"  skip confusion for {m}: no unpacked dir")
            continue
        path = out_dir / f"confusion_{_short(m)}.png"
        plot_confusion(m, run_dir, class_names, path)
        print(f"  wrote {path}")

    pcr = args.aggregate_dir / "per_class_recall.csv"
    if pcr.is_file():
        plot_per_class_recall_heatmap(pcr, top_models,
                                      out_dir / f"per_class_recall_top{args.top_k}.png")
        print(f"  wrote {out_dir / f'per_class_recall_top{args.top_k}.png'}")
        plot_worst_classes(pcr, top_models,
                           out_dir / f"worst_classes_top{args.top_k}.png")
        print(f"  wrote {out_dir / f'worst_classes_top{args.top_k}.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
