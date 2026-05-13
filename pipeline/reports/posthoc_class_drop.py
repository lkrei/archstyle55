
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from .figures_aggregate import _as_float, _load_run, _read_summary, _short


def _eval(logits: np.ndarray, labels: np.ndarray) -> dict:
    preds = logits.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_res"))
    parser.add_argument("--aggregate-dir", type=Path,
                        default=Path("runs_res/aggregate"))
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--drop", action="append", required=True,
                        help="имя класса для удаления (можно несколько раз)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_csv = args.out or (args.aggregate_dir / "posthoc_class_drop.csv")

    idx_to_class = json.loads(args.classes.read_text(encoding="utf-8"))
    name_to_idx = {v: int(k) for k, v in idx_to_class.items()}
    drop_idx = []
    for n in args.drop:
        if n not in name_to_idx:
            raise SystemExit(f"class not found in idx_to_class: {n!r}")
        drop_idx.append(name_to_idx[n])
    drop_idx_set = set(drop_idx)
    keep_classes = sorted(set(name_to_idx.values()) - drop_idx_set)

    summary = _read_summary(args.aggregate_dir / "summary_table.csv")
    summary.sort(key=lambda r: _as_float(r.get("accuracy")), reverse=True)

    rows: list[dict] = []
    unpacked = args.runs_dir / "_unpacked"
    for r in summary:
        m = r["model"]
        if m.startswith("zeroshot_"):
            continue
        run_dir = _load_run(unpacked, m)
        if run_dir is None:
            continue
        npz = run_dir / "test_logits.npz"
        if not npz.is_file():
            continue
        data = np.load(npz, allow_pickle=False)
        logits = np.asarray(data["logits"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.int64)

        full = _eval(logits, labels)

        keep_mask = np.array([y not in drop_idx_set for y in labels])
        if keep_mask.sum() == 0:
            continue
        sub_logits = logits[keep_mask][:, keep_classes]
        sub_labels = labels[keep_mask]
        # перенумерация старого индекса → нового индекса в keep_classes
        idx_remap = {old: new for new, old in enumerate(keep_classes)}
        sub_labels_remap = np.array([idx_remap[y] for y in sub_labels],
                                    dtype=np.int64)
        sub = _eval(sub_logits, sub_labels_remap)

        delta_acc = sub["accuracy"] - full["accuracy"]
        delta_f1 = sub["macro_f1"] - full["macro_f1"]
        rows.append({
            "model": m,
            "n_full": int(len(labels)),
            "n_keep": int(keep_mask.sum()),
            "acc_full": full["accuracy"],
            "acc_drop": sub["accuracy"],
            "delta_acc": delta_acc,
            "macro_f1_full": full["macro_f1"],
            "macro_f1_drop": sub["macro_f1"],
            "delta_macro_f1": delta_f1,
            "bal_acc_full": full["balanced_accuracy"],
            "bal_acc_drop": sub["balanced_accuracy"],
        })
        print(f"  {_short(m):28s}  acc {full['accuracy']:.4f} -> {sub['accuracy']:.4f}  "
              f"(Δ {delta_acc:+.4f})  F1 {full['macro_f1']:.4f} -> {sub['macro_f1']:.4f}  "
              f"(Δ {delta_f1:+.4f})")

    if not rows:
        return 1

    keys = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row_out = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(row_out)
    print(f"\nwrote {out_csv}")

    md = ["| model | acc full | acc ablation | Δacc | F1 full | F1 ablation | ΔF1 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(
            f"| {_short(r['model'])} "
            f"| {r['acc_full']:.4f} | {r['acc_drop']:.4f} | {r['delta_acc']:+.4f} "
            f"| {r['macro_f1_full']:.4f} | {r['macro_f1_drop']:.4f} | {r['delta_macro_f1']:+.4f} |"
        )
    out_md = out_csv.with_suffix(".md")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
