
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .class_aliases import apply_aliases

DROP = "Moscow Luzhkov style architecture"


def _load_logits(run: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(run / "test_logits.npz")
    return d["logits"], d["labels"]


def _per_class_recall(logits: np.ndarray, labels: np.ndarray, n_cls: int) -> np.ndarray:
    pred = logits.argmax(axis=1)
    rec = np.zeros(n_cls, dtype=np.float64)
    for c in range(n_cls):
        mask = labels == c
        if mask.sum() == 0:
            rec[c] = float("nan")
            continue
        rec[c] = float((pred[mask] == c).mean())
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--abl", type=Path, required=True)
    p.add_argument("--classes-full", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    classes_full = json.loads(args.classes_full.read_text(encoding="utf-8"))
    idx2name = {int(k): v for k, v in classes_full.items()}
    drop_idx = next(i for i, n in idx2name.items() if n == DROP)
    keep_full = sorted(i for i in idx2name if i != drop_idx)
    map_full_to_abl = {old: new for new, old in enumerate(keep_full)}

    logits_full, labels_full = _load_logits(args.full)
    logits_abl, labels_abl = _load_logits(args.abl)

    n_full, n_abl = logits_full.shape[1], logits_abl.shape[1]
    rec_full_55 = _per_class_recall(logits_full, labels_full, n_full)
    rec_abl_54 = _per_class_recall(logits_abl, labels_abl, n_abl)

    keep_mask = labels_full != drop_idx
    pred_full = logits_full.argmax(axis=1)
    acc_full_on_overlap = float((pred_full[keep_mask] == labels_full[keep_mask]).mean())

    pred_abl = logits_abl.argmax(axis=1)
    acc_abl = float((pred_abl == labels_abl).mean())

    deltas = []
    for old_idx in keep_full:
        new_idx = map_full_to_abl[old_idx]
        deltas.append({
            "class": apply_aliases(idx2name[old_idx]),
            "recall_full_55": float(rec_full_55[old_idx]),
            "recall_abl_54": float(rec_abl_54[new_idx]),
            "delta": float(rec_abl_54[new_idx] - rec_full_55[old_idx]),
        })
    deltas.sort(key=lambda r: r["delta"])

    summary = {
        "drop_class": apply_aliases(DROP),
        "full_55": {"accuracy": float((pred_full == labels_full).mean())},
        "ablation_54": {"accuracy": acc_abl},
        "full_55_excl_drop": {"accuracy": acc_full_on_overlap},
        "delta_acc_overlap_vs_abl54": acc_abl - acc_full_on_overlap,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.out / "ablation_summary.json").write_text(
        json.dumps({"summary": summary, "per_class": deltas}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_lines = [
        "| class | recall (full 55) | recall (abl 54) | Δ |",
        "| --- | --- | --- | --- |",
    ]
    for r in deltas:
        md_lines.append(f"| {r['class']} | {r['recall_full_55']:.3f} | "
                        f"{r['recall_abl_54']:.3f} | {r['delta']:+.3f} |")
    (args.out / "ablation_per_class.md").write_text("\n".join(md_lines), encoding="utf-8")

    sorted_by_delta = sorted(deltas, key=lambda r: r["delta"])
    names = [r["class"] for r in sorted_by_delta]
    vals = np.array([r["delta"] for r in sorted_by_delta])
    colors = ["#cf2e2e" if v < 0 else "#1f9d55" for v in vals]
    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.18)))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Δ recall (abl 54 − full 55)")
    ax.set_title(f"V2-S: per-class recall change after dropping «{apply_aliases(DROP)}»",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(args.out / "ablation_per_class_delta.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
