"""Hybrid classifier: DINOv2 embeddings concat with SegFormer attributes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


def _key_from_path(p: str) -> str:

    parts = str(p).replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _join_attrs(paths: np.ndarray, attr_map: pd.DataFrame,
                feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (X_attr, mask_found) для списка путей."""
    keys = [_key_from_path(p) for p in paths]
    df = attr_map.reindex(keys)
    mask_found = ~df.iloc[:, 0].isna().to_numpy()
    df = df.fillna(0.0)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    return X, mask_found


def _load_split_emb(emb_dir: Path, split: str) -> dict[str, np.ndarray]:
    npz = emb_dir / f"embeddings_{split}.npz"
    if not npz.is_file():
        raise SystemExit(f"missing {npz}; run pipeline.reports.embeddings --split {split}")
    data = np.load(npz, allow_pickle=True)
    out = {
        "features": np.asarray(data["features"], dtype=np.float32),
        "labels":   np.asarray(data["labels"],   dtype=np.int64),
    }
    if "paths" in data.files:
        out["paths"] = np.asarray(data["paths"], dtype=object)
    return out


def _eval(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _fit_eval(name: str, Xtr, ytr, Xte, yte, max_iter: int) -> dict:
    t0 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.06, max_depth=6,
        l2_regularization=1e-3, random_state=42,
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics = _eval(yte, pred) | {
        "name": name,
        "n_train": int(len(ytr)),
        "n_test":  int(len(yte)),
        "n_features": int(Xtr.shape[1]),
        "fit_seconds": round(time.time() - t0, 1),
    }
    return metrics, pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", type=Path, required=True,
                        help="каталог с embeddings_train.npz / val / test")
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--no-val", action="store_true",
                        help="не использовать val (по умолчанию train+val в trainset)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = _load_split_emb(args.emb_dir, "train")
    test  = _load_split_emb(args.emb_dir, "test")
    if not args.no_val:
        try:
            val = _load_split_emb(args.emb_dir, "val")
            train = {
                "features": np.concatenate([train["features"], val["features"]], axis=0),
                "labels":   np.concatenate([train["labels"],   val["labels"]],   axis=0),
                "paths":    (np.concatenate([train["paths"], val["paths"]], axis=0)
                             if "paths" in train and "paths" in val else None),
            }
            print(f"  trainset: train+val = {len(train['labels'])} samples")
        except SystemExit as exc:
            print(f"  no val embeddings, training on train only ({exc})")

    if "paths" not in train or train["paths"] is None or "paths" not in test:
        raise SystemExit("embeddings *.npz must contain 'paths' (rerun pipeline.reports.embeddings)")

    idx_to_class = json.loads(args.classes.read_text(encoding="utf-8"))
    class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]

    attrs = pd.read_csv(args.attributes)
    feat_cols = [c for c in attrs.columns if c not in {"path", "split", "label"}]
    attrs["key"] = attrs["path"].map(_key_from_path)
    attr_map = attrs.set_index("key")
    print(f"  attributes: {attr_map.shape[0]} rows, {len(feat_cols)} features")

    Xtr_attr, found_tr = _join_attrs(train["paths"], attr_map, feat_cols)
    Xte_attr, found_te = _join_attrs(test["paths"],  attr_map, feat_cols)
    print(f"  attr coverage: train {found_tr.mean():.3f}, test {found_te.mean():.3f}")

    scaler = StandardScaler().fit(Xtr_attr)
    Xtr_attr = scaler.transform(Xtr_attr).astype(np.float32)
    Xte_attr = scaler.transform(Xte_attr).astype(np.float32)

    Xtr_emb = train["features"]
    Xte_emb = test["features"]
    ytr = train["labels"]
    yte = test["labels"]

    summaries = []

    print("\n[1/3] attr_only ...")
    m_attr, pred_attr = _fit_eval("attr_only", Xtr_attr, ytr, Xte_attr, yte, args.max_iter)
    summaries.append(m_attr); print("       ", m_attr)

    print("[2/3] emb_only ...")
    m_emb, pred_emb = _fit_eval("emb_only", Xtr_emb, ytr, Xte_emb, yte, args.max_iter)
    summaries.append(m_emb); print("       ", m_emb)

    print("[3/3] hybrid (emb || attr) ...")
    Xtr_hyb = np.concatenate([Xtr_emb, Xtr_attr], axis=1)
    Xte_hyb = np.concatenate([Xte_emb, Xte_attr], axis=1)
    m_hyb, pred_hyb = _fit_eval("hybrid", Xtr_hyb, ytr, Xte_hyb, yte, args.max_iter)
    summaries.append(m_hyb); print("       ", m_hyb)

    cm = confusion_matrix(yte, pred_hyb, labels=list(range(len(class_names))))
    np.save(args.out_dir / "test_confusion.npy", cm)

    report = classification_report(yte, pred_hyb,
                                   target_names=class_names,
                                   zero_division=0, output_dict=True)
    (args.out_dir / "test_report_hybrid.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    np.savez_compressed(args.out_dir / "predictions.npz",
                        y_true=yte, attr_only=pred_attr,
                        emb_only=pred_emb, hybrid=pred_hyb)
    (args.out_dir / "hybrid_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    md_lines = ["| variant | acc | macro_F1 | bal_acc | n_features | fit, s |",
                "| --- | --- | --- | --- | --- | --- |"]
    for s in summaries:
        md_lines.append(
            f"| {s['name']} | {s['accuracy']:.4f} | {s['macro_f1']:.4f} "
            f"| {s['balanced_accuracy']:.4f} | {s['n_features']} | {s['fit_seconds']} |"
        )
    (args.out_dir / "hybrid_summary.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8",
    )

    print(f"\nwrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
