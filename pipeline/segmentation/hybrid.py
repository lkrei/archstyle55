
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from ..config import RESULTS_DIR, SEGMENT_DIR


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _load_logits(npz_path: Path):
    bundle = np.load(npz_path, allow_pickle=True)
    return bundle["logits"], bundle["labels"].astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-logits", type=Path, required=True,
                        help="logits на val (npz: logits, labels)")
    parser.add_argument("--test-logits", type=Path, required=True,
                        help="logits на test (npz: logits, labels)")
    parser.add_argument("--attributes", type=Path, default=SEGMENT_DIR / "attributes.csv")
    parser.add_argument("--val-paths", type=Path, required=True,
                        help="json со списком sample-словарей val (path, label)")
    parser.add_argument("--test-paths", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "hybrid")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_logits, val_labels = _load_logits(args.val_logits)
    test_logits, test_labels = _load_logits(args.test_logits)
    val_probs = softmax(val_logits)
    test_probs = softmax(test_logits)

    attrs = pd.read_csv(args.attributes)
    feat_cols = [c for c in attrs.columns if c not in {"path", "split", "label"}]
    attr_map = attrs.set_index("path")[feat_cols]

    val_paths = [s["path"] for s in json.loads(args.val_paths.read_text())]
    test_paths = [s["path"] for s in json.loads(args.test_paths.read_text())]

    val_attr = attr_map.reindex(val_paths).fillna(0.0).to_numpy(dtype=np.float32)
    test_attr = attr_map.reindex(test_paths).fillna(0.0).to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    val_attr = scaler.fit_transform(val_attr)
    test_attr = scaler.transform(test_attr)

    Xtr = np.concatenate([val_probs, val_attr], axis=1)
    Xte = np.concatenate([test_probs, test_attr], axis=1)

    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=6)
    clf.fit(Xtr, val_labels)
    pred = clf.predict(Xte)

    image_only_pred = test_logits.argmax(axis=1)
    metrics = {
        "image_only_accuracy": float(accuracy_score(test_labels, image_only_pred)),
        "image_only_macro_f1": float(f1_score(test_labels, image_only_pred,
                                              average="macro", zero_division=0)),
        "hybrid_accuracy": float(accuracy_score(test_labels, pred)),
        "hybrid_macro_f1": float(f1_score(test_labels, pred,
                                          average="macro", zero_division=0)),
        "n_features_image": int(val_probs.shape[1]),
        "n_features_attr": int(val_attr.shape[1]),
    }
    (args.out_dir / "hybrid_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
