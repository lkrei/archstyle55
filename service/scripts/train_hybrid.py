from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

def _default_emb_dir() -> Path:
    candidates = [
        Path(os.environ.get("HYBRID_EMB_DIR", "")),
        Path("/runs_res/embeddings"),
        Path("/runs_res/aggregate/embeddings"),
    ]
    for c in candidates:
        if str(c) and c.is_dir():
            return c
    return Path("/runs_res/embeddings")


EMB_DIR = _default_emb_dir()
ATTR_CSV = Path(os.environ.get("HYBRID_ATTR_CSV", "/repo/pipeline/results/segmentation/attributes.csv"))
OUT_DEFAULT = Path(os.environ.get("HYBRID_OUT", "/data/models/hybrid_histgbm.pkl"))


def _key_from_path(p: str) -> str:
    parts = Path(p).parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return Path(p).name


def _load_attr_table() -> tuple[dict[str, np.ndarray], list[str]]:
    if not ATTR_CSV.is_file():
        raise FileNotFoundError(f"missing {ATTR_CSV}")
    with ATTR_CSV.open() as f:
        reader = csv.DictReader(f)
        feature_cols = [c for c in reader.fieldnames if c not in {"path", "split", "label"}]
        table: dict[str, np.ndarray] = {}
        for row in reader:
            key = _key_from_path(row["path"])
            vec = np.array([float(row[c]) for c in feature_cols], dtype=np.float32)
            table[key] = vec
    return table, feature_cols


def _build_split(emb_path: Path, attr_table: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(emb_path, allow_pickle=True)
    feats = d["features"].astype(np.float32)
    labels = d["labels"].astype(np.int64)
    paths = d["paths"]
    rows = []
    keep_labels = []
    n_skip = 0
    for i, p in enumerate(paths):
        key = _key_from_path(str(p))
        attr = attr_table.get(key)
        if attr is None:
            n_skip += 1
            continue
        rows.append(np.concatenate([feats[i], attr], axis=0))
        keep_labels.append(labels[i])
    if n_skip:
        print(f"  skipped {n_skip} rows missing attributes ({emb_path.name})")
    return np.stack(rows, axis=0), np.array(keep_labels, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.06)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("loading attribute table...")
    attr_table, feat_cols = _load_attr_table()
    print(f"attributes per image: {len(feat_cols)}")

    print("building train split...")
    Xtr, ytr = _build_split(EMB_DIR / "embeddings_train.npz", attr_table)
    print(f"  train: {Xtr.shape}")
    print("building val split...")
    Xv, yv = _build_split(EMB_DIR / "embeddings_val.npz", attr_table)
    print(f"  val: {Xv.shape}")

    print("training HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.lr,
        max_depth=None,
        l2_regularization=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    clf.fit(Xtr, ytr)

    yp = clf.predict(Xv)
    acc = accuracy_score(yv, yp)
    f1 = f1_score(yv, yp, average="macro")
    print(f"val acc {acc:.4f} | macro F1 {f1:.4f}")

    joblib.dump(clf, args.out, compress=3)
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
