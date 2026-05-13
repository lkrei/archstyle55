
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _key(p: str) -> str:
    parts = str(p).replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _load_logits(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    npz = run_dir / "test_logits.npz"
    if not npz.is_file():
        raise SystemExit(f"missing {npz}")
    data = np.load(npz, allow_pickle=False)
    return np.asarray(data["logits"], dtype=np.float64), \
           np.asarray(data["labels"], dtype=np.int64)


def _eval(y_true, y_pred) -> dict:
    return {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "macro_f1":           float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_true, y_pred)),
    }


def _mcnemar(a, b, y) -> dict:
    a_ok = (a == y); b_ok = (b == y)
    n01 = int(((~a_ok) & b_ok).sum())
    n10 = int((a_ok & (~b_ok)).sum())
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p": float(p)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True,
                        help="имена backbone-папок (run_*_seed42 -> сами слаги)")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--attributes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logits = []
    labels_ref = None
    for m in args.models:
        run_dir = args.runs_dir / f"run_{m}_seed42" / f"{m}_seed42"
        L, y = _load_logits(run_dir)
        if labels_ref is None:
            labels_ref = y
        elif not np.array_equal(y, labels_ref):
            raise SystemExit(f"label order differs for {m}")
        logits.append(L)
    probs = [softmax(L) for L in logits]
    print(f"  loaded {len(probs)} models, test size {labels_ref.shape[0]}")

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    test_paths = np.array([_key(s["path"]) for s in splits["test"]])
    if len(test_paths) != len(labels_ref):
        raise SystemExit("splits/test size != logits size")

    attrs = pd.read_csv(args.attributes)
    feat_cols = [c for c in attrs.columns if c not in {"path", "split", "label"}]
    attrs["key"] = attrs["path"].map(_key)
    attr_map = attrs.set_index("key")
    df = attr_map.reindex(test_paths)
    found = ~df.iloc[:, 0].isna().to_numpy()
    print(f"  attr coverage: {found.mean():.3f}")
    df = df.fillna(0.0)
    attrs_X = df[feat_cols].to_numpy(dtype=np.float32)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    train_idx, eval_idx = next(sss.split(np.zeros_like(labels_ref), labels_ref))
    print(f"  split: stack-train {len(train_idx)}, eval {len(eval_idx)}")

    y_train = labels_ref[train_idx]
    y_eval  = labels_ref[eval_idx]

    scaler = StandardScaler().fit(attrs_X[train_idx])
    attrs_train = scaler.transform(attrs_X[train_idx]).astype(np.float32)
    attrs_eval  = scaler.transform(attrs_X[eval_idx]).astype(np.float32)

    probs_train = np.concatenate([P[train_idx] for P in probs], axis=1)
    probs_eval  = np.concatenate([P[eval_idx]  for P in probs], axis=1)

    pred_v2s_eval     = probs[0][eval_idx].argmax(axis=1)
    ens_probs_eval    = np.mean([P[eval_idx] for P in probs], axis=0)
    pred_ensemble     = ens_probs_eval.argmax(axis=1)

    def fit_and_eval(name: str, Xtr, Xev) -> tuple[dict, np.ndarray]:
        clf = HistGradientBoostingClassifier(
            max_iter=args.max_iter, learning_rate=0.06, max_depth=6,
            l2_regularization=1e-3, random_state=args.seed,
        )
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xev)
        m = _eval(y_eval, pred)
        m["name"] = name
        m["n_train"] = len(y_train)
        m["n_eval"]  = len(y_eval)
        m["n_features"] = Xtr.shape[1]
        return m, pred

    rows = []

    rows.append({"name": "single_v2s_argmax",
                 **_eval(y_eval, pred_v2s_eval),
                 "n_train": 0, "n_eval": len(y_eval),
                 "n_features": probs[0].shape[1]})

    rows.append({"name": "ensemble_top3_uniform_argmax",
                 **_eval(y_eval, pred_ensemble),
                 "n_train": 0, "n_eval": len(y_eval),
                 "n_features": probs[0].shape[1] * len(probs)})

    m_attr,  p_attr  = fit_and_eval("attrs_only", attrs_train, attrs_eval)
    rows.append(m_attr)

    m_pt,    p_pt    = fit_and_eval("stacker_probs_top3", probs_train, probs_eval)
    rows.append(m_pt)

    Xtr_full = np.concatenate([probs_train, attrs_train], axis=1)
    Xev_full = np.concatenate([probs_eval,  attrs_eval],  axis=1)
    m_pt_a,  p_pt_a  = fit_and_eval("stacker_probs_top3_attrs", Xtr_full, Xev_full)
    rows.append(m_pt_a)

    Xtr_v2a = np.concatenate([probs[0][train_idx], attrs_train], axis=1)
    Xev_v2a = np.concatenate([probs[0][eval_idx],  attrs_eval],  axis=1)
    m_va,    p_va    = fit_and_eval("stacker_v2s_attrs", Xtr_v2a, Xev_v2a)
    rows.append(m_va)

    mc = {
        "v2s_vs_ensemble":            _mcnemar(pred_v2s_eval, pred_ensemble, y_eval),
        "ensemble_vs_stacker_pt":     _mcnemar(pred_ensemble, p_pt, y_eval),
        "ensemble_vs_stacker_pt_atr": _mcnemar(pred_ensemble, p_pt_a, y_eval),
        "stacker_pt_vs_pt_atr":       _mcnemar(p_pt, p_pt_a, y_eval),
        "v2s_vs_stacker_v2s_atr":     _mcnemar(pred_v2s_eval, p_va, y_eval),
    }

    np.savez_compressed(args.out_dir / "predictions.npz",
                        y_true=y_eval, eval_idx=eval_idx,
                        v2s=pred_v2s_eval, ensemble=pred_ensemble,
                        attrs_only=p_attr, stacker_probs_top3=p_pt,
                        stacker_probs_top3_attrs=p_pt_a,
                        stacker_v2s_attrs=p_va)
    (args.out_dir / "stacking_summary.json").write_text(
        json.dumps({"rows": rows, "mcnemar": mc}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    pd.DataFrame(rows)[["name", "accuracy", "macro_f1", "balanced_accuracy",
                        "n_train", "n_eval", "n_features"]
                      ].to_csv(args.out_dir / "stacking_summary.csv", index=False)

    md = ["| вариант | acc | macro F1 | bal. acc | n_features | n_train |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(
            f"| {r['name']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} "
            f"| {r['balanced_accuracy']:.4f} | {r['n_features']} | {r['n_train']} |"
        )
    md.append("")
    md.append("McNemar (на той же 50% eval-выборке):")
    for k, v in mc.items():
        md.append(f"* `{k}`: chi2={v['chi2']:.3f}, p={v['p']:.4f}, n01={v['n01']}, n10={v['n10']}")
    (args.out_dir / "stacking_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print()
    for r in rows:
        print(f"  {r['name']:32s}  acc {r['accuracy']:.4f}  F1 {r['macro_f1']:.4f}")
    print("\n  McNemar:")
    for k, v in mc.items():
        print(f"  {k:36s}  chi2={v['chi2']:.3f}  p={v['p']:.4f}")
    print(f"\n  wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
