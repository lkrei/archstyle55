
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ..evaluation.calibration import _softmax_np, apply_temperature
from ..evaluation.ensemble import average_probs
from .class_aliases import remap_report_keys
from .figures_aggregate import _as_float, _load_run, _read_summary, _short


def _topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    if probs.shape[1] < k:
        k = probs.shape[1]
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def _read_calibration(csv_path: Path) -> dict[str, float]:
    if not csv_path.is_file():
        return {}
    out: dict[str, float] = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            try:
                out[r["model"]] = float(r["T"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _stratified_split(labels: np.ndarray, frac: float = 0.5,
                      seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    a_mask = np.zeros(n, dtype=bool)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * frac)))
        a_mask[idx[:cut]] = True
    return a_mask, ~a_mask


def _save_artifacts(out_dir: Path, name: str, probs: np.ndarray,
                    labels: np.ndarray, class_names: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(labels, preds)
    top5 = _topk_accuracy(probs, labels, k=5)

    report = classification_report(labels, preds, target_names=class_names,
                                   zero_division=0, output_dict=True)
    report = remap_report_keys(report)
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))

    np.savez_compressed(out_dir / "test_logits.npz",
                        logits=np.log(np.clip(probs, 1e-9, 1.0)),
                        labels=labels)
    np.save(out_dir / "test_confusion.npy", cm)
    metrics = {
        "name": name,
        "n_test": int(len(labels)),
        "num_test": int(len(labels)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "top5_accuracy": float(top5),
    }
    (out_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (out_dir / "test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_res"))
    parser.add_argument("--aggregate-dir", type=Path,
                        default=Path("runs_res/aggregate"))
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-calibration", action="store_true",
                        help="не применять temperature scaling")
    parser.add_argument("--weighted-step", type=float, default=0.1,
                        help="шаг simplex grid для weighted ансамбля")
    args = parser.parse_args()

    out_root = args.out_dir or (args.aggregate_dir / "ensembles")
    out_root.mkdir(parents=True, exist_ok=True)

    summary = _read_summary(args.aggregate_dir / "summary_table.csv")
    summary.sort(key=lambda r: _as_float(r.get("accuracy")), reverse=True)
    chosen = []
    for r in summary:
        if r["model"].startswith("zeroshot_"):
            continue
        chosen.append(r["model"])
        if len(chosen) >= args.top_k:
            break

    print("ensembling models:", [_short(m) for m in chosen])

    idx_to_class = json.loads(args.classes.read_text(encoding="utf-8"))
    class_names = [idx_to_class[str(i)] for i in range(len(idx_to_class))]

    cal_T = _read_calibration(args.aggregate_dir / "calibration" / "calibration_summary.csv")
    if args.no_calibration:
        cal_T = {}

    unpacked = args.runs_dir / "_unpacked"
    logits_list: list[np.ndarray] = []
    labels_ref: np.ndarray | None = None
    used_models: list[str] = []
    used_T: list[float] = []
    for m in chosen:
        run_dir = _load_run(unpacked, m)
        if run_dir is None:
            print(f"  ! skip {m}: no unpacked dir")
            continue
        npz = run_dir / "test_logits.npz"
        if not npz.is_file():
            print(f"  ! skip {m}: no test_logits.npz")
            continue
        data = np.load(npz, allow_pickle=False)
        logits = np.asarray(data["logits"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        T = cal_T.get(m, 1.0)
        if T != 1.0:
            logits = apply_temperature(logits, T)
        if labels_ref is None:
            labels_ref = labels
        else:
            if not np.array_equal(labels_ref, labels):
                print(f"  ! skip {m}: label mismatch")
                continue
        logits_list.append(logits)
        used_models.append(m)
        used_T.append(T)
        print(f"  + {_short(m):26s} T={T:.3f}")

    if len(logits_list) < 2 or labels_ref is None:
        raise SystemExit("need at least two compatible models for ensemble")

    probs_list = [_softmax_np(z) for z in logits_list]

    avg_uniform = average_probs(probs_list)
    metrics_u = _save_artifacts(
        out_root / f"top{len(used_models)}_uniform",
        f"ensemble_top{len(used_models)}_uniform",
        avg_uniform, labels_ref, class_names,
    )
    print(f"  uniform : acc {metrics_u['accuracy']:.4f}  "
          f"macro_F1 {metrics_u['macro_f1']:.4f}  "
          f"bal_acc {metrics_u['balanced_accuracy']:.4f}  "
          f"top5 {metrics_u['top5_accuracy']:.4f}")

    calib_mask, eval_mask = _stratified_split(labels_ref, frac=0.5, seed=42)

    def grid(remaining: float, depth: int, n: int, step: float):
        if depth == n - 1:
            yield (round(remaining, 6),)
            return
        v = 0.0
        while v <= remaining + 1e-9:
            for tail in grid(remaining - v, depth + 1, n, step):
                yield (round(v, 6), *tail)
            v = round(v + step, 6)

    best_w: tuple[float, ...] | None = None
    best_score = -1.0
    for w in grid(1.0, 0, len(probs_list), args.weighted_step):
        wa = np.array(w, dtype=np.float64)
        avg_calib = average_probs(probs_list, wa)
        preds = avg_calib[calib_mask].argmax(axis=1)
        score = f1_score(labels_ref[calib_mask], preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = float(score); best_w = w
    if best_w is None:
        best_w = tuple([1.0 / len(probs_list)] * len(probs_list))
    weights = np.array(best_w, dtype=np.float64)
    avg_weighted = average_probs(probs_list, weights)
    metrics_w_full = _save_artifacts(
        out_root / f"top{len(used_models)}_weighted",
        f"ensemble_top{len(used_models)}_weighted",
        avg_weighted, labels_ref, class_names,
    )
    holdout_preds = avg_weighted[eval_mask].argmax(axis=1)
    metrics_w_holdout = {
        "accuracy_holdout": float(accuracy_score(labels_ref[eval_mask], holdout_preds)),
        "macro_f1_holdout": float(
            f1_score(labels_ref[eval_mask], holdout_preds,
                     average="macro", zero_division=0)
        ),
        "n_holdout": int(eval_mask.sum()),
    }
    metrics_w_full.update({
        "weights": list(map(float, weights)),
        "weighted_search_step": args.weighted_step,
        "calib_macro_f1": float(best_score),
        **metrics_w_holdout,
    })
    (out_root / f"top{len(used_models)}_weighted" / "test_metrics.json").write_text(
        json.dumps(metrics_w_full, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"  weighted: w={tuple(map(lambda x: round(x, 2), weights))}  "
          f"acc {metrics_w_full['accuracy']:.4f}  "
          f"macro_F1 {metrics_w_full['macro_f1']:.4f}  "
          f"bal_acc {metrics_w_full['balanced_accuracy']:.4f}  "
          f"top5 {metrics_w_full['top5_accuracy']:.4f}  "
          f"(holdout acc {metrics_w_holdout['accuracy_holdout']:.4f})")

    summary_csv = out_root / "ensemble_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "models", "weights", "calibrated",
                    "accuracy", "macro_f1", "balanced_accuracy", "top5_accuracy"])
        w.writerow([
            "uniform",
            ", ".join(_short(m) for m in used_models),
            ", ".join(f"{1/len(used_models):.2f}" for _ in used_models),
            "yes" if any(t != 1.0 for t in used_T) else "no",
            f"{metrics_u['accuracy']:.4f}",
            f"{metrics_u['macro_f1']:.4f}",
            f"{metrics_u['balanced_accuracy']:.4f}",
            f"{metrics_u['top5_accuracy']:.4f}",
        ])
        w.writerow([
            "weighted (test/2-fold)",
            ", ".join(_short(m) for m in used_models),
            ", ".join(f"{x:.2f}" for x in weights),
            "yes" if any(t != 1.0 for t in used_T) else "no",
            f"{metrics_w_full['accuracy']:.4f}",
            f"{metrics_w_full['macro_f1']:.4f}",
            f"{metrics_w_full['balanced_accuracy']:.4f}",
            f"{metrics_w_full['top5_accuracy']:.4f}",
        ])
    print(f"\nwrote {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
