
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..evaluation.calibration import (
    apply_temperature,
    calibration_report,
    fit_temperature,
)
from .figures_aggregate import _as_float, _load_run, _read_summary


def _short(name: str) -> str:
    return name.removesuffix("_seed42")


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


def _plot_reliability(rep_pre, rep_post, T: float | None,
                      title: str, out_path: Path) -> None:
    bins = np.linspace(0, 1, rep_pre.n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    width = 1.0 / rep_pre.n_bins

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(centers, rep_pre.bin_acc, width=width, alpha=0.45,
           label=f"pre ECE={rep_pre.ece:.3f}", edgecolor="#3a3a3a")
    if rep_post is not None:
        ax.bar(centers, rep_post.bin_acc, width=width, alpha=0.55,
               label=f"post ECE={rep_post.ece:.3f}  (T={T:.2f})",
               edgecolor="#882020")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def calibrate_model(name: str, run_dir: Path, out_dir: Path,
                    n_bins: int = 15) -> dict | None:
    npz = run_dir / "test_logits.npz"
    if not npz.is_file():
        return None
    data = np.load(npz, allow_pickle=False)
    logits = np.asarray(data["logits"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)

    calib_mask, eval_mask = _stratified_split(labels, frac=0.5, seed=42)

    rep_pre = calibration_report(logits[eval_mask], labels[eval_mask], n_bins=n_bins)
    T = fit_temperature(logits[calib_mask], labels[calib_mask])
    cal_logits = apply_temperature(logits[eval_mask], T)
    rep_post = calibration_report(cal_logits, labels[eval_mask], n_bins=n_bins)

    short = _short(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": name,
        "temperature": T,
        "n_eval": int(eval_mask.sum()),
        "n_calib": int(calib_mask.sum()),
        "pre":  {"ece": rep_pre.ece,  "mce": rep_pre.mce},
        "post": {"ece": rep_post.ece, "mce": rep_post.mce},
    }
    (out_dir / f"calibration_{short}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    _plot_reliability(rep_pre, rep_post, T,
                      title=f"{short} — temperature scaling",
                      out_path=out_dir / f"reliability_{short}.png")
    return {
        "model": name,
        "T": T,
        "ece_pre": rep_pre.ece,
        "ece_post": rep_post.ece,
        "mce_pre": rep_pre.mce,
        "mce_post": rep_post.mce,
        "n_eval": int(eval_mask.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_res"))
    parser.add_argument("--aggregate-dir", type=Path,
                        default=Path("runs_res/aggregate"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5,
                        help="сколько лучших моделей калибровать (по accuracy)")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--include-zeroshot-dir", type=Path, default=None,
                        help="каталог с zeroshot-результатами (test_logits.npz)")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.aggregate_dir / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_summary(args.aggregate_dir / "summary_table.csv")
    summary.sort(key=lambda r: _as_float(r.get("accuracy")), reverse=True)
    chosen_supervised = [r["model"] for r in summary[: args.top_k]
                         if not r["model"].startswith("zeroshot_")]

    rows: list[dict] = []
    unpacked = args.runs_dir / "_unpacked"
    for m in chosen_supervised:
        run_dir = _load_run(unpacked, m)
        if run_dir is None:
            print(f"  ! skip {m}: no unpacked dir")
            continue
        r = calibrate_model(m, run_dir, out_dir, n_bins=args.n_bins)
        if r is None:
            print(f"  ! skip {m}: no test_logits.npz")
            continue
        rows.append(r)
        print(f"  + {_short(m):28s} T={r['T']:.3f}  "
              f"ECE {r['ece_pre']:.3f} -> {r['ece_post']:.3f}  "
              f"MCE {r['mce_pre']:.3f} -> {r['mce_post']:.3f}")

    if args.include_zeroshot_dir is not None and args.include_zeroshot_dir.is_dir():
        for sub in sorted(p for p in args.include_zeroshot_dir.iterdir() if p.is_dir()):
            if not (sub / "test_logits.npz").is_file():
                continue
            name = f"zeroshot_{sub.name}"
            r = calibrate_model(name, sub, out_dir, n_bins=args.n_bins)
            if r is None:
                continue
            rows.append(r)
            print(f"  + {name:28s} T={r['T']:.3f}  "
                  f"ECE {r['ece_pre']:.3f} -> {r['ece_post']:.3f}  "
                  f"MCE {r['mce_pre']:.3f} -> {r['mce_post']:.3f}")

    if rows:
        csv_path = out_dir / "calibration_summary.csv"
        keys = ["model", "T", "ece_pre", "ece_post", "mce_pre", "mce_post", "n_eval"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
        print(f"\nwrote {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
