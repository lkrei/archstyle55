
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import SPLITS_DIR
from ..evaluation.calibration import (
    apply_temperature,
    calibration_report,
    fit_temperature,
    write_calibration_report,
)
from ..evaluation.stat_tests import bootstrap_metric
from .figures import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_reliability,
    plot_training_curves,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=SPLITS_DIR / "data_splits.json")
    parser.add_argument("--classes", type=Path, default=SPLITS_DIR / "idx_to_class.json")
    parser.add_argument("--balance", type=Path, default=SPLITS_DIR / "class_balance.csv")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_fig = run_dir / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.is_file():
        plot_training_curves(metrics_csv, out_fig / "training_curves.png")

    if args.balance.is_file():
        plot_class_distribution(args.balance, out_fig / "class_distribution.png")

    logits_path = run_dir / "test_logits.npz"
    confusion_path = run_dir / "test_confusion.npy"
    report_path = run_dir / "test_report.json"

    if logits_path.is_file() and confusion_path.is_file():
        bundle = np.load(logits_path, allow_pickle=True)
        logits = bundle["logits"]
        labels = bundle["labels"].astype(np.int64)
        class_names = list(bundle["class_names"])
        cm = np.load(confusion_path)

        plot_confusion_matrix(cm, class_names, out_fig / "confusion_matrix.png")

        if report_path.is_file():
            plot_per_class_f1(json.loads(report_path.read_text()),
                              out_fig / "per_class_f1.png")

        pre = calibration_report(logits, labels)
        T = fit_temperature(logits, labels)
        post = calibration_report(apply_temperature(logits, T), labels)
        plot_reliability(pre, post, out_fig / "reliability.png",
                         title="reliability before/after temperature scaling")
        write_calibration_report(run_dir / "calibration.json",
                                 label="post-train", pre=pre, post=post, T=T)

        boot_acc = bootstrap_metric(labels, logits.argmax(axis=1),
                                    metric="accuracy", n_resamples=1000)
        boot_f1 = bootstrap_metric(labels, logits.argmax(axis=1),
                                   metric="macro_f1", n_resamples=1000)
        (run_dir / "bootstrap.json").write_text(json.dumps({
            "accuracy": boot_acc.__dict__,
            "macro_f1": boot_f1.__dict__,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"figures saved to {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
