
from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..evaluation.stat_tests import bootstrap_metric, mcnemar_test
from .class_aliases import remap_report_keys


@dataclass
class RunArtifacts:
    name: str
    run_dir: Path
    metrics: dict
    report: dict | None
    logits: np.ndarray | None = None
    labels: np.ndarray | None = None
    summary: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def preds(self) -> np.ndarray | None:
        if self.logits is None:
            return None
        return self.logits.argmax(axis=1)


def _safe_load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _safe_load_npz(path: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not path.is_file():
        return None, None
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None, None
    return data["logits"], data["labels"]


def _short_name_from_zip(zip_path: Path) -> str:
    name = zip_path.stem
    name = re.sub(r"^run_", "", name)
    return name


def _unpack_zip(zip_path: Path, dst_root: Path) -> Path | None:
    out_dir = dst_root / zip_path.stem
    if not out_dir.is_dir() or not any(out_dir.iterdir()):
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(out_dir)
        except zipfile.BadZipFile:
            return None
    children = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return out_dir


def load_run(zip_path: Path, cache_dir: Path) -> RunArtifacts | None:
    base = _unpack_zip(zip_path, cache_dir)
    if base is None:
        return None
    metrics = _safe_load_json(base / "test_metrics.json")
    if not metrics:
        return None
    report = _safe_load_json(base / "test_report.json") or None
    if report:
        report = remap_report_keys(report)
    logits, labels = _safe_load_npz(base / "test_logits.npz")
    summary = _safe_load_json(base / "summary.json")
    config = _safe_load_json(base / "config.json")
    return RunArtifacts(
        name=_short_name_from_zip(zip_path),
        run_dir=base,
        metrics=metrics,
        report=report,
        logits=logits,
        labels=labels,
        summary=summary,
        config=config,
    )


def _row_for_run(run: RunArtifacts, n_bootstrap: int) -> dict:
    row: dict = {
        "model": run.name,
        "accuracy": float(run.metrics.get("accuracy", float("nan"))),
        "macro_f1": float(run.metrics.get("macro_f1", float("nan"))),
        "balanced_accuracy": float(run.metrics.get("balanced_accuracy", float("nan"))),
        "best_epoch": run.summary.get("best_epoch"),
        "best_val_macro_f1": run.summary.get("best_val_macro_f1"),
        "n_test": run.metrics.get("num_test"),
    }
    if "top5_accuracy" in run.metrics:
        row["top5_accuracy"] = float(run.metrics["top5_accuracy"])
    if run.logits is not None and run.labels is not None and n_bootstrap > 0:
        preds = run.preds
        truths = run.labels
        ci_acc = bootstrap_metric(truths, preds, "accuracy",
                                  n_resamples=n_bootstrap)
        ci_f1 = bootstrap_metric(truths, preds, "macro_f1",
                                 n_resamples=n_bootstrap)
        row.update({
            "accuracy_ci_lo": ci_acc.ci_lo,
            "accuracy_ci_hi": ci_acc.ci_hi,
            "macro_f1_ci_lo": ci_f1.ci_lo,
            "macro_f1_ci_hi": ci_f1.ci_hi,
        })
    return row


def _format_md(table: list[dict]) -> str:
    cols = [
        ("model", "Model"),
        ("accuracy", "acc"),
        ("accuracy_ci_lo", "acc_lo"),
        ("accuracy_ci_hi", "acc_hi"),
        ("macro_f1", "macro_F1"),
        ("macro_f1_ci_lo", "F1_lo"),
        ("macro_f1_ci_hi", "F1_hi"),
        ("balanced_accuracy", "bal_acc"),
        ("top5_accuracy", "top5"),
        ("best_epoch", "best_ep"),
        ("n_test", "N"),
    ]
    headers = " | ".join(label for _, label in cols)
    sep = " | ".join("---" for _ in cols)
    rows = [f"| {headers} |", f"| {sep} |"]
    for r in table:
        cells = []
        for key, _ in cols:
            v = r.get(key)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    lines = [",".join(keys)]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.6f}")
            else:
                cells.append(str(v).replace(",", ";"))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def per_class_recall_matrix(runs: list[RunArtifacts]) -> tuple[list[str], list[str], list[list[float]]]:
    classes: list[str] = []
    for r in runs:
        if not r.report:
            continue
        for k in r.report:
            if isinstance(r.report.get(k), dict) and "recall" in r.report[k] and k not in {
                "accuracy", "macro avg", "weighted avg", "micro avg",
            } and k not in classes:
                classes.append(k)
    classes.sort()
    model_names = [r.name for r in runs]
    matrix = []
    for cls in classes:
        row = []
        for r in runs:
            v = float("nan")
            if r.report and isinstance(r.report.get(cls), dict):
                v = float(r.report[cls].get("recall", float("nan")))
            row.append(v)
        matrix.append(row)
    return classes, model_names, matrix


def pairwise_mcnemar(runs: list[RunArtifacts]) -> list[dict]:
    eligible = [r for r in runs if r.preds is not None and r.labels is not None]
    out = []
    for i, a in enumerate(eligible):
        for j, b in enumerate(eligible):
            if i >= j:
                continue
            if len(a.labels) != len(b.labels):
                continue
            if not np.array_equal(a.labels, b.labels):
                continue
            res = mcnemar_test(a.preds, b.preds, a.labels)
            out.append({
                "model_a": a.name,
                "model_b": b.name,
                "b": res["b"],
                "c": res["c"],
                "n": res["n"],
                "p_value": res["p_value"],
                "p_chi_approx": res["chi_approx_p"],
                "delta_acc": float((a.preds == a.labels).mean() -
                                   (b.preds == b.labels).mean()),
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs_res"),
                        help="папка с run_*.zip и/или уже распакованными подпапками")
    parser.add_argument("--out-dir", type=Path, default=Path("results/aggregate"))
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="куда распаковывать (по умолчанию <runs-dir>/_unpacked)")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="число bootstrap-семплов (0 — выключить CI)")
    parser.add_argument("--include-zeroshot", type=Path, default=None,
                        help="каталог с зеро-шот результатами "
                             "(test_metrics.json внутри подпапок) — добавляется к таблице")
    args = parser.parse_args()

    runs_dir = args.runs_dir
    cache_dir = args.cache_dir or (runs_dir / "_unpacked")
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted(runs_dir.glob("run_*.zip"))
    if not zip_paths:
        raise SystemExit(f"no run_*.zip in {runs_dir}")

    runs: list[RunArtifacts] = []
    for zp in zip_paths:
        run = load_run(zp, cache_dir)
        if run is None:
            print(f"  ! skip {zp.name}: no test_metrics.json")
            continue
        print(f"  + {run.name:40s} "
              f"acc={run.metrics.get('accuracy', float('nan')):.4f}")
        runs.append(run)

    if args.include_zeroshot is not None and args.include_zeroshot.is_dir():
        for sub in sorted(p for p in args.include_zeroshot.iterdir() if p.is_dir()):
            metrics = _safe_load_json(sub / "test_metrics.json")
            if not metrics:
                continue
            report = _safe_load_json(sub / "test_report.json") or None
            if report:
                report = remap_report_keys(report)
            logits, labels = _safe_load_npz(sub / "test_logits.npz")
            zs = RunArtifacts(
                name=f"zeroshot_{sub.name}",
                run_dir=sub,
                metrics=metrics,
                report=report,
                logits=logits,
                labels=labels,
            )
            print(f"  + {zs.name:40s} "
                  f"acc={zs.metrics.get('accuracy', float('nan')):.4f} (zero-shot)")
            runs.append(zs)

    rows = [_row_for_run(r, args.bootstrap) for r in runs]
    rows.sort(key=lambda r: r.get("accuracy", 0.0), reverse=True)

    _write_csv(rows, args.out_dir / "summary_table.csv")
    (args.out_dir / "summary_table.md").write_text(_format_md(rows), encoding="utf-8")

    classes, model_names, mat = per_class_recall_matrix(runs)
    pcr_rows = []
    for cls, row in zip(classes, mat):
        item = {"class": cls}
        for m, v in zip(model_names, row):
            item[m] = v
        pcr_rows.append(item)
    _write_csv(pcr_rows, args.out_dir / "per_class_recall.csv")

    if args.bootstrap > 0:
        pw = pairwise_mcnemar(runs)
        _write_csv(pw, args.out_dir / "pairwise_mcnemar.csv")

    print(f"\nwrote summary_table.csv / .md to {args.out_dir}")
    print(f"      per_class_recall.csv ({len(classes)} classes × {len(model_names)} models)")
    if args.bootstrap > 0:
        print(f"      pairwise_mcnemar.csv  ({len(pw)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
