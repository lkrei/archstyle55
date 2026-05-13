
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .figures_aggregate import _as_float, _read_summary, _short

_EXTRA_COSTS: dict[str, dict] = {
    "dinov2_vitb14_linear": {
        "image_size": 224,
        "params_m": 86.6,
        "gflops": 17.5,
        "inference_ms": None,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path,
                        default=Path("runs_res/aggregate"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.aggregate_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_path = args.aggregate_dir / "compute_cost.json"
    if not cost_path.is_file():
        raise SystemExit(f"missing {cost_path} (run pipeline.reports.compute_cost first)")
    cost_rows = json.loads(cost_path.read_text(encoding="utf-8"))
    cost_by_model: dict[str, dict] = {r["model"]: r for r in cost_rows
                                      if "error" not in r}
    for k, v in _EXTRA_COSTS.items():
        cost_by_model.setdefault(k, {"model": k, **v})

    summary = _read_summary(args.aggregate_dir / "summary_table.csv")
    summary_by_model: dict[str, dict] = {}
    for r in summary:
        key = _short(r["model"])
        summary_by_model[key] = r

    rows = []
    for name, cost in cost_by_model.items():
        s = summary_by_model.get(name)
        if s is None:
            continue
        rows.append({
            "model": name,
            "image_size": cost.get("image_size"),
            "params_m": cost.get("params_m"),
            "gflops": cost.get("gflops"),
            "inference_ms_cpu": cost.get("inference_ms"),
            "accuracy": _as_float(s.get("accuracy")),
            "accuracy_ci_lo": _as_float(s.get("accuracy_ci_lo")),
            "accuracy_ci_hi": _as_float(s.get("accuracy_ci_hi")),
            "macro_f1": _as_float(s.get("macro_f1")),
        })

    rows.sort(key=lambda r: r["accuracy"], reverse=True)

    csv_path = out_dir / "compute_cost_table.csv"
    keys = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})

    md_path = out_dir / "compute_cost_table.md"
    md = ["| model | img | params, M | GFLOPs | acc | F1 |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['image_size']} "
            f"| {r['params_m']:.1f} | {(r['gflops'] or 0):.2f} "
            f"| {r['accuracy']:.4f} | {r['macro_f1']:.4f} |"
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")

    have_gflops = [r for r in rows if r.get("gflops") is not None]
    if have_gflops:
        fig, ax = plt.subplots(figsize=(9, 6))
        for r in have_gflops:
            size = max(40.0, 10.0 * float(r["params_m"]))
            ax.scatter(r["gflops"], r["accuracy"], s=size, alpha=0.55,
                       edgecolors="black", linewidths=0.5)
            ax.annotate(r["model"],
                        (r["gflops"], r["accuracy"]),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("GFLOPs (log scale)")
        ax.set_ylabel("test accuracy")
        ax.set_title("Compute cost vs quality (диаметр маркера = params)")
        ax.grid(True, alpha=0.3)
        ax.axhline(1 / 55, color="grey", linestyle=":", linewidth=1,
                   label="random (1/55)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        png = out_dir / "compute_cost_bubble.png"
        fig.savefig(png, dpi=200)
        plt.close(fig)
        print(f"wrote {png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
