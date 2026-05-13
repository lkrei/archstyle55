
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev


def run_seed(model: str, seed: int, run_dir: Path, extra: list[str]) -> None:
    cmd = [sys.executable, "-m", "pipeline.training.run",
           "--model", model, "--seed", str(seed), "--run-dir", str(run_dir), *extra]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def collect(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.is_file():
        return None
    return json.loads(metrics_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2025])
    parser.add_argument("--root", type=Path, required=True,
                        help="директория, куда пишутся run-папки")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-train", action="store_true",
                        help="не запускать обучение, только агрегировать существующие run'ы")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    if not args.no_train:
        for seed in args.seeds:
            run_dir = args.root / f"{args.model}_seed{seed}"
            run_seed(args.model, seed, run_dir, args.extra)
            subprocess.run([sys.executable, "-m", "pipeline.evaluation.evaluate",
                            "--run-dir", str(run_dir),
                            "--checkpoint", "last_ema.pt"], check=True)

    rows = []
    for seed in args.seeds:
        run_dir = args.root / f"{args.model}_seed{seed}"
        m = collect(run_dir)
        if m is not None:
            rows.append({"seed": seed, **m})

    summary: dict = {"model": args.model, "seeds": args.seeds, "runs": rows}
    if rows:
        accs = [r["accuracy"] for r in rows]
        f1s = [r["macro_f1"] for r in rows]
        bals = [r["balanced_accuracy"] for r in rows]
        summary.update({
            "accuracy_mean": mean(accs), "accuracy_std": pstdev(accs),
            "macro_f1_mean": mean(f1s), "macro_f1_std": pstdev(f1s),
            "balanced_acc_mean": mean(bals), "balanced_acc_std": pstdev(bals),
        })

    out = args.out or (args.root / "multi_seed.json")
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
