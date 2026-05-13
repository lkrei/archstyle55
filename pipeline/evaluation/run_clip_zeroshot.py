"""CLI wrapper around the CLIP / SigLIP zero-shot baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .clip_zeroshot import zero_shot_predict


def _resolve_paths(samples: list[dict], images_dir: Path | None) -> list[dict]:
    """Если ``samples[i]["path"]`` относительный, склеить его с ``images_dir``."""
    if images_dir is None:
        return samples
    resolved = []
    for s in samples:
        p = Path(s["path"])
        if not p.is_absolute():
            p = images_dir / p
        resolved.append({**s, "path": str(p)})
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, required=True,
                        help="data_splits.json with train/val/test lists")
    parser.add_argument("--classes", type=Path, required=True,
                        help="idx_to_class.json")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="prefix to prepend to relative sample paths")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16",
                        help="HF model id, e.g. openai/clip-vit-base-patch16 "
                             "or google/siglip-base-patch16-224")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    idx_to_class = json.loads(args.classes.read_text())
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[str(i)] for i in range(num_classes)]

    splits = json.loads(args.splits.read_text())
    test = _resolve_paths(splits["test"], args.images_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    res = zero_shot_predict(
        samples=test,
        class_names=class_names,
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
    )

    logits = res.logits
    labels = res.labels
    preds = logits.argmax(axis=1)

    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    report = classification_report(
        labels, preds, target_names=class_names,
        output_dict=True, zero_division=0,
    )

    np.savez_compressed(args.out_dir / "test_logits.npz",
                        logits=logits, labels=labels,
                        class_names=np.array(class_names))
    np.save(args.out_dir / "test_confusion.npy", cm)
    (args.out_dir / "test_metrics.json").write_text(
        json.dumps({
            "accuracy": res.accuracy,
            "top5_accuracy": res.top5_accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": bal_acc,
            "model_id": args.model_id,
            "num_test": int(len(labels)),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out_dir / "test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[zero-shot {args.model_id}] "
        f"acc {res.accuracy:.4f} | top5 {res.top5_accuracy:.4f} | "
        f"macro_f1 {macro_f1:.4f} | bal_acc {bal_acc:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
