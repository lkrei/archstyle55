"""Тестовая оценка модели. Сохраняет логиты, метрики и confusion matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from ..config import SPLITS_DIR
from ..data.dataset import ArchitecturalStyleDataset, DatasetConfig
from ..data.transforms import build_eval_transform
from ..models.factory import DEFAULT_HPARAMS, build_model


@torch.no_grad()
def collect_logits(model, loader, device: str):
    model.eval()
    all_logits = []
    all_labels = []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs).detach().cpu()
        all_logits.append(logits)
        all_labels.append(labels)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    preds = logits.argmax(axis=1)
    accuracy = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(labels, preds))
    report = classification_report(
        labels, preds, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--model", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--splits", type=Path, default=SPLITS_DIR / "data_splits.json")
    parser.add_argument("--classes", type=Path, default=SPLITS_DIR / "idx_to_class.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg_path = args.run_dir / "config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    model_name = args.model or config.get("model")
    if model_name is None:
        raise SystemExit("model name not provided and config.json missing")
    image_size = args.image_size or config.get("image_size") or DEFAULT_HPARAMS[model_name].image_size

    idx_to_class = json.loads(args.classes.read_text())
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[str(i)] for i in range(num_classes)]
    splits = json.loads(args.splits.read_text())

    val_tf = build_eval_transform(image_size=image_size)
    test_ds = ArchitecturalStyleDataset(splits["test"], transform_full=val_tf,
                                        config=DatasetConfig(mode="full"))
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=num_classes).to(device)
    state = torch.load(args.run_dir / args.checkpoint, map_location=device)
    model.load_state_dict(state["state"] if isinstance(state, dict) and "state" in state else state)

    logits, labels = collect_logits(model, loader, device)
    metrics = metrics_from_logits(logits, labels, class_names)

    out_logits = args.run_dir / "test_logits.npz"
    np.savez_compressed(out_logits, logits=logits, labels=labels,
                        class_names=np.array(class_names))
    cm = confusion_matrix(labels, logits.argmax(axis=1), labels=list(range(num_classes)))
    np.save(args.run_dir / "test_confusion.npy", cm)
    (args.run_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (args.run_dir / "test_report.json").write_text(
        json.dumps(metrics["per_class"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"acc {metrics['accuracy']:.4f} | "
        f"macro_f1 {metrics['macro_f1']:.4f} | "
        f"bal_acc {metrics['balanced_accuracy']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
