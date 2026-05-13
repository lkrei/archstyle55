
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import RUNS_DIR, SPLITS_DIR
from ..data.dataset import ArchitecturalStyleDataset, DatasetConfig
from ..data.transforms import AugConfig, build_eval_transform, build_train_transform
from ..models.factory import DEFAULT_HPARAMS, build_model
from .logger import RunLogger
from .loop import TrainConfig, train_run
from .repro import build_repro_snapshot, write_repro


def _make_loaders(splits, num_classes: int, image_size: int, batch_size: int,
                  num_workers: int):
    train_tf = build_train_transform(AugConfig(image_size=image_size))
    val_tf = build_eval_transform(image_size=image_size)
    train_ds = ArchitecturalStyleDataset(splits["train"], transform_full=train_tf,
                                         config=DatasetConfig(mode="full"))
    val_ds = ArchitecturalStyleDataset(splits["val"], transform_full=val_tf,
                                       config=DatasetConfig(mode="full"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(DEFAULT_HPARAMS))
    parser.add_argument("--splits", type=Path, default=SPLITS_DIR / "data_splits.json")
    parser.add_argument("--classes", type=Path, default=SPLITS_DIR / "idx_to_class.json")
    parser.add_argument("--manifest", type=Path, default=SPLITS_DIR / "manifest.csv")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb-project", default="archstyle-vkr")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--base-lr", type=float, default=None,
                        help="override base_lr from ModelSpec")
    parser.add_argument("--backbone-lr-mult", type=float, default=None,
                        help="override backbone_lr_mult from ModelSpec")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="override weight_decay from ModelSpec")
    parser.add_argument("--label-smoothing", type=float, default=None,
                        help="override label_smoothing in TrainConfig")
    parser.add_argument("--mixup-alpha", type=float, default=None,
                        help="override mixup_alpha in TrainConfig (0 disables MixUp)")
    parser.add_argument("--cutmix-alpha", type=float, default=None,
                        help="override cutmix_alpha in TrainConfig (0 disables CutMix)")
    parser.add_argument("--mixer-apply-p", type=float, default=None,
                        help="probability of applying MixUp/CutMix per batch")
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="override EMA decay (e.g. 0.999 instead of 0.9999)")
    parser.add_argument("--warmup-ratio", type=float, default=None,
                        help="warmup fraction of total steps")
    args = parser.parse_args()

    spec = DEFAULT_HPARAMS[args.model]
    epochs = args.epochs or spec.epochs
    image_size = args.image_size or spec.image_size
    batch_size = args.batch_size or spec.batch_size
    base_lr = args.base_lr if args.base_lr is not None else spec.base_lr
    backbone_lr_mult = args.backbone_lr_mult if args.backbone_lr_mult is not None else spec.backbone_lr_mult
    weight_decay = args.weight_decay if args.weight_decay is not None else spec.weight_decay

    splits = json.loads(args.splits.read_text())
    idx_to_class = json.loads(args.classes.read_text())
    num_classes = len(idx_to_class)

    run_dir = args.run_dir or (RUNS_DIR / f"{args.model}_seed{args.seed}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = _make_loaders(
        splits, num_classes, image_size, batch_size, args.num_workers,
    )

    model = build_model(args.model, num_classes=num_classes)

    cfg_kwargs = {
        "epochs": epochs,
        "base_lr": base_lr,
        "backbone_lr_mult": backbone_lr_mult,
        "weight_decay": weight_decay,
        "seed": args.seed,
        "use_amp": not args.no_amp,
        "use_ema": not args.no_ema,
    }
    if args.label_smoothing is not None:
        cfg_kwargs["label_smoothing"] = args.label_smoothing
    if args.mixup_alpha is not None:
        cfg_kwargs["mixup_alpha"] = args.mixup_alpha
    if args.cutmix_alpha is not None:
        cfg_kwargs["cutmix_alpha"] = args.cutmix_alpha
    if args.mixer_apply_p is not None:
        cfg_kwargs["mixer_apply_p"] = args.mixer_apply_p
    if args.ema_decay is not None:
        cfg_kwargs["ema_decay"] = args.ema_decay
    if args.warmup_ratio is not None:
        cfg_kwargs["warmup_ratio"] = args.warmup_ratio
    cfg = TrainConfig(**cfg_kwargs)

    config_payload = {
        "model": args.model,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_classes": num_classes,
        "spec": asdict(spec),
        "train": asdict(cfg),
    }
    write_repro(run_dir, build_repro_snapshot(manifest_csv=args.manifest,
                                              extra=config_payload))

    logger = RunLogger(
        run_dir=run_dir,
        project=args.wandb_project,
        run_name=f"{args.model}_seed{args.seed}",
        config=config_payload,
        enable_wandb=not args.no_wandb,
    )

    outputs = train_run(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        run_dir=run_dir,
        cfg=cfg,
        logger=logger,
        device=device,
    )

    summary = {
        "model": args.model,
        "best_val_macro_f1": outputs.best_val_macro_f1,
        "best_epoch": outputs.best_epoch,
        "best_checkpoint": outputs.best_checkpoint,
        "last_checkpoint": outputs.last_checkpoint,
        "last_ema_checkpoint": outputs.last_ema_checkpoint,
    }
    logger.close(summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
