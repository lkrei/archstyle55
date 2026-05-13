
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from ..models.factory import param_groups
from .ema import ModelEMA
from .logger import EpochMetrics, RunLogger
from .mixers import BatchMixer, MixerConfig
from .repro import fix_seeds, write_repro
from .scheduler import cosine_with_warmup


@dataclass
class TrainConfig:
    epochs: int = 30
    base_lr: float = 1e-3
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mixer_apply_p: float = 0.5
    mixer_off_last_epochs: int = 5
    early_stop_patience: int = 8
    seed: int = 42


@dataclass
class TrainOutputs:
    best_val_macro_f1: float
    best_epoch: int
    best_checkpoint: str | None
    last_checkpoint: str | None
    last_ema_checkpoint: str | None
    history: list = field(default_factory=list)


def _two_group_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    groups = param_groups(model, cfg.base_lr, cfg.backbone_lr_mult, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def _soft_ce(logits: torch.Tensor, soft_target: torch.Tensor,
             label_smoothing: float = 0.0) -> torch.Tensor:
    if label_smoothing > 0:
        n = soft_target.size(-1)
        soft_target = soft_target * (1.0 - label_smoothing) + label_smoothing / n
    log_p = torch.log_softmax(logits, dim=-1)
    return -(soft_target * log_p).sum(dim=-1).mean()


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float, float, float]:
    model.eval()
    losses, all_logits, all_labels = [], [], []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        losses.append(nn.functional.cross_entropy(logits, labels).item())
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(dim=1).numpy()
    truths = labels.numpy()
    acc = float((preds == truths).mean())
    macro_f1 = float(f1_score(truths, preds, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(truths, preds))
    return float(sum(losses) / max(1, len(losses))), acc, macro_f1, bal_acc


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum())
    return float(total ** 0.5)


def train_run(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    run_dir: Path,
    cfg: TrainConfig,
    logger: RunLogger,
    device: str = "cuda",
) -> TrainOutputs:
    fix_seeds(cfg.seed)
    write_repro(run_dir, snapshot=_repro(cfg))

    model = model.to(device)
    optimizer = _two_group_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = cosine_with_warmup(optimizer, total_steps, warmup_steps)

    mixer = BatchMixer(MixerConfig(
        mixup_alpha=cfg.mixup_alpha,
        cutmix_alpha=cfg.cutmix_alpha,
        apply_p=cfg.mixer_apply_p,
        num_classes=num_classes,
    ))
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device == "cuda")

    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    last_ema_path = run_dir / "last_ema.pt"
    history = []

    for epoch in range(1, cfg.epochs + 1):
        if cfg.mixer_off_last_epochs and epoch > cfg.epochs - cfg.mixer_off_last_epochs:
            mixer.set_active(False)

        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_n = 0
        last_grad = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            inputs, soft_target = mixer(inputs, labels)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                    enabled=cfg.use_amp and device == "cuda"):
                logits = model(inputs)
                loss = _soft_ce(logits, soft_target, cfg.label_smoothing)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            last_grad = _grad_norm(model)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            if ema is not None:
                ema.update(model)

            running_loss += float(loss.detach()) * inputs.size(0)
            running_n += inputs.size(0)

        train_loss = running_loss / max(1, running_n)
        eval_module = ema.module if ema is not None else model
        val_loss, val_acc, val_f1, val_bal = _evaluate(eval_module, val_loader, device)

        lrs = [g["lr"] for g in optimizer.param_groups]
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            val_macro_f1=val_f1,
            val_balanced_acc=val_bal,
            lr_head=lrs[-1],
            lr_backbone=lrs[0] if len(lrs) > 1 else lrs[-1],
            grad_norm=last_grad,
            epoch_time_s=time.time() - t0,
        )
        logger.log_epoch(metrics)
        history.append(asdict(metrics))

        improved = val_f1 > best_f1 + 1e-4
        if improved:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save({"state": model.state_dict(), "config": asdict(cfg)}, best_path)
            no_improve = 0
        else:
            no_improve += 1

        torch.save({"state": model.state_dict(), "config": asdict(cfg)}, last_path)
        if ema is not None:
            torch.save({"state": ema.state_dict(), "config": asdict(cfg)}, last_ema_path)

        if no_improve >= cfg.early_stop_patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return TrainOutputs(
        best_val_macro_f1=best_f1,
        best_epoch=best_epoch,
        best_checkpoint=str(best_path) if best_path.is_file() else None,
        last_checkpoint=str(last_path) if last_path.is_file() else None,
        last_ema_checkpoint=str(last_ema_path) if last_ema_path.is_file() else None,
        history=history,
    )


def _repro(cfg: TrainConfig) -> dict:
    return {"train_config": asdict(cfg)}
