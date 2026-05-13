"""Калибровка вероятностей: ECE, MCE и temperature scaling.

Temperature scaling по Guo et al., 2017: один скаляр T,
оптимизируем NLL на валидации, проверяем падение ECE на тесте.

Использование (из Python):

    from pipeline.evaluation.calibration import (
        ece_score, mce_score, fit_temperature, apply_temperature,
    )

    T = fit_temperature(val_logits, val_labels)
    cal_logits = apply_temperature(test_logits, T)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CalibrationReport:
    n_bins: int
    ece: float
    mce: float
    bin_acc: list
    bin_conf: list
    bin_count: list


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def calibration_report(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> CalibrationReport:
    probs = _softmax_np(logits)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_acc, bin_conf, bin_count = [], [], []
    ece = 0.0
    mce = 0.0
    n = len(labels)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-9)
        if not mask.any():
            bin_acc.append(0.0); bin_conf.append(0.0); bin_count.append(0)
            continue
        a = correct[mask].mean()
        c = conf[mask].mean()
        m = int(mask.sum())
        bin_acc.append(float(a))
        bin_conf.append(float(c))
        bin_count.append(m)
        gap = abs(a - c)
        ece += (m / n) * gap
        mce = max(mce, gap)

    return CalibrationReport(
        n_bins=n_bins, ece=float(ece), mce=float(mce),
        bin_acc=bin_acc, bin_conf=bin_conf, bin_count=bin_count,
    )


def ece_score(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    return calibration_report(logits, labels, n_bins=n_bins).ece


def mce_score(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    return calibration_report(logits, labels, n_bins=n_bins).mce


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 200,
    lr: float = 0.05,
) -> float:
    z = torch.from_numpy(logits).double()
    y = torch.from_numpy(labels).long()
    log_T = torch.zeros(1, requires_grad=True, dtype=torch.float64)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        T = log_T.exp()
        scaled = z / T
        loss = F.cross_entropy(scaled, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.detach().exp().item())


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return logits / max(1e-6, T)


def write_calibration_report(
    out_path: Path,
    label: str,
    pre: CalibrationReport,
    post: CalibrationReport | None,
    T: float | None,
) -> None:
    payload = {
        "label": label,
        "pre": pre.__dict__,
        "post": post.__dict__ if post else None,
        "temperature": T,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
