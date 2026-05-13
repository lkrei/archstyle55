"""Ансамблирование вероятностей.

"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import _softmax_np


@dataclass
class EnsembleResult:
    weights: np.ndarray
    val_macro_f1: float
    test_macro_f1: float | None = None


def average_probs(prob_matrices: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    stacked = np.stack(prob_matrices, axis=0)
    if weights is None:
        return stacked.mean(axis=0)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(1e-9, weights.sum())
    return (stacked * weights[:, None, None]).sum(axis=0)


def logits_list_to_probs(logits_list: list[np.ndarray]) -> list[np.ndarray]:
    return [_softmax_np(z) for z in logits_list]


def weighted_simplex_search(
    val_logits_list: list[np.ndarray],
    val_labels: np.ndarray,
    step: float = 0.1,
    metric_fn=None,
) -> EnsembleResult:
    """Грубый перебор весов по сетке шагом step (без замены, на симплексе)."""
    from sklearn.metrics import f1_score

    if metric_fn is None:
        def metric_fn(probs, labels):
            preds = probs.argmax(axis=1)
            return float(f1_score(labels, preds, average="macro", zero_division=0))

    probs = logits_list_to_probs(val_logits_list)
    n = len(probs)
    best = None

    def grid(remaining: float, depth: int):
        if depth == n - 1:
            yield (round(remaining, 6),)
            return
        v = 0.0
        while v <= remaining + 1e-9:
            for tail in grid(remaining - v, depth + 1):
                yield (round(v, 6), *tail)
            v = round(v + step, 6)

    for w in grid(1.0, 0):
        weights = np.array(w, dtype=np.float64)
        avg = average_probs(probs, weights)
        score = metric_fn(avg, val_labels)
        if best is None or score > best.val_macro_f1:
            best = EnsembleResult(weights=weights, val_macro_f1=score)
    return best
