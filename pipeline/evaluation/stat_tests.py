"""Бутстрэп CI и парный McNemar test.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, lgamma, log

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class BootstrapCI:
    estimate: float
    ci_lo: float
    ci_hi: float
    n_resamples: int
    alpha: float


def bootstrap_metric(
    truths: np.ndarray,
    preds: np.ndarray,
    metric: str = "accuracy",
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    rng = np.random.default_rng(seed)
    n = len(truths)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        t = truths[idx]
        p = preds[idx]
        if metric == "accuracy":
            samples[i] = float((t == p).mean())
        elif metric == "macro_f1":
            samples[i] = float(f1_score(t, p, average="macro", zero_division=0))
        else:
            raise ValueError(f"unknown metric: {metric}")
    estimate = float(samples.mean())
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1.0 - alpha / 2))
    return BootstrapCI(estimate=estimate, ci_lo=lo, ci_hi=hi,
                       n_resamples=n_resamples, alpha=alpha)


def _log_binom(n: int, k: int) -> float:
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _binom_two_sided_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    log_p_n = -n * log(2.0)
    log_terms = [_log_binom(n, i) + log_p_n for i in range(k + 1)]
    m = max(log_terms)
    log_one_sided = m + log(sum(exp(t - m) for t in log_terms))
    p_one = exp(log_one_sided)
    return min(1.0, 2.0 * p_one)


def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, truths: np.ndarray) -> dict:
    correct_a = (preds_a == truths)
    correct_b = (preds_b == truths)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c >= 25:
        chi = (abs(b - c) - 1) ** 2 / float(b + c)
        p = float(np.exp(-chi / 2))  # rough; for thesis use the exact p below in any case
    else:
        p = _binom_two_sided_pvalue(b, c)
    return {"b": b, "c": c, "n": b + c, "p_value": float(_binom_two_sided_pvalue(b, c)),
            "chi_approx_p": float(p)}
