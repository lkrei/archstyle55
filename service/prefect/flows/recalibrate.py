from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import create_engine, text

from prefect import flow, get_run_logger, task

DATABASE_URL = os.environ.get(
    "DATABASE_URL_SYNC",
    "postgresql+psycopg://archstyle:archstyle@postgres:5432/archstyle",
)


@task
def collect_recent(window_days: int) -> list[dict]:
    eng = create_engine(DATABASE_URL, pool_pre_ping=True)
    horizon = datetime.now(tz=UTC) - timedelta(days=window_days)
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT p.top1_prob AS prob, f.is_correct AS is_correct
                FROM predictions p
                JOIN feedback f ON f.prediction_id = p.id
                WHERE f.is_correct IS NOT NULL AND p.created_at > :h
                """
            ),
            {"h": horizon},
        ).mappings().all()
    return [dict(r) for r in rows]


@task
def estimate_temperature(items: list[dict]) -> float:
    if len(items) < 30:
        return 1.0
    probs = np.array([r["prob"] for r in items])
    is_correct = np.array([1.0 if r["is_correct"] else 0.0 for r in items])
    eps = 1e-6
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))

    best_t = 1.0
    best_nll = float("inf")
    for t in np.linspace(0.5, 3.0, 26):
        scaled = logits / t
        cal = 1.0 / (1.0 + np.exp(-scaled))
        cal = np.clip(cal, eps, 1 - eps)
        nll = -(is_correct * np.log(cal) + (1 - is_correct) * np.log(1 - cal)).mean()
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


@flow(name="weekly_recalibrate")
def weekly_recalibrate(window_days: int = 14) -> dict:
    log = get_run_logger()
    items = collect_recent(window_days)
    log.info(f"feedback samples: {len(items)}")
    t = estimate_temperature(items)
    log.info(f"temperature estimate: {t:.3f}")
    return {"n": len(items), "temperature": t}


if __name__ == "__main__":
    weekly_recalibrate.serve(name="weekly-recalibrate", cron="0 4 * * 1")
