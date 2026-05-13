from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text

from prefect import flow, get_run_logger, task

DATABASE_URL = os.environ.get(
    "DATABASE_URL_SYNC",
    "postgresql+psycopg://archstyle:archstyle@postgres:5432/archstyle",
)
ARTIFACT_DIR = Path(os.environ.get("DRIFT_OUT", "/runs_res/aggregate/drift"))


@task
def fetch_predictions(days: int) -> list[str]:
    eng = create_engine(DATABASE_URL, pool_pre_ping=True)
    horizon = datetime.now(tz=UTC) - timedelta(days=days)
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT top1_class FROM predictions
                WHERE created_at > :h
                """
            ),
            {"h": horizon},
        ).all()
    return [r[0] for r in rows]


@task
def fetch_baseline() -> Counter:
    eng = create_engine(DATABASE_URL, pool_pre_ping=True)
    with eng.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT i.style_label, COUNT(*) AS c
                FROM images i WHERE i.source LIKE 'split:test'
                GROUP BY i.style_label
            """)
        ).all()
    return Counter({r[0]: int(r[1]) for r in rows if r[0]})


@task
def kl(a: Counter, b: Counter, smoothing: float = 1.0) -> dict:
    keys = sorted(set(a) | set(b))
    pa = np.array([a.get(k, 0) + smoothing for k in keys], dtype=np.float64)
    pb = np.array([b.get(k, 0) + smoothing for k in keys], dtype=np.float64)
    pa /= pa.sum()
    pb /= pb.sum()
    kl_ab = float((pa * np.log(pa / pb)).sum())
    js = 0.5 * float((pa * np.log(pa / (0.5 * (pa + pb)))).sum()) + \
         0.5 * float((pb * np.log(pb / (0.5 * (pa + pb)))).sum())
    return {"kl_divergence": kl_ab, "js_divergence": js, "n_classes": len(keys)}


@flow(name="drift_report")
def drift_report(window_days: int = 7) -> dict:
    log = get_run_logger()
    preds = fetch_predictions(window_days)
    if not preds:
        log.warning("no recent predictions")
        return {"status": "empty"}
    pred_dist = Counter(preds)
    base_dist = fetch_baseline()
    metrics = kl(pred_dist, base_dist)
    out = {
        "window_days": window_days,
        "n_predictions": len(preds),
        "metrics": metrics,
        "top_predicted": pred_dist.most_common(10),
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = ARTIFACT_DIR / f"drift_{datetime.now(tz=UTC).strftime('%Y%m%d')}.json"
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info(f"wrote {out_file}")
    return out


if __name__ == "__main__":
    drift_report.serve(name="drift-report", cron="0 6 * * *")
