from __future__ import annotations

import os

import httpx
import requests
from sqlalchemy import create_engine, text

from prefect import flow, get_run_logger, task

DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://backend:8000")
DATABASE_URL = os.environ.get(
    "DATABASE_URL_SYNC",
    "postgresql+psycopg://archstyle:archstyle@postgres:5432/archstyle",
)


@task
def list_missing(limit: int) -> list[dict]:
    eng = create_engine(DATABASE_URL, pool_pre_ping=True)
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT i.id::text AS id, i.blob_url, i.style_label
                FROM images i
                LEFT JOIN embeddings e
                  ON e.image_id = i.id AND e.model = 'dinov2_vitb14'
                WHERE i.blob_url IS NOT NULL AND e.image_id IS NULL
                ORDER BY i.created_at DESC
                LIMIT :n
                """
            ),
            {"n": limit},
        ).mappings().all()
    return [dict(r) for r in rows]


@task(retries=2, retry_delay_seconds=5)
def reembed(backend_url: str, item: dict) -> dict | None:
    url = item["blob_url"]
    if url is None:
        return None
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return None
        with httpx.Client(timeout=60.0) as c:
            files = {"file": ("img.jpg", r.content, "image/jpeg")}
            res = c.post(f"{backend_url}/search/similar", files=files, params={"k": 1})
            res.raise_for_status()
            embed = res.json()["embedding_norm"]
        return {"image_id": item["id"], "norm": embed}
    except Exception:
        return None


@flow(name="nightly_reembed")
def nightly_reembed(backend_url: str = DEFAULT_BACKEND, limit: int = 200) -> int:
    log = get_run_logger()
    missing = list_missing(limit)
    log.info(f"images without embedding: {len(missing)}")
    n_done = 0
    for item in missing:
        if reembed(backend_url, item):
            n_done += 1
    log.info(f"re-embedded: {n_done}")
    return n_done


if __name__ == "__main__":
    nightly_reembed.serve(name="nightly-reembed", cron="30 2 * * *")
