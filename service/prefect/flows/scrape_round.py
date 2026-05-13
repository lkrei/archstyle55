from __future__ import annotations

import os
import random

import httpx

from prefect import flow, get_run_logger, task

DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://backend:8000")
DEFAULT_QUERIES = [
    "Stalinist architecture facade Moscow",
    "Bauhaus Dessau facade",
    "Art Deco facade New York",
    "Brutalist concrete facade",
    "Naryshkin Baroque church facade",
    "Russian eclectic mansion facade",
    "Constructivist apartment Moscow facade",
    "Italianate villa facade",
    "Tudor Revival house facade",
    "Soviet modernist apartment facade",
]


@task(retries=2, retry_delay_seconds=10)
def fetch_classes(backend_url: str) -> list[dict]:
    with httpx.Client(timeout=30.0) as c:
        r = c.get(f"{backend_url}/meta/classes")
        r.raise_for_status()
        return r.json()


@task(retries=1)
def kick_off(backend_url: str, style: str, query: str, n_target: int) -> str:
    with httpx.Client(timeout=30.0) as c:
        r = c.post(
            f"{backend_url}/scrape/start",
            json={"style": style, "query": query, "n_target": n_target},
        )
        r.raise_for_status()
        return r.json()["job_id"]


@flow(name="daily_scrape_round")
def daily_scrape_round(
    backend_url: str = DEFAULT_BACKEND,
    n_styles: int = 5,
    n_per_style: int = 5,
    seed: int = 0,
) -> list[str]:
    log = get_run_logger()
    classes = fetch_classes(backend_url)
    chosen = random.Random(seed).sample(classes, k=min(n_styles, len(classes)))
    job_ids = []
    for cls in chosen:
        query = f"{cls['display_name']} facade"
        job_id = kick_off(backend_url, cls["name"], query, n_per_style)
        log.info(f"queued {job_id} for {cls['name']}")
        job_ids.append(job_id)
    return job_ids


if __name__ == "__main__":
    daily_scrape_round.serve(
        name="daily-scrape-round",
        cron="0 7 * * *",
    )
