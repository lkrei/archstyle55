from __future__ import annotations

import asyncio
import json
import secrets

import redis.asyncio as aredis
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from rq import Queue
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_settings
from ..db.session import get_session
from ..tasks.scrape import run_scrape_job

router = APIRouter()
settings = get_settings()


def _queue() -> Queue:
    import redis as redis_sync
    conn = redis_sync.Redis.from_url(settings.redis_url)
    return Queue("archstyle", connection=conn)


class ScrapeIn(BaseModel):
    style: str
    query: str
    n_target: int = Field(default=20, ge=1, le=100)


@router.post("/start")
async def start(payload: ScrapeIn, session: AsyncSession = Depends(get_session)):
    job_id = secrets.token_hex(8)
    await session.execute(
        text(
            """
            INSERT INTO scrape_jobs (id, style, query, n_target, status)
            VALUES (:id, :style, :query, :n, 'queued')
            """
        ),
        {"id": job_id, "style": payload.style, "query": payload.query, "n": payload.n_target},
    )
    await session.commit()

    q = _queue()
    q.enqueue(
        run_scrape_job,
        job_id, payload.style, payload.query, payload.n_target,
        job_id=f"scrape-{job_id}",
        result_ttl=86400,
        failure_ttl=86400,
        job_timeout=900,
    )
    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}")
async def status(job_id: str, session: AsyncSession = Depends(get_session)):
    row = (await session.execute(
        text("SELECT id, style, query, n_target, n_done, status, log, started_at, finished_at "
             "FROM scrape_jobs WHERE id=:i"),
        {"i": job_id},
    )).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {**row}


@router.get("")
async def list_jobs(limit: int = 20, session: AsyncSession = Depends(get_session)):
    rows = (await session.execute(
        text("SELECT id, style, query, n_target, n_done, status, started_at, finished_at "
             "FROM scrape_jobs ORDER BY started_at DESC LIMIT :n"),
        {"n": limit},
    )).mappings().all()
    return list(rows)


@router.websocket("/ws/{job_id}")
async def progress_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    r = aredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = r.pubsub()
    await pubsub.subscribe(f"scrape:{job_id}")

    history = await r.lrange(f"scrape:log:{job_id}", 0, -1)
    for entry in reversed(history):
        await websocket.send_text(entry)

    try:
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30)
            if msg is None:
                await websocket.send_text(json.dumps({"event": "ping"}))
                continue
            payload = msg["data"]
            await websocket.send_text(payload if isinstance(payload, str) else payload.decode())
            try:
                data = json.loads(payload)
                if data.get("event") in {"done", "error"}:
                    await asyncio.sleep(0.5)
                    break
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"scrape:{job_id}")
        await pubsub.close()
        await r.aclose()
