from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root() -> dict:
    return {"endpoints": ["/scrape/{job_id}/progress", "/predict/stream"]}
