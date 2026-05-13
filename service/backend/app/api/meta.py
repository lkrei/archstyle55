from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import ClassRow, ModelMetaRow
from ..db.session import get_session

router = APIRouter()


@router.get("/classes")
async def list_classes(session: AsyncSession = Depends(get_session)) -> list[dict]:
    rows = (await session.execute(select(ClassRow).order_by(ClassRow.idx))).scalars().all()
    return [
        {
            "idx": r.idx,
            "name": r.name,
            "display_name": r.display_name,
            "description": r.description,
            "signature": r.signature,
            "examples_paths": r.examples_paths,
        }
        for r in rows
    ]


@router.get("/models")
async def list_models(session: AsyncSession = Depends(get_session)) -> list[dict]:
    rows = (
        await session.execute(select(ModelMetaRow).order_by(ModelMetaRow.accuracy.desc().nullslast()))
    ).scalars().all()
    return [
        {
            "name": r.name,
            "family": r.family,
            "params_m": r.params_m,
            "gflops": r.gflops,
            "accuracy": r.accuracy,
            "macro_f1": r.macro_f1,
            "bal_acc": r.bal_acc,
            "inference_ms": r.inference_ms,
            "image_size": r.image_size,
            "hf_repo": r.hf_repo,
        }
        for r in rows
    ]


@router.get("/leaderboard")
async def leaderboard(session: AsyncSession = Depends(get_session)) -> list[dict]:
    return await list_models(session)
