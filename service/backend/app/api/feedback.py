from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import FeedbackRow, PredictionRow
from ..db.session import get_session

router = APIRouter()


class FeedbackIn(BaseModel):
    prediction_id: UUID
    is_correct: bool | None = None
    user_label: str | None = None
    comment: str | None = None


@router.post("")
async def post_feedback(payload: FeedbackIn, session: AsyncSession = Depends(get_session)) -> dict:
    pred = (
        await session.execute(select(PredictionRow).where(PredictionRow.id == payload.prediction_id))
    ).scalar_one_or_none()
    if pred is None:
        raise HTTPException(status_code=404, detail="prediction not found")
    row = FeedbackRow(
        prediction_id=payload.prediction_id,
        is_correct=payload.is_correct,
        user_label=payload.user_label,
        comment=payload.comment,
    )
    session.add(row)
    await session.commit()
    return {"ok": True, "feedback_id": row.id}


@router.get("/stats")
async def stats(session: AsyncSession = Depends(get_session)) -> dict:
    total = (await session.execute(select(FeedbackRow))).scalars().all()
    n = len(total)
    n_correct = sum(1 for r in total if r.is_correct is True)
    n_wrong = sum(1 for r in total if r.is_correct is False)
    return {
        "n_total": n,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "accuracy_on_user_uploads": (n_correct / max(1, n_correct + n_wrong)),
    }
