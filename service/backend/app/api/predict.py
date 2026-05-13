from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_settings
from ..db.models import ImageRow, PredictionRow
from ..db.session import get_session
from ..ml.classify import (
    ClassifyResult,
    ensemble_models,
    list_runtime_models,
    predict,
    predict_all_real,
    predict_ensemble,
)
from ..ml.hybrid import predict_hybrid
from ..ml.zeroshot import predict_zeroshot
from ..schemas.predict import HybridResponse, PredictionResponse, ZeroShotResponse
from ..utils.images import load_pil, sha256_hex

router = APIRouter()
settings = get_settings()


async def _persist(session: AsyncSession, *, sha: str, width: int, height: int,
                   res: ClassifyResult, source: str = "upload") -> tuple[ImageRow, PredictionRow]:
    stmt = (
        pg_insert(ImageRow)
        .values(id=uuid4(), sha256=sha, source=source, width=width, height=height)
        .on_conflict_do_nothing(index_elements=["sha256"])
    )
    await session.execute(stmt)
    img_obj = (await session.execute(
        select(ImageRow).where(ImageRow.sha256 == sha)
    )).scalar_one()

    pred = PredictionRow(
        id=uuid4(),
        image_id=img_obj.id,
        model=res.model,
        top1_class=res.top1_class,
        top1_prob=res.top1_prob,
        top5=res.top5,
        latency_ms=res.latency_ms,
    )
    session.add(pred)
    await session.flush()
    return img_obj, pred


def _to_resp(image_id, prediction_id, res: ClassifyResult, cache: bool = False) -> dict:
    return {
        "prediction_id": str(prediction_id),
        "image_id": str(image_id),
        "model": res.model,
        "top1_class": res.top1_class,
        "top1_prob": res.top1_prob,
        "top5": res.top5,
        "latency_ms": res.latency_ms,
        "cache": cache,
    }


@router.get("")
async def info() -> dict:
    return {
        "real_models": list_runtime_models(),
        "ensemble_components": ensemble_models(),
        "endpoints": ["/single", "/ensemble", "/hybrid", "/zeroshot", "/all"],
    }


@router.post("/single", response_model=PredictionResponse)
async def predict_single(
    request: Request,
    model: str = "efficientnet_v2_s",
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    try:
        data = await file.read()
        img = load_pil(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if model not in list_runtime_models():
        raise HTTPException(status_code=400, detail=f"unknown model: {model}")

    sha = sha256_hex(data)
    res = predict(model, img)
    img_row, pred_row = await _persist(
        session, sha=sha, width=img.width, height=img.height, res=res,
    )
    await session.commit()
    return _to_resp(img_row.id, pred_row.id, res)


@router.post("/ensemble", response_model=PredictionResponse)
async def predict_ensemble_ep(
    file: UploadFile = File(...),
    mode: str = "uniform",
    session: AsyncSession = Depends(get_session),
):
    if mode not in {"uniform", "weighted"}:
        raise HTTPException(status_code=400, detail="mode must be uniform|weighted")
    try:
        data = await file.read()
        img = load_pil(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    sha = sha256_hex(data)
    res = predict_ensemble(img, mode=mode)
    img_row, pred_row = await _persist(
        session, sha=sha, width=img.width, height=img.height, res=res,
    )
    await session.commit()
    return _to_resp(img_row.id, pred_row.id, res)


@router.post("/hybrid", response_model=HybridResponse)
async def predict_hybrid_ep(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    try:
        data = await file.read()
        img = load_pil(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    sha = sha256_hex(data)
    try:
        res = predict_hybrid(img)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    proxy = ClassifyResult(
        model="hybrid_dinov2_segformer_histgbm",
        top1_class=res.top1_class,
        top1_prob=res.top1_prob,
        top5=res.top5,
        latency_ms=res.latency_ms,
        logits=[],
    )
    img_row, pred_row = await _persist(
        session, sha=sha, width=img.width, height=img.height, res=proxy,
    )
    await session.commit()
    return {
        **_to_resp(img_row.id, pred_row.id, proxy),
        "attributes": res.attributes,
        "embedding_norm": res.embedding_norm,
    }


@router.post("/zeroshot", response_model=ZeroShotResponse)
async def predict_zeroshot_ep(
    file: UploadFile = File(...),
    prompt: str = "a photograph of {}",
    session: AsyncSession = Depends(get_session),
):
    try:
        data = await file.read()
        img = load_pil(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    sha = sha256_hex(data)
    res = predict_zeroshot(img, prompt_template=prompt)
    proxy = ClassifyResult(
        model="clip_zeroshot",
        top1_class=res.top1_class,
        top1_prob=res.top1_prob,
        top5=res.top5,
        latency_ms=res.latency_ms,
        logits=[],
    )
    img_row, pred_row = await _persist(
        session, sha=sha, width=img.width, height=img.height, res=proxy,
    )
    await session.commit()
    return {**_to_resp(img_row.id, pred_row.id, proxy), "prompt": res.prompt}


@router.post("/all")
async def predict_all_ep(
    file: UploadFile = File(...),
):
    try:
        data = await file.read()
        img = load_pil(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    out = {}
    for r in predict_all_real(img):
        out[r.model] = {
            "top1": r.top1_class,
            "prob": r.top1_prob,
            "latency_ms": r.latency_ms,
            "top5": r.top5,
        }
    out["ensemble_top3_uniform"] = (
        lambda r: {"top1": r.top1_class, "prob": r.top1_prob, "latency_ms": r.latency_ms, "top5": r.top5}
    )(predict_ensemble(img, mode="uniform"))
    out["ensemble_top3_weighted"] = (
        lambda r: {"top1": r.top1_class, "prob": r.top1_prob, "latency_ms": r.latency_ms, "top5": r.top5}
    )(predict_ensemble(img, mode="weighted"))
    return out
