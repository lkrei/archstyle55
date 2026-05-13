from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.session import get_session
from ..ml.embed import embed_image
from ..ml.labels import display
from ..utils.images import load_pil

router = APIRouter()


@router.post("/similar")
async def similar(
    file: UploadFile = File(...),
    k: int = 10,
    session: AsyncSession = Depends(get_session),
):
    if k < 1 or k > 100:
        raise HTTPException(status_code=400, detail="k must be in [1, 100]")
    try:
        data = await file.read()
        img = load_pil(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    emb = embed_image(img)
    vec_literal = "[" + ",".join(f"{v:.6f}" for v in emb.vector) + "]"

    rows = (await session.execute(
        text(
            """
            SELECT i.id::text AS image_id,
                   i.style_label,
                   i.blob_url,
                   i.source,
                   1 - (e.vec <=> CAST(:q AS vector)) AS score
            FROM embeddings e
            JOIN images i ON i.id = e.image_id
            WHERE e.model = 'dinov2_vitb14'
            ORDER BY e.vec <=> CAST(:q AS vector)
            LIMIT :k
            """
        ),
        {"q": vec_literal, "k": k},
    )).mappings().all()

    return {
        "k": k,
        "matches": [
            {
                "image_id": r["image_id"],
                "style": display(r["style_label"]) if r["style_label"] else None,
                "score": float(r["score"]),
                "blob_url": r["blob_url"],
                "source": r["source"],
            }
            for r in rows
        ],
        "embedding_norm": emb.norm,
    }


@router.get("/atlas")
async def atlas(
    sample: int = 600,
    session: AsyncSession = Depends(get_session),
):
    if sample < 50 or sample > 5000:
        raise HTTPException(status_code=400, detail="sample must be in [50, 5000]")
    rows = (await session.execute(
        text(
            """
            SELECT i.style_label, e.vec::text AS vec, i.id::text AS image_id
            FROM embeddings e
            JOIN images i ON i.id = e.image_id
            WHERE e.model = 'dinov2_vitb14' AND i.style_label IS NOT NULL
            ORDER BY random()
            LIMIT :n
            """
        ),
        {"n": sample},
    )).mappings().all()
    return {
        "n": len(rows),
        "items": [
            {
                "image_id": r["image_id"],
                "style": display(r["style_label"]),
                "raw_style": r["style_label"],
                "vec": r["vec"],
            }
            for r in rows
        ],
    }
