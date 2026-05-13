from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from ..ml.segment import segment_image
from ..utils.images import load_pil

router = APIRouter()


@router.post("")
async def segment_ep(file: UploadFile = File(...), output: str = "json"):
    try:
        data = await file.read()
        img = load_pil(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    res = segment_image(img)
    if output == "overlay":
        return Response(content=res.overlay_png, media_type="image/png")
    if output == "mask":
        return Response(content=res.mask_png, media_type="image/png")
    return {
        "attributes": res.attributes,
        "shape": list(res.mask.shape),
        "categories": [
            "wall", "window", "door", "roof", "balcony",
            "column", "sky", "vegetation", "ground", "other",
        ],
    }


@router.post("/color")
async def palette_ep(file: UploadFile = File(...)):
    from ..ml.color import palette
    try:
        data = await file.read()
        img = load_pil(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    items = palette(img, k=5)
    return [{"hex": e.rgb_hex, "rgb": list(e.rgb), "share": e.share} for e in items]
