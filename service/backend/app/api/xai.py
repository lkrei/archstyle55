from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from ..ml.xai_cnn import gradcam_pp, supports_cnn
from ..ml.xai_vit import attention_rollout_dinov2, supports_transformer
from ..utils.images import load_pil

router = APIRouter()


@router.post("/cnn")
async def cnn_xai(
    model: str = "efficientnet_v2_s",
    target_class: int | None = None,
    file: UploadFile = File(...),
    output: str = "overlay",
):
    if not supports_cnn(model):
        raise HTTPException(status_code=400, detail=f"{model} is not a supported CNN")
    try:
        data = await file.read()
        img = load_pil(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    res = gradcam_pp(model, img, target_class=target_class)
    if output == "overlay":
        return Response(content=res.overlay_png, media_type="image/png")
    return {
        "method": res.method,
        "target_class": res.target_class,
        "target_class_name": res.target_class_name,
        "latency_ms": res.latency_ms,
    }


@router.post("/transformer")
async def transformer_xai(
    model: str = "dinov2_vitb14_linear",
    file: UploadFile = File(...),
    output: str = "overlay",
):
    if not supports_transformer(model):
        raise HTTPException(status_code=400, detail=f"{model} is not a supported ViT")
    try:
        data = await file.read()
        img = load_pil(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    res = attention_rollout_dinov2(img)
    if output == "overlay":
        return Response(content=res.overlay_png, media_type="image/png")
    return {
        "method": res.method,
        "target_class": res.target_class,
        "target_class_name": res.target_class_name,
        "latency_ms": res.latency_ms,
    }
