from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .preprocess import to_tensor_batch
from .registry import registry


def _target_layer_for(name: str, module: torch.nn.Module) -> torch.nn.Module:
    backbone = module.backbone
    if name.startswith("efficientnet"):
        return backbone.features[-1]
    if name == "convnext_small":
        return backbone.features[-1]
    if name == "resnet50":
        return backbone.layer4[-1]
    raise ValueError(f"target layer not configured for {name}")


def _heatmap_to_rgba(cam: np.ndarray, image: Image.Image, alpha: float = 0.55) -> np.ndarray:
    cam = np.clip(cam, 0, 1)
    cmap = (np.stack([cam, np.zeros_like(cam), 1 - cam], axis=-1) * 255).astype(np.uint8)
    arr = np.array(image.convert("RGB"))
    blended = ((1 - alpha) * arr + alpha * cmap).astype(np.uint8)
    return blended


@dataclass
class CamResult:
    method: str
    target_class: int
    target_class_name: str
    overlay_png: bytes
    raw: list[list[float]]
    latency_ms: float


def gradcam_pp(name: str, img: Image.Image, target_class: int | None = None) -> CamResult:
    import time

    bundle = registry.get(name)
    model = bundle["module"]
    image_size = bundle["image_size"]
    layer = _target_layer_for(name, model)

    from pipeline.xai.cam import grad_cam_pp

    with torch.enable_grad():
        x = to_tensor_batch(img, image_size).clone().detach().requires_grad_(True)
        started = time.perf_counter()
        cam = grad_cam_pp(model, x, layer, target_class=target_class)
        elapsed = (time.perf_counter() - started) * 1000.0

    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)) / 255.0
    overlay = _heatmap_to_rgba(cam_resized, img)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")

    target_class_resolved = target_class
    if target_class_resolved is None:
        with torch.inference_mode():
            logits = model(to_tensor_batch(img, image_size))
            target_class_resolved = int(logits.argmax(dim=-1).item())
    from .labels import class_names, display
    return CamResult(
        method="grad_cam_pp",
        target_class=target_class_resolved,
        target_class_name=display(class_names()[target_class_resolved]),
        overlay_png=buf.getvalue(),
        raw=cam.tolist(),
        latency_ms=round(elapsed, 2),
    )


def supports_cnn(name: str) -> bool:
    return name in {"efficientnet_v2_s", "efficientnet_b3", "convnext_small"}
