from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .preprocess import to_tensor_batch
from .registry import registry


@dataclass
class ViTResult:
    method: str
    target_class: int
    target_class_name: str
    overlay_png: bytes
    latency_ms: float


def _to_overlay(att: np.ndarray, img: Image.Image, alpha: float = 0.55) -> bytes:
    h_w = img.size
    att = (att - att.min()) / (att.max() - att.min() + 1e-8)
    cam = np.array(Image.fromarray((att * 255).astype(np.uint8)).resize(h_w, Image.BILINEAR)) / 255.0
    color = (np.stack([cam, np.zeros_like(cam), 1 - cam], axis=-1) * 255).astype(np.uint8)
    arr = np.array(img.convert("RGB"))
    blended = ((1 - alpha) * arr + alpha * color).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return buf.getvalue()


@torch.inference_mode()
def attention_rollout_dinov2(img: Image.Image) -> ViTResult:
    bundle = registry.get("dinov2_vitb14_linear")
    module = bundle["module"]
    image_size = bundle["image_size"]
    backbone = module.backbone

    x = to_tensor_batch(img, image_size)
    started = time.perf_counter()
    out = backbone(x, output_attentions=True)
    elapsed = (time.perf_counter() - started) * 1000.0

    attentions = out.attentions
    eye = torch.eye(attentions[0].shape[-1])
    rollout = eye.clone()
    for att in attentions:
        a = att.mean(dim=1)[0]
        a = a + torch.eye(a.shape[-1])
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout
    cls_to_patches = rollout[0, 1:]
    side = int(np.sqrt(cls_to_patches.numel()))
    grid = cls_to_patches.reshape(side, side).cpu().numpy()

    logits = module(x)
    target = int(logits.argmax(dim=-1).item())
    from .labels import class_names, display
    return ViTResult(
        method="attention_rollout",
        target_class=target,
        target_class_name=display(class_names()[target]),
        overlay_png=_to_overlay(grid, img),
        latency_ms=round(elapsed, 2),
    )


def supports_transformer(name: str) -> bool:
    return name == "dinov2_vitb14_linear"
