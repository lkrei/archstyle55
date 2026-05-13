from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .registry import registry

DEVICE = "cpu"
PALETTE = np.array([
    [180, 180, 180],
    [220, 90,  90],
    [120, 60,  220],
    [240, 200, 100],
    [100, 230, 160],
    [255, 130, 220],
    [120, 200, 240],
    [60,  170, 100],
    [80,  80,  80],
    [40,  40,  40],
], dtype=np.uint8)


def _load_segformer():
    from pipeline.segmentation.segmentor import FacadeSegmentor
    return FacadeSegmentor(backend="segformer", device=DEVICE)


registry.register("segformer_b2_facade", lambda: {"module": _load_segformer()})


@dataclass
class SegmentationOutput:
    mask: np.ndarray
    overlay_png: bytes
    mask_png: bytes
    attributes: dict[str, float]


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = PALETTE[: int(mask.max()) + 1]
    return palette[mask]


def overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    color = colorize_mask(mask)
    return ((1 - alpha) * arr + alpha * color).astype(np.uint8)


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def attributes_from_mask(mask: np.ndarray) -> dict[str, float]:
    from pipeline.segmentation.extract_attributes import extract
    return extract(mask)


@torch.inference_mode()
def segment_image(img: Image.Image) -> SegmentationOutput:
    bundle = registry.get("segformer_b2_facade")
    seg = bundle["module"]
    res = seg.segment(img)
    mask = res.mask.astype(np.int64)
    overlay_arr = overlay(img, mask)
    mask_color = colorize_mask(mask)
    attrs = attributes_from_mask(mask)
    return SegmentationOutput(
        mask=mask,
        overlay_png=_png(overlay_arr),
        mask_png=_png(mask_color),
        attributes={k: float(v) for k, v in attrs.items()},
    )
