from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None
ALLOWED_EXT = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_BYTES = 10 * 1024 * 1024
TARGET_SHORT = 512


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pil(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise ValueError("image too large")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("unsupported image format") from exc
    return img


def to_short_edge(img: Image.Image, short: int = TARGET_SHORT) -> Image.Image:
    w, h = img.size
    if min(w, h) <= short:
        return img
    if w < h:
        new_w = short
        new_h = int(h * short / w)
    else:
        new_h = short
        new_w = int(w * short / h)
    return img.resize((new_w, new_h), Image.BILINEAR)


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img, dtype=np.uint8)
