from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class PaletteEntry:
    rgb_hex: str
    rgb: tuple[int, int, int]
    share: float


def _kmeans_rgb(pixels: np.ndarray, k: int = 5, iters: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = len(pixels)
    if n == 0:
        return np.zeros((k, 3), dtype=np.uint8), np.zeros(k, dtype=np.float32)
    if n > 4096:
        pixels = pixels[rng.choice(n, size=4096, replace=False)]
        n = 4096
    centers = pixels[rng.choice(n, size=k, replace=False)].astype(np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = d.argmin(axis=1)
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = pixels[mask].mean(axis=0)
    counts = np.bincount(labels, minlength=k).astype(np.float32)
    return centers.astype(np.uint8), counts / max(1, counts.sum())


def palette(img: Image.Image, k: int = 5) -> list[PaletteEntry]:
    arr = np.array(img.convert("RGB").resize((256, 256), Image.BILINEAR))
    pixels = arr.reshape(-1, 3)
    centers, shares = _kmeans_rgb(pixels, k=k)
    order = np.argsort(shares)[::-1]
    return [
        PaletteEntry(
            rgb_hex="#{:02x}{:02x}{:02x}".format(*centers[i]),
            rgb=tuple(int(c) for c in centers[i]),
            share=float(shares[i]),
        )
        for i in order
    ]
