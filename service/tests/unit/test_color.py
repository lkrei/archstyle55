from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.ml.color import palette  # noqa: E402


def test_palette_returns_5_entries_with_normalized_shares():
    img = Image.fromarray((np.random.rand(128, 128, 3) * 255).astype(np.uint8))
    pal = palette(img, k=5)
    assert len(pal) == 5
    assert abs(sum(p.share for p in pal) - 1.0) < 1e-3
    for entry in pal:
        assert entry.rgb_hex.startswith("#") and len(entry.rgb_hex) == 7


def test_palette_handles_constant_image():
    img = Image.new("RGB", (96, 96), color=(120, 80, 220))
    pal = palette(img, k=3)
    assert len(pal) == 3
    assert pal[0].share > 0.7
