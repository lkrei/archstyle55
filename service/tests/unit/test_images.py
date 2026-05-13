from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from app.utils.images import load_pil, sha256_hex, to_short_edge  # noqa: E402


def _bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def test_load_pil_reads_jpeg():
    img = Image.new("RGB", (256, 192), color=(10, 20, 30))
    img2 = load_pil(_bytes(img))
    assert img2.size == (256, 192)
    assert img2.mode == "RGB"


def test_load_pil_rejects_nonimage():
    with pytest.raises(ValueError):
        load_pil(b"not an image")


def test_short_edge_resize():
    img = Image.new("RGB", (1024, 512))
    out = to_short_edge(img, short=256)
    assert min(out.size) == 256


def test_sha256_stable():
    a = sha256_hex(b"a")
    b = sha256_hex(b"a")
    assert a == b
    assert len(a) == 64
