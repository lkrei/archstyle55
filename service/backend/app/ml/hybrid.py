from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from ..core.config import get_settings
from ..core.logging import get_logger
from .embed import embed_image
from .labels import class_names, display
from .registry import registry
from .segment import segment_image

log = get_logger(__name__)


def _model_path() -> Path:
    return get_settings().model_cache_dir / "hybrid_histgbm.pkl"


def _attribute_order() -> list[str]:
    from pipeline.segmentation.extract_attributes import feature_names
    return feature_names()


def _load_classifier():
    path = _model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"hybrid HistGBM not found at {path}. Run scripts.train_hybrid first."
        )
    log.info("hybrid.load", path=str(path))
    return joblib.load(path)


registry.register("hybrid_histgbm", lambda: {"module": _load_classifier()})


@dataclass
class HybridResult:
    top1_class: str
    top1_prob: float
    top5: list[dict]
    latency_ms: float
    attributes: dict
    embedding_norm: float


def _build_features(emb_vec: np.ndarray, attrs: dict[str, float]) -> np.ndarray:
    order = _attribute_order()
    attr_vec = np.array([attrs.get(name, 0.0) for name in order], dtype=np.float32)
    return np.concatenate([emb_vec, attr_vec], axis=0).astype(np.float32)


def predict_hybrid(img: Image.Image, top_k: int = 5) -> HybridResult:
    started = time.perf_counter()
    emb = embed_image(img)
    seg = segment_image(img)
    feats = _build_features(np.asarray(emb.vector, dtype=np.float32), seg.attributes)
    bundle = registry.get("hybrid_histgbm")
    clf = bundle["module"]
    probs = clf.predict_proba(feats.reshape(1, -1))[0]
    elapsed = (time.perf_counter() - started) * 1000.0

    classes = class_names()
    order = np.argsort(probs)[::-1][:top_k]
    items = [{"cls": display(classes[int(i)]), "prob": float(probs[i])} for i in order]
    return HybridResult(
        top1_class=items[0]["cls"],
        top1_prob=items[0]["prob"],
        top5=items,
        latency_ms=round(elapsed, 2),
        attributes=seg.attributes,
        embedding_norm=emb.norm,
    )
