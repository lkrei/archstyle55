from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..core.logging import get_logger
from ..core.metrics import INFERENCE_LATENCY
from .labels import class_names, display
from .preprocess import to_tensor_batch
from .registry import registry
from .weights import maybe_unwrap, resolve_checkpoint

log = get_logger(__name__)
NUM_CLASSES = 55
DEVICE = "cpu"


@dataclass
class ClassifyResult:
    model: str
    top1_class: str
    top1_prob: float
    top5: list[dict]
    latency_ms: float
    logits: list[float]


def _build_torchvision(name: str) -> nn.Module:
    from pipeline.models.factory import build_model

    model = build_model(name, NUM_CLASSES)
    return model


def _build_dinov2_linear() -> nn.Module:
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained("facebook/dinov2-base")
    head = nn.Linear(backbone.config.hidden_size, NUM_CLASSES)

    class DinoLinear(nn.Module):
        def __init__(self, b, h):
            super().__init__()
            self.backbone = b
            self.head = h
            self.feature_dim = backbone.config.hidden_size

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            out = self.backbone(x)
            return out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.forward_features(x))

    return DinoLinear(backbone, head)


def _load_checkpoint(model: nn.Module, run_name: str) -> nn.Module:
    ckpt_path = resolve_checkpoint(run_name)
    raw = torch.load(ckpt_path, map_location="cpu")
    state = maybe_unwrap(raw if isinstance(raw, dict) else {"state_dict": raw})
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("ckpt.missing_keys", run=run_name, n=len(missing), example=missing[:3])
    if unexpected:
        log.warning("ckpt.unexpected_keys", run=run_name, n=len(unexpected), example=unexpected[:3])
    model.eval()
    return model


_BUILDERS = {
    "efficientnet_v2_s": ("efficientnet_v2_s_seed42", lambda: _build_torchvision("efficientnet_v2_s"), 384),
    "convnext_small":    ("convnext_small_seed42",    lambda: _build_torchvision("convnext_small"), 224),
    "efficientnet_b3":   ("efficientnet_b3_seed42",   lambda: _build_torchvision("efficientnet_b3"), 300),
    "dinov2_vitb14_linear": ("dinov2_vitb14_linear_seed42", _build_dinov2_linear, 224),
}


def _factory(name: str):
    run, builder, image_size = _BUILDERS[name]

    def loader():
        model = builder()
        try:
            _load_checkpoint(model, run)
        except FileNotFoundError as exc:
            log.warning("classify.no_ckpt", model=name, error=str(exc))
        return {"module": model.to(DEVICE).eval(), "image_size": image_size, "run": run}

    return loader


for name_ in _BUILDERS:
    registry.register(name_, _factory(name_))


@torch.inference_mode()
def predict(name: str, img: Image.Image, top_k: int = 5) -> ClassifyResult:
    bundle = registry.get(name)
    model = bundle["module"]
    image_size = bundle["image_size"]
    x = to_tensor_batch(img, image_size).to(DEVICE)
    started = time.perf_counter()
    logits = model(x).squeeze(0).float()
    elapsed = (time.perf_counter() - started) * 1000.0
    INFERENCE_LATENCY.labels(name).observe(elapsed / 1000.0)

    probs = F.softmax(logits, dim=-1)
    top_p, top_i = torch.topk(probs, k=top_k)
    classes = class_names()
    items = [
        {"cls": display(classes[int(i)]), "prob": float(p)}
        for p, i in zip(top_p.tolist(), top_i.tolist())
    ]
    return ClassifyResult(
        model=name,
        top1_class=items[0]["cls"],
        top1_prob=items[0]["prob"],
        top5=items,
        latency_ms=round(elapsed, 2),
        logits=logits.tolist(),
    )


_ENSEMBLE_MODELS = ("efficientnet_v2_s", "dinov2_vitb14_linear", "convnext_small")
_ENSEMBLE_WEIGHTS = (0.412, 0.318, 0.270)


@torch.inference_mode()
def predict_ensemble(img: Image.Image, mode: str = "uniform", top_k: int = 5) -> ClassifyResult:
    started = time.perf_counter()
    weights = (1 / 3, 1 / 3, 1 / 3) if mode == "uniform" else _ENSEMBLE_WEIGHTS
    accum = torch.zeros(NUM_CLASSES)
    parts: list[float] = []
    for name, w in zip(_ENSEMBLE_MODELS, weights):
        result = predict(name, img, top_k=NUM_CLASSES)
        accum += w * F.softmax(torch.tensor(result.logits), dim=-1)
        parts.append(result.latency_ms)
    elapsed = (time.perf_counter() - started) * 1000.0
    INFERENCE_LATENCY.labels(f"ensemble_{mode}").observe(elapsed / 1000.0)

    probs = accum / accum.sum()
    top_p, top_i = torch.topk(probs, k=top_k)
    classes = class_names()
    items = [
        {"cls": display(classes[int(i)]), "prob": float(p)}
        for p, i in zip(top_p.tolist(), top_i.tolist())
    ]
    return ClassifyResult(
        model=f"ensemble_top3_{mode}",
        top1_class=items[0]["cls"],
        top1_prob=items[0]["prob"],
        top5=items,
        latency_ms=round(elapsed, 2),
        logits=probs.log().tolist(),
    )


@torch.inference_mode()
def predict_all_real(img: Image.Image, top_k: int = 5) -> list[ClassifyResult]:
    out: list[ClassifyResult] = []
    for name in _BUILDERS:
        out.append(predict(name, img, top_k=top_k))
    return out


def list_runtime_models() -> list[str]:
    return list(_BUILDERS.keys())


def ensemble_models() -> list[str]:
    return list(_ENSEMBLE_MODELS)
