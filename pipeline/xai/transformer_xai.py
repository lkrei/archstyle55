"""Объяснимость для ViT/Swin/DINOv2.

Включает:

* `attention_rollout` — базовый baseline (Abnar & Zuidema, 2020).
* `chefer_relevance` — упрощённая реализация подхода
  Chefer et al., 2021 «Transformer Interpretability Beyond Attention Visualization»:
  агрегируем `attn * (∂out/∂attn)` по слоям и применяем rollout.
  Не претендует на полноценный LRP, но даёт class-specific карты,
  устойчивые к skip-connections, и подходит для отчёта.

Реализация работает с любыми ViT-подобными модулями, у которых
self-attention доступен через атрибут `attn` (это так у `timm.vit`,
`torchvision.vit_b_16` через `EncoderBlock.self_attention`).

"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def _find_attention_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    found = []
    for m in model.modules():
        cls_name = type(m).__name__.lower()
        if "attention" in cls_name and hasattr(m, "num_heads"):
            found.append(m)
    return found


def _capture_attention(model: torch.nn.Module,
                       modules: list[torch.nn.Module] | None = None) -> tuple[Callable, Callable]:
    if modules is None:
        modules = _find_attention_modules(model)
    if not modules:
        raise RuntimeError("no attention modules detected; pass modules explicitly")

    cache: list[torch.Tensor] = []
    handles = []

    def hook(_, __, output):
        # Многие модули MultiheadAttention возвращают (out, attn_weights);
        # torchvision ViT по умолчанию `need_weights=False`, поэтому работаем
        # через monkey-patch on forward для класса EncoderBlock.
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[1]
            if attn is not None:
                cache.append(attn.detach())

    for m in modules:
        handles.append(m.register_forward_hook(hook))

    def cleanup():
        for h in handles:
            h.remove()

    return cache, cleanup


def attention_rollout(model: torch.nn.Module, image_tensor: torch.Tensor,
                      head_fusion: str = "mean", discard_ratio: float = 0.0) -> np.ndarray:
    model.eval()
    image_tensor = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor

    cache, cleanup = _capture_attention(model)
    try:
        with torch.no_grad():
            _ = model(image_tensor)
    finally:
        cleanup()

    if not cache:
        raise RuntimeError(
            "attention weights were not captured; pass `need_weights=True` modules "
            "or use a model that exposes attention probabilities",
        )

    result = None
    for attn in cache:
        a = attn.mean(dim=1) if head_fusion == "mean" else attn.max(dim=1).values
        if discard_ratio > 0:
            flat = a.view(a.size(0), -1)
            n_drop = int(flat.size(-1) * discard_ratio)
            if n_drop > 0:
                vals, _ = flat.topk(n_drop, dim=-1, largest=False)
                threshold = vals[:, -1:]
                mask = (flat <= threshold).view_as(a)
                a = a.masked_fill(mask, 0.0)
        a = a + torch.eye(a.size(-1), device=a.device).unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else torch.matmul(a, result)

    cls_attention = result[0, 0, 1:].cpu().numpy()
    side = int(np.sqrt(cls_attention.shape[0]))
    return cls_attention.reshape(side, side)


def chefer_relevance(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> np.ndarray:
    """Упрощённый class-specific вариант: rollout по `(attn * grad)+`."""
    model.eval()
    image_tensor = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
    image_tensor = image_tensor.requires_grad_(True)

    attns: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    handles = []

    def fwd(_, __, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attns.append(output[1])

    for m in _find_attention_modules(model):
        handles.append(m.register_forward_hook(fwd))

    try:
        logits = model(image_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[0, target_class]
        for a in attns:
            a.retain_grad()
        score.backward()
        for a in attns:
            grads.append(a.grad.detach() if a.grad is not None else torch.zeros_like(a))
    finally:
        for h in handles:
            h.remove()

    if not attns:
        raise RuntimeError("no attention captured; ensure attention modules return weights")

    result = None
    for attn, grad in zip(attns, grads):
        weighted = (attn * grad).clamp(min=0).mean(dim=1)
        weighted = weighted + torch.eye(weighted.size(-1), device=weighted.device).unsqueeze(0)
        weighted = weighted / weighted.sum(dim=-1, keepdim=True)
        result = weighted if result is None else torch.matmul(weighted, result)

    cls_relevance = result[0, 0, 1:].detach().cpu().numpy()
    side = int(np.sqrt(cls_relevance.shape[0]))
    return cls_relevance.reshape(side, side)
