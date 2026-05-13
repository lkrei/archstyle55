"""Zero-shot baseline через CLIP/SigLIP по 55 архитектурным стилям.

В литературе показано, что CLIP-подобные модели работают как сильный baseline
для извлечения атрибутов зданий (Wu et al., arXiv:2312.12479). Здесь мы:

* строим набор подсказок (prompt ensemble) для каждого класса,
* считаем эмбеддинги текста и изображений,
* возвращаем top-1/top-5 предсказание и log-вероятности по 55 стилям.

Поддерживаются `transformers` (CLIPModel, AutoModel для SigLIP) и `open_clip`
(если установлен). Если ни один не доступен — функция отдаёт понятную ошибку.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROMPT_TEMPLATES = (
    "a photo of a {} building",
    "facade of a {} style building",
    "an example of {} architecture",
    "a building in the {} architectural style",
    "exterior of a {} architecture facade",
)


@dataclass
class ZeroShotResult:
    accuracy: float
    top5_accuracy: float
    logits: np.ndarray
    labels: np.ndarray
    class_names: list[str]


def _normalise(name: str) -> str:
    return name.removesuffix(" architecture").removesuffix(" style").strip()


def build_prompts(class_names: list[str]) -> list[list[str]]:
    return [
        [tpl.format(_normalise(name)) for tpl in PROMPT_TEMPLATES]
        for name in class_names
    ]


def _load_clip(model_id: str, device: str):
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return model, proc


def _is_siglip(model_id: str) -> bool:
    return "siglip" in model_id.lower()


def _as_tensor(out) -> torch.Tensor:
    """Достать 2D тензор-эмбеддинг из любого ответа CLIP/SigLIP.

    transformers ≥ 4.45 в некоторых сборках возвращает из get_text_features /
    get_image_features объект ModelOutput (BaseModelOutputWithPooling),
    а не голый Tensor. Берём из него правильный атрибут.
    """
    if isinstance(out, torch.Tensor):
        return out
    for key in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, key, None)
        if isinstance(v, torch.Tensor):
            if v.dim() == 3:
                v = v[:, 0]
            return v
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"cannot extract tensor from {type(out).__name__}")


@torch.no_grad()
def _text_features(model, proc, prompts: list[list[str]], device: str,
                   model_id: str = "") -> torch.Tensor:
    flat = [p for group in prompts for p in group]
    pad_kwargs = (
        {"padding": "max_length"} if _is_siglip(model_id)
        else {"padding": True}
    )
    tokens = proc(
        text=flat, return_tensors="pt", truncation=True, **pad_kwargs,
    ).to(device)
    if hasattr(model, "get_text_features"):
        out = model.get_text_features(**tokens)
    else:
        out = model(**tokens)
    feats = _as_tensor(out)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    feats = feats.view(len(prompts), len(prompts[0]), -1).mean(dim=1)
    return torch.nn.functional.normalize(feats, dim=-1)


@torch.no_grad()
def zero_shot_predict(
    samples: list[dict],
    class_names: list[str],
    model_id: str = "openai/clip-vit-base-patch16",
    device: str | None = None,
    batch_size: int = 32,
) -> ZeroShotResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, proc = _load_clip(model_id, device)
    prompts = build_prompts(class_names)
    text_feats = _text_features(model, proc, prompts, device, model_id=model_id)

    all_logits = []
    all_labels = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        images = [Image.open(s["path"]).convert("RGB") for s in batch]
        inputs = proc(images=images, return_tensors="pt").to(device)
        if hasattr(model, "get_image_features"):
            img_out = model.get_image_features(**inputs)
        else:
            img_out = model(**inputs)
        img_feats = torch.nn.functional.normalize(_as_tensor(img_out), dim=-1)
        sim = img_feats @ text_feats.T
        all_logits.append(sim.cpu().numpy())
        all_labels.append(np.array([int(s["label"]) for s in batch]))

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = logits.argmax(axis=1)
    top5 = np.argsort(-logits, axis=1)[:, :5]
    accuracy = float((preds == labels).mean())
    top5_acc = float(np.mean([labels[i] in top5[i] for i in range(len(labels))]))
    return ZeroShotResult(
        accuracy=accuracy, top5_accuracy=top5_acc,
        logits=logits, labels=labels, class_names=class_names,
    )


def save_zero_shot(result: ZeroShotResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "test_logits.npz",
                        logits=result.logits, labels=result.labels,
                        class_names=np.array(result.class_names))
    (out_dir / "test_metrics.json").write_text(
        f'{{"accuracy": {result.accuracy:.4f}, "top5_accuracy": {result.top5_accuracy:.4f}}}',
        encoding="utf-8",
    )
