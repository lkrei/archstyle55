from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from .labels import class_names, display
from .registry import registry

DEVICE = "cpu"
CLIP_MODEL = "openai/clip-vit-base-patch16"


def _load_clip():
    from transformers import AutoProcessor, CLIPModel
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(CLIP_MODEL)
    prompts = [f"a photograph of {n.lower()}" for n in class_names()]
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.inference_mode():
        text_feats = model.get_text_features(**inputs)
    text_feats = F.normalize(text_feats, dim=-1)
    return {"model": model, "processor": processor, "text_feats": text_feats}


registry.register("clip_zeroshot", lambda: {"module": _load_clip()})


@dataclass
class ZeroShotResult:
    top1_class: str
    top1_prob: float
    top5: list[dict]
    latency_ms: float
    prompt: str


@torch.inference_mode()
def predict_zeroshot(img: Image.Image, prompt_template: str = "a photograph of {}",
                     top_k: int = 5) -> ZeroShotResult:
    started = time.perf_counter()
    bundle = registry.get("clip_zeroshot")
    inner = bundle["module"]
    model = inner["model"]
    processor = inner["processor"]

    if prompt_template == "a photograph of {}":
        text_feats = inner["text_feats"]
    else:
        prompts = [prompt_template.format(n.lower()) for n in class_names()]
        toks = processor.tokenizer(prompts, return_tensors="pt", padding=True)
        text_feats = model.get_text_features(**toks)
        text_feats = F.normalize(text_feats, dim=-1)

    img_inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    img_feats = model.get_image_features(**img_inputs)
    img_feats = F.normalize(img_feats, dim=-1)
    logits = img_feats @ text_feats.T
    probs = F.softmax(logits.squeeze(0) * 100.0, dim=-1)
    elapsed = (time.perf_counter() - started) * 1000.0

    top_p, top_i = torch.topk(probs, k=top_k)
    classes = class_names()
    items = [
        {"cls": display(classes[int(i)]), "prob": float(p)}
        for p, i in zip(top_p.tolist(), top_i.tolist())
    ]
    return ZeroShotResult(
        top1_class=items[0]["cls"],
        top1_prob=items[0]["prob"],
        top5=items,
        latency_ms=round(elapsed, 2),
        prompt=prompt_template,
    )
