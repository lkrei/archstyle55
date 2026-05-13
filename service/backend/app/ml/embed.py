from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .preprocess import to_tensor_batch
from .registry import registry

DEVICE = "cpu"
EMBED_DIM = 768
EMBED_MODEL = "dinov2_vitb14"


def _load_embedder():
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
    return model


registry.register("dinov2_embedder", lambda: {"module": _load_embedder(), "image_size": 224})


@dataclass
class EmbedResult:
    vector: list[float]
    norm: float


@torch.inference_mode()
def embed_image(img: Image.Image) -> EmbedResult:
    bundle = registry.get("dinov2_embedder")
    model = bundle["module"]
    x = to_tensor_batch(img, bundle["image_size"]).to(DEVICE)
    out = model(x)
    cls = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
    cls = cls.squeeze(0).float()
    norm = float(cls.norm().item())
    cls = F.normalize(cls, dim=-1)
    return EmbedResult(vector=cls.tolist(), norm=norm)


@torch.inference_mode()
def embed_batch(imgs: list[Image.Image]) -> np.ndarray:
    bundle = registry.get("dinov2_embedder")
    model = bundle["module"]
    tensors = torch.cat([to_tensor_batch(im, bundle["image_size"]) for im in imgs], dim=0).to(DEVICE)
    out = model(tensors)
    cls = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
    cls = F.normalize(cls.float(), dim=-1)
    return cls.cpu().numpy()
