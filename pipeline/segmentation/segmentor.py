
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image

FACADE_CATEGORIES = (
    "wall", "window", "door", "roof", "balcony",
    "column", "sky", "vegetation", "ground", "other",
)

ADE20K_TO_FACADE = {
    0: 0, 1: 0, 25: 0, 48: 0, 79: 0, 84: 0,
    8: 1, 63: 1,
    14: 2, 58: 2,
    5: 3, 86: 3, 106: 3,
    38: 4, 95: 4,
    42: 5, 93: 5,
    2: 6,
    4: 7, 9: 7, 17: 7, 66: 7, 72: 7,
    3: 8, 6: 8, 11: 8, 13: 8, 29: 8, 46: 8, 52: 8, 53: 8, 59: 8,
}
OTHER_CLASS = 9


@dataclass
class SegmentationResult:
    mask: np.ndarray
    backend: str


def _build_remap(num_classes: int = 151) -> np.ndarray:
    table = np.full(num_classes, OTHER_CLASS, dtype=np.int64)
    for ade, facade in ADE20K_TO_FACADE.items():
        table[ade] = facade
    return table


class FacadeSegmentor:
    def __init__(
        self,
        backend: Literal["segformer", "mask2former"] = "segformer",
        model_id: str | None = None,
        device: str | None = None,
    ):
        from transformers import (
            AutoImageProcessor,
            Mask2FormerForUniversalSegmentation,
            SegformerForSemanticSegmentation,
        )

        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 ("mps" if torch.backends.mps.is_available() else "cpu"))
        if backend == "segformer":
            mid = model_id or "nvidia/segformer-b2-finetuned-ade-512-512"
            self.processor = AutoImageProcessor.from_pretrained(mid)
            self.model = SegformerForSemanticSegmentation.from_pretrained(mid).to(self.device).eval()
        elif backend == "mask2former":
            mid = model_id or "facebook/mask2former-swin-small-ade-semantic"
            self.processor = AutoImageProcessor.from_pretrained(mid)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(mid).to(self.device).eval()
        else:
            raise ValueError(backend)

        self.remap = _build_remap()

    @torch.no_grad()
    def segment(self, image: Image.Image | str) -> SegmentationResult:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        if self.backend == "segformer":
            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(
                logits, size=image.size[::-1], mode="bilinear", align_corners=False,
            )
            ade_mask = upsampled.argmax(dim=1)[0].cpu().numpy()
        else:
            seg = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]],
            )[0]
            ade_mask = seg.cpu().numpy()
        ade_mask = ade_mask.clip(0, len(self.remap) - 1)
        facade_mask = self.remap[ade_mask]
        return SegmentationResult(mask=facade_mask.astype(np.int64), backend=self.backend)
