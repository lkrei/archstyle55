
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class MixerConfig:
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    apply_p: float = 0.5
    cutmix_share: float = 0.5
    num_classes: int = 0


class BatchMixer:
    def __init__(self, cfg: MixerConfig):
        if cfg.num_classes <= 0:
            raise ValueError("MixerConfig.num_classes must be set")
        self.cfg = cfg
        self.active = True

    def set_active(self, value: bool) -> None:
        self.active = value

    def _one_hot(self, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(target, self.cfg.num_classes).float()

    def __call__(
        self,
        images: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_oh = self._one_hot(target)
        if not self.active or torch.rand(1).item() > self.cfg.apply_p:
            return images, target_oh

        cutmix_on = self.cfg.cutmix_alpha > 0
        mixup_on = self.cfg.mixup_alpha > 0
        if not (cutmix_on or mixup_on):
            return images, target_oh
        if cutmix_on and mixup_on:
            use_cutmix = torch.rand(1).item() < self.cfg.cutmix_share
        else:
            use_cutmix = cutmix_on
        if use_cutmix:
            return self._cutmix(images, target_oh)
        return self._mixup(images, target_oh)

    def _mixup(self, images: torch.Tensor, target_oh: torch.Tensor):
        lam = float(np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_alpha))
        perm = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1.0 - lam) * images[perm]
        mixed_t = lam * target_oh + (1.0 - lam) * target_oh[perm]
        return mixed, mixed_t

    def _cutmix(self, images: torch.Tensor, target_oh: torch.Tensor):
        lam = float(np.random.beta(self.cfg.cutmix_alpha, self.cfg.cutmix_alpha))
        perm = torch.randperm(images.size(0), device=images.device)
        b, _, h, w = images.shape
        cx, cy = int(np.random.uniform(0, w)), int(np.random.uniform(0, h))
        cut_ratio = float(np.sqrt(1.0 - lam))
        cw, ch = int(w * cut_ratio), int(h * cut_ratio)
        x0, x1 = max(0, cx - cw // 2), min(w, cx + cw // 2)
        y0, y1 = max(0, cy - ch // 2), min(h, cy + ch // 2)
        images = images.clone()
        images[:, :, y0:y1, x0:x1] = images[perm, :, y0:y1, x0:x1]
        actual_lam = 1.0 - ((x1 - x0) * (y1 - y0) / float(h * w))
        mixed_t = actual_lam * target_oh + (1.0 - actual_lam) * target_oh[perm]
        return images, mixed_t
