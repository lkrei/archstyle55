"""Exponential moving average весов модели.

"""
from __future__ import annotations

from copy import deepcopy

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for ema_p, src_p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(src_p.detach(), alpha=1.0 - d)
        for ema_b, src_b in zip(self.module.buffers(), model.buffers()):
            if src_b.dtype.is_floating_point:
                ema_b.mul_(d).add_(src_b.detach(), alpha=1.0 - d)
            else:
                ema_b.copy_(src_b)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state)
