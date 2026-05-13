"""Прототипная интерпретация (упрощённый ProtoPNet).

Описывается оригинал — Chen et al., NeurIPS 2019; современные варианты —
ProtoPNeXt (Willard et al., 2024) и EPPNet (Saralajew et al., 2024).
Здесь реализован компактный прототипный слой на основе `cosine`-сходства,
обучаемый совместно с фиксированным backbone (typical setup для archaeology
/ fine-grained), без push-проекции и cluster/sep-loss из полного ProtoPNet.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ProtoConfig:
    num_classes: int
    prototypes_per_class: int = 10
    proto_dim: int = 256
    feature_dim: int = 768
    head_temperature: float = 0.05


class ProtoLayer(nn.Module):
    def __init__(self, cfg: ProtoConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(cfg.feature_dim, cfg.proto_dim, kernel_size=1)
        self.prototypes = nn.Parameter(
            torch.randn(cfg.num_classes * cfg.prototypes_per_class, cfg.proto_dim) * 0.01,
        )

    def similarity_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        z = self.proj(feature_map)
        z = F.normalize(z, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        b, c, h, w = z.shape
        z_flat = z.view(b, c, h * w)
        sim = torch.einsum("pd,bdn->bpn", p, z_flat).view(b, p.size(0), h, w)
        return sim

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sim = self.similarity_map(feature_map)
        sim_pooled = sim.amax(dim=(2, 3))
        logits = sim_pooled.view(
            -1, self.cfg.num_classes, self.cfg.prototypes_per_class,
        ).mean(dim=2) / self.cfg.head_temperature
        return logits, sim


class PrototypeClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_extractor_fn, cfg: ProtoConfig):
        super().__init__()
        self.backbone = backbone
        self.feature_fn = feature_extractor_fn
        self.proto = ProtoLayer(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.feature_fn(self.backbone, x)
        return self.proto(feature_map)
