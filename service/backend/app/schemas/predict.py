from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class TopK(BaseModel):
    cls: str
    prob: float


class PredictionResponse(BaseModel):
    prediction_id: UUID
    image_id: UUID
    model: str
    top1_class: str
    top1_prob: float
    top5: list[TopK]
    latency_ms: float
    cache: bool = False


class EnsembleMode(BaseModel):
    mode: Literal["uniform", "weighted"] = "uniform"


class HybridResponse(PredictionResponse):
    attributes: dict = Field(default_factory=dict)
    embedding_norm: float


class ZeroShotResponse(PredictionResponse):
    prompt: str
