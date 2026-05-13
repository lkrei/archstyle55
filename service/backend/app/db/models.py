from __future__ import annotations

import datetime as dt
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class ImageRow(Base):
    __tablename__ = "images"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(32), default="upload")
    style_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    blob_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    embeddings: Mapped[list[EmbeddingRow]] = relationship(back_populates="image", cascade="all, delete-orphan")
    predictions: Mapped[list[PredictionRow]] = relationship(back_populates="image", cascade="all, delete-orphan")


class EmbeddingRow(Base):
    __tablename__ = "embeddings"
    __table_args__ = (UniqueConstraint("image_id", "model", name="uq_embeddings_image_model"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), index=True)
    model: Mapped[str] = mapped_column(String(64), default="dinov2_vitb14", index=True)
    vec: Mapped[list[float]] = mapped_column(Vector(768))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    image: Mapped[ImageRow] = relationship(back_populates="embeddings")


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    image_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), index=True)
    model: Mapped[str] = mapped_column(String(64), index=True)
    top1_class: Mapped[str] = mapped_column(String(128))
    top1_prob: Mapped[float] = mapped_column(Float)
    top5: Mapped[list] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    image: Mapped[ImageRow] = relationship(back_populates="predictions")
    feedback: Mapped[list[FeedbackRow]] = relationship(back_populates="prediction", cascade="all, delete-orphan")


class FeedbackRow(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"), index=True)
    user_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    prediction: Mapped[PredictionRow] = relationship(back_populates="feedback")


class ClassRow(Base):
    __tablename__ = "classes"

    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    display_name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    signature: Mapped[dict] = mapped_column(JSON, default=dict)
    examples_paths: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)


class ScrapeJobRow(Base):
    __tablename__ = "scrape_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    style: Mapped[str] = mapped_column(String(128), index=True)
    query: Mapped[str] = mapped_column(Text)
    n_target: Mapped[int] = mapped_column(Integer, default=20)
    n_done: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    log: Mapped[list] = mapped_column(JSON, default=list)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModelMetaRow(Base):
    __tablename__ = "model_meta"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    family: Mapped[str] = mapped_column(String(32))
    params_m: Mapped[float] = mapped_column(Float)
    gflops: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    bal_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_size: Mapped[int] = mapped_column(Integer, default=224)
    hf_repo: Mapped[str | None] = mapped_column(String(128), nullable=True)
