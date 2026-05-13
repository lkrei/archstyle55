"""initial schema with pgvector

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-13
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "images",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("source", sa.String(32), nullable=False, server_default="upload"),
        sa.Column("style_label", sa.String(128), nullable=True),
        sa.Column("width", sa.Integer, nullable=True),
        sa.Column("height", sa.Integer, nullable=True),
        sa.Column("blob_url", sa.Text, nullable=True),
        sa.Column("extra", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_images_style_label", "images", ["style_label"])

    op.create_table(
        "embeddings",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("image_id", UUID(as_uuid=True), sa.ForeignKey("images.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model", sa.String(64), nullable=False, server_default="dinov2_vitb14"),
        sa.Column("vec", Vector(768), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("image_id", "model", name="uq_embeddings_image_model"),
    )
    op.create_index("ix_embeddings_image_id", "embeddings", ["image_id"])
    op.create_index("ix_embeddings_model", "embeddings", ["model"])
    op.execute(
        "CREATE INDEX ix_embeddings_vec_hnsw ON embeddings USING hnsw (vec vector_cosine_ops) "
        "WITH (m=16, ef_construction=64)"
    )

    op.create_table(
        "predictions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("image_id", UUID(as_uuid=True), sa.ForeignKey("images.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("top1_class", sa.String(128), nullable=False),
        sa.Column("top1_prob", sa.Float, nullable=False),
        sa.Column("top5", JSONB, nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_predictions_image_id", "predictions", ["image_id"])
    op.create_index("ix_predictions_model", "predictions", ["model"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("prediction_id", UUID(as_uuid=True), sa.ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_label", sa.String(128), nullable=True),
        sa.Column("is_correct", sa.Boolean, nullable=True),
        sa.Column("comment", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_feedback_prediction_id", "feedback", ["prediction_id"])

    op.create_table(
        "classes",
        sa.Column("idx", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("display_name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("signature", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("examples_paths", ARRAY(sa.Text), nullable=False, server_default="{}"),
    )

    op.create_table(
        "scrape_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("style", sa.String(128), nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("n_target", sa.Integer, nullable=False, server_default="20"),
        sa.Column("n_done", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("log", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_scrape_jobs_style", "scrape_jobs", ["style"])
    op.create_index("ix_scrape_jobs_status", "scrape_jobs", ["status"])

    op.create_table(
        "model_meta",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("family", sa.String(32), nullable=False),
        sa.Column("params_m", sa.Float, nullable=False),
        sa.Column("gflops", sa.Float, nullable=True),
        sa.Column("accuracy", sa.Float, nullable=True),
        sa.Column("macro_f1", sa.Float, nullable=True),
        sa.Column("bal_acc", sa.Float, nullable=True),
        sa.Column("inference_ms", sa.Float, nullable=True),
        sa.Column("image_size", sa.Integer, nullable=False, server_default="224"),
        sa.Column("hf_repo", sa.String(128), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("model_meta")
    op.drop_table("scrape_jobs")
    op.drop_table("classes")
    op.drop_table("feedback")
    op.drop_table("predictions")
    op.execute("DROP INDEX IF EXISTS ix_embeddings_vec_hnsw")
    op.drop_table("embeddings")
    op.drop_table("images")
    op.execute("DROP EXTENSION IF EXISTS vector")
