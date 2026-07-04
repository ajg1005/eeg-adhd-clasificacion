"""add trained models registry

Revision ID: 20260703_0003
Revises: 20260618_0002
Create Date: 2026-07-03
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260703_0003"
down_revision: Union[str, None] = "20260618_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "trained_models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("model_type", sa.String(length=20), nullable=False),
        sa.Column("model_name", sa.String(length=80), nullable=False),
        sa.Column("model_family", sa.String(length=40), nullable=False),
        sa.Column("artifact_path", sa.String(length=500), nullable=False),
        sa.Column("feature_columns_path", sa.String(length=500), nullable=True),
        sa.Column("n_features", sa.Integer(), nullable=True),
        sa.Column("n_epochs_training", sa.Integer(), nullable=False),
        sa.Column("n_subjects_training", sa.Integer(), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("model_metadata", sa.JSON(), nullable=False),
        sa.Column("is_selected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("experiment_id", name="uq_trained_models_experiment_id"),
    )
    op.create_index("ix_trained_models_experiment_id", "trained_models", ["experiment_id"])
    op.create_index("ix_trained_models_model_family", "trained_models", ["model_family"])
    op.create_index("ix_trained_models_model_name", "trained_models", ["model_name"])
    op.create_index("ix_trained_models_model_type", "trained_models", ["model_type"])


def downgrade() -> None:
    op.drop_index("ix_trained_models_model_type", table_name="trained_models")
    op.drop_index("ix_trained_models_model_name", table_name="trained_models")
    op.drop_index("ix_trained_models_model_family", table_name="trained_models")
    op.drop_index("ix_trained_models_experiment_id", table_name="trained_models")
    op.drop_table("trained_models")