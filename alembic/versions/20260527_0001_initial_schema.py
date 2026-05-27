"""initial backend schema

Revision ID: 20260527_0001
Revises:
Create Date: 2026-05-27
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260527_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_hash", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("columns", sa.Integer(), nullable=False),
        sa.Column("n_subjects", sa.Integer(), nullable=False),
        sa.Column("class_distribution", sa.JSON(), nullable=False),
        sa.Column("eeg_columns", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dataset_hash", name="uq_datasets_hash"),
    )
    op.create_index(op.f("ix_datasets_dataset_hash"), "datasets", ["dataset_hash"], unique=False)

    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("model_type", sa.String(length=20), nullable=False),
        sa.Column("model_name", sa.String(length=80), nullable=False),
        sa.Column("evaluation_mode", sa.String(length=255), nullable=False),
        sa.Column("training_time_seconds", sa.Float(), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=False),
        sa.Column("balanced_accuracy", sa.Float(), nullable=False),
        sa.Column("precision", sa.Float(), nullable=False),
        sa.Column("recall", sa.Float(), nullable=False),
        sa.Column("f1_score", sa.Float(), nullable=False),
        sa.Column("eeg_params", sa.JSON(), nullable=False),
        sa.Column("model_params", sa.JSON(), nullable=False),
        sa.Column("training_params", sa.JSON(), nullable=False),
        sa.Column("confusion_matrix", sa.JSON(), nullable=False),
        sa.Column("classification_report", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_experiments_dataset_id"), "experiments", ["dataset_id"], unique=False)
    op.create_index(op.f("ix_experiments_model_name"), "experiments", ["model_name"], unique=False)
    op.create_index(op.f("ix_experiments_model_type"), "experiments", ["model_type"], unique=False)

    op.create_table(
        "experiment_folds",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("fold", sa.Integer(), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=False),
        sa.Column("balanced_accuracy", sa.Float(), nullable=False),
        sa.Column("precision", sa.Float(), nullable=False),
        sa.Column("recall", sa.Float(), nullable=False),
        sa.Column("f1_score", sa.Float(), nullable=False),
        sa.Column("n_train_subjects", sa.Integer(), nullable=True),
        sa.Column("n_val_subjects", sa.Integer(), nullable=True),
        sa.Column("n_test_subjects", sa.Integer(), nullable=True),
        sa.Column("best_threshold", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiments.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_experiment_folds_experiment_id"), "experiment_folds", ["experiment_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_experiment_folds_experiment_id"), table_name="experiment_folds")
    op.drop_table("experiment_folds")
    op.drop_index(op.f("ix_experiments_model_type"), table_name="experiments")
    op.drop_index(op.f("ix_experiments_model_name"), table_name="experiments")
    op.drop_index(op.f("ix_experiments_dataset_id"), table_name="experiments")
    op.drop_table("experiments")
    op.drop_index(op.f("ix_datasets_dataset_hash"), table_name="datasets")
    op.drop_table("datasets")
