"""store uploaded training datasets

Revision ID: 20260618_0002
Revises: 20260527_0001
Create Date: 2026-06-18
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260618_0002"
down_revision: Union[str, None] = "20260527_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("original_filename", sa.String(length=255), nullable=True))
    op.add_column("datasets", sa.Column("storage_path", sa.String(length=500), nullable=True))
    op.add_column("datasets", sa.Column("file_size_bytes", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("datasets", "file_size_bytes")
    op.drop_column("datasets", "storage_path")
    op.drop_column("datasets", "original_filename")
