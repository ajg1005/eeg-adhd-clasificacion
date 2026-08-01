"""add background job ownership

Revision ID: 20260801_0006
Revises: 20260731_0005
Create Date: 2026-08-01
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260801_0006"
down_revision: Union[str, None] = "20260731_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "background_jobs",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("task_name", sa.String(length=100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name="fk_background_jobs_owner_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_background_jobs_owner_id",
        "background_jobs",
        ["owner_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_background_jobs_owner_id", table_name="background_jobs")
    op.drop_table("background_jobs")
