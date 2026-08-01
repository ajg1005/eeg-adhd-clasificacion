"""add dataset access and experiment ownership

Revision ID: 20260731_0005
Revises: 52a779fdbb8d
Create Date: 2026-07-31
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260731_0005"
down_revision: Union[str, None] = "52a779fdbb8d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dataset_accesses",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            name="fk_dataset_accesses_dataset_id_datasets",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_dataset_accesses_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("user_id", "dataset_id"),
    )
    op.create_index(
        "ix_dataset_accesses_dataset_id",
        "dataset_accesses",
        ["dataset_id"],
    )

    with op.batch_alter_table("experiments") as batch_op:
        batch_op.add_column(sa.Column("owner_id", sa.Integer(), nullable=True))
        batch_op.create_index("ix_experiments_owner_id", ["owner_id"])
        batch_op.create_foreign_key(
            "fk_experiments_owner_id_users",
            "users",
            ["owner_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("experiments") as batch_op:
        batch_op.drop_constraint(
            "fk_experiments_owner_id_users",
            type_="foreignkey",
        )
        batch_op.drop_index("ix_experiments_owner_id")
        batch_op.drop_column("owner_id")

    op.drop_index("ix_dataset_accesses_dataset_id", table_name="dataset_accesses")
    op.drop_table("dataset_accesses")
