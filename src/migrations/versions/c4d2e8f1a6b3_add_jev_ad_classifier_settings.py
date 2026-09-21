"""add jev ad classifier settings

Revision ID: c4d2e8f1a6b3
Revises: b8e4f1c2d3a9
Create Date: 2026-09-21 17:15:47.438648

"""

import sqlalchemy as sa
from alembic import op

revision = "c4d2e8f1a6b3"
down_revision = "b8e4f1c2d3a9"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("llm_settings", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "ad_classifier_backend", sa.Text(), server_default="llm", nullable=False
            )
        )
        batch_op.add_column(sa.Column("jev_api_key", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("jev_base_url", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("jev_model", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("llm_settings", schema=None) as batch_op:
        batch_op.drop_column("jev_model")
        batch_op.drop_column("jev_base_url")
        batch_op.drop_column("jev_api_key")
        batch_op.drop_column("ad_classifier_backend")
