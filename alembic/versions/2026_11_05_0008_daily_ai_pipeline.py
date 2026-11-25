"""Add durable daily AI campaign queue fields.

Revision ID: 2026_11_05_0008
Revises: 2026_10_23_0007
"""

from alembic import op
import sqlalchemy as sa


revision = "2026_11_05_0008"
down_revision = "2026_10_23_0007"
branch_labels = None
depends_on = None


def _add_missing(table_name, columns):
    existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}
    for name, column in columns.items():
        if name not in existing:
            op.add_column(table_name, column)


def upgrade():
    _add_missing("business_schedules", {
        "image_count": sa.Column("image_count", sa.Integer(), nullable=False, server_default="1"),
        "video_count": sa.Column("video_count", sa.Integer(), nullable=False, server_default="1"),
        "cadence_days": sa.Column("cadence_days", sa.Integer(), nullable=False, server_default="1"),
    })
    _add_missing("pipeline_runs", {
        "attempts": sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        "lease_token": sa.Column("lease_token", sa.String(64), nullable=True),
        "lease_expires_at": sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
        "next_attempt_at": sa.Column("next_attempt_at", sa.DateTime(), nullable=True),
    })


def downgrade():
    pass
