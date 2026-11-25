"""add tenant media categories

Revision ID: 2026_11_06_0010
Revises: 2026_11_06_0009
"""

from alembic import op
import sqlalchemy as sa


revision = "2026_11_06_0010"
down_revision = "2026_11_06_0009"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("media_categories"):
        op.create_table(
            "media_categories",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("color", sa.String(length=20), nullable=True),
            sa.Column("organization_id", sa.Integer(), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("organization_id", "name", name="uq_media_categories_organization_name"),
        )
        op.create_index("ix_media_categories_organization_id", "media_categories", ["organization_id"])

    columns = {column["name"] for column in inspector.get_columns("media_assets")}
    if "category_id" not in columns:
        op.add_column("media_assets", sa.Column("category_id", sa.Integer(), nullable=True))
        op.create_foreign_key("fk_media_assets_category_id", "media_assets", "media_categories", ["category_id"], ["id"], ondelete="SET NULL")
        op.create_index("ix_media_assets_category_id", "media_assets", ["category_id"])


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("media_assets")}
    if "category_id" in columns:
        op.drop_index("ix_media_assets_category_id", table_name="media_assets")
        op.drop_constraint("fk_media_assets_category_id", "media_assets", type_="foreignkey")
        op.drop_column("media_assets", "category_id")
    if inspector.has_table("media_categories"):
        op.drop_index("ix_media_categories_organization_id", table_name="media_categories")
        op.drop_table("media_categories")
