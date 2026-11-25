"""merge the scheduled-post schema branch with the main migration history

Revision ID: 2026_11_06_0009
Revises: 2026_11_05_0008, 2026_10_15_0004_scheduled_posts
"""

revision = "2026_11_06_0009"
down_revision = ("2026_11_05_0008", "2026_10_15_0004_scheduled_posts")
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
