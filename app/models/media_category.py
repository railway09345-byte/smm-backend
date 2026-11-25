from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base import Base


class MediaCategory(Base):
    """Tenant-owned, reusable classification for media assets and AI campaigns."""

    __tablename__ = "media_categories"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_media_categories_organization_name"),
    )

    name = Column(String(100), nullable=False)
    color = Column(String(20), nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)

    organization = relationship("Organization")
    assets = relationship("MediaAsset", back_populates="category")
