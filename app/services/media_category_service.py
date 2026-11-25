from typing import List

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.media_asset import MediaAsset
from app.models.media_category import MediaCategory
from app.schemas.media_category import MediaCategoryCreate, MediaCategoryUpdate


class MediaCategoryService:
    @staticmethod
    def _serialize(category: MediaCategory, asset_count: int = 0) -> dict:
        return {
            "id": category.id,
            "name": category.name,
            "color": category.color,
            "organization_id": category.organization_id,
            "asset_count": asset_count,
            "created_at": category.created_at,
            "updated_at": category.updated_at,
        }

    @classmethod
    def list_categories(cls, db: Session, organization_id: int) -> List[dict]:
        rows = (
            db.query(MediaCategory, func.count(MediaAsset.id))
            .outerjoin(MediaAsset, MediaAsset.category_id == MediaCategory.id)
            .filter(MediaCategory.organization_id == organization_id)
            .group_by(MediaCategory.id)
            .order_by(MediaCategory.name.asc())
            .all()
        )
        return [cls._serialize(category, count) for category, count in rows]

    @classmethod
    def create_category(cls, db: Session, organization_id: int, payload: MediaCategoryCreate) -> dict:
        name = payload.name.strip()
        existing = db.query(MediaCategory).filter(
            MediaCategory.organization_id == organization_id,
            func.lower(MediaCategory.name) == name.lower(),
        ).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A media category with this name already exists.")
        category = MediaCategory(organization_id=organization_id, name=name, color=payload.color)
        db.add(category)
        db.commit()
        db.refresh(category)
        return cls._serialize(category)

    @classmethod
    def update_category(cls, db: Session, category_id: int, organization_id: int, payload: MediaCategoryUpdate) -> dict:
        category = cls._get_category(db, category_id, organization_id)
        if payload.name is not None:
            name = payload.name.strip()
            duplicate = db.query(MediaCategory).filter(
                MediaCategory.organization_id == organization_id,
                func.lower(MediaCategory.name) == name.lower(),
                MediaCategory.id != category_id,
            ).first()
            if duplicate:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="A media category with this name already exists.")
            category.name = name
        if payload.color is not None:
            category.color = payload.color
        db.commit()
        db.refresh(category)
        return cls._serialize(category, len(category.assets))

    @classmethod
    def delete_category(cls, db: Session, category_id: int, organization_id: int) -> None:
        category = cls._get_category(db, category_id, organization_id)
        for asset in category.assets:
            asset.category_id = None
        db.delete(category)
        db.commit()

    @staticmethod
    def _get_category(db: Session, category_id: int, organization_id: int) -> MediaCategory:
        category = db.query(MediaCategory).filter(
            MediaCategory.id == category_id,
            MediaCategory.organization_id == organization_id,
        ).first()
        if not category:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media category not found.")
        return category
