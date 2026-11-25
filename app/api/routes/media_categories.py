from typing import List

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.permissions import get_current_active_user, require_creator
from app.db.dependencies import get_db
from app.models.user import User
from app.schemas.media_category import MediaCategoryCreate, MediaCategoryResponse, MediaCategoryUpdate
from app.services.media_category_service import MediaCategoryService

router = APIRouter(prefix="/media-categories", tags=["Media Categories"])


@router.get("/", response_model=List[MediaCategoryResponse])
def list_media_categories(db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    return MediaCategoryService.list_categories(db, current_user.organization_id)


@router.post("/", response_model=MediaCategoryResponse, status_code=status.HTTP_201_CREATED)
def create_media_category(payload: MediaCategoryCreate, db: Session = Depends(get_db), current_user: User = Depends(require_creator)):
    return MediaCategoryService.create_category(db, current_user.organization_id, payload)


@router.patch("/{category_id}", response_model=MediaCategoryResponse)
def update_media_category(category_id: int, payload: MediaCategoryUpdate, db: Session = Depends(get_db), current_user: User = Depends(require_creator)):
    return MediaCategoryService.update_category(db, category_id, current_user.organization_id, payload)


@router.delete("/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_media_category(category_id: int, db: Session = Depends(get_db), current_user: User = Depends(require_creator)):
    MediaCategoryService.delete_category(db, category_id, current_user.organization_id)
    return None
