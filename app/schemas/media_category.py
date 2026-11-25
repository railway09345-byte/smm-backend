from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MediaCategoryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    color: Optional[str] = Field(default=None, max_length=20)


class MediaCategoryUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    color: Optional[str] = Field(default=None, max_length=20)


class MediaCategoryResponse(BaseModel):
    id: int
    name: str
    color: Optional[str] = None
    organization_id: int
    asset_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
