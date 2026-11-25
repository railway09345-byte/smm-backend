from copy import deepcopy
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.campaign import Campaign
from app.models.brand_profile import BrandProfile
from app.models.content import Content
from app.models.scheduled_post import ScheduledPost
from app.utils.enums import ContentStatus
from app.schemas.campaign import CampaignCreate, CampaignUpdate


class CampaignService:
    @staticmethod
    def progress_for_campaign(db: Session, campaign: Campaign) -> Dict[str, int]:
        contents = db.query(Content).filter(Content.campaign_id == campaign.id).all()
        scheduled = db.query(ScheduledPost).join(Content, ScheduledPost.content_id == Content.id).filter(Content.campaign_id == campaign.id).all()
        generated = sum(1 for content in contents if content.media_asset_id)
        completed = sum(1 for content in contents if (content.meta_info or {}).get("plan_status") in {"completed", "published"})
        return {
            "planned": len(contents),
            "generated": generated,
            "scheduled": len(scheduled),
            "published": sum(1 for post in scheduled if post.status in {"published", "posted"}),
            "completed": completed,
            "failed": sum(1 for post in scheduled if post.status == "failed"),
        }

    @classmethod
    def serialize_campaign(cls, db: Session, campaign: Campaign) -> dict:
        return {
            "id": campaign.id,
            "title": campaign.title,
            "objective": campaign.objective,
            "start_date": campaign.start_date,
            "end_date": campaign.end_date,
            "tags": campaign.tags or {},
            "status": campaign.status,
            "organization_id": campaign.organization_id,
            "brand_profile_id": campaign.brand_profile_id,
            "created_at": campaign.created_at,
            "updated_at": campaign.updated_at,
            "progress": cls.progress_for_campaign(db, campaign),
        }

    @staticmethod
    def list_campaigns(db: Session, organization_id: Optional[int]) -> List[dict]:
        if not organization_id:
            return []
        campaigns = db.query(Campaign).filter(Campaign.organization_id == organization_id).order_by(Campaign.id.desc()).all()
        return [CampaignService.serialize_campaign(db, campaign) for campaign in campaigns]

    @staticmethod
    def get_campaign(db: Session, campaign_id: int, organization_id: Optional[int]) -> Campaign:
        if not organization_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign not found.")
        campaign = db.query(Campaign).filter(
            Campaign.id == campaign_id,
            Campaign.organization_id == organization_id
        ).first()
        if not campaign:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign not found.")
        return campaign

    @staticmethod
    def create_campaign(db: Session, organization_id: int, payload: CampaignCreate) -> Campaign:
        brand_id = payload.brand_profile_id
        if not brand_id:
            brand = db.query(BrandProfile).filter(BrandProfile.organization_id == organization_id).first()
            if not brand:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No Brand Profile found for organization.")
            brand_id = brand.id
        else:
            # Verify brand exists for this organization
            brand = db.query(BrandProfile).filter(
                BrandProfile.id == brand_id,
                BrandProfile.organization_id == organization_id
            ).first()
            if not brand:
                fallback_brand = db.query(BrandProfile).filter(BrandProfile.organization_id == organization_id).first()
                if fallback_brand:
                    brand_id = fallback_brand.id
                else:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Brand Profile for organization.")

        campaign = Campaign(
            organization_id=organization_id,
            brand_profile_id=brand_id,
            title=payload.title,
            objective=payload.objective,
            start_date=payload.start_date,
            end_date=payload.end_date,
            tags=payload.tags,
            status=payload.status
        )
        db.add(campaign)
        db.commit()
        db.refresh(campaign)
        return campaign

    @staticmethod
    def update_campaign(db: Session, campaign_id: int, organization_id: int, payload: CampaignUpdate) -> Campaign:
        campaign = CampaignService.get_campaign(db, campaign_id, organization_id)
        update_data = payload.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(campaign, field, value)
        db.commit()
        db.refresh(campaign)
        return campaign

    @staticmethod
    def delete_campaign(db: Session, campaign_id: int, organization_id: int) -> bool:
        campaign = CampaignService.get_campaign(db, campaign_id, organization_id)
        db.delete(campaign)
        db.commit()
        return True

    @staticmethod
    def set_lifecycle_status(db: Session, campaign_id: int, organization_id: int, lifecycle_status: str) -> Campaign:
        campaign = CampaignService.get_campaign(db, campaign_id, organization_id)
        if lifecycle_status not in {"active", "paused", "cancelled"}:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unsupported campaign lifecycle status.")
        campaign.status = lifecycle_status
        posts = db.query(ScheduledPost).join(Content, ScheduledPost.content_id == Content.id).filter(
            Content.campaign_id == campaign.id,
            Content.organization_id == organization_id,
            ScheduledPost.status.in_(["scheduled", "pending_approval", "ready_to_publish"]),
        ).all()
        if lifecycle_status == "paused":
            for post in posts:
                post.status = "paused"
        elif lifecycle_status == "cancelled":
            for post in posts:
                post.status = "cancelled"
        else:
            for post in posts:
                if post.status == "paused":
                    post.status = "scheduled"
        db.commit()
        db.refresh(campaign)
        return campaign

    @staticmethod
    def duplicate_campaign(db: Session, campaign_id: int, organization_id: int) -> Campaign:
        source = CampaignService.get_campaign(db, campaign_id, organization_id)
        duplicate = Campaign(
            title=f"{source.title} (Copy)",
            objective=source.objective,
            start_date=source.start_date,
            end_date=source.end_date,
            tags=deepcopy(source.tags or {}),
            status="active",
            organization_id=source.organization_id,
            brand_profile_id=source.brand_profile_id,
        )
        db.add(duplicate)
        db.flush()
        for source_content in source.contents:
            meta = deepcopy(source_content.meta_info or {})
            meta["plan_status"] = "planned"
            db.add(Content(
                title=source_content.title,
                caption=source_content.caption,
                hashtags=deepcopy(source_content.hashtags or []),
                target_platforms=deepcopy(source_content.target_platforms or []),
                status=ContentStatus.DRAFT,
                meta_info=meta,
                brand_profile_id=source_content.brand_profile_id,
                organization_id=source_content.organization_id,
                campaign_id=duplicate.id,
                creator_id=source_content.creator_id,
                media_asset_id=source_content.media_asset_id,
            ))
        db.commit()
        db.refresh(duplicate)
        return duplicate
