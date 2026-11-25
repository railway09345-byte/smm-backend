from datetime import datetime
from pydantic import BaseModel
from app.utils.enums import SubscriptionTier, SubscriptionStatus


class SubscriptionBase(BaseModel):
    tier: SubscriptionTier = SubscriptionTier.FREE
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    monthly_generation_limit: int = 50
    monthly_posts_limit: int = 100


class SubscriptionCreate(SubscriptionBase):
    organization_id: int
    razorpay_order_id: str | None = None
    razorpay_payment_id: str | None = None


class SubscriptionUpdate(BaseModel):
    tier: SubscriptionTier | None = None
    status: SubscriptionStatus | None = None
    monthly_generation_limit: int | None = None
    monthly_posts_limit: int | None = None
    current_period_end: datetime | None = None


class SubscriptionResponse(SubscriptionBase):
    id: int
    organization_id: int
    razorpay_order_id: str | None = None
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    created_at: datetime
    updated_at: datetime

    # Map repurposed DB columns to Razorpay field names
    @classmethod
    def from_orm_with_razorpay(cls, obj):
        data = {
            "id": obj.id,
            "organization_id": obj.organization_id,
            "tier": obj.tier,
            "status": obj.status,
            "monthly_generation_limit": obj.monthly_generation_limit,
            "monthly_posts_limit": obj.monthly_posts_limit,
            "razorpay_order_id": obj.stripe_customer_id,  # repurposed column
            "current_period_start": obj.current_period_start,
            "current_period_end": obj.current_period_end,
            "created_at": obj.created_at,
            "updated_at": obj.updated_at,
        }
        return cls(**data)

    class Config:
        from_attributes = True
