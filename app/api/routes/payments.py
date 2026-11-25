"""
Razorpay Payment Routes
Handles order creation and payment signature verification for subscription upgrades.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.db.dependencies import get_db
from app.models.user import User
from app.core.permissions import get_current_active_user
from app.core.config import settings
from app.core.logging import logger
from app.integrations.payments.payment_provider import PaymentProvider
from app.services.subscription_service import SubscriptionService
from app.utils.enums import SubscriptionTier

router = APIRouter(prefix="/payments", tags=["Payments & Billing"])

# Pricing map: tier → amount in paise (1 INR = 100 paise)
TIER_PRICING_PAISE = {
    "starter": 199900,    # ₹1,999 / month
    "pro": 499900,        # ₹4,999 / month
    "enterprise": 999900, # ₹9,999 / month
}


class CreateOrderRequest(BaseModel):
    tier: str  # "starter" | "pro" | "enterprise"


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    tier: str  # tier to activate on successful verification


@router.post("/create-order")
def create_razorpay_order(
    body: CreateOrderRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Step 1 of Razorpay checkout flow.
    Creates a Razorpay order and returns order details to the frontend.
    The frontend uses these details to open the Razorpay checkout modal.
    """
    tier_key = body.tier.lower()
    amount_paise = TIER_PRICING_PAISE.get(tier_key)

    if not amount_paise:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid subscription tier '{body.tier}'. Valid values: starter, pro, enterprise."
        )

    receipt = f"org_{current_user.organization_id}_tier_{tier_key}"
    notes = {
        "organization_id": str(current_user.organization_id),
        "tier": tier_key,
        "user_email": current_user.email or "",
    }

    try:
        order = PaymentProvider.create_order(
            amount_paise=amount_paise,
            currency="INR",
            receipt=receipt,
            notes=notes
        )
        return {
            "razorpay_order_id": order["razorpay_order_id"],
            "amount": order["amount"],
            "currency": order["currency"],
            "razorpay_key_id": settings.RAZORPAY_KEY_ID,
            "tier": tier_key,
            "test_mode": order.get("test_mode", False),
        }
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e)
        )


@router.post("/verify")
def verify_razorpay_payment(
    body: VerifyPaymentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Step 2 of Razorpay checkout flow.
    Verifies the payment signature returned by Razorpay checkout modal.
    On success, upgrades the organization's subscription tier.
    """
    tier_key = body.tier.lower()

    # Validate tier
    try:
        target_tier = SubscriptionTier(tier_key)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid subscription tier '{body.tier}'."
        )

    # Verify signature
    is_valid = PaymentProvider.verify_payment_signature(
        razorpay_order_id=body.razorpay_order_id,
        razorpay_payment_id=body.razorpay_payment_id,
        razorpay_signature=body.razorpay_signature
    )

    if not is_valid:
        logger.warning(
            f"[PaymentVerify] Invalid signature for org #{current_user.organization_id} "
            f"| order={body.razorpay_order_id} | payment={body.razorpay_payment_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment signature verification failed. Please contact support if the amount was deducted."
        )

    logger.info(
        f"[PaymentVerify] Valid payment for org #{current_user.organization_id} "
        f"| tier={tier_key} | payment={body.razorpay_payment_id}"
    )

    # Upgrade subscription
    subscription = SubscriptionService.upgrade_tier(
        db=db,
        organization_id=current_user.organization_id,
        new_tier=target_tier,
        verified_payment=True,
        razorpay_order_id=body.razorpay_order_id,
        razorpay_payment_id=body.razorpay_payment_id
    )

    return {
        "status": "SUCCESS",
        "message": f"Subscription upgraded to {target_tier.value.title()} successfully!",
        "tier": subscription.tier.value,
        "subscription_id": subscription.id,
        "razorpay_payment_id": body.razorpay_payment_id,
    }
