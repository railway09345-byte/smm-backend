from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models.subscription import Subscription
from app.schemas.subscription import SubscriptionUpdate
from app.utils.enums import SubscriptionTier, SubscriptionStatus
from app.core.config import settings
from app.core.logging import logger


class SubscriptionService:
    TIER_LIMITS = {
        SubscriptionTier.FREE: {"generations": 50, "posts": 100},
        SubscriptionTier.STARTER: {"generations": 300, "posts": 500},
        SubscriptionTier.PRO: {"generations": 2000, "posts": 5000},
        SubscriptionTier.ENTERPRISE: {"generations": 10000, "posts": 50000},
    }

    # In-memory idempotency cache for verified Razorpay payment IDs to prevent replay attacks
    _processed_razorpay_payments: Set[str] = set()

    @staticmethod
    def get_subscription(db: Session, organization_id: int) -> Subscription:
        sub = db.query(Subscription).filter(Subscription.organization_id == organization_id).first()
        if not sub:
            sub = Subscription(
                organization_id=organization_id,
                tier=SubscriptionTier.FREE,
                status=SubscriptionStatus.ACTIVE,
                monthly_generation_limit=50,
                monthly_posts_limit=100
            )
            db.add(sub)
            db.commit()
            db.refresh(sub)
        return sub

    @classmethod
    def upgrade_tier(
        cls,
        db: Session,
        organization_id: int,
        new_tier: SubscriptionTier,
        verified_payment: bool = False,
        razorpay_order_id: Optional[str] = None,
        razorpay_payment_id: Optional[str] = None
    ) -> Subscription:
        """
        Secure subscription tier upgrade handler:
        - Free tier downgrades/resets are permitted without payment verification.
        - Paid tiers (STARTER, PRO, ENTERPRISE) strictly require verified Razorpay payment.
        - Direct client requests attempting to bypass payment raise HTTP 501 or HTTP 400.
        """
        # 1. Payment Verification Enforcement for Paid Tiers
        if new_tier != SubscriptionTier.FREE and not verified_payment:
            is_razorpay_configured = bool(
                getattr(settings, "RAZORPAY_KEY_ID", "")
                and getattr(settings, "RAZORPAY_KEY_SECRET", "")
            )
            if not is_razorpay_configured:
                logger.warning(
                    f"[Security] Rejected direct paid tier upgrade to '{new_tier.value}' for org #{organization_id}: "
                    "Razorpay payment gateway is not configured."
                )
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail=f"Payment processing is not configured on this server. Direct subscription upgrades to paid tiers ({new_tier.value}) are disabled until Razorpay credentials (RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET) are active."
                )

            logger.warning(
                f"[Security] Blocked unverified direct upgrade request to '{new_tier.value}' for org #{organization_id}."
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Paid tier upgrades ({new_tier.value}) require completed payment verification via Razorpay. Direct client tier upgrades are prohibited."
            )

        # 2. Apply Verified Subscription Upgrade
        sub = cls.get_subscription(db, organization_id)
        limits = cls.TIER_LIMITS.get(new_tier, {"generations": 50, "posts": 100})

        sub.tier = new_tier
        sub.status = SubscriptionStatus.ACTIVE
        sub.monthly_generation_limit = limits["generations"]
        sub.monthly_posts_limit = limits["posts"]
        sub.current_period_start = datetime.utcnow()
        sub.current_period_end = datetime.utcnow() + timedelta(days=30)

        # Store Razorpay IDs in existing columns (repurposed from Stripe)
        if razorpay_order_id:
            sub.stripe_customer_id = razorpay_order_id      # repurposed: stores Razorpay order ID
        if razorpay_payment_id:
            sub.stripe_subscription_id = razorpay_payment_id  # repurposed: stores Razorpay payment ID

        db.commit()
        db.refresh(sub)
        logger.info(f"[Subscription] Successfully upgraded organization #{organization_id} to tier '{new_tier.value}' (verified={verified_payment}).")
        return sub

    @classmethod
    def process_razorpay_webhook(
        cls,
        db: Session,
        raw_body: bytes,
        signature_header: str
    ) -> Dict[str, Any]:
        """
        Cryptographically verifies Razorpay webhook signatures and applies paid upgrades idempotently.
        Razorpay sends x-razorpay-signature header (HMAC-SHA256 of raw body with RAZORPAY_WEBHOOK_SECRET).
        """
        from app.integrations.payments.payment_provider import PaymentProvider
        import json

        webhook_secret = getattr(settings, "RAZORPAY_WEBHOOK_SECRET", "")

        if not webhook_secret:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Razorpay webhook secret is not configured on the backend."
            )

        # Verify signature
        if not PaymentProvider.verify_webhook_signature(raw_body, signature_header):
            logger.error("[RazorpayWebhook] Cryptographic signature verification failed.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Razorpay webhook signature."
            )

        try:
            event = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            logger.error(f"[RazorpayWebhook] Failed to parse webhook body: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid webhook payload."
            )

        event_id = event.get("id", "")
        # Idempotency check: prevent replay attacks
        if event_id in cls._processed_razorpay_payments:
            logger.info(f"[RazorpayWebhook] Duplicate event #{event_id} ignored (idempotent).")
            return {"status": "DUPLICATE_IGNORED", "event_id": event_id}

        event_type = event.get("event", "")
        logger.info(f"[RazorpayWebhook] Processing event: {event_type} (id={event_id})")

        # payment.captured is the main success event from Razorpay
        if event_type == "payment.captured":
            payment_obj = event.get("payload", {}).get("payment", {}).get("entity", {})
            notes = payment_obj.get("notes", {})
            org_id_str = notes.get("organization_id")
            tier_str = notes.get("tier", "starter")
            razorpay_payment_id = payment_obj.get("id")
            razorpay_order_id = payment_obj.get("order_id")

            if org_id_str:
                try:
                    org_id = int(org_id_str)
                    target_tier = SubscriptionTier(tier_str.lower())

                    cls.upgrade_tier(
                        db=db,
                        organization_id=org_id,
                        new_tier=target_tier,
                        verified_payment=True,
                        razorpay_order_id=razorpay_order_id,
                        razorpay_payment_id=razorpay_payment_id
                    )
                    cls._processed_razorpay_payments.add(event_id)
                    return {
                        "status": "SUCCESS",
                        "event_id": event_id,
                        "organization_id": org_id,
                        "tier": target_tier.value
                    }
                except (ValueError, KeyError) as e:
                    logger.warning(f"[RazorpayWebhook] Unprocessable organization or tier data: {e}")

        cls._processed_razorpay_payments.add(event_id)
        return {"status": "EVENT_ACKNOWLEDGED", "event_id": event_id}
