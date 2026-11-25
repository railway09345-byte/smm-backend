import hmac
import hashlib
import base64
import json
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, Response, Query, status
from fastapi.responses import PlainTextResponse, JSONResponse

from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/webhooks", tags=["Universal Social Media Webhooks"])


# =========================================================================
# 1. META (FACEBOOK & INSTAGRAM) WEBHOOKS
# =========================================================================

@router.get("/meta", response_class=PlainTextResponse)
@router.get("/facebook", response_class=PlainTextResponse)
@router.get("/instagram", response_class=PlainTextResponse)
def meta_webhook_verification(
    request: Request,
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge")
):
    """
    Meta (Facebook & Instagram) Webhook Handshake Verification.
    Meta's servers send a GET request with hub.mode=subscribe, hub.verify_token, and hub.challenge.
    Returns hub.challenge as plain text upon token verification.
    """
    expected_token = getattr(settings, "META_WEBHOOK_VERIFY_TOKEN", "social_ai_meta_verify_token_2026_994757e6")

    logger.info(f"[MetaWebhook] Verification attempt: mode={hub_mode}, token={hub_verify_token}")

    if hub_mode == "subscribe" and hub_verify_token == expected_token:
        logger.info(f"[MetaWebhook] Verification successful! Returning challenge: {hub_challenge}")
        return PlainTextResponse(content=str(hub_challenge), status_code=200)

    logger.warning(f"[MetaWebhook] Verification failed! Expected: '{expected_token}', Received: '{hub_verify_token}'")
    return Response(content="Forbidden: Invalid verify token", status_code=403)


@router.post("/meta")
@router.post("/facebook")
@router.post("/instagram")
async def meta_webhook_events(request: Request):
    """
    Meta Real-Time Event Ingestion (Page posts, Instagram mentions, feed comments, publish updates).
    """
    try:
        body = await request.body()
        payload = json.loads(body.decode("utf-8")) if body else {}
    except Exception as e:
        logger.warning(f"[MetaWebhook] Failed to parse payload: {e}")
        payload = {}

    logger.info(f"[MetaWebhook] Event received: object={payload.get('object')}")
    return {"status": "EVENT_RECEIVED", "object": payload.get("object")}


# =========================================================================
# 2. GOOGLE / YOUTUBE PUBSUB WEBHOOKS
# =========================================================================

@router.get("/youtube", response_class=PlainTextResponse)
def youtube_webhook_verification(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    hub_topic: Optional[str] = Query(None, alias="hub.topic")
):
    """
    YouTube PubSubHubbub Verification Handshake.
    """
    if hub_challenge:
        return PlainTextResponse(content=str(hub_challenge), status_code=200)
    return PlainTextResponse(content="OK", status_code=200)


@router.post("/youtube")
async def youtube_webhook_events(request: Request):
    """
    YouTube video upload and channel update event feed.
    """
    try:
        body = await request.body()
    except Exception:
        body = b""
    return {"status": "YOUTUBE_EVENT_RECEIVED", "bytes": len(body)}


# =========================================================================
# 3. RAZORPAY PAYMENT GATEWAY WEBHOOKS
# =========================================================================

@router.post("/razorpay")
async def razorpay_webhook_events(
    request: Request
):
    """
    Cryptographically verified Razorpay Webhook Event Ingestion.
    Dispatches payment.captured events and upgrades subscriptions idempotently.
    Verifies the x-razorpay-signature header with HMAC-SHA256.
    """
    from app.db.session import SessionLocal
    from app.services.subscription_service import SubscriptionService

    sig_header = request.headers.get("x-razorpay-signature", "")
    body = await request.body()

    db = SessionLocal()
    try:
        result = SubscriptionService.process_razorpay_webhook(
            db=db,
            raw_body=body,
            signature_header=sig_header
        )
        return result
    finally:
        db.close()
