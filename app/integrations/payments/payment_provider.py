import hmac
import hashlib
import json
from typing import Dict, Any, Optional
from app.core.config import settings
from app.core.logging import logger


class PaymentProvider:
    """Razorpay Payment Integration Provider."""


    @classmethod
    def _get_client(cls):
        """Return an authenticated Razorpay client."""
        import razorpay
        return razorpay.Client(
            auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
        )

    @classmethod
    def create_order(
        cls,
        amount_paise: int,
        currency: str = "INR",
        receipt: str = "order_receipt",
        notes: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a Razorpay order.
        Returns: { razorpay_order_id, amount, currency, receipt }
        Amount must be in paise (1 INR = 100 paise).
        """
        if not settings.RAZORPAY_KEY_ID or not settings.RAZORPAY_KEY_SECRET:
            logger.warning("[PaymentProvider] Razorpay credentials not configured — returning test order.")
            return {
                "razorpay_order_id": f"order_test_{receipt}",
                "amount": amount_paise,
                "currency": currency,
                "receipt": receipt,
                "test_mode": True,
            }

        try:
            client = cls._get_client()
            order_data = {
                "amount": amount_paise,
                "currency": currency,
                "receipt": receipt,
                "notes": notes or {},
                "payment_capture": 1,  # auto-capture on payment
            }
            order = client.order.create(data=order_data)
            logger.info(f"[PaymentProvider] Razorpay order created: {order['id']} | ₹{amount_paise/100:.2f}")
            return {
                "razorpay_order_id": order["id"],
                "amount": order["amount"],
                "currency": order["currency"],
                "receipt": order.get("receipt", receipt),
            }
        except Exception as e:
            logger.error(f"[PaymentProvider] Razorpay order creation failed: {e}")
            raise RuntimeError(f"Razorpay API error: {str(e)}")

    @classmethod
    def verify_payment_signature(
        cls,
        razorpay_order_id: str,
        razorpay_payment_id: str,
        razorpay_signature: str
    ) -> bool:
        """
        Verify Razorpay payment signature using HMAC-SHA256.
        Returns True if the signature is valid, False otherwise.
        """
        try:
            message = f"{razorpay_order_id}|{razorpay_payment_id}"
            expected = hmac.new(
                settings.RAZORPAY_KEY_SECRET.encode("utf-8"),
                message.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(expected, razorpay_signature)
        except Exception as e:
            logger.error(f"[PaymentProvider] Signature verification error: {e}")
            return False

    @classmethod
    def verify_webhook_signature(
        cls,
        raw_body: bytes,
        signature_header: str
    ) -> bool:
        """
        Verify the x-razorpay-signature header for incoming webhook events.
        Uses HMAC-SHA256 with RAZORPAY_WEBHOOK_SECRET.
        """
        try:
            expected = hmac.new(
                settings.RAZORPAY_WEBHOOK_SECRET.encode("utf-8"),
                raw_body,
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(expected, signature_header)
        except Exception as e:
            logger.error(f"[PaymentProvider] Webhook signature verification error: {e}")
            return False
