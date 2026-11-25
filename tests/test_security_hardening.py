"""
Comprehensive Security Hardening and Production Safety Verification Suite.
Validates:
1. Test Database Safety Guard (proves production database rejection).
2. CORS Security (untrusted origin rejection, trusted Vercel/local origin allowance).
3. JWT Secret Security (rejection of default, placeholder, or short keys in production).
4. Approval Queue Role Security (creator rejection, manager approval, cross-tenant 403).
5. Subscription Payment Security (rejection of unverified paid upgrades).
"""
import os
import unittest
import uuid
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.main import app
from app.db.base import Base
from app.core.config import Settings
from app.models.organization import Organization
from app.models.user import User
from app.models.brand_profile import BrandProfile
from app.models.content import Content
from app.models.approval import Approval
from app.models.subscription import Subscription
from app.schemas.approval import ApprovalReview, ApprovalCreate
from app.services.approval_service import ApprovalService
from app.services.subscription_service import SubscriptionService
from app.utils.enums import UserRole, ApprovalStatus, ContentStatus, SubscriptionTier
from tests.conftest import assert_test_database_safety, test_engine, TestSessionLocal


class SecurityHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=test_engine)

    def setUp(self):
        self.db: Session = TestSessionLocal()
        self.client = TestClient(app)

        # Seed two distinct organizations for tenancy tests
        self.org1 = Organization(name="Sec Org 1", slug=f"sec-org-1-{uuid.uuid4().hex[:6]}", settings={})
        self.org2 = Organization(name="Sec Org 2", slug=f"sec-org-2-{uuid.uuid4().hex[:6]}", settings={})
        self.db.add_all([self.org1, self.org2])
        self.db.flush()

        # Users for Org 1
        self.creator1 = User(
            email=f"creator1_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hash",
            full_name="Creator One",
            role=UserRole.CREATOR,
            organization_id=self.org1.id,
            is_active=True
        )
        self.manager1 = User(
            email=f"manager1_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hash",
            full_name="Manager One",
            role=UserRole.MANAGER,
            organization_id=self.org1.id,
            is_active=True
        )
        self.admin1 = User(
            email=f"admin1_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hash",
            full_name="Admin One",
            role=UserRole.ADMIN,
            organization_id=self.org1.id,
            is_active=True
        )

        # Users for Org 2
        self.manager2 = User(
            email=f"manager2_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="hash",
            full_name="Manager Two",
            role=UserRole.MANAGER,
            organization_id=self.org2.id,
            is_active=True
        )

        # Brands
        self.brand1 = BrandProfile(name="Brand 1", organization_id=self.org1.id)
        self.brand2 = BrandProfile(name="Brand 2", organization_id=self.org2.id)

        self.db.add_all([self.creator1, self.manager1, self.admin1, self.manager2, self.brand1, self.brand2])
        self.db.flush()

        # Content and Approvals
        self.content1 = Content(
            organization_id=self.org1.id,
            brand_profile_id=self.brand1.id,
            title="Post 1",
            caption="Caption 1",
            status=ContentStatus.PENDING_APPROVAL
        )
        self.content2 = Content(
            organization_id=self.org2.id,
            brand_profile_id=self.brand2.id,
            title="Post 2",
            caption="Caption 2",
            status=ContentStatus.PENDING_APPROVAL
        )
        self.db.add_all([self.content1, self.content2])
        self.db.flush()

        self.approval1 = Approval(
            content_id=self.content1.id,
            requester_id=self.creator1.id,
            status=ApprovalStatus.PENDING,
            comment="Please approve"
        )
        self.approval2 = Approval(
            content_id=self.content2.id,
            requester_id=self.manager2.id,
            status=ApprovalStatus.PENDING,
            comment="Org 2 approval"
        )
        self.db.add_all([self.approval1, self.approval2])
        self.db.commit()

    def tearDown(self):
        self.db.close()

    # =========================================================================
    # FIX 1: Test Database Safety Guard Tests
    # =========================================================================
    def test_database_safety_guard_permits_isolated_test_database(self):
        """Verifies that assert_test_database_safety permits the verified isolated SQLite engine."""
        try:
            assert_test_database_safety(test_engine)
        except Exception as e:
            self.fail(f"Safety guard unexpectedly rejected valid isolated test engine: {e}")

    def test_database_safety_guard_rejects_production_mysql_engine(self):
        """Proves that any attempt to connect to production/development MySQL raises RuntimeError."""
        prod_mysql_engine = create_engine("mysql+pymysql://root:secret@db.prod.company.internal:3306/social_prod")
        with self.assertRaises(RuntimeError) as ctx:
            assert_test_database_safety(prod_mysql_engine)
        self.assertIn("FATAL DATABASE SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("production/development SQL engine", str(ctx.exception))

    def test_database_safety_guard_rejects_arbitrary_unlabeled_engine(self):
        """Proves that arbitrary unlabeled database URLs are rejected."""
        unknown_engine = create_engine("postgresql+psycopg2://admin:pass@db.internal:5432/live_warehouse")
        with self.assertRaises(RuntimeError) as ctx:
            assert_test_database_safety(unknown_engine)
        self.assertIn("FATAL DATABASE SAFETY VIOLATION", str(ctx.exception))

    # =========================================================================
    # FIX 2: CORS Security Tests
    # =========================================================================
    def test_cors_rejects_untrusted_origins(self):
        """Verifies that untrusted external origins are denied CORS credentials and headers."""
        headers = {
            "Origin": "https://malicious-attacker-site.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type"
        }
        response = self.client.options("/api/v1/auth/login", headers=headers)
        # CORSMiddleware does NOT echo the untrusted origin
        allow_origin = response.headers.get("access-control-allow-origin")
        self.assertNotEqual(allow_origin, "https://malicious-attacker-site.com")
        self.assertIsNone(allow_origin)

    def test_cors_permits_trusted_vercel_frontend(self):
        """Verifies that the legitimate Vercel production frontend receives proper CORS headers."""
        headers = {
            "Origin": "https://smm-frontend-ten.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type"
        }
        response = self.client.options("/api/v1/auth/login", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "https://smm-frontend-ten.vercel.app")
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")

    def test_cors_permits_trusted_local_development_hosts(self):
        """Verifies that development hosts (localhost:3000) are allowed during development/testing."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
        response = self.client.options("/health", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:3000")

    # =========================================================================
    # FIX 3: JWT Secret Security Tests
    # =========================================================================
    def test_jwt_secret_validation_rejects_insecure_defaults_in_production(self):
        """Proves that running in production with default/placeholder/short secret fails safely."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            # Known insecure placeholder
            with self.assertRaises(ValidationError) as ctx:
                Settings(
                    SECRET_KEY="temporary-super-secret-key-change-in-production-1234567890",
                    _env_file=None
                )
            self.assertIn("FATAL SECURITY CONFIGURATION ERROR", str(ctx.exception))

            # Short secret (< 32 chars)
            with self.assertRaises(ValidationError) as ctx2:
                Settings(SECRET_KEY="short-secret-key-too-brief", _env_file=None)
            self.assertIn("minimum 32 characters", str(ctx2.exception))

            # Empty secret
            with self.assertRaises(ValidationError) as ctx3:
                Settings(SECRET_KEY="", _env_file=None)
            self.assertIn("FATAL SECURITY CONFIGURATION ERROR", str(ctx3.exception))

    def test_jwt_secret_validation_accepts_strong_secret_in_production(self):
        """Proves that a strong 256-bit secret succeeds in production."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            strong_secret = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"
            s = Settings(SECRET_KEY=strong_secret, _env_file=None)
            self.assertEqual(s.SECRET_KEY, strong_secret)

    # =========================================================================
    # FIX 4: Approval Queue Role Security & Tenancy Tests
    # =========================================================================
    def test_creator_cannot_review_or_approve_content(self):
        """Proves that CREATOR role is strictly forbidden from approving content."""
        payload = ApprovalReview(status=ApprovalStatus.APPROVED, comment="Self-approval attempt")
        with self.assertRaises(HTTPException) as ctx:
            ApprovalService.review_approval(
                db=self.db,
                approval_id=self.approval1.id,
                reviewer=self.creator1,
                payload=payload
            )
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("lacks permission", ctx.exception.detail)

    def test_manager_can_review_and_approve_content(self):
        """Verifies that MANAGER role can approve content within their organization."""
        payload = ApprovalReview(status=ApprovalStatus.APPROVED, comment="Manager approved")
        result = ApprovalService.review_approval(
            db=self.db,
            approval_id=self.approval1.id,
            reviewer=self.manager1,
            payload=payload
        )
        self.assertEqual(result.status, ApprovalStatus.APPROVED)
        self.assertEqual(result.reviewer_id, self.manager1.id)

        # Content status should sync to APPROVED
        refreshed_content = self.db.query(Content).filter(Content.id == self.content1.id).first()
        self.assertEqual(refreshed_content.status, ContentStatus.APPROVED)

    def test_admin_can_review_and_reject_content(self):
        """Verifies that ADMIN role can reject content within their organization."""
        payload = ApprovalReview(status=ApprovalStatus.REJECTED, comment="Revising angle needed")
        result = ApprovalService.review_approval(
            db=self.db,
            approval_id=self.approval1.id,
            reviewer=self.admin1,
            payload=payload
        )
        self.assertEqual(result.status, ApprovalStatus.REJECTED)
        self.assertEqual(result.reviewer_id, self.admin1.id)

        refreshed_content = self.db.query(Content).filter(Content.id == self.content1.id).first()
        self.assertEqual(refreshed_content.status, ContentStatus.REJECTED)

    def test_cross_tenant_approval_review_is_rejected_with_403(self):
        """Proves that Manager of Org 2 cannot review an approval belonging to Org 1."""
        payload = ApprovalReview(status=ApprovalStatus.APPROVED, comment="Cross-tenant invasion")
        with self.assertRaises(HTTPException) as ctx:
            ApprovalService.review_approval(
                db=self.db,
                approval_id=self.approval1.id,  # Belongs to Org 1
                reviewer=self.manager2,          # Belongs to Org 2
                payload=payload
            )
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("Cross-tenant approval reviews are strictly prohibited", ctx.exception.detail)

    def test_cross_tenant_approval_request_is_rejected_with_403(self):
        """Proves that a user cannot submit an approval request for another tenant's content."""
        payload = ApprovalCreate(content_id=self.content1.id, comment="Foreign request")
        with self.assertRaises(HTTPException) as ctx:
            ApprovalService.request_approval(
                db=self.db,
                requester=self.manager2,  # Org 2 user requesting Org 1 content!
                payload=payload
            )
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("Cross-tenant approval requests are strictly prohibited", ctx.exception.detail)

    # =========================================================================
    # FIX 5: Subscription Payment Security Tests
    # =========================================================================
    def test_unverified_paid_subscription_upgrade_is_blocked(self):
        """Proves that direct unverified requests to upgrade to paid tiers are blocked."""
        with self.assertRaises(HTTPException) as ctx:
            SubscriptionService.upgrade_tier(
                db=self.db,
                organization_id=self.org1.id,
                new_tier=SubscriptionTier.ENTERPRISE,
                verified_payment=False
            )
        # Should be 501 (when Stripe unconfigured) or 400 (when Stripe configured)
        self.assertIn(ctx.exception.status_code, [400, 501])
        self.assertTrue(
            "disabled" in ctx.exception.detail.lower() or "prohibited" in ctx.exception.detail.lower()
        )

    def test_free_tier_downgrade_is_permitted_without_payment(self):
        """Verifies that downgrading/resetting to FREE tier is permitted."""
        sub = SubscriptionService.upgrade_tier(
            db=self.db,
            organization_id=self.org1.id,
            new_tier=SubscriptionTier.FREE,
            verified_payment=False
        )
        self.assertEqual(sub.tier, SubscriptionTier.FREE)
        self.assertEqual(sub.monthly_generation_limit, 50)

    def test_verified_payment_webhook_successfully_upgrades_tier(self):
        """Verifies that verified payment authorization upgrades the organization tier."""
        sub = SubscriptionService.upgrade_tier(
            db=self.db,
            organization_id=self.org1.id,
            new_tier=SubscriptionTier.PRO,
            verified_payment=True,
            stripe_customer_id="cus_test123",
            stripe_subscription_id="sub_test123"
        )
        self.assertEqual(sub.tier, SubscriptionTier.PRO)
        self.assertEqual(sub.monthly_generation_limit, 2000)
        self.assertEqual(sub.stripe_customer_id, "cus_test123")
        self.assertEqual(sub.stripe_subscription_id, "sub_test123")


if __name__ == "__main__":
    unittest.main()

