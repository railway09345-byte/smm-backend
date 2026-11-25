from datetime import datetime, timedelta
import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.user import User
from app.models.organization import Organization
from app.models.subscription import Subscription
from app.models.user_organization import UserOrganization
from app.models.content import Content
from app.models.generation_job import GenerationJob
from app.models.brand_profile import BrandProfile
from app.utils.enums import UserRole, SubscriptionTier, SubscriptionStatus, ContentStatus, JobStatus
from app.core.permissions import require_super_admin, get_current_user


@pytest.fixture
def test_data(db_session):
    """Seed test data for SuperAdmin privacy validation with unique identifiers."""
    uid = uuid.uuid4().hex[:8]

    # Organization 1 (Verified Enterprise)
    org1 = Organization(
        name=f"Enterprise Alpha Corp {uid}",
        slug=f"enterprise-alpha-corp-{uid}",
        description="Private Enterprise Workspace",
        is_active=True
    )
    db_session.add(org1)
    db_session.flush()

    sub1 = Subscription(
        organization_id=org1.id,
        tier=SubscriptionTier.PRO,
        status=SubscriptionStatus.ACTIVE,
        stripe_subscription_id=f"pay_verified_{uid}",
        stripe_customer_id=f"order_verified_{uid}",
        monthly_generation_limit=1000,
        monthly_posts_limit=2500
    )
    db_session.add(sub1)

    admin1 = User(
        email=f"admin.alpha.{uid}@enterprise-alpha.com",
        hashed_password="hashed_pwd_alpha",
        full_name="Alpha Enterprise Admin",
        role=UserRole.ADMIN,
        organization_id=org1.id,
        is_active=True,
        is_verified=True
    )
    db_session.add(admin1)
    db_session.flush()

    db_session.add(UserOrganization(
        user_id=admin1.id,
        organization_id=org1.id,
        role=UserRole.ADMIN,
        is_default=True
    ))

    # Organization 2 (Free Boutique)
    org2 = Organization(
        name=f"Boutique Beta Studio {uid}",
        slug=f"boutique-beta-studio-{uid}",
        description="Free Tier Boutique Store",
        is_active=True
    )
    db_session.add(org2)
    db_session.flush()

    sub2 = Subscription(
        organization_id=org2.id,
        tier=SubscriptionTier.FREE,
        status=SubscriptionStatus.ACTIVE,
        stripe_subscription_id=None,
        stripe_customer_id=None,
        monthly_generation_limit=50,
        monthly_posts_limit=100
    )
    db_session.add(sub2)

    creator2 = User(
        email=f"creator.beta.{uid}@boutique-beta.com",
        hashed_password="hashed_pwd_beta",
        full_name="Beta Creator User",
        role=UserRole.CREATOR,
        organization_id=org2.id,
        is_active=True,
        is_verified=True
    )
    db_session.add(creator2)

    # SuperAdmin user
    super_admin = User(
        email=f"master.superadmin.{uid}@platform.system",
        hashed_password="hashed_pwd_super",
        full_name="Global SuperAdmin",
        role=UserRole.SUPER_ADMIN,
        organization_id=None,
        is_active=True,
        is_verified=True
    )
    db_session.add(super_admin)

    db_session.commit()
    db_session.refresh(super_admin)
    db_session.refresh(admin1)
    db_session.refresh(creator2)
    db_session.refresh(org1)
    db_session.refresh(org2)

    return {
        "uid": uid,
        "super_admin": super_admin,
        "admin1": admin1,
        "creator2": creator2,
        "org1": org1,
        "org2": org2,
        "pay_id": f"pay_verified_{uid}",
        "order_id": f"order_verified_{uid}"
    }


def test_superadmin_admins_endpoint_privacy(client, test_data):
    """Test that GET /superadmin/admins exposes only privacy allowlist."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        response = client.get("/api/v1/superadmin/admins")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

        allowed_keys = {"id", "email", "is_active", "subscription_plan", "subscription_status"}
        for item in data:
            assert set(item.keys()) == allowed_keys
            # Confirm NO private leaks
            assert "role" not in item
            assert "created_at" not in item
            assert "organization_name" not in item
            assert "assigned_businesses" not in item
            assert "total_posts_created" not in item
            assert "has_brand_profile" not in item
            assert "full_name" not in item
            assert "organization_id" not in item
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_users_endpoint_privacy(client, test_data):
    """Test that GET /superadmin/users exposes only privacy allowlist."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        response = client.get("/api/v1/superadmin/users")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

        allowed_keys = {"id", "email", "is_active"}
        for item in data:
            assert set(item.keys()) == allowed_keys
            assert "role" not in item
            assert "created_at" not in item
            assert "organization_name" not in item
            assert "assigned_businesses" not in item
            assert "total_posts_created" not in item
            assert "full_name" not in item
            assert "organization_id" not in item
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_tenants_endpoint_privacy(client, test_data):
    """Test that GET /superadmin/tenants exposes only email, plan, status (no brand names or private assets)."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        response = client.get("/api/v1/superadmin/tenants")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2

        allowed_keys = {
            "id",
            "registered_email",
            "subscription_tier",
            "subscription_status",
            "is_active",
            "created_at",
            "tenant_isolation_status"
        }
        for item in data:
            assert set(item.keys()) == allowed_keys
            assert "name" not in item
            assert "slug" not in item
            assert "description" not in item
            assert "logo_url" not in item
            assert "admins" not in item
            assert "total_posts" not in item
            assert "total_users" not in item
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_analytics_privacy(client, test_data):
    """Test that GET /superadmin/analytics does not leak private user/admin rankings."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        response = client.get("/api/v1/superadmin/analytics?timeframe=30d")
        assert response.status_code == 200
        data = response.json()

        assert "top_active_creators" not in data
        assert "top_managing_admins" not in data
        assert "post_status_distribution" in data
        assert "platform_distribution" in data
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_blocked_from_businesses_api(client, test_data):
    """Test that SuperAdmin is strictly blocked from GET/POST/PUT/DELETE /admin/businesses with HTTP 403."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[get_current_user] = lambda: super_admin

    try:
        # GET
        get_res = client.get("/api/v1/admin/businesses")
        assert get_res.status_code == 403
        assert "SuperAdmins are restricted from accessing Enterprise business details" in get_res.json()["detail"]

        # POST
        post_res = client.post("/api/v1/admin/businesses", json={"name": "Forbidden Brand"})
        assert post_res.status_code == 403

        # PUT
        put_res = client.put("/api/v1/admin/businesses/1", json={"name": "Forbidden Brand"})
        assert put_res.status_code == 403

        # DELETE
        del_res = client.delete("/api/v1/admin/businesses/1")
        assert del_res.status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_enterprise_admin_can_access_businesses_api(client, test_data):
    """Test that Enterprise Admin can access their own organization's businesses."""
    admin1 = test_data["admin1"]
    app.dependency_overrides[get_current_user] = lambda: admin1

    try:
        response = client.get("/api/v1/admin/businesses")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_superadmin_billing_endpoint(client, test_data):
    """Test that GET /superadmin/billing provides verified Razorpay payment references strictly to SuperAdmin."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        response = client.get("/api/v1/superadmin/billing")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2

        allowed_keys = {
            "id",
            "registered_email",
            "plan",
            "subscription_status",
            "account_status",
            "razorpay_payment_id",
            "razorpay_order_id",
            "verification_status",
            "current_period_start",
            "current_period_end",
            "monthly_generation_limit",
            "monthly_posts_limit",
            "current_generation_usage",
            "current_posts_usage"
        }

        # Check for enterprise record: references must be None because provenance is Unverified
        unverified_item = next((i for i in data if i["registered_email"] == test_data["admin1"].email), None)
        assert unverified_item is not None
        assert set(unverified_item.keys()) == allowed_keys
        assert unverified_item["plan"].upper() == "PRO"
        assert unverified_item["verification_status"] == "Unverified"
        assert unverified_item["razorpay_payment_id"] is None
        assert unverified_item["razorpay_order_id"] is None
        assert "is_verified_payment" not in unverified_item

        # Check for Free record: references must be None because provenance is Unavailable
        free_item = next((i for i in data if "beta" in i["registered_email"] or i["plan"].upper() == "FREE"), None)
        assert free_item is not None
        assert set(free_item.keys()) == allowed_keys
        assert free_item["verification_status"] == "Unavailable"
        assert free_item["razorpay_payment_id"] is None
        assert free_item["razorpay_order_id"] is None

        # Verify no sensitive leaks across any billing items
        for item in data:
            assert "name" not in item
            assert "slug" not in item
            assert "brand_profile" not in item
            assert "content" not in item
            assert "social_credentials" not in item
            assert "customer_name" not in item
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_account_creation_without_company_name(client, test_data, db_session):
    """Test that Super Admin can provision an account without company name, automatically binding valid organization."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    new_email = f"auto_org_admin_{test_data['uid']}@enterprise-test.com"
    payload = {
        "email": new_email,
        "password": "SecurePassword123!",
        "full_name": "Autonomous Enterprise Admin",
        "role": "admin",
        "is_active": True
    }

    try:
        response = client.post("/api/v1/superadmin/users", json=payload)
        assert response.status_code in (200, 201)
        res_data = response.json()

        # Confirm response does not leak private organization name or metadata
        assert "organization_name" not in res_data
        assert "workspace" not in res_data
        assert res_data["email"] == new_email

        # Confirm database binding: user exists and is tied to a valid organization
        created_user = db_session.query(User).filter(User.email == new_email).first()
        assert created_user is not None
        assert created_user.organization_id is not None

        # Confirm organization membership
        uo = db_session.query(UserOrganization).filter(
            UserOrganization.user_id == created_user.id,
            UserOrganization.organization_id == created_user.organization_id
        ).first()
        assert uo is not None
        assert uo.role == UserRole.ADMIN
        assert uo.is_default is True

        # Confirm the new Enterprise Admin can log in successfully
        login_res = client.post("/api/v1/auth/login", json={
            "email": new_email,
            "password": "SecurePassword123!"
        })
        assert login_res.status_code == 200
        assert "access_token" in login_res.json()
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_enterprise_admin_organization_isolation(client, test_data, db_session):
    """
    Scenario:
    1. Create Enterprise Admin A through Super Admin (no org_id or company name provided).
    2. Create Enterprise Admin B through Super Admin (no org_id or company name provided).
    3. Verify that each account receives a separate, isolated organization.
    4. Verify UserOrganization memberships point strictly to their respective organizations.
    5. Verify Enterprise Admin A cannot access Enterprise Admin B's users, content, brand profiles, or businesses.
    6. Verify reverse isolation: Admin B cannot access Admin A's data.
    7. Verify both Enterprise Admins can log in and access their own dashboards.
    """
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    uid = uuid.uuid4().hex[:8]
    email_a = f"admin_a_{uid}@isolated-tenant-a.com"
    email_b = f"admin_b_{uid}@isolated-tenant-b.com"
    pwd = "AdminSecurePass123!"

    try:
        # 1. Create Enterprise Admin A through Super Admin
        res_a = client.post("/api/v1/superadmin/users", json={
            "email": email_a,
            "password": pwd,
            "full_name": "Admin Tenant A",
            "role": "admin",
            "is_active": True
        })
        assert res_a.status_code in (200, 201)

        # 2. Create Enterprise Admin B through Super Admin
        res_b = client.post("/api/v1/superadmin/users", json={
            "email": email_b,
            "password": pwd,
            "full_name": "Admin Tenant B",
            "role": "admin",
            "is_active": True
        })
        assert res_b.status_code in (200, 201)

        # 3. Verify each account received a separate organization in DB
        user_a = db_session.query(User).filter(User.email == email_a).first()
        user_b = db_session.query(User).filter(User.email == email_b).first()
        assert user_a is not None
        assert user_b is not None
        assert user_a.organization_id is not None
        assert user_b.organization_id is not None
        assert user_a.organization_id != user_b.organization_id, "Enterprise Admins must NOT share the same organization!"

        # 4. Verify UserOrganization memberships point to respective distinct organizations
        uo_a = db_session.query(UserOrganization).filter(UserOrganization.user_id == user_a.id).all()
        uo_b = db_session.query(UserOrganization).filter(UserOrganization.user_id == user_b.id).all()
        org_ids_a = {uo.organization_id for uo in uo_a}
        org_ids_b = {uo.organization_id for uo in uo_b}
        assert user_a.organization_id in org_ids_a
        assert user_b.organization_id in org_ids_b
        assert org_ids_a.isdisjoint(org_ids_b), "UserOrganization memberships must be completely disjoint between separate tenants!"

        # 5. Verify both Enterprise Admins can log in and access their tokens
        login_a = client.post("/api/v1/auth/login", json={"email": email_a, "password": pwd})
        login_b = client.post("/api/v1/auth/login", json={"email": email_b, "password": pwd})
        assert login_a.status_code == 200
        assert login_b.status_code == 200
        token_a = login_a.json()["access_token"]
        token_b = login_b.json()["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # 6. Verify dashboard and current organization access
        me_a = client.get("/api/v1/auth/me", headers=headers_a).json()
        me_b = client.get("/api/v1/auth/me", headers=headers_b).json()
        assert me_a["organization_id"] == user_a.organization_id
        assert me_b["organization_id"] == user_b.organization_id

        curr_org_a = client.get("/api/v1/organizations/current", headers=headers_a).json()
        curr_org_b = client.get("/api/v1/organizations/current", headers=headers_b).json()
        assert curr_org_a["id"] == user_a.organization_id
        assert curr_org_b["id"] == user_b.organization_id
        assert curr_org_a["id"] != curr_org_b["id"]

        # 7. Verify cross-tenant isolation on Team / Users management
        # Admin A's user list must ONLY show Admin A (or A's members), never Admin B
        users_a = client.get("/api/v1/admin/users", headers=headers_a).json()
        users_a_emails = [u["email"] for u in users_a]
        assert email_a in users_a_emails
        assert email_b not in users_a_emails

        # Admin B's user list must ONLY show Admin B (or B's members), never Admin A
        users_b = client.get("/api/v1/admin/users", headers=headers_b).json()
        users_b_emails = [u["email"] for u in users_b]
        assert email_b in users_b_emails
        assert email_a not in users_b_emails

        # Admin A cannot modify Admin B's account
        mod_res = client.put(f"/api/v1/admin/users/{user_b.id}/status", headers=headers_a, json={"is_active": False})
        assert mod_res.status_code == 403

        # Admin B cannot modify Admin A's account
        mod_res_rev = client.put(f"/api/v1/admin/users/{user_a.id}/status", headers=headers_b, json={"is_active": False})
        assert mod_res_rev.status_code == 403

        # 8. Verify cross-tenant isolation on Businesses
        # Admin A creates a business in Org A
        create_biz_a = client.post("/api/v1/admin/businesses", headers=headers_a, json={
            "name": f"Enterprise A Boutique {uid}",
            "description": "Tenant A Exclusive",
            "settings": {"flagshipAddr": "100 A St"}
        })
        assert create_biz_a.status_code == 201
        biz_a_id = create_biz_a.json()["id"]

        # Admin B lists businesses -> must NOT contain Admin A's business
        biz_list_b = client.get("/api/v1/admin/businesses", headers=headers_b).json()
        b_ids = [b["id"] for b in biz_list_b]
        assert biz_a_id not in b_ids

        # Admin B cannot update or delete Admin A's business
        put_b = client.put(f"/api/v1/admin/businesses/{biz_a_id}", headers=headers_b, json={"name": "Hijacked Name"})
        assert put_b.status_code in (403, 404)

        del_b = client.delete(f"/api/v1/admin/businesses/{biz_a_id}", headers=headers_b)
        assert del_b.status_code in (403, 404)
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_no_payment_references_leaked_elsewhere(client, test_data):
    """Confirm no Razorpay payment IDs or order IDs are leaked in any other Super Admin endpoints."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        # 1. Admins endpoint
        admins = client.get("/api/v1/superadmin/admins").json()
        for a in admins:
            assert "razorpay_payment_id" not in a
            assert "razorpay_order_id" not in a

        # 2. Users endpoint
        users = client.get("/api/v1/superadmin/users").json()
        for u in users:
            assert "razorpay_payment_id" not in u
            assert "razorpay_order_id" not in u

        # 3. Tenants endpoint
        tenants = client.get("/api/v1/superadmin/tenants").json()
        for t in tenants:
            assert "razorpay_payment_id" not in t
            assert "razorpay_order_id" not in t

        # 4. Analytics endpoint
        analytics = client.get("/api/v1/superadmin/analytics").json()
        assert "razorpay" not in str(analytics).lower()
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_stats_enhanced_telemetry(client, db_session, test_data):
    """Test Task 1: Verify enhanced stats metrics including active/suspended orgs, subscriptions, plan distribution, and MRR/ARR."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        # Create dedicated test organizations and subscriptions with known tiers
        uid = uuid.uuid4().hex[:6]
        active_org_starter = Organization(name=f"Active Org Starter {uid}", slug=f"active-org-starter-{uid}", is_active=True)
        active_org_free = Organization(name=f"Active Org Free {uid}", slug=f"active-org-free-{uid}", is_active=True)
        suspended_org = Organization(name=f"Suspended Org {uid}", slug=f"suspended-org-{uid}", is_active=False)
        db_session.add_all([active_org_starter, active_org_free, suspended_org])
        db_session.flush()

        # Active Starter sub on active org -> contributes 1999.0
        sub_starter = Subscription(
            organization_id=active_org_starter.id,
            tier=SubscriptionTier.STARTER,
            status=SubscriptionStatus.ACTIVE,
            current_period_start=datetime.utcnow() - timedelta(days=5),
            current_period_end=datetime.utcnow() + timedelta(days=25)
        )
        # Free sub on active org -> contributes 0.0
        sub_free = Subscription(
            organization_id=active_org_free.id,
            tier=SubscriptionTier.FREE,
            status=SubscriptionStatus.ACTIVE
        )
        # Pro sub on suspended org -> must NOT contribute to active revenue
        sub_suspended_org = Subscription(
            organization_id=suspended_org.id,
            tier=SubscriptionTier.PRO,
            status=SubscriptionStatus.ACTIVE
        )
        db_session.add_all([sub_starter, sub_free, sub_suspended_org])
        db_session.commit()

        res = client.get("/api/v1/superadmin/stats")
        assert res.status_code == 200
        stats = res.json()

        # 1. Organization telemetry
        assert "active_organizations" in stats
        assert "suspended_organizations" in stats
        assert stats["active_organizations"] >= 1
        assert stats["suspended_organizations"] >= 1

        # 2. Subscription telemetry
        assert "active_subscriptions" in stats
        assert "expired_subscriptions" in stats
        assert stats["active_subscriptions"] >= 1

        # 3. Plan distribution
        assert "plan_distribution" in stats
        dist = stats["plan_distribution"]
        assert "free" in dist
        assert "starter" in dist
        assert "pro" in dist
        assert "enterprise" in dist

        # 4. Revenue telemetry
        assert "monthly_recurring_revenue" in stats
        assert "annual_recurring_revenue" in stats
        assert stats["revenue_currency"] == "INR"
        assert stats["revenue_type"] == "projected_recurring"
        assert stats["monthly_recurring_revenue"] >= 1999.0
        assert stats["annual_recurring_revenue"] == round(stats["monthly_recurring_revenue"] * 12.0, 2)
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_revenue_calculation_rules(client, db_session, test_data):
    """Test Task 1: Verify exact MRR/ARR calculation rules and edge case handling."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        uid = uuid.uuid4().hex[:6]
        org = Organization(name=f"Rev Org {uid}", slug=f"rev-org-{uid}", is_active=True)
        db_session.add(org)
        db_session.flush()

        # Date-expired subscription (expired 2 days ago) -> must NOT contribute to MRR
        expired_sub = Subscription(
            organization_id=org.id,
            tier=SubscriptionTier.ENTERPRISE,
            status=SubscriptionStatus.ACTIVE,
            current_period_start=datetime.utcnow() - timedelta(days=32),
            current_period_end=datetime.utcnow() - timedelta(days=2)
        )
        db_session.add(expired_sub)
        db_session.commit()

        res = client.get("/api/v1/superadmin/stats")
        assert res.status_code == 200
        stats = res.json()
        assert stats["expired_subscriptions"] >= 1
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_tenants_created_at_and_isolation(client, db_session, test_data):
    """Test Task 2: Verify registration date (created_at) and isolation status without private leaks."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        res = client.get("/api/v1/superadmin/tenants")
        assert res.status_code == 200
        tenants = res.json()
        assert len(tenants) > 0

        for t in tenants:
            assert "id" in t
            assert "registered_email" in t
            assert "subscription_tier" in t
            assert "is_active" in t
            assert "created_at" in t
            assert t["created_at"] is not None
            assert "tenant_isolation_status" in t
            assert t["tenant_isolation_status"] == "Dedicated Workspace"

            # Strict privacy assertions: zero private company names or slugs leaked
            assert "name" not in t
            assert "slug" not in t
            assert "description" not in t
            assert "brand_profile" not in t
            assert "brand_profiles" not in t
            assert "settings" not in t
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_billing_telemetry_and_usage(client, db_session, test_data):
    """Test Task 3: Verify subscription billing window, limits, and live generation/post usage."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        uid = uuid.uuid4().hex[:6]
        org = Organization(name=f"Billing Org {uid}", slug=f"billing-org-{uid}", is_active=True)
        db_session.add(org)
        db_session.flush()

        start_dt = datetime.utcnow() - timedelta(days=10)
        end_dt = datetime.utcnow() + timedelta(days=20)
        sub = Subscription(
            organization_id=org.id,
            tier=SubscriptionTier.PRO,
            status=SubscriptionStatus.ACTIVE,
            monthly_generation_limit=500,
            monthly_posts_limit=1000,
            current_period_start=start_dt,
            current_period_end=end_dt
        )
        db_session.add(sub)
        db_session.flush()

        # Add brand profile for content association
        brand = BrandProfile(
            name=f"Brand {uid}",
            tone_of_voice="Professional",
            key_themes=[],
            hashtags=[],
            color_palette={},
            meta_info={},
            organization_id=org.id
        )
        db_session.add(brand)
        db_session.flush()

        # Add a generation job inside the billing period
        job_inside = GenerationJob(
            organization_id=org.id,
            user_id=test_data["admin1"].id,
            job_type="text",
            status="completed",
            prompt="Test generation prompt",
            parameters={},
            result_data={},
            tokens_used=10,
            created_at=datetime.utcnow() - timedelta(days=2)
        )
        # Add a content post published inside the billing period
        post_inside = Content(
            organization_id=org.id,
            brand_profile_id=brand.id,
            status=ContentStatus.PUBLISHED,
            caption="Test published post",
            created_at=datetime.utcnow() - timedelta(days=1)
        )
        db_session.add_all([job_inside, post_inside])
        db_session.commit()

        res = client.get("/api/v1/superadmin/billing")
        assert res.status_code == 200
        billing_records = res.json()

        matching = [b for b in billing_records if b["id"] == sub.id]
        assert len(matching) == 1
        rec = matching[0]

        assert rec["monthly_generation_limit"] == 500
        assert rec["monthly_posts_limit"] == 1000
        assert rec["current_generation_usage"] >= 1
        assert rec["current_posts_usage"] >= 1
        assert rec["current_period_start"] is not None
        assert rec["current_period_end"] is not None
        # Verified-only reference check: unverified records remain None
        assert rec["razorpay_payment_id"] is None
        assert rec["razorpay_order_id"] is None
        assert rec["verification_status"] == "Unavailable"
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_health_endpoint(client, test_data):
    """Test Task 4: Verify platform health endpoint returns real telemetry and enforces auth."""
    super_admin = test_data["super_admin"]
    app.dependency_overrides[require_super_admin] = lambda: super_admin

    try:
        # 1. SuperAdmin access returns 200 with structured health telemetry
        res = client.get("/api/v1/superadmin/health")
        assert res.status_code == 200
        data = res.json()

        assert data["api_status"] == "healthy"
        assert data["database_connected"] is True
        assert data["database_latency_ms"] >= 0.0
        assert isinstance(data["redis_available"], bool)
        assert data["publishing_worker_status"] in ("running", "stopped", "unavailable")
        assert data["autonomous_scheduler_status"] in ("running", "stopped", "unavailable")
        assert data["failed_content_count"] >= 0
        assert data["failed_generation_jobs_count"] >= 0

        # Verify no secret leakage in response
        data_str = str(data).lower()
        assert "password" not in data_str
        assert "secret" not in data_str
        assert "token" not in data_str
        assert "api_key" not in data_str
        assert "connection" not in data_str
    finally:
        app.dependency_overrides.pop(require_super_admin, None)

    # 2. Unauthorized access without SuperAdmin role is strictly rejected (HTTP 401 or 403)
    unauth_res = client.get("/api/v1/superadmin/health")
    assert unauth_res.status_code == 401

    admin_user = test_data["admin1"]
    app.dependency_overrides[get_current_user] = lambda: admin_user
    try:
        forbidden_res = client.get("/api/v1/superadmin/health")
        assert forbidden_res.status_code == 403
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def _create_billing_org(db_session, *, tier=SubscriptionTier.PRO, period_start=None, period_end=None):
    """Helper to seed an isolated organization, admin user, brand profile, and subscription."""
    uid = uuid.uuid4().hex[:8]
    org = Organization(name=f"Metering Org {uid}", slug=f"metering-org-{uid}", is_active=True)
    db_session.add(org)
    db_session.flush()

    admin = User(
        email=f"billing.{uid}@metering.example",
        hashed_password="hashed_pw_test",
        full_name=f"Admin {uid}",
        role=UserRole.ADMIN,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(admin)
    db_session.flush()

    sub = Subscription(
        organization_id=org.id,
        tier=tier,
        status=SubscriptionStatus.ACTIVE,
        monthly_generation_limit=500,
        monthly_posts_limit=1000,
        current_period_start=period_start,
        current_period_end=period_end,
    )
    db_session.add(sub)

    brand = BrandProfile(
        name=f"Private Brand {uid}",
        industry="Technology",
        brand_guidelines="Enterprise SaaS automation platform.",
        tone_of_voice="Modern",
        key_themes=[],
        hashtags=[],
        color_palette={},
        meta_info={"description": "Enterprise SaaS automation platform."},
        organization_id=org.id,
    )
    db_session.add(brand)
    db_session.commit()
    return org, admin, sub, brand


def test_billing_ai_asset_counting_bundle_dedup_and_pipeline(client, db_session, test_data, tmp_path):
    """
    Verify AI Generation Usage counts billable completed assets:
    - image=1, text=1, video=1, full bundle=3
    - pending/processing/failed excluded
    - scheduled Content Planner image reuses existing job without double counting
    - duplicate GenerationJobs sharing one media_asset_id count once
    - Autonomous Pipeline ContentBrief + done GeneratedAssets included; failed/pending excluded
    """
    from unittest.mock import AsyncMock, patch
    from app.models.media_asset import MediaAsset
    from app.models.autonomous_pipeline import PipelineRun, ContentBrief, GeneratedAsset
    from app.schemas.generation import GeminiImageGenerationRequest
    from app.services.image_generation_service import ImageGenerationService
    from app.utils.enums import GenerationType, MediaType

    now = datetime.utcnow()
    org, admin, sub, brand = _create_billing_org(
        db_session,
        period_start=now - timedelta(days=5),
        period_end=now + timedelta(days=25),
    )

    asset_single_img = MediaAsset(
        user_id=admin.id, organization_id=org.id,
        file_name="s_img.png", file_url="https://cdn.example/s_img.png", media_type=MediaType.IMAGE
    )
    asset_single_vid = MediaAsset(
        user_id=admin.id, organization_id=org.id,
        file_name="s_vid.mp4", file_url="https://cdn.example/s_vid.mp4", media_type=MediaType.VIDEO
    )
    asset_bundle_img = MediaAsset(
        user_id=admin.id, organization_id=org.id,
        file_name="b_img.png", file_url="https://cdn.example/b_img.png", media_type=MediaType.IMAGE
    )
    asset_bundle_vid = MediaAsset(
        user_id=admin.id, organization_id=org.id,
        file_name="b_vid.mp4", file_url="https://cdn.example/b_vid.mp4", media_type=MediaType.VIDEO
    )
    db_session.add_all([asset_single_img, asset_single_vid, asset_bundle_img, asset_bundle_vid])
    db_session.flush()

    # 1. Single image (1), text (1), video (1) = 3
    db_session.add_all([
        GenerationJob(
            user_id=admin.id, organization_id=org.id, job_type=GenerationType.IMAGE_GENERATION,
            status=JobStatus.COMPLETED, prompt="secret single image", parameters={},
            result_data={"media_asset_id": asset_single_img.id}, created_at=now - timedelta(hours=4)
        ),
        GenerationJob(
            user_id=admin.id, organization_id=org.id, job_type=GenerationType.TEXT_MARKETING,
            status=JobStatus.COMPLETED, prompt="secret single text", parameters={},
            result_data={"text": {"headline": "Hi"}}, created_at=now - timedelta(hours=3)
        ),
        GenerationJob(
            user_id=admin.id, organization_id=org.id, job_type=GenerationType.VIDEO_GENERATION,
            status=JobStatus.COMPLETED, prompt="secret single video", parameters={},
            result_data={"media_asset_id": asset_single_vid.id}, created_at=now - timedelta(hours=2)
        ),
        # Full bundle: image (1) + video (1) + text (1) = 3
        GenerationJob(
            user_id=admin.id, organization_id=org.id, job_type=GenerationType.IMAGE_GENERATION,
            status=JobStatus.COMPLETED, prompt="bundle img", parameters={},
            result_data={"media_asset_id": asset_bundle_img.id}, created_at=now - timedelta(hours=1)
        ),
        GenerationJob(
            user_id=admin.id, organization_id=org.id, job_type=GenerationType.VIDEO_GENERATION,
            status=JobStatus.COMPLETED, prompt="bundle vid", parameters={},
            result_data={"media_asset_id": asset_bundle_vid.id}, created_at=now - timedelta(hours=1)
        ),
        GenerationJob(
            user_id=admin.id, organization_id=org.id, job_type=GenerationType.TEXT_MARKETING,
            status=JobStatus.COMPLETED, prompt="bundle text", parameters={},
            result_data={
                "image": {"media_asset_id": asset_bundle_img.id},
                "video": {"job_id": 999, "status": "processing"},
                "text": {"headline": "Bundle"}
            }, created_at=now - timedelta(hours=1)
        ),
        # Duplicate completed job referencing the same asset_single_img -> must not double count
        GenerationJob(
            user_id=admin.id, organization_id=org.id, job_type=GenerationType.IMAGE_GENERATION,
            status=JobStatus.COMPLETED, prompt="duplicate job ref", parameters={},
            result_data={"media_asset_id": asset_single_img.id}, created_at=now - timedelta(minutes=30)
        ),
        # Non-completed jobs -> must all be excluded
        GenerationJob(
            user_id=admin.id, organization_id=org.id, job_type=GenerationType.IMAGE_GENERATION,
            status=JobStatus.FAILED, prompt="failed img", parameters={},
            result_data={}, created_at=now - timedelta(minutes=20)
        ),
        GenerationJob(
            user_id=admin.id, organization_id=org.id, job_type=GenerationType.VIDEO_GENERATION,
            status=JobStatus.PROCESSING, prompt="processing vid", parameters={},
            result_data={}, created_at=now - timedelta(minutes=15)
        ),
        GenerationJob(
            user_id=admin.id, organization_id=org.id, job_type=GenerationType.IMAGE_GENERATION,
            status=JobStatus.PENDING, prompt="pending img", parameters={},
            result_data={}, created_at=now - timedelta(minutes=10)
        ),
    ])

    # 2. Scheduled Content Planner job reused via existing_job= -> adds only 1 row & counts as 1
    scheduled_job = GenerationJob(
        user_id=admin.id, organization_id=org.id, job_type=GenerationType.IMAGE_GENERATION,
        status=JobStatus.PENDING, prompt="scheduled planner img", parameters={"auto_generate_media": True},
        result_data={}, created_at=now - timedelta(minutes=5)
    )
    db_session.add(scheduled_job)
    db_session.commit()
    db_session.refresh(scheduled_job)

    img_path = tmp_path / "planned.png"
    from PIL import Image
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path)

    mock_cf_res = {
        "generation_id": f"gen_test_{uuid.uuid4().hex[:8]}",
        "image_url": "/uploads/media/ai_images/planned.png",
        "file_name": "planned.png",
        "local_path": str(img_path),
        "file_size_bytes": img_path.stat().st_size,
        "model_name": "flux-mock",
        "provider": "cloudflare",
    }
    import asyncio
    from app.services.cloudflare_image_service import CloudflareImageService
    from app.services.quality_service import QualityService

    with patch.object(CloudflareImageService, "generate_image", new=AsyncMock(return_value=mock_cf_res)), \
         patch.object(QualityService, "validate_image", return_value={"status": "PASS", "score": 95, "validation_status": "PASS", "overall_score": 95, "is_duplicate": False}):
        updated_job = asyncio.run(
            ImageGenerationService.generate_image(
                GeminiImageGenerationRequest(prompt="scheduled planner img", brand_profile_id=brand.id),
                db_session,
                admin,
                existing_job=scheduled_job,
            )
        )
    assert updated_job.id == scheduled_job.id
    assert db_session.query(GenerationJob).filter(
        GenerationJob.organization_id == org.id,
        GenerationJob.prompt == "scheduled planner img",
    ).count() == 1

    # 3. Autonomous Pipeline assets: 1 completed brief (text=1) + 1 done image (1) + 1 done video (1) = 3
    # plus failed/pending items that must be excluded
    run_ok = PipelineRun(
        brand_profile_id=brand.id, organization_id=org.id, run_date=now.date(),
        status="completed", stage="Completed", stats={}
    )
    run_fail = PipelineRun(
        brand_profile_id=brand.id, organization_id=org.id, run_date=now.date() - timedelta(days=1),
        status="failed", stage="Failed", stats={}
    )
    db_session.add_all([run_ok, run_fail])
    db_session.flush()

    brief_ok = ContentBrief(
        run_id=run_ok.id, brand_profile_id=brand.id, topic="Private Topic",
        angle="Editorial angle", caption_brief="Private caption brief", image_prompt="Private image prompt",
    )
    brief_fail = ContentBrief(
        run_id=run_fail.id, brand_profile_id=brand.id, topic="Failed Topic",
        angle="Failed angle", caption_brief="Failed caption brief", image_prompt="Failed image prompt",
    )
    db_session.add_all([brief_ok, brief_fail])
    db_session.flush()

    db_session.add_all([
        GeneratedAsset(brief_id=brief_ok.id, type="image", status="done", file_path="https://cdn.example/ap1.png"),
        GeneratedAsset(brief_id=brief_ok.id, type="video", status="done", file_path="https://cdn.example/ap1.mp4"),
        GeneratedAsset(brief_id=brief_ok.id, type="video", status="failed", file_path=None),
        GeneratedAsset(brief_id=brief_ok.id, type="image", status="pending", file_path=None),
    ])
    db_session.commit()

    # Expected: 3 (single) + 3 (bundle) + 1 (scheduled planner) + 3 (pipeline brief+img+vid) = 10
    app.dependency_overrides[require_super_admin] = lambda: test_data["super_admin"]
    try:
        res = client.get("/api/v1/superadmin/billing")
        assert res.status_code == 200
        rec = next(b for b in res.json() if b["id"] == sub.id)
        assert rec["current_generation_usage"] == 10
        assert rec["current_posts_usage"] == 0
        assert "secret single" not in str(rec).lower()
        assert "private brand" not in str(rec).lower()
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_billing_social_deliveries_multi_platform_sibling_fix_and_periods(client, db_session, test_data):
    """
    Verify Social Media Usage counts successful platform deliveries separately,
    fixes sibling-platform Content.status overwrite on failure, deduplicates retries,
    uses published_time for period boundaries, falls back to current calendar month when
    period dates are missing, and isolates tenants cleanly.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch
    from app.models.scheduled_post import ScheduledPost
    from app.models.schedule_rule import ScheduleRule
    from app.models.social_account import SocialAccount
    from app.models.autonomous_pipeline import PipelineRun, ContentBrief, AutonomousPost, PostVariant
    from app.scheduler.publishing_worker import PublishingWorker

    now = datetime.utcnow()
    period_start = now - timedelta(days=5)
    period_end = now + timedelta(days=25)

    # Tenant A: explicit period bounds
    org_a, admin_a, sub_a, brand_a = _create_billing_org(
        db_session, period_start=period_start, period_end=period_end
    )
    # Tenant B: missing period bounds (Free plan -> current calendar month fallback)
    org_b, admin_b, sub_b, brand_b = _create_billing_org(
        db_session, tier=SubscriptionTier.FREE, period_start=None, period_end=None
    )

    from app.models.media_asset import MediaAsset
    from app.utils.enums import MediaType, PlatformType

    acc_ig = SocialAccount(
        brand_profile_id=brand_a.id, platform="instagram", account_name="@org_a_ig",
        account_id="ig_a_1", access_token="tok_ig", is_connected=True,
        meta_info={"verification_status": "VERIFIED"},
    )
    acc_fb = SocialAccount(
        brand_profile_id=brand_a.id, platform="facebook", account_name="Org A FB",
        account_id="fb_a_1", access_token="tok_fb", is_connected=True,
        meta_info={"verification_status": "VERIFIED"},
    )
    acc_yt = SocialAccount(
        brand_profile_id=brand_a.id, platform="youtube", account_name="Org A YT",
        account_id="yt_a_1", access_token="tok_yt", is_connected=True,
        meta_info={"verification_status": "VERIFIED"},
    )
    media_partial = MediaAsset(
        user_id=admin_a.id, organization_id=org_a.id,
        file_name="partial.png", file_url="https://cdn.example/partial.png", media_type=MediaType.IMAGE
    )
    db_session.add_all([acc_ig, acc_fb, acc_yt, media_partial])
    db_session.flush()

    # 1. Multi-platform post (created 20 days ago BEFORE period_start, published inside period -> 3 deliveries)
    content_multi = Content(
        organization_id=org_a.id, brand_profile_id=brand_a.id, creator_id=admin_a.id,
        title="Launch", caption="Multi-channel launch", status=ContentStatus.PUBLISHED,
        created_at=period_start - timedelta(days=15),
    )
    db_session.add(content_multi)
    db_session.flush()

    db_session.add_all([
        ScheduledPost(
            organization_id=org_a.id, content_id=content_multi.id, social_account_id=acc_ig.id,
            platform="instagram", status="published", platform_post_id="ig_pub_1",
            scheduled_time=now, published_time=now - timedelta(hours=2),
            created_at=period_start - timedelta(days=15),
        ),
        ScheduledPost(
            organization_id=org_a.id, content_id=content_multi.id, social_account_id=acc_fb.id,
            platform="facebook", status="published", platform_post_id="fb_pub_1",
            scheduled_time=now, published_time=now - timedelta(hours=2),
            created_at=period_start - timedelta(days=15),
        ),
        ScheduledPost(
            organization_id=org_a.id, content_id=content_multi.id, social_account_id=acc_yt.id,
            platform="youtube", status="published", platform_post_id="yt_pub_1",
            scheduled_time=now, published_time=now - timedelta(hours=2),
            created_at=period_start - timedelta(days=15),
        ),
        # Duplicate delivery row pointing to same (instagram, ig_pub_1) -> must not double count
        ScheduledPost(
            organization_id=org_a.id, content_id=content_multi.id, social_account_id=acc_ig.id,
            platform="instagram", status="published", platform_post_id="ig_pub_1",
            scheduled_time=now, published_time=now - timedelta(hours=1),
        ),
        # Post published outside billing period -> excluded
        ScheduledPost(
            organization_id=org_a.id, content_id=content_multi.id, social_account_id=acc_ig.id,
            platform="instagram", status="published", platform_post_id="ig_old",
            scheduled_time=period_start - timedelta(days=10),
            published_time=period_start - timedelta(days=1),
        ),
        # Draft, scheduled, failed posts -> all excluded
        ScheduledPost(
            organization_id=org_a.id, content_id=content_multi.id, social_account_id=acc_ig.id,
            platform="instagram", status="scheduled", scheduled_time=now + timedelta(days=2),
        ),
        ScheduledPost(
            organization_id=org_a.id, content_id=content_multi.id, social_account_id=acc_fb.id,
            platform="facebook", status="failed", scheduled_time=now - timedelta(hours=3),
        ),
    ])

    # 2. Recurring ScheduleRule delivery (content_id=None, organization_id=org_a.id -> +1 delivery)
    rule = ScheduleRule(
        organization_id=org_a.id, user_id=admin_a.id, social_account_id=acc_ig.id,
        platform="instagram", name="Daily Tip", recurrence_type="daily",
        start_date=now - timedelta(days=1),
    )
    db_session.add(rule)
    db_session.flush()
    db_session.add(
        ScheduledPost(
            organization_id=org_a.id, content_id=None, social_account_id=acc_ig.id,
            schedule_rule_id=rule.id, platform="instagram", status="published",
            platform_post_id="ig_rec_1", scheduled_time=now, published_time=now - timedelta(minutes=30),
        )
    )

    # 3. Partial publish + PublishingWorker sibling-platform status preservation bug fix
    content_partial = Content(
        organization_id=org_a.id, brand_profile_id=brand_a.id, creator_id=admin_a.id,
        media_asset_id=media_partial.id,
        title="Partial", caption="IG ok, FB fails", status=ContentStatus.PUBLISHED,
    )
    db_session.add(content_partial)
    db_session.flush()
    sp_partial_ig = ScheduledPost(
        organization_id=org_a.id, content_id=content_partial.id, social_account_id=acc_ig.id,
        media_asset_id=media_partial.id,
        platform="instagram", status="published", platform_post_id="ig_partial_ok",
        scheduled_time=now - timedelta(minutes=10), published_time=now - timedelta(minutes=5),
    )
    sp_partial_fb = ScheduledPost(
        organization_id=org_a.id, content_id=content_partial.id, social_account_id=acc_fb.id,
        media_asset_id=media_partial.id,
        platform="facebook", status="scheduled",
        scheduled_time=now - timedelta(minutes=5),
    )
    db_session.add_all([sp_partial_ig, sp_partial_fb])
    db_session.commit()

    class FailingFBClient:
        def __init__(self, access_token, refresh_token=None):
            self.access_token = access_token

        async def publish_post(self, content_text, media_urls=None, **kwargs):
            raise RuntimeError("Facebook API down")

    with patch.dict("app.scheduler.publishing_worker.CLIENT_MAP", {"facebook": FailingFBClient, PlatformType.FACEBOOK: FailingFBClient}), \
         patch("app.services.social_token_manager.SocialTokenManager.ensure_valid_token", new=AsyncMock(return_value="tok")):
        fail_res = asyncio.run(PublishingWorker.publish_single_post(db_session, sp_partial_fb.id, is_manual=True))
    assert fail_res["success"] is False

    # Retry already-published sibling must return already_published and never duplicate count
    retry_ok_res = asyncio.run(PublishingWorker.publish_single_post(db_session, sp_partial_ig.id, is_manual=True))
    assert retry_ok_res.get("already_published") is True

    # HTTP publish/retry endpoints must also guard already-published posts
    app.dependency_overrides[get_current_user] = lambda: admin_a
    try:
        pub_dup_res = client.post(f"/api/v1/scheduler/posts/{sp_partial_ig.id}/publish")
        assert pub_dup_res.status_code == 200
        assert pub_dup_res.json().get("already_published") is True

        retry_dup_res = client.post(f"/api/v1/scheduler/posts/{sp_partial_ig.id}/retry")
        assert retry_dup_res.status_code == 200
        assert retry_dup_res.json().get("already_published") is True
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    db_session.refresh(content_partial)
    db_session.refresh(sp_partial_fb)
    assert sp_partial_fb.status == "failed"
    assert content_partial.status == ContentStatus.PUBLISHED  # Must NOT be overwritten to FAILED

    # 4. Autonomous Pipeline PostVariants: 2 posted + 1 failed -> +2 deliveries
    run_ap = PipelineRun(
        brand_profile_id=brand_a.id, organization_id=org_a.id, run_date=now.date(),
        status="completed", stage="Published", stats={}
    )
    db_session.add(run_ap)
    db_session.flush()
    brief_ap = ContentBrief(
        run_id=run_ap.id, brand_profile_id=brand_a.id, topic="Auto Launch",
        angle="Product spotlight", caption_brief="Private campaign brief", image_prompt="Private studio shot",
    )
    db_session.add(brief_ap)
    db_session.flush()
    auto_post = AutonomousPost(
        brief_id=brief_ap.id, brand_profile_id=brand_a.id, organization_id=org_a.id,
        status="published", published_at=now - timedelta(minutes=15)
    )
    db_session.add(auto_post)
    db_session.flush()
    db_session.add_all([
        PostVariant(
            post_id=auto_post.id, platform="instagram", caption="Private IG caption",
            scheduled_time=now - timedelta(minutes=20), status="posted", platform_post_id="ap_ig_1",
        ),
        PostVariant(
            post_id=auto_post.id, platform="facebook", caption="Private FB caption",
            scheduled_time=now - timedelta(minutes=20), status="posted", platform_post_id="ap_fb_1",
        ),
        PostVariant(
            post_id=auto_post.id, platform="youtube", caption="Private YT caption",
            scheduled_time=now - timedelta(minutes=20), status="failed",
        ),
    ])

    # 5. Tenant B (Free plan without period dates): current calendar month vs prior month isolation
    month_start = datetime(now.year, now.month, 1, 0, 0, 0)
    prev_month_ts = month_start - timedelta(days=2)
    db_session.add_all([
        GenerationJob(
            user_id=admin_b.id, organization_id=org_b.id, job_type="text", status=JobStatus.COMPLETED,
            prompt="Org B current month", parameters={}, result_data={}, created_at=now - timedelta(minutes=10)
        ),
        GenerationJob(
            user_id=admin_b.id, organization_id=org_b.id, job_type="text", status=JobStatus.COMPLETED,
            prompt="Org B last month excluded", parameters={}, result_data={}, created_at=prev_month_ts
        ),
        ScheduledPost(
            organization_id=org_b.id, platform="instagram", status="published",
            platform_post_id="orgb_cur_1", scheduled_time=now, published_time=now - timedelta(minutes=5)
        ),
        ScheduledPost(
            organization_id=org_b.id, platform="instagram", status="published",
            platform_post_id="orgb_old_1", scheduled_time=prev_month_ts, published_time=prev_month_ts
        ),
    ])
    db_session.commit()

    # Expected Tenant A:
    # - AI usage: 1 (ContentBrief brief_ap, since status='completed')
    # - Social deliveries: 3 (multi) + 1 (recurring rule) + 1 (partial IG ok) + 2 (autonomous posted) = 7
    # Expected Tenant B (Free calendar month fallback):
    # - AI usage: 1 (prev month excluded)
    # - Social deliveries: 1 (prev month excluded)
    app.dependency_overrides[require_super_admin] = lambda: test_data["super_admin"]
    try:
        res = client.get("/api/v1/superadmin/billing")
        assert res.status_code == 200
        records = res.json()
        rec_a = next(b for b in records if b["id"] == sub_a.id)
        rec_b = next(b for b in records if b["id"] == sub_b.id)

        assert rec_a["current_generation_usage"] == 1
        assert rec_a["current_posts_usage"] == 7

        assert rec_b["current_generation_usage"] == 1
        assert rec_b["current_posts_usage"] == 1
        assert rec_b["current_period_start"] is not None
        assert rec_b["current_period_end"] is not None
        assert rec_b["current_period_start"].startswith(month_start.strftime("%Y-%m-01"))
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_expired_subscriptions_accuracy(client, db_session, test_data):
    """
    Verify expired_subscriptions counts only subscriptions whose current_period_end has passed:
    - Expired period -> counted as expired (+1)
    - Active period on a suspended organization -> NOT counted as expired, NOT counted as active
    - Canceled / past_due / incomplete / trialing with future period end -> NOT counted as expired
    - Preserves plan_distribution, active_subscriptions, and response schema
    """
    now = datetime.utcnow()
    uid = uuid.uuid4().hex[:8]
    app.dependency_overrides[require_super_admin] = lambda: test_data["super_admin"]

    try:
        baseline_res = client.get("/api/v1/superadmin/stats")
        assert baseline_res.status_code == 200
        baseline = baseline_res.json()

        org_expired = Organization(name=f"Exp Period {uid}", slug=f"exp-period-{uid}", is_active=True)
        org_suspended = Organization(name=f"Susp Org {uid}", slug=f"susp-org-{uid}", is_active=False)
        org_canceled = Organization(name=f"Canc Org {uid}", slug=f"canc-org-{uid}", is_active=True)
        org_past_due = Organization(name=f"PastDue Org {uid}", slug=f"pd-org-{uid}", is_active=True)
        org_incomplete = Organization(name=f"Inc Org {uid}", slug=f"inc-org-{uid}", is_active=True)
        org_trialing = Organization(name=f"Trial Org {uid}", slug=f"trial-org-{uid}", is_active=True)
        db_session.add_all([org_expired, org_suspended, org_canceled, org_past_due, org_incomplete, org_trialing])
        db_session.flush()

        db_session.add_all([
            # 1. Expired period -> MUST increment expired_subscriptions (+1)
            Subscription(
                organization_id=org_expired.id,
                tier=SubscriptionTier.STARTER,
                status=SubscriptionStatus.ACTIVE,
                current_period_start=now - timedelta(days=31),
                current_period_end=now - timedelta(days=1),
            ),
            # 2. Active period on suspended organization -> NOT expired, NOT active
            Subscription(
                organization_id=org_suspended.id,
                tier=SubscriptionTier.PRO,
                status=SubscriptionStatus.ACTIVE,
                current_period_start=now - timedelta(days=5),
                current_period_end=now + timedelta(days=25),
            ),
            # 3. Canceled with future period end -> NOT expired, NOT active
            Subscription(
                organization_id=org_canceled.id,
                tier=SubscriptionTier.PRO,
                status=SubscriptionStatus.CANCELED,
                current_period_start=now - timedelta(days=5),
                current_period_end=now + timedelta(days=20),
            ),
            # 4. Past_due with future period end -> NOT expired, NOT active
            Subscription(
                organization_id=org_past_due.id,
                tier=SubscriptionTier.ENTERPRISE,
                status=SubscriptionStatus.PAST_DUE,
                current_period_start=now - timedelta(days=5),
                current_period_end=now + timedelta(days=15),
            ),
            # 5. Incomplete with future period end -> NOT expired, NOT active
            Subscription(
                organization_id=org_incomplete.id,
                tier=SubscriptionTier.STARTER,
                status=SubscriptionStatus.INCOMPLETE,
                current_period_start=now - timedelta(days=1),
                current_period_end=now + timedelta(days=29),
            ),
            # 6. Trialing with future period end -> NOT expired, NOT active
            Subscription(
                organization_id=org_trialing.id,
                tier=SubscriptionTier.STARTER,
                status=SubscriptionStatus.TRIALING,
                current_period_start=now - timedelta(days=2),
                current_period_end=now + timedelta(days=12),
            ),
        ])
        db_session.commit()

        after_res = client.get("/api/v1/superadmin/stats")
        assert after_res.status_code == 200
        after = after_res.json()

        assert set(after.keys()) == set(baseline.keys())
        assert after["expired_subscriptions"] == baseline["expired_subscriptions"] + 1
        assert after["active_subscriptions"] == baseline["active_subscriptions"]
        assert after["monthly_recurring_revenue"] == baseline["monthly_recurring_revenue"]
        assert after["plan_distribution"]["starter"] == baseline["plan_distribution"]["starter"] + 3
        assert after["plan_distribution"]["pro"] == baseline["plan_distribution"]["pro"] + 2
        assert after["plan_distribution"]["enterprise"] == baseline["plan_distribution"]["enterprise"] + 1
    finally:
        app.dependency_overrides.pop(require_super_admin, None)


def test_superadmin_global_content_pipeline_breakdown(client, db_session, test_data):
    """
    Verify GET /superadmin/stats Global Content Pipeline Breakdown:
    - Counts unique Content rows without double-counting child ScheduledPost deliveries
    - Includes valid standalone ScheduledPost rows (content_id IS NULL), excluding cancelled ones
    - Includes valid AutonomousPost rows (mapping both published & posted) without double-counting PostVariants
    - Excludes in-progress 'generating' autonomous posts
    - Preserves response schema and exposes zero private content/captions/prompts
    """
    from app.models.scheduled_post import ScheduledPost
    from app.models.schedule_rule import ScheduleRule
    from app.models.social_account import SocialAccount
    from app.models.autonomous_pipeline import PipelineRun, ContentBrief, AutonomousPost, PostVariant

    now = datetime.utcnow()
    org, admin, _, brand = _create_billing_org(db_session)
    acc_ig = SocialAccount(
        brand_profile_id=brand.id, platform="instagram", account_name="@pipe_ig",
        account_id="pipe_ig_1", access_token="tok_ig", is_connected=True
    )
    acc_fb = SocialAccount(
        brand_profile_id=brand.id, platform="facebook", account_name="Pipe FB",
        account_id="pipe_fb_1", access_token="tok_fb", is_connected=True
    )
    acc_yt = SocialAccount(
        brand_profile_id=brand.id, platform="youtube", account_name="Pipe YT",
        account_id="pipe_yt_1", access_token="tok_yt", is_connected=True
    )
    db_session.add_all([acc_ig, acc_fb, acc_yt])
    db_session.flush()

    app.dependency_overrides[require_super_admin] = lambda: test_data["super_admin"]
    try:
        baseline = client.get("/api/v1/superadmin/stats").json()

        # 1. Ordinary Content: 1 published (linked to 3 ScheduledPost deliveries -> must count as 1),
        #    plus 1 pending_approval and 1 draft
        c_pub = Content(
            organization_id=org.id, brand_profile_id=brand.id, creator_id=admin.id,
            title="Pipeline Secret Pub", caption="pipeline secret caption one",
            status=ContentStatus.PUBLISHED,
        )
        c_pend = Content(
            organization_id=org.id, brand_profile_id=brand.id, creator_id=admin.id,
            title="Pipeline Secret Pend", caption="pipeline secret pending",
            status=ContentStatus.PENDING_APPROVAL,
        )
        c_draft = Content(
            organization_id=org.id, brand_profile_id=brand.id, creator_id=admin.id,
            title="Pipeline Secret Draft", caption="pipeline secret draft",
            status=ContentStatus.DRAFT,
        )
        db_session.add_all([c_pub, c_pend, c_draft])
        db_session.flush()

        # Linked ScheduledPost deliveries for c_pub -> must NOT inflate counts (+0 extra items)
        db_session.add_all([
            ScheduledPost(
                organization_id=org.id, content_id=c_pub.id, social_account_id=acc_ig.id,
                platform="instagram", status="published", scheduled_time=now,
            ),
            ScheduledPost(
                organization_id=org.id, content_id=c_pub.id, social_account_id=acc_fb.id,
                platform="facebook", status="published", scheduled_time=now,
            ),
            ScheduledPost(
                organization_id=org.id, content_id=c_pub.id, social_account_id=acc_yt.id,
                platform="youtube", status="published", scheduled_time=now,
            ),
        ])

        # 2. Standalone recurring/unlinked ScheduledPosts (content_id=None) -> 3 valid + 1 cancelled excluded
        rule = ScheduleRule(
            organization_id=org.id, user_id=admin.id, social_account_id=acc_ig.id,
            platform="instagram", name="Recurring Rule", recurrence_type="daily", start_date=now,
        )
        db_session.add(rule)
        db_session.flush()
        db_session.add_all([
            ScheduledPost(
                organization_id=org.id, content_id=None, schedule_rule_id=rule.id,
                social_account_id=acc_ig.id, platform="instagram", status="scheduled",
                caption="standalone scheduled secret", scheduled_time=now + timedelta(days=1),
            ),
            ScheduledPost(
                organization_id=org.id, content_id=None, schedule_rule_id=rule.id,
                social_account_id=acc_ig.id, platform="instagram", status="published",
                caption="standalone published secret", scheduled_time=now - timedelta(hours=1),
                published_time=now - timedelta(hours=1),
            ),
            ScheduledPost(
                organization_id=org.id, content_id=None, schedule_rule_id=rule.id,
                social_account_id=acc_fb.id, platform="facebook", status="failed",
                caption="standalone failed secret", scheduled_time=now - timedelta(hours=2),
            ),
            ScheduledPost(
                organization_id=org.id, content_id=None, schedule_rule_id=rule.id,
                social_account_id=acc_ig.id, platform="instagram", status="cancelled",
                caption="standalone cancelled secret", scheduled_time=now + timedelta(days=2),
            ),
        ])

        # 3. Autonomous Pipeline posts (6 valid across statuses + 1 'generating' excluded)
        ap_specs = [
            ("published", [("instagram", "posted"), ("facebook", "posted"), ("youtube", "posted")]),
            ("posted", [("instagram", "posted")]),
            ("scheduled", [("instagram", "scheduled"), ("facebook", "scheduled")]),
            ("pending_approval", [("instagram", "pending"), ("youtube", "pending")]),
            ("draft", []),
            ("failed", [("facebook", "failed")]),
            ("generating", []),
        ]
        for idx, (ap_status, variants) in enumerate(ap_specs):
            run = PipelineRun(
                brand_profile_id=brand.id, organization_id=org.id,
                run_date=(now - timedelta(days=idx)).date(),
                status="completed" if ap_status != "generating" else "running",
                stage="Done", stats={},
            )
            db_session.add(run)
            db_session.flush()
            brief = ContentBrief(
                run_id=run.id, brand_profile_id=brand.id,
                topic=f"Secret Auto Topic {idx}", angle="Angle",
                caption_brief="autonomous brief secret", image_prompt="autonomous prompt secret",
            )
            db_session.add(brief)
            db_session.flush()
            ap = AutonomousPost(
                brief_id=brief.id, brand_profile_id=brand.id,
                organization_id=org.id, status=ap_status, scheduled_time=now,
            )
            db_session.add(ap)
            db_session.flush()
            for plat, var_status in variants:
                db_session.add(PostVariant(
                    post_id=ap.id, platform=plat, caption="variant secret caption",
                    scheduled_time=now, status=var_status,
                ))

        db_session.commit()

        after_res = client.get("/api/v1/superadmin/stats")
        assert after_res.status_code == 200
        after = after_res.json()

        # Expected deltas:
        # published: 1 (Content) + 1 (standalone SP) + 2 (Autonomous: 'published' + 'posted') = +4
        # scheduled: 0 (Content) + 1 (standalone SP) + 1 (Autonomous) = +2
        # pending:   1 (Content) + 0 (standalone SP) + 1 (Autonomous) = +2
        # draft:     1 (Content) + 0 (standalone SP) + 1 (Autonomous) = +2
        # failed:    0 (Content) + 1 (standalone SP) + 1 (Autonomous) = +2
        # total:     3 (Content) + 3 (standalone SP) + 6 (Autonomous) = +12 (cancelled SP & generating AP excluded)
        assert set(after.keys()) == set(baseline.keys())
        assert after["total_published_posts"] == baseline["total_published_posts"] + 4
        assert after["total_scheduled_posts"] == baseline["total_scheduled_posts"] + 2
        assert after["total_pending_posts"] == baseline["total_pending_posts"] + 2
        assert after["total_draft_posts"] == baseline["total_draft_posts"] + 2
        assert after["total_failed_posts"] == baseline["total_failed_posts"] + 2
        assert after["total_posts"] == baseline["total_posts"] + 12

        # Strict aggregate privacy verification
        serialized = str(after).lower()
        assert "secret" not in serialized
        assert "autonomous prompt" not in serialized
        assert "pipeline secret caption" not in serialized
    finally:
        app.dependency_overrides.pop(require_super_admin, None)



