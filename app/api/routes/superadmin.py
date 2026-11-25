"""
SuperAdmin Master Management & Governance Router.
Provides platform-level executive oversight, full admin & user detail management,
comparative analytics between admin and user activities, and master system controls.
"""
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, text, String, cast, exists

from app.db.dependencies import get_db
from app.models.user import User
from app.models.organization import Organization
from app.models.subscription import Subscription
from app.models.brand_profile import BrandProfile
from app.models.user_organization import UserOrganization
from app.models.content import Content
from app.models.campaign import Campaign
from app.models.scheduled_post import ScheduledPost
from app.models.generation_job import GenerationJob
from app.models.schedule_rule import ScheduleRule
from app.models.autonomous_pipeline import (
    PipelineRun,
    ContentBrief,
    GeneratedAsset,
    AutonomousPost,
    PostVariant,
)
from app.models.social_account import SocialAccount
from app.models.audit_log import AuditLog
from app.core.permissions import require_super_admin
from app.core.security import get_password_hash
from app.utils.enums import UserRole, ContentStatus, PlatformType, SubscriptionTier, SubscriptionStatus, JobStatus
from app.api.routes.payments import TIER_PRICING_PAISE
from app.services.redis_manager import RedisManager
from app.scheduler.publishing_worker import PublishingWorker
from app.scheduler.autonomous_scheduler import AutonomousScheduler

router = APIRouter(prefix="/superadmin", tags=["SuperAdmin Executive Portal"])


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------

class SuperAdminStatsResponse(BaseModel):
    total_superadmins: int
    total_admins: int
    total_users: int
    active_admins: int
    inactive_admins: int
    active_users: int
    inactive_users: int
    total_organizations: int
    active_organizations: int
    suspended_organizations: int
    active_subscriptions: int
    expired_subscriptions: int
    plan_distribution: Dict[str, int]
    monthly_recurring_revenue: float
    annual_recurring_revenue: float
    revenue_currency: str = "INR"
    revenue_type: str = "projected_recurring"
    total_posts: int
    total_published_posts: int
    total_pending_posts: int
    total_scheduled_posts: int
    total_draft_posts: int
    total_failed_posts: int
    total_generation_jobs: int
    total_social_accounts: int
    new_users_last_7_days: int
    new_users_last_30_days: int
    new_admins_last_30_days: int


class AdminDetailResponse(BaseModel):
    id: int
    email: str
    is_active: bool
    subscription_plan: Optional[str] = None
    subscription_status: Optional[str] = None


class UserDetailResponse(BaseModel):
    id: int
    email: str
    is_active: bool


class BillingSubscriptionResponse(BaseModel):
    id: int
    registered_email: str
    plan: str
    subscription_status: str
    account_status: bool
    razorpay_payment_id: Optional[str] = None
    razorpay_order_id: Optional[str] = None
    verification_status: str = "Unverified"
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    monthly_generation_limit: int = 50
    monthly_posts_limit: int = 100
    current_generation_usage: int = 0
    current_posts_usage: int = 0


class PlatformHealthResponse(BaseModel):
    api_status: str = "healthy"
    database_connected: bool
    database_latency_ms: float
    redis_available: bool
    publishing_worker_status: str
    autonomous_scheduler_status: str
    failed_content_count: int
    failed_generation_jobs_count: int


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.CREATOR
    organization_id: Optional[int] = None
    is_active: bool = True


class UpdateUserRoleRequest(BaseModel):
    role: UserRole


class UpdateUserStatusRequest(BaseModel):
    is_active: bool


class SuperAdminControls(BaseModel):
    maintenance_mode: bool = False
    open_registration: bool = True
    default_role: str = "creator"
    autonomous_scheduler_active: bool = True
    publishing_worker_active: bool = True
    active_ai_provider: str = "Google Gemini 2.5 (Veo & Imagen 3)"
    max_upload_size_mb: int = 50
    auto_approval_enabled: bool = False


class AuditLogItem(BaseModel):
    id: int
    action: str
    entity_type: str
    entity_id: Optional[int] = None
    user_id: Optional[int] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    ip_address: Optional[str] = None
    details: Dict[str, Any] = {}
    created_at: Optional[datetime] = None


# Persistent runtime controls state
_RUNTIME_CONTROLS = {
    "maintenance_mode": False,
    "open_registration": True,
    "default_role": "creator",
    "autonomous_scheduler_active": True,
    "publishing_worker_active": True,
    "active_ai_provider": "Google Gemini 2.5 (Veo & Imagen 3)",
    "max_upload_size_mb": 50,
    "auto_approval_enabled": False,
}


# ---------------------------------------------------------
# 1. Executive Platform Stats
# ---------------------------------------------------------

_PIPELINE_STATUS_BUCKETS = {
    "published": "published",
    "posted": "published",
    "pending_approval": "pending",
    "scheduled": "scheduled",
    "draft": "draft",
    "failed": "failed",
}


def _compute_global_content_pipeline_counts(db: Session) -> Dict[str, int]:
    """
    Aggregate unique content items across Content, standalone ScheduledPost
    (content_id IS NULL), and AutonomousPost without double-counting per-platform
    delivery records (linked ScheduledPost or PostVariant rows).
    """
    counts = {
        "total": 0,
        "published": 0,
        "pending": 0,
        "scheduled": 0,
        "draft": 0,
        "failed": 0,
    }

    def _accumulate(raw_status: Any) -> None:
        counts["total"] += 1
        bucket = _PIPELINE_STATUS_BUCKETS.get(str(raw_status or "").strip().lower())
        if bucket:
            counts[bucket] += 1

    # 1. Ordinary Content rows (1 item per Content; child ScheduledPost rows are platform deliveries)
    for (status_str,) in db.query(func.lower(cast(Content.status, String))).all():
        _accumulate(status_str)

    # 2. Standalone recurring/unlinked ScheduledPosts (content_id IS NULL)
    standalone_rows = (
        db.query(
            func.lower(cast(ScheduledPost.status, String)),
            ScheduledPost.publish_metadata,
        )
        .filter(ScheduledPost.content_id.is_(None))
        .all()
    )
    for status_str, publish_meta in standalone_rows:
        norm_status = (status_str or "").strip().lower()
        is_cancelled = norm_status == "cancelled" or (
            isinstance(publish_meta, dict) and publish_meta.get("cancelled") is True
        )
        if is_cancelled:
            continue
        _accumulate(norm_status)

    # 3. Autonomous Pipeline consolidated posts (1 item per AutonomousPost; PostVariant rows are platform deliveries)
    for (status_str,) in db.query(func.lower(cast(AutonomousPost.status, String))).all():
        norm_status = (status_str or "").strip().lower()
        if norm_status == "generating":
            continue
        _accumulate(norm_status)

    return counts


@router.get("/stats", response_model=SuperAdminStatsResponse)
def get_superadmin_stats(
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Retrieve top-level platform statistics covering Superadmins, Admins, and Users."""
    now = datetime.utcnow()
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    # SuperAdmin counts
    total_superadmins = db.query(User).filter(User.role == UserRole.SUPER_ADMIN).count()

    # Admin counts
    total_admins = db.query(User).filter(User.role == UserRole.ADMIN).count()
    active_admins = db.query(User).filter(User.role.in_([UserRole.ADMIN, UserRole.SUPER_ADMIN]), User.is_active == True).count()
    inactive_admins = (total_admins + total_superadmins) - active_admins

    # Regular user counts (Creator, Manager, Viewer)
    user_roles = [UserRole.CREATOR, UserRole.MANAGER, UserRole.VIEWER]
    total_users = db.query(User).filter(User.role.in_(user_roles)).count()
    active_users = db.query(User).filter(User.role.in_(user_roles), User.is_active == True).count()
    inactive_users = total_users - active_users

    total_orgs = db.query(Organization).count()
    active_orgs = db.query(Organization).filter(Organization.is_active == True).count()
    suspended_orgs = db.query(Organization).filter(Organization.is_active == False).count()

    active_org_ids = {o[0] for o in db.query(Organization.id).filter(Organization.is_active == True).all()}
    all_subs = db.query(Subscription).all()

    plan_dist = {"free": 0, "starter": 0, "pro": 0, "enterprise": 0}
    active_subs_count = 0
    expired_subs_count = 0
    mrr = 0.0

    # Pricing in INR per month from actual payment gateway tier config
    monthly_prices_inr = {
        "starter": TIER_PRICING_PAISE.get("starter", 199900) / 100.0,
        "pro": TIER_PRICING_PAISE.get("pro", 499900) / 100.0,
        "enterprise": TIER_PRICING_PAISE.get("enterprise", 999900) / 100.0
    }

    for sub in all_subs:
        tier_val = (sub.tier.value if hasattr(sub.tier, "value") else str(sub.tier)).lower()
        if tier_val in plan_dist:
            plan_dist[tier_val] += 1
        else:
            plan_dist[tier_val] = 1

        status_val = (sub.status.value if hasattr(sub.status, "value") else str(sub.status)).lower()
        period_end = _to_naive_utc(sub.current_period_end)
        is_date_expired = bool(period_end and period_end < now)
        is_active_sub = (status_val == "active") and not is_date_expired and (sub.organization_id in active_org_ids)

        if is_active_sub:
            active_subs_count += 1
            if tier_val != "free":
                mrr += monthly_prices_inr.get(tier_val, 0.0)

        # Count only subscriptions whose current_period_end has actually passed
        if is_date_expired:
            expired_subs_count += 1

    arr = round(mrr * 12.0, 2)
    mrr = round(mrr, 2)

    # Content pipeline breakdowns across Content, standalone ScheduledPost, and AutonomousPost
    pipeline_counts = _compute_global_content_pipeline_counts(db)
    total_posts = pipeline_counts["total"]
    total_published = pipeline_counts["published"]
    total_pending = pipeline_counts["pending"]
    total_scheduled = pipeline_counts["scheduled"]
    total_drafts = pipeline_counts["draft"]
    total_failed = pipeline_counts["failed"]

    total_jobs = db.query(GenerationJob).count()
    allowed_superadmin_platforms = [PlatformType.YOUTUBE, PlatformType.FACEBOOK, PlatformType.INSTAGRAM]
    total_social_accounts = db.query(SocialAccount).filter(SocialAccount.platform.in_(allowed_superadmin_platforms)).count()

    new_users_7d = db.query(User).filter(User.role.in_(user_roles), User.created_at >= seven_days_ago).count()
    new_users_30d = db.query(User).filter(User.role.in_(user_roles), User.created_at >= thirty_days_ago).count()
    new_admins_30d = db.query(User).filter(User.role.in_([UserRole.ADMIN, UserRole.SUPER_ADMIN]), User.created_at >= thirty_days_ago).count()

    return SuperAdminStatsResponse(
        total_superadmins=total_superadmins,
        total_admins=total_admins,
        total_users=total_users,
        active_admins=active_admins,
        inactive_admins=inactive_admins,
        active_users=active_users,
        inactive_users=inactive_users,
        total_organizations=total_orgs,
        active_organizations=active_orgs,
        suspended_organizations=suspended_orgs,
        active_subscriptions=active_subs_count,
        expired_subscriptions=expired_subs_count,
        plan_distribution=plan_dist,
        monthly_recurring_revenue=mrr,
        annual_recurring_revenue=arr,
        revenue_currency="INR",
        revenue_type="projected_recurring",
        total_posts=total_posts,
        total_published_posts=total_published,
        total_pending_posts=total_pending,
        total_scheduled_posts=total_scheduled,
        total_draft_posts=total_drafts,
        total_failed_posts=total_failed,
        total_generation_jobs=total_jobs,
        total_social_accounts=total_social_accounts,
        new_users_last_7_days=new_users_7d,
        new_users_last_30_days=new_users_30d,
        new_admins_last_30_days=new_admins_30d
    )


# ---------------------------------------------------------
# 2. Admin Details Management
# ---------------------------------------------------------

@router.get("/admins", response_model=List[AdminDetailResponse])
def get_all_admins(
    search: Optional[str] = None,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Retrieve details of all Administrators and SuperAdmins (strictly privacy-sanitized)."""
    query = db.query(User).filter(User.role.in_([UserRole.SUPER_ADMIN, UserRole.ADMIN]))

    if search:
        st = f"%{search.strip()}%"
        query = query.filter(User.email.ilike(st))

    if status_filter == "active":
        query = query.filter(User.is_active == True)
    elif status_filter == "inactive":
        query = query.filter(User.is_active == False)

    admins = query.order_by(User.role.asc(), User.created_at.desc()).all()

    results = []
    for adm in admins:
        sub_plan = "Pro Growth Plan"
        sub_status = "Active"

        if adm.organization_id:
            sub = db.query(Subscription).filter(Subscription.organization_id == adm.organization_id).first()
            if sub:
                tier_str = sub.tier.value if hasattr(sub.tier, 'value') else str(sub.tier)
                tier_map = {
                    "FREE": "Starter Boutique",
                    "STARTER": "Growth Starter Plan",
                    "PRO": "Pro Growth Plan",
                    "ENTERPRISE": "Enterprise Scale Plan"
                }
                sub_plan = tier_map.get(tier_str.upper(), f"{tier_str.title()} Plan")
                sub_status = sub.status.value if hasattr(sub.status, 'value') else str(sub.status).capitalize()

        results.append(AdminDetailResponse(
            id=adm.id,
            email=adm.email,
            is_active=adm.is_active,
            subscription_plan=sub_plan,
            subscription_status=sub_status
        ))

    return results


# ---------------------------------------------------------
# 3. Regular User Details Management
# ---------------------------------------------------------

@router.get("/users", response_model=List[UserDetailResponse])
def get_all_users(
    search: Optional[str] = None,
    role_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    organization_id: Optional[int] = None,
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Retrieve details of all Regular Users (Creators, Managers, Viewers) (strictly privacy-sanitized)."""
    user_roles = [UserRole.CREATOR, UserRole.MANAGER, UserRole.VIEWER]
    query = db.query(User).filter(User.role.in_(user_roles))

    if search:
        st = f"%{search.strip()}%"
        query = query.filter(User.email.ilike(st))

    if role_filter and role_filter != "all":
        query = query.filter(User.role == role_filter)

    if status_filter == "active":
        query = query.filter(User.is_active == True)
    elif status_filter == "inactive":
        query = query.filter(User.is_active == False)

    if organization_id:
        query = query.filter(User.organization_id == organization_id)

    users = query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()

    results = []
    for u in users:
        results.append(UserDetailResponse(
            id=u.id,
            email=u.email,
            is_active=u.is_active
        ))

    return results


# ---------------------------------------------------------
# 4. SuperAdmin User / Admin Creation & Role Management
# ---------------------------------------------------------

@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user_or_admin(
    payload: CreateUserRequest,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Directly provision a new Admin or User account with organizational assignment."""
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email '{payload.email}' already exists."
        )

    org_id = payload.organization_id
    role_val = payload.role.value if hasattr(payload.role, "value") else str(payload.role).lower()
    is_admin_role = role_val in ("admin", "enterprise_admin")
    is_super_admin_role = role_val in ("super_admin", "superadmin")

    if not org_id and not is_super_admin_role:
        if is_admin_role:
            # Each Enterprise Admin provisioned without an explicit organization receives their OWN separate, isolated organization
            uid_suffix = uuid.uuid4().hex[:8]
            new_org = Organization(
                name=f"Enterprise Workspace {uid_suffix}",
                slug=f"enterprise-{uid_suffix}",
                description="Dedicated Enterprise Workspace",
                is_active=True,
                settings={"category": "Enterprise Workspace"}
            )
            db.add(new_org)
            db.flush()
            org_id = new_org.id

            # Provision default subscription for this new isolated enterprise organization
            db.add(Subscription(
                organization_id=new_org.id,
                tier=SubscriptionTier.FREE,
                status=SubscriptionStatus.ACTIVE,
                monthly_generation_limit=50,
                monthly_posts_limit=100
            ))
            db.flush()
        else:
            # Non-admin team members (Creator, Manager, Viewer) default to first available organization if not explicitly specified
            first_org = db.query(Organization).filter(Organization.is_active == True).first()
            if first_org:
                org_id = first_org.id
            else:
                uid_suffix = uuid.uuid4().hex[:8]
                new_org = Organization(
                    name=f"Default Workspace {uid_suffix}",
                    slug=f"workspace-{uid_suffix}",
                    is_active=True
                )
                db.add(new_org)
                db.flush()
                org_id = new_org.id

    new_user = User(
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        full_name=payload.full_name,
        role=payload.role,
        organization_id=org_id,
        is_active=payload.is_active,
        is_verified=True
    )
    db.add(new_user)
    db.flush()

    if org_id:
        db.add(UserOrganization(
            user_id=new_user.id,
            organization_id=org_id,
            role=payload.role,
            is_default=True
        ))

    # Log action in AuditLog
    db.add(AuditLog(
        action="superadmin_create_account",
        entity_type="user",
        entity_id=new_user.id,
        user_id=current_superadmin.id,
        details={
            "created_email": new_user.email,
            "created_role": new_user.role.value if hasattr(new_user.role, 'value') else str(new_user.role),
            "created_by": current_superadmin.email
        }
    ))

    db.commit()
    db.refresh(new_user)

    return {
        "message": f"Account '{new_user.email}' created successfully with role '{new_user.role.value}'.",
        "id": new_user.id,
        "email": new_user.email,
        "role": new_user.role.value
    }


@router.put("/users/{user_id}/role")
def update_user_role(
    user_id: int,
    payload: UpdateUserRoleRequest,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Update role for any user (promote user to Admin/SuperAdmin, or demote)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    # Guard: prevent demoting oneself if you are the only super_admin
    if current_superadmin.id == user_id and payload.role != UserRole.SUPER_ADMIN:
        super_count = db.query(User).filter(User.role == UserRole.SUPER_ADMIN, User.is_active == True).count()
        if super_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote the only remaining active SuperAdmin."
            )

    old_role = user.role.value if hasattr(user.role, 'value') else str(user.role)
    user.role = payload.role

    # Update assigned organization roles
    db.query(UserOrganization).filter(UserOrganization.user_id == user_id).update({"role": payload.role})

    db.add(AuditLog(
        action="superadmin_update_role",
        entity_type="user",
        entity_id=user.id,
        user_id=current_superadmin.id,
        details={"old_role": old_role, "new_role": payload.role.value, "target_email": user.email}
    ))

    db.commit()
    db.refresh(user)

    return {
        "message": f"User '{user.email}' role updated to '{user.role.value}'.",
        "user_id": user.id,
        "email": user.email,
        "role": user.role.value
    }


@router.put("/users/{user_id}/status")
def update_user_status(
    user_id: int,
    payload: UpdateUserStatusRequest,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Activate or deactivate any user or administrator account."""
    if current_superadmin.id == user_id and not payload.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot deactivate your own SuperAdmin account."
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    user.is_active = payload.is_active

    db.add(AuditLog(
        action="superadmin_update_status",
        entity_type="user",
        entity_id=user.id,
        user_id=current_superadmin.id,
        details={"is_active": payload.is_active, "target_email": user.email}
    ))

    db.commit()
    db.refresh(user)

    return {
        "message": f"User '{user.email}' status updated to {'Active' if user.is_active else 'Deactivated'}.",
        "user_id": user.id,
        "is_active": user.is_active
    }


@router.delete("/users/{user_id}")
def delete_user_or_admin(
    user_id: int,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Permanently delete a user or administrator account."""
    if current_superadmin.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own SuperAdmin account."
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    target_email = user.email
    db.delete(user)

    db.add(AuditLog(
        action="superadmin_delete_user",
        entity_type="user",
        entity_id=user_id,
        user_id=current_superadmin.id,
        details={"deleted_email": target_email}
    ))

    db.commit()
    return {"message": f"Account '{target_email}' (ID #{user_id}) permanently removed."}


# ---------------------------------------------------------
# 5. Comparative Analytics: Admin vs User
# ---------------------------------------------------------

@router.get("/analytics")
def get_superadmin_analytics(
    timeframe: str = Query("30d", pattern="^(7d|30d|90d|all)$"),
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """
    Retrieve comparative analytics contrasting Admin operations against User/Creator activities,
    including content generation rates, post status ratios, platform breakdown, and top performers.
    """
    # 1. Admin operations summary
    admins = db.query(User).filter(User.role.in_([UserRole.ADMIN, UserRole.SUPER_ADMIN])).all()
    admin_ids = [a.id for a in admins]

    admin_post_count = db.query(Content).filter(Content.creator_id.in_(admin_ids)).count() if admin_ids else 0
    admin_gen_count = db.query(GenerationJob).filter(GenerationJob.user_id.in_(admin_ids)).count() if admin_ids else 0

    # 2. User operations summary
    user_roles = [UserRole.CREATOR, UserRole.MANAGER, UserRole.VIEWER]
    users = db.query(User).filter(User.role.in_(user_roles)).all()
    user_ids = [u.id for u in users]

    user_post_count = db.query(Content).filter(Content.creator_id.in_(user_ids)).count() if user_ids else 0
    user_gen_count = db.query(GenerationJob).filter(GenerationJob.user_id.in_(user_ids)).count() if user_ids else 0

    # 3. Post status breakdown
    post_status_dist = {}
    for st in ContentStatus:
        count = db.query(Content).filter(Content.status == st).count()
        post_status_dist[st.value] = count

    # 4. Social platform breakdown (strictly YouTube, Facebook, Instagram)
    allowed_platforms = [PlatformType.YOUTUBE, PlatformType.FACEBOOK, PlatformType.INSTAGRAM]
    platform_dist = {}
    for plat in allowed_platforms:
        count = db.query(SocialAccount).filter(SocialAccount.platform == plat).count()
        platform_dist[plat.value] = count

    # 4b. Timeline trend series for SuperAdmin Line Chart (dynamic data across timeframe)
    days_map = {"7d": 7, "30d": 30, "90d": 90, "all": 30}
    num_days = days_map.get(timeframe, 30)
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=num_days - 1)

    contents_in_range = db.query(Content.created_at, Content.target_platforms).filter(Content.created_at >= start_date).all()
    jobs_in_range = db.query(GenerationJob.created_at).filter(GenerationJob.created_at >= start_date).all()

    timeline_dict = {}
    cur = start_date
    while cur <= today:
        d_str = cur.strftime("%Y-%m-%d")
        timeline_dict[d_str] = {
            "date": d_str,
            "label": cur.strftime("%b %d"),
            "youtube": 0,
            "facebook": 0,
            "instagram": 0,
            "total_posts": 0,
            "generations": 0
        }
        cur += timedelta(days=1)

    for c in contents_in_range:
        if c[0]:
            d_str = c[0].date().strftime("%Y-%m-%d")
            if d_str in timeline_dict:
                timeline_dict[d_str]["total_posts"] += 1
                plats = [str(p).lower() for p in (c[1] or [])]
                if any("youtube" in p for p in plats):
                    timeline_dict[d_str]["youtube"] += 1
                if any("facebook" in p for p in plats):
                    timeline_dict[d_str]["facebook"] += 1
                if any("instagram" in p for p in plats):
                    timeline_dict[d_str]["instagram"] += 1

    for j in jobs_in_range:
        if j[0]:
            d_str = j[0].date().strftime("%Y-%m-%d")
            if d_str in timeline_dict:
                timeline_dict[d_str]["generations"] += 1

    activity_timeline = list(timeline_dict.values())

    # 5. Generation jobs by type
    gen_jobs = db.query(GenerationJob.job_type, func.count(GenerationJob.id)).group_by(GenerationJob.job_type).all()
    gen_by_type = {str(job_type.value if hasattr(job_type, "value") else job_type): count for job_type, count in gen_jobs}

    return {
        "admin_summary": {
            "total_admins": len(admins),
            "posts_created_by_admins": admin_post_count,
            "generations_by_admins": admin_gen_count,
        },
        "user_summary": {
            "total_users": len(users),
            "posts_created_by_users": user_post_count,
            "generations_by_users": user_gen_count,
        },
        "ratio_user_to_admin_posts": round(user_post_count / max(admin_post_count, 1), 2),
        "post_status_distribution": post_status_dist,
        "platform_distribution": platform_dist,
        "activity_timeline": activity_timeline,
        "generation_types_distribution": gen_by_type,
        "timeframe": timeframe
    }


# ---------------------------------------------------------
# 6. Platform Governance & Master Controls
# ---------------------------------------------------------

@router.get("/controls", response_model=SuperAdminControls)
def get_system_controls(
    current_superadmin: User = Depends(require_super_admin)
):
    """Retrieve global platform governance toggles and engine states."""
    return SuperAdminControls(**_RUNTIME_CONTROLS)


@router.put("/controls", response_model=SuperAdminControls)
def update_system_controls(
    payload: SuperAdminControls,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Update global system controls (maintenance mode, auto-scheduling, AI provider limits)."""
    global _RUNTIME_CONTROLS
    _RUNTIME_CONTROLS.update(payload.dict())

    db.add(AuditLog(
        action="superadmin_update_system_controls",
        entity_type="system",
        entity_id=0,
        user_id=current_superadmin.id,
        details=payload.dict()
    ))
    db.commit()

    return SuperAdminControls(**_RUNTIME_CONTROLS)


# ---------------------------------------------------------
# 7. Platform Audit Logs
# ---------------------------------------------------------

@router.get("/audit-logs", response_model=List[AuditLogItem])
def get_platform_audit_logs(
    limit: int = Query(50, ge=1, le=200),
    action_filter: Optional[str] = None,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """View chronological platform-wide audit log trail for governance and security."""
    query = db.query(AuditLog)
    if action_filter:
        query = query.filter(AuditLog.action.ilike(f"%{action_filter.strip()}%"))

    logs = query.order_by(AuditLog.id.desc()).limit(limit).all()

    user_ids = [l.user_id for l in logs if l.user_id]
    users_db = {u.id: u for u in db.query(User).filter(User.id.in_(user_ids)).all()} if user_ids else {}

    results = []
    for log in logs:
        u = users_db.get(log.user_id)
        results.append(AuditLogItem(
            id=log.id,
            action=log.action,
            entity_type=log.entity_type,
            entity_id=log.entity_id,
            user_id=log.user_id,
            user_email=u.email if u else None,
            user_role=u.role.value if u and hasattr(u.role, 'value') else (str(u.role) if u else None),
            ip_address=log.ip_address,
            details=log.details or {},
            created_at=log.created_at
        ))

    return results


# ---------------------------------------------------------
# 8. Multi-Tenant Enterprise Management & Isolation Control
# ---------------------------------------------------------

class CreateTenantRequest(BaseModel):
    name: str
    slug: Optional[str] = None
    description: Optional[str] = None
    admin_email: Optional[EmailStr] = None
    admin_password: Optional[str] = None
    admin_full_name: Optional[str] = None
    subscription_tier: Optional[SubscriptionTier] = SubscriptionTier.FREE


class UpdateTenantSubscriptionRequest(BaseModel):
    tier: SubscriptionTier
    status: Optional[SubscriptionStatus] = SubscriptionStatus.ACTIVE
    monthly_generation_limit: Optional[int] = None
    monthly_posts_limit: Optional[int] = None


class TenantDetailResponse(BaseModel):
    id: int
    registered_email: Optional[str] = None
    subscription_tier: Optional[str] = None
    subscription_status: Optional[str] = None
    is_active: bool
    created_at: Optional[datetime] = None
    tenant_isolation_status: str = "Dedicated Workspace"


@router.get("/tenants", response_model=List[TenantDetailResponse])
def get_all_tenants(
    search: Optional[str] = None,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Platform-wide listing of all enterprise tenants (strictly privacy-sanitized: registered email, plan, status, created_at)."""
    query = db.query(Organization)
    if status_filter == "active":
        query = query.filter(Organization.is_active == True)
    elif status_filter == "suspended":
        query = query.filter(Organization.is_active == False)

    orgs = query.order_by(Organization.id.asc()).all()
    results = []
    for org in orgs:
        admin_user = db.query(User).filter(
            User.organization_id == org.id,
            User.role.in_([UserRole.ADMIN, UserRole.SUPER_ADMIN])
        ).order_by(User.id.asc()).first()
        registered_email = admin_user.email if admin_user else None

        if search and registered_email:
            if search.strip().lower() not in registered_email.lower():
                continue
        elif search and not registered_email:
            continue

        sub = db.query(Subscription).filter(Subscription.organization_id == org.id).first()
        sub_tier = sub.tier.value if (sub and hasattr(sub.tier, 'value')) else (str(sub.tier) if sub else None)
        sub_status = sub.status.value if (sub and hasattr(sub.status, 'value')) else (str(sub.status) if sub else None)

        results.append(TenantDetailResponse(
            id=org.id,
            registered_email=registered_email,
            subscription_tier=sub_tier,
            subscription_status=sub_status,
            is_active=org.is_active,
            created_at=org.created_at,
            tenant_isolation_status="Dedicated Workspace"
        ))
    return results


@router.post("/tenants", response_model=TenantDetailResponse, status_code=status.HTTP_201_CREATED)
def create_tenant(
    payload: CreateTenantRequest,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Create a new enterprise tenant workspace with isolated storage and optional initial Enterprise Admin."""
    clean_slug = (payload.slug or re.sub(r'[^a-zA-Z0-9_-]', '_', payload.name.lower())).strip('_')
    existing = db.query(Organization).filter(or_(Organization.name == payload.name, Organization.slug == clean_slug)).first()
    if existing:
        clean_slug = f"{clean_slug}_{int(datetime.utcnow().timestamp())}"

    org = Organization(
        name=payload.name,
        slug=clean_slug,
        description=payload.description,
        is_active=True,
        settings={"category": "Enterprise Workspace"}
    )
    db.add(org)
    db.flush()

    # Create default subscription
    limits = {
        SubscriptionTier.FREE: (50, 100),
        SubscriptionTier.STARTER: (200, 500),
        SubscriptionTier.PRO: (1000, 2500),
        SubscriptionTier.ENTERPRISE: (10000, 50000)
    }
    gen_lim, post_lim = limits.get(payload.subscription_tier, (50, 100))
    sub = Subscription(
        organization_id=org.id,
        tier=payload.subscription_tier,
        status=SubscriptionStatus.ACTIVE,
        monthly_generation_limit=gen_lim,
        monthly_posts_limit=post_lim
    )
    db.add(sub)

    # Optional initial Enterprise Admin
    registered_email = None
    if payload.admin_email and payload.admin_password:
        existing_user = db.query(User).filter(User.email == payload.admin_email).first()
        if existing_user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"User with email '{payload.admin_email}' already exists.")
        admin_user = User(
            email=payload.admin_email,
            hashed_password=get_password_hash(payload.admin_password),
            full_name=payload.admin_full_name or "Enterprise Admin",
            role=UserRole.ADMIN,
            organization_id=org.id,
            is_active=True,
            is_verified=True
        )
        db.add(admin_user)
        db.flush()
        db.add(UserOrganization(
            user_id=admin_user.id,
            organization_id=org.id,
            role=UserRole.ADMIN,
            is_default=True
        ))
        registered_email = admin_user.email

    db.add(AuditLog(
        action="superadmin_create_tenant",
        entity_type="tenant",
        entity_id=org.id,
        organization_id=org.id,
        user_id=current_superadmin.id,
        details={"tenant_id": org.id, "created_by": current_superadmin.email}
    ))
    db.commit()
    db.refresh(org)

    return TenantDetailResponse(
        id=org.id,
        registered_email=registered_email,
        subscription_tier=sub.tier.value if hasattr(sub.tier, 'value') else str(sub.tier),
        subscription_status=sub.status.value if hasattr(sub.status, 'value') else str(sub.status),
        is_active=org.is_active,
        created_at=org.created_at,
        tenant_isolation_status="Dedicated Workspace"
    )


@router.get("/tenants/{tenant_id}", response_model=TenantDetailResponse)
def get_tenant_details(
    tenant_id: int,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Retrieve metadata for a single enterprise tenant (strictly privacy-sanitized: registered email, plan, status)."""
    org = db.query(Organization).filter(Organization.id == tenant_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tenant #{tenant_id} not found.")

    admin_user = db.query(User).filter(
        User.organization_id == org.id,
        User.role.in_([UserRole.ADMIN, UserRole.SUPER_ADMIN])
    ).order_by(User.id.asc()).first()
    registered_email = admin_user.email if admin_user else None

    sub = db.query(Subscription).filter(Subscription.organization_id == org.id).first()
    sub_tier = sub.tier.value if (sub and hasattr(sub.tier, 'value')) else (str(sub.tier) if sub else None)
    sub_status = sub.status.value if (sub and hasattr(sub.status, 'value')) else (str(sub.status) if sub else None)

    return TenantDetailResponse(
        id=org.id,
        registered_email=registered_email,
        subscription_tier=sub_tier,
        subscription_status=sub_status,
        is_active=org.is_active,
        created_at=org.created_at,
        tenant_isolation_status="Dedicated Workspace"
    )


@router.put("/tenants/{tenant_id}/suspend")
def suspend_tenant(
    tenant_id: int,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Suspend an enterprise tenant workspace, immediately revoking API access for all its users."""
    org = db.query(Organization).filter(Organization.id == tenant_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tenant #{tenant_id} not found.")

    org.is_active = False

    db.add(AuditLog(
        action="superadmin_suspend_tenant",
        entity_type="tenant",
        entity_id=org.id,
        organization_id=org.id,
        user_id=current_superadmin.id,
        details={"tenant_id": org.id, "action": "suspend", "suspended_by": current_superadmin.email}
    ))
    db.commit()
    db.refresh(org)

    return {
        "message": f"Enterprise tenant #{org.id} has been suspended.",
        "tenant_id": org.id,
        "is_active": org.is_active
    }


@router.put("/tenants/{tenant_id}/activate")
def activate_tenant(
    tenant_id: int,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Activate / unsuspend an enterprise tenant workspace, restoring API access."""
    org = db.query(Organization).filter(Organization.id == tenant_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tenant #{tenant_id} not found.")

    org.is_active = True

    db.add(AuditLog(
        action="superadmin_activate_tenant",
        entity_type="tenant",
        entity_id=org.id,
        organization_id=org.id,
        user_id=current_superadmin.id,
        details={"tenant_name": org.name, "action": "activate", "activated_by": current_superadmin.email}
    ))
    db.commit()
    db.refresh(org)

    return {
        "message": f"Enterprise tenant '{org.name}' (ID #{org.id}) has been activated.",
        "tenant_id": org.id,
        "is_active": org.is_active
    }


@router.put("/tenants/{tenant_id}/subscription")
def update_tenant_subscription(
    tenant_id: int,
    payload: UpdateTenantSubscriptionRequest,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Update subscription plan, generation limits, and status for an enterprise tenant."""
    org = db.query(Organization).filter(Organization.id == tenant_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tenant #{tenant_id} not found.")

    sub = db.query(Subscription).filter(Subscription.organization_id == tenant_id).first()
    if not sub:
        sub = Subscription(organization_id=tenant_id, tier=payload.tier, status=payload.status or SubscriptionStatus.ACTIVE)
        db.add(sub)
    else:
        sub.tier = payload.tier
        if payload.status:
            sub.status = payload.status

    if payload.monthly_generation_limit is not None:
        sub.monthly_generation_limit = payload.monthly_generation_limit
    if payload.monthly_posts_limit is not None:
        sub.monthly_posts_limit = payload.monthly_posts_limit

    db.add(AuditLog(
        action="superadmin_update_subscription",
        entity_type="tenant",
        entity_id=org.id,
        organization_id=org.id,
        user_id=current_superadmin.id,
        details={
            "tenant_name": org.name,
            "new_tier": payload.tier.value if hasattr(payload.tier, 'value') else str(payload.tier),
            "new_status": payload.status.value if hasattr(payload.status, 'value') else str(payload.status)
        }
    ))
    db.commit()
    db.refresh(sub)

    return {
        "message": f"Tenant #{tenant_id} subscription updated to '{sub.tier.value}'.",
        "tenant_id": tenant_id,
        "tier": sub.tier.value,
        "status": sub.status.value,
        "monthly_generation_limit": sub.monthly_generation_limit,
        "monthly_posts_limit": sub.monthly_posts_limit
    }


@router.delete("/tenants/{tenant_id}")
def delete_tenant(
    tenant_id: int,
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """Permanently delete an enterprise tenant workspace and all its scoped data."""
    org = db.query(Organization).filter(Organization.id == tenant_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tenant #{tenant_id} not found.")

    target_name = org.name
    db.delete(org)

    db.add(AuditLog(
        action="superadmin_delete_tenant",
        entity_type="tenant",
        entity_id=tenant_id,
        user_id=current_superadmin.id,
        details={"deleted_tenant_name": target_name, "deleted_by": current_superadmin.email}
    ))
    db.commit()
    return {"message": f"Enterprise tenant '{target_name}' (ID #{tenant_id}) permanently removed."}


# ---------------------------------------------------------
# 9. SuperAdmin Billing & Subscriptions Dedicated View
# ---------------------------------------------------------

def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime value to naive UTC for consistent database comparisons."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    """Parse an ISO-formatted timestamp string into naive UTC, returning None on invalid input."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return _to_naive_utc(parsed)
    except ValueError:
        return None


def _resolve_billing_period(
    sub: Subscription,
    now: Optional[datetime] = None,
) -> tuple[datetime, datetime, datetime, datetime, bool]:
    """
    Resolve effective billing query window and display period bounds.
    Uses subscription period dates when configured; falls back to the current
    calendar month when missing so plans without explicit dates never report lifetime usage.
    """
    ref_now = _to_naive_utc(now) or datetime.utcnow()
    if sub.current_period_start and sub.current_period_end:
        start = _to_naive_utc(sub.current_period_start)
        end = _to_naive_utc(sub.current_period_end)
        return start, end, start, end, True

    if sub.current_period_start:
        start = _to_naive_utc(sub.current_period_start)
        end = max(ref_now, start)
        return start, end, start, end, True

    month_start = datetime(ref_now.year, ref_now.month, 1, 0, 0, 0)
    if ref_now.month == 12:
        next_month_start = datetime(ref_now.year + 1, 1, 1, 0, 0, 0)
    else:
        next_month_start = datetime(ref_now.year, ref_now.month + 1, 1, 0, 0, 0)
    display_end = next_month_start - timedelta(seconds=1)
    return month_start, next_month_start, month_start, display_end, False


def _in_billing_window(
    ts: Optional[datetime],
    period_start: datetime,
    period_end: datetime,
    end_inclusive: bool,
) -> bool:
    """Check whether a timestamp falls inside the resolved billing window."""
    naive_ts = _to_naive_utc(ts)
    if naive_ts is None:
        return False
    if naive_ts < period_start:
        return False
    return naive_ts <= period_end if end_inclusive else naive_ts < period_end


def _count_billable_ai_assets(
    db: Session,
    org_id: int,
    period_start: datetime,
    period_end: datetime,
    end_inclusive: bool,
) -> int:
    """
    Count completed, billable AI-generated assets (image, video, text) for one organization.
    Excludes pending/processing/failed jobs and deduplicates shared asset references.
    """
    count = 0
    seen_media_asset_ids: set[str] = set()

    completed_jobs = (
        db.query(
            GenerationJob.id,
            GenerationJob.result_data,
            GenerationJob.created_at,
        )
        .outerjoin(User, GenerationJob.user_id == User.id)
        .filter(
            func.coalesce(GenerationJob.organization_id, User.organization_id) == org_id,
            func.lower(cast(GenerationJob.status, String)) == "completed",
        )
        .order_by(GenerationJob.id.asc())
        .all()
    )

    for job_id, result_data_raw, created_at in completed_jobs:
        result_data = result_data_raw if isinstance(result_data_raw, dict) else {}
        effective_ts = _parse_iso_datetime(result_data.get("completed_at")) or created_at
        if not _in_billing_window(effective_ts, period_start, period_end, end_inclusive):
            continue

        media_asset_id = result_data.get("media_asset_id")
        if media_asset_id is not None and str(media_asset_id).strip():
            asset_key = str(media_asset_id).strip()
            if asset_key in seen_media_asset_ids:
                continue
            seen_media_asset_ids.add(asset_key)

        count += 1

    # Include completed Autonomous Pipeline text assets (1 ContentBrief per completed/active brief)
    briefs = (
        db.query(ContentBrief)
        .join(BrandProfile, ContentBrief.brand_profile_id == BrandProfile.id)
        .outerjoin(PipelineRun, ContentBrief.run_id == PipelineRun.id)
        .filter(
            func.coalesce(PipelineRun.organization_id, BrandProfile.organization_id) == org_id,
            or_(PipelineRun.id.is_(None), PipelineRun.status.notin_(["failed", "pending"])),
        )
        .all()
    )
    for brief in briefs:
        if _in_billing_window(brief.created_at, period_start, period_end, end_inclusive):
            count += 1

    # Include completed Autonomous Pipeline visual assets (status='done' with stored media)
    pipeline_assets = (
        db.query(GeneratedAsset)
        .join(ContentBrief, GeneratedAsset.brief_id == ContentBrief.id)
        .join(BrandProfile, ContentBrief.brand_profile_id == BrandProfile.id)
        .outerjoin(PipelineRun, ContentBrief.run_id == PipelineRun.id)
        .filter(
            func.coalesce(PipelineRun.organization_id, BrandProfile.organization_id) == org_id,
            GeneratedAsset.status == "done",
            GeneratedAsset.file_path.isnot(None),
            GeneratedAsset.file_path != "",
        )
        .all()
    )
    seen_pipeline_keys: set[tuple] = set()
    for asset in pipeline_assets:
        if not _in_billing_window(asset.created_at, period_start, period_end, end_inclusive):
            continue
        dedup_key = (asset.brief_id, (asset.type or "").lower(), asset.file_path.strip())
        if dedup_key in seen_pipeline_keys:
            continue
        seen_pipeline_keys.add(dedup_key)
        count += 1

    return count


def _count_successful_deliveries(
    db: Session,
    org_id: int,
    period_start: datetime,
    period_end: datetime,
    end_inclusive: bool,
) -> int:
    """
    Count successful per-platform social deliveries across ScheduledPost
    (direct/scheduled/campaign/recurring) and Autonomous Pipeline PostVariant records.
    """
    seen_deliveries: set[tuple] = set()

    scheduled_rows = (
        db.query(ScheduledPost)
        .outerjoin(Content, ScheduledPost.content_id == Content.id)
        .outerjoin(BrandProfile, Content.brand_profile_id == BrandProfile.id)
        .outerjoin(ScheduleRule, ScheduledPost.schedule_rule_id == ScheduleRule.id)
        .filter(
            func.coalesce(
                ScheduledPost.organization_id,
                Content.organization_id,
                BrandProfile.organization_id,
                ScheduleRule.organization_id,
            ) == org_id,
            func.lower(ScheduledPost.status) == "published",
        )
        .all()
    )

    for post in scheduled_rows:
        pub_ts = post.published_time or post.created_at
        if not _in_billing_window(pub_ts, period_start, period_end, end_inclusive):
            continue
        plat = str(getattr(post.platform, "value", post.platform) or "").strip().lower()
        ext_id = (post.platform_post_id or "").strip()
        delivery_key = ("sp_ext", plat, ext_id) if ext_id else ("sp_row", post.id)
        seen_deliveries.add(delivery_key)

    variant_rows = (
        db.query(PostVariant, AutonomousPost)
        .join(AutonomousPost, PostVariant.post_id == AutonomousPost.id)
        .join(BrandProfile, AutonomousPost.brand_profile_id == BrandProfile.id)
        .filter(
            func.coalesce(AutonomousPost.organization_id, BrandProfile.organization_id) == org_id,
            func.lower(PostVariant.status).in_(["posted", "published"]),
        )
        .all()
    )

    for variant, auto_post in variant_rows:
        pub_ts = auto_post.published_at or variant.scheduled_time or variant.created_at
        if not _in_billing_window(pub_ts, period_start, period_end, end_inclusive):
            continue
        plat = str(getattr(variant.platform, "value", variant.platform) or "").strip().lower()
        ext_id = (variant.platform_post_id or "").strip()
        delivery_key = ("ap_ext", plat, ext_id) if ext_id else ("ap_row", variant.id)
        seen_deliveries.add(delivery_key)

    # Legacy/standalone published Content rows that have no linked ScheduledPost rows
    standalone_contents = (
        db.query(Content.id, Content.created_at)
        .outerjoin(BrandProfile, Content.brand_profile_id == BrandProfile.id)
        .filter(
            func.coalesce(Content.organization_id, BrandProfile.organization_id) == org_id,
            func.lower(cast(Content.status, String)) == "published",
            ~exists().where(ScheduledPost.content_id == Content.id),
        )
        .all()
    )
    for content_id, created_at in standalone_contents:
        if _in_billing_window(created_at, period_start, period_end, end_inclusive):
            seen_deliveries.add(("content_standalone", content_id))

    return len(seen_deliveries)


@router.get("/billing", response_model=List[BillingSubscriptionResponse])
def get_superadmin_billing(
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """
    Dedicated view for Super Admin to inspect verified Razorpay billing and subscription references.
    Exposes ONLY registered email, plan, status, billing period, limits, and live asset/delivery usage.
    """
    subscriptions = db.query(Subscription).order_by(Subscription.id.desc()).all()
    results = []
    now_utc = datetime.utcnow()

    for sub in subscriptions:
        admin_user = db.query(User).filter(
            User.organization_id == sub.organization_id,
            User.role.in_([UserRole.ADMIN, UserRole.SUPER_ADMIN])
        ).order_by(User.id.asc()).first()

        registered_email = admin_user.email if admin_user else f"org_{sub.organization_id}@system.local"

        org = db.query(Organization).filter(Organization.id == sub.organization_id).first()
        account_status = org.is_active if org else True

        tier_val = sub.tier.value if hasattr(sub.tier, "value") else str(sub.tier)
        status_val = sub.status.value if hasattr(sub.status, "value") else str(sub.status)

        # Do not treat payment ID prefixes, order ID prefixes, or ACTIVE subscription status as proof of verification.
        # Provenance is Unverified/Unavailable because cryptographic signature audit is not persisted in DB.
        # When verification status is Unverified or Unavailable, razorpay_payment_id and razorpay_order_id must NOT be returned.
        has_payment_ref = bool(sub.stripe_subscription_id)
        verif_status = "Unverified" if has_payment_ref else "Unavailable"

        # Return both references only if reliable, persisted verification evidence can actually establish that the payment was verified.
        # Without persisted signature proofs, references remain None.
        payment_ref = None
        order_ref = None

        period_start, period_end, display_start, display_end, end_inclusive = _resolve_billing_period(sub, now=now_utc)
        gen_usage = _count_billable_ai_assets(db, sub.organization_id, period_start, period_end, end_inclusive)
        posts_usage = _count_successful_deliveries(db, sub.organization_id, period_start, period_end, end_inclusive)

        results.append(BillingSubscriptionResponse(
            id=sub.id,
            registered_email=registered_email,
            plan=tier_val,
            subscription_status=status_val,
            account_status=account_status,
            razorpay_payment_id=payment_ref,
            razorpay_order_id=order_ref,
            verification_status=verif_status,
            current_period_start=display_start,
            current_period_end=display_end,
            monthly_generation_limit=sub.monthly_generation_limit if sub.monthly_generation_limit is not None else 50,
            monthly_posts_limit=sub.monthly_posts_limit if sub.monthly_posts_limit is not None else 100,
            current_generation_usage=gen_usage,
            current_posts_usage=posts_usage
        ))

    return results


# ---------------------------------------------------------
# 10. Platform Health Endpoint
# ---------------------------------------------------------

@router.get("/health", response_model=PlatformHealthResponse)
def get_superadmin_health(
    db: Session = Depends(get_db),
    current_superadmin: User = Depends(require_super_admin)
):
    """
    Unified platform health and infrastructure telemetry for the Super Admin Owner Console.
    Inspects actual database ping latency, Redis availability, background workers, and failed job counts.
    """
    # 1. Database connectivity & latency check
    start_time = time.perf_counter()
    try:
        db.execute(text("SELECT 1"))
        db_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        db_connected = True
    except Exception:
        db_connected = False
        db_latency_ms = 0.0

    # 2. Redis availability
    try:
        redis_ok = RedisManager.is_available()
    except Exception:
        redis_ok = False

    # 3. Publishing worker runtime status
    try:
        is_worker_running = getattr(PublishingWorker, "_is_running", False)
        publishing_status = "running" if is_worker_running else "stopped"
    except Exception:
        publishing_status = "unavailable"

    # 4. Autonomous scheduler runtime status
    try:
        is_scheduler_running = getattr(AutonomousScheduler, "_is_running", False)
        scheduler_status = "running" if is_scheduler_running else "stopped"
    except Exception:
        scheduler_status = "unavailable"

    # 5. Failed background content & generation jobs
    failed_contents = db.query(Content).filter(Content.status == ContentStatus.FAILED).count()
    failed_jobs = db.query(GenerationJob).filter(GenerationJob.status == JobStatus.FAILED).count()

    return PlatformHealthResponse(
        api_status="healthy",
        database_connected=db_connected,
        database_latency_ms=db_latency_ms,
        redis_available=redis_ok,
        publishing_worker_status=publishing_status,
        autonomous_scheduler_status=scheduler_status,
        failed_content_count=failed_contents,
        failed_generation_jobs_count=failed_jobs
    )


__all__ = ["router"]
