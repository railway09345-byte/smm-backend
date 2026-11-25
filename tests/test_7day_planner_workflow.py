import pytest
import asyncio
from datetime import datetime, timedelta
from app.db.session import SessionLocal
from app.models.organization import Organization
from app.models.user import User
from app.models.brand_profile import BrandProfile
from app.models.campaign import Campaign
from app.models.content import Content
from app.models.scheduled_post import ScheduledPost
from app.models.approval import Approval
from app.utils.enums import UserRole, ContentStatus, ApprovalStatus
from app.services.content_planner_service import ContentPlannerService
from app.services.approval_service import ApprovalService
from app.schemas.approval import ApprovalReview


@pytest.fixture
def test_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.mark.asyncio
async def test_7day_instant_descriptions_and_scheduling(test_db):
    db = test_db
    
    # Setup test Org, User & Brand Profile matching user's account details
    org = db.query(Organization).filter(Organization.slug == "test-zenvora-org").first()
    if not org:
        org = Organization(name="Test Zenvora Org", slug="test-zenvora-org")
        db.add(org)
        db.commit()
        db.refresh(org)

    user = db.query(User).filter(User.email == "test_zenvora_user@example.com").first()
    if not user:
        user = User(
            email="test_zenvora_user@example.com",
            hashed_password="fakehashpassword123",
            role=UserRole.ADMIN,
            organization_id=org.id
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.role = UserRole.ADMIN
        db.commit()

    brand = db.query(BrandProfile).filter(BrandProfile.organization_id == org.id).first()
    if not brand:
        brand = BrandProfile(
            organization_id=org.id,
            user_id=user.id,
            name="Zenvora Technologies Pvt. Ltd.",
            industry="Technology & Software",
            tagline="Clean Cloud Automation for Modern Teams",
            location="Chennai, India",
            tone_of_voice="Professional & Inspiring",
            target_audience="Small and mid-sized business owners",
            brand_guidelines="Zenvora builds cloud-based tools for small and mid-sized companies that want automation without enterprise complexity.",
            cta="Schedule your free cloud automation walkthrough today.",
            avoid="cliches, deceptive claims, confusing tech jargon",
            key_themes=["Cloud Automation", "Workflow Simplicity", "Team Productivity"],
            meta_info={
                "category": "Technology & Software",
                "description": "Zenvora builds cloud-based tools for small and mid-sized companies that want automation without enterprise complexity.",
                "usp": "Intuitive automation without enterprise bloat",
                "products": [
                    {"name": "Zenvora Flow", "description": "Automated workflow builder for small teams", "price": "49"},
                    {"name": "Zenvora Sync", "description": "Cloud data synchronization engine", "price": "99"}
                ]
            }
        )
        db.add(brand)
        db.commit()
        db.refresh(brand)

    # 1. Generate 7-day Plan
    plan_result = await ContentPlannerService.generate_plan(
        db=db,
        organization_id=org.id,
        brand_profile_id=brand.id,
        duration_days=7,
        posting_frequency="1_per_day",
        campaign_goal="product_launch"
    )

    assert plan_result is not None
    assert len(plan_result["ideas"]) == 7

    # Verify every day has distinct topic and instant full description (not empty or trivial)
    seen_topics = set()
    for idea in plan_result["ideas"]:
        assert idea["title"]
        assert idea["title"] not in seen_topics
        seen_topics.add(idea["title"])

        # Check instant description
        assert "description" in idea
        assert "caption" in idea
        assert len(idea["description"]) > 80, f"Description too short for idea {idea['title']}"
        assert idea["cta"]
        assert len(idea["hashtags"]) >= 3

    # 2. Save Plan to Database
    save_result = ContentPlannerService.save_plan_to_database(
        db=db,
        organization_id=org.id,
        current_user=user,
        plan_data=plan_result
    )

    campaign_id = save_result["campaign_id"]
    assert campaign_id is not None
    assert len(save_result["ideas"]) == 7

    # Verify database contents saved with full descriptions
    saved_contents = db.query(Content).filter(Content.campaign_id == campaign_id).all()
    assert len(saved_contents) == 7
    for sc in saved_contents:
        assert sc.caption is not None and len(sc.caption) > 80

    # 3. Test Auto-Schedule Plan with Auto-Post (Instant Auto-Approve)
    schedule_res_auto = ContentPlannerService.schedule_campaign_plan(
        db=db,
        organization_id=org.id,
        campaign_id=campaign_id,
        current_user=user,
        auto_approve=True,
        timezone_str="Asia/Kolkata",
        target_platform="instagram"
    )

    assert schedule_res_auto["success"] is True
    assert schedule_res_auto["mode"] == "auto_post"
    assert schedule_res_auto["scheduled_count"] == 7

    posts = db.query(ScheduledPost).filter(
        ScheduledPost.organization_id == org.id,
        ScheduledPost.publish_metadata["campaign_id"].as_integer() == campaign_id
    ).all()
    assert len(posts) == 7
    for p in posts:
        assert p.status == "scheduled"
        assert p.scheduled_time is not None
        assert p.publish_metadata.get("auto_generate_media") is True

    # 4. Test Auto-Schedule Plan with Approval-Required Mode
    schedule_res_approval = ContentPlannerService.schedule_campaign_plan(
        db=db,
        organization_id=org.id,
        campaign_id=campaign_id,
        current_user=user,
        auto_approve=False,
        timezone_str="Asia/Kolkata",
        target_platform="instagram"
    )

    assert schedule_res_approval["success"] is True
    assert schedule_res_approval["mode"] == "approval_required"
    assert schedule_res_approval["scheduled_count"] == 7

    posts_after = db.query(ScheduledPost).filter(
        ScheduledPost.organization_id == org.id,
        ScheduledPost.publish_metadata["campaign_id"].as_integer() == campaign_id
    ).all()
    assert len(posts_after) == 7
    for p in posts_after:
        assert p.status == "pending_approval"

    # Verify Approvals were created
    first_content = saved_contents[0]
    apprv = db.query(Approval).filter(Approval.content_id == first_content.id).first()
    assert apprv is not None
    assert apprv.status == ApprovalStatus.PENDING

    # 5. Test Approval Review: Approving promotes linked ScheduledPost to "scheduled"
    review_req = ApprovalReview(
        status=ApprovalStatus.APPROVED,
        comment="Approved for 7-day automated publication!"
    )
    ApprovalService.review_approval(
        db=db,
        approval_id=apprv.id,
        reviewer=user,
        payload=review_req
    )

    db.refresh(first_content)
    assert first_content.status == ContentStatus.APPROVED

    linked_sp = db.query(ScheduledPost).filter(ScheduledPost.content_id == first_content.id).first()
    assert linked_sp.status == "scheduled", f"Expected scheduled post to be 'scheduled', got '{linked_sp.status}'"
    print("\n[ALL 7-DAY PLANNER & PUBLISHING WORKFLOW TESTS PASSED CLEANLY!]")
