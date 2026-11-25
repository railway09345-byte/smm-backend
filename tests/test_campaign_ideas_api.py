import pytest
import time
from unittest.mock import patch
from datetime import datetime, date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

from app.db.base import Base
from app.models.organization import Organization
from app.models.user import User
from app.models.brand_profile import BrandProfile
from app.models.campaign import Campaign
from app.models.content import Content
from app.utils.enums import UserRole, ContentStatus, PlatformType
from app.services.content_planner_service import ContentPlannerService


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed orgs
    org1 = Organization(id=1, name="Test Org 1", slug="test-org-1")
    org2 = Organization(id=2, name="Test Org 2", slug="test-org-2")
    session.add_all([org1, org2])
    session.flush()

    # Seed users
    u1 = User(id=1, email="user1@example.com", hashed_password="pwd", full_name="User 1", role=UserRole.CREATOR, organization_id=1, is_active=True)
    u2 = User(id=2, email="user2@example.com", hashed_password="pwd", full_name="User 2", role=UserRole.CREATOR, organization_id=2, is_active=True)
    session.add_all([u1, u2])
    session.flush()

    # Seed brand
    bp = BrandProfile(id=1, name="Clay & Co.", industry="Ceramics", organization_id=1)
    session.add(bp)
    session.flush()

    yield session
    session.close()


def test_get_campaign_ideas_empty_campaign(db_session):
    """Test retrieving campaign ideas when campaign has 0 ideas."""
    camp = Campaign(
        id=10,
        title="Empty Campaign",
        objective="Brand Awareness",
        start_date=date(2026, 11, 1),
        end_date=date(2026, 11, 7),
        tags={"duration_days": 7, "campaign_goal": "awareness"},
        organization_id=1,
        brand_profile_id=1,
        status="active"
    )
    db_session.add(camp)
    db_session.commit()

    t0 = time.time()
    res = ContentPlannerService.get_campaign_ideas(db_session, organization_id=1, campaign_id=10)
    elapsed_ms = (time.time() - t0) * 1000

    assert res["campaign_id"] == 10
    assert res["total_ideas"] == 0
    assert len(res["ideas"]) == 0
    assert res["title"] == "Empty Campaign"
    assert elapsed_ms < 50.0, f"Query took too long: {elapsed_ms}ms"


def test_get_campaign_ideas_many_ideas(db_session):
    """Test retrieving campaign ideas for 30-day campaign with 30 ideas."""
    camp = Campaign(
        id=20,
        title="30-Day Omnichannel Campaign",
        objective="Lead Generation",
        start_date=date(2026, 12, 1),
        end_date=date(2026, 12, 30),
        tags={"duration_days": 30, "campaign_goal": "lead_generation"},
        organization_id=1,
        brand_profile_id=1,
        status="active"
    )
    db_session.add(camp)
    db_session.flush()

    # Create 30 content ideas
    contents = []
    for day in range(1, 31):
        c = Content(
            id=100 + day,
            title=f"Day {day}: Topic Idea",
            caption=f"Full caption for day {day}",
            hashtags=["#ceramics", f"#day{day}"],
            target_platforms=[PlatformType.INSTAGRAM, PlatformType.FACEBOOK],
            status=ContentStatus.DRAFT,
            meta_info={
                "is_planned_idea": True,
                "day_number": day,
                "hook": f"Hook for day {day}",
                "caption_brief": f"Brief for day {day}",
                "description": f"Description for day {day}",
                "content_format": "reel" if day % 2 == 0 else "image",
                "suggested_date": f"2026-12-{day:02d}",
                "suggested_time": "10:00",
                "plan_status": "planned"
            },
            organization_id=1,
            brand_profile_id=1,
            campaign_id=20,
            creator_id=1
        )
        contents.append(c)

    db_session.add_all(contents)
    db_session.commit()

    t0 = time.time()
    res = ContentPlannerService.get_campaign_ideas(db_session, organization_id=1, campaign_id=20)
    elapsed_ms = (time.time() - t0) * 1000

    assert res["campaign_id"] == 20
    assert res["total_ideas"] == 30
    assert len(res["ideas"]) == 30
    assert res["ideas"][0]["day_number"] == 1
    assert res["ideas"][29]["day_number"] == 30
    assert elapsed_ms < 50.0, f"Query took too long: {elapsed_ms}ms"


def test_get_campaign_ideas_no_ai_invocations(db_session):
    """Verify loading campaign ideas performs 0 AI calls (pure DB query)."""
    camp = Campaign(
        id=30,
        title="Zero AI Test Campaign",
        objective="Verification",
        organization_id=1,
        brand_profile_id=1,
        status="active"
    )
    c = Content(
        id=301,
        title="Existing Idea",
        caption="Existing Caption",
        hashtags=["#test"],
        target_platforms=[PlatformType.INSTAGRAM],
        meta_info={"day_number": 1, "description": "Existing Caption"},
        organization_id=1,
        brand_profile_id=1,
        campaign_id=30,
        creator_id=1
    )
    db_session.add_all([camp, c])
    db_session.commit()

    with patch("app.services.gemini_service.GeminiService.generate_text") as mock_gemini, \
         patch("app.services.ai_provider_manager.AIProviderManager.generate_json_with_resilience") as mock_ai_mgr:
        res = ContentPlannerService.get_campaign_ideas(db_session, organization_id=1, campaign_id=30)
        assert mock_gemini.call_count == 0
        assert mock_ai_mgr.call_count == 0
        assert res["total_ideas"] == 1


def test_get_campaign_ideas_tenant_isolation(db_session):
    """Verify tenant isolation: organization 2 cannot access organization 1's campaign."""
    camp = Campaign(
        id=40,
        title="Org 1 Private Campaign",
        objective="Secret",
        organization_id=1,
        brand_profile_id=1,
        status="active"
    )
    db_session.add(camp)
    db_session.commit()

    # Org 2 attempts to fetch Org 1's campaign -> must raise 404
    with pytest.raises(HTTPException) as exc_info:
        ContentPlannerService.get_campaign_ideas(db_session, organization_id=2, campaign_id=40)
    assert exc_info.value.status_code == 404


def test_get_campaign_ideas_non_existent(db_session):
    """Verify 404 when campaign does not exist."""
    with pytest.raises(HTTPException) as exc_info:
        ContentPlannerService.get_campaign_ideas(db_session, organization_id=1, campaign_id=999999)
    assert exc_info.value.status_code == 404
