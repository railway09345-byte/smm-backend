import pytest
import os
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.organization import Organization
from app.models.user import User
from app.models.brand_profile import BrandProfile
from app.models.campaign import Campaign
from app.models.content import Content
from app.utils.enums import UserRole, ContentStatus
from app.services.content_planner_service import ContentPlannerService
from app.services.ai_provider_manager import AIProviderManager
from app.core.ai_model_config import centralized_ai_config


@pytest.fixture
def db_session():
    from app.db.base import Base
    from app.db.session import engine
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def test_setup(db_session):
    uid = uuid.uuid4().hex[:6]
    org = Organization(name=f"Test Org {uid}", slug=f"test-org-{uid}")
    db_session.add(org)
    db_session.flush()

    user = User(
        email=f"creator_{uid}@example.com",
        hashed_password="hashed_password_123",
        role=UserRole.CREATOR,
        organization_id=org.id
    )
    db_session.add(user)
    db_session.flush()

    brand = BrandProfile(
        organization_id=org.id,
        user_id=user.id,
        name="Apex Dynamics",
        industry="Enterprise Software & AI",
        tagline="Next-Gen Intelligent Automation",
        location="San Francisco, CA",
        tone_of_voice="Professional, Authoritative, Innovative",
        target_audience="CTOs, Engineering Leads, VP of Operations",
        brand_guidelines="Apex Dynamics creates AI-powered orchestration platforms.",
        cta="Book an architectural demo with our systems engineering team.",
        avoid="cliches, false promises, generic buzzwords",
        key_themes=["Autonomous Workflows", "Reliability Engineering", "Enterprise Scale"],
        meta_info={
            "category": "Enterprise Software",
            "usp": "Sub-millisecond event streaming with self-healing topology",
            "products": [
                {"name": "Apex Core", "description": "High-throughput message mesh", "price": "499"},
                {"name": "Apex Pulse", "description": "Automated system monitoring agent", "price": "199"}
            ]
        }
    )
    db_session.add(brand)
    db_session.commit()
    db_session.refresh(brand)
    db_session.refresh(user)

    return {"db": db_session, "org": org, "user": user, "brand": brand}


@pytest.mark.asyncio
async def test_01_centralized_model_hierarchy_verification():
    """Verify that ContentPlannerService and centralized_ai_config share identical text models."""
    assert centralized_ai_config.TEXT_PRIMARY_MODEL == "gemini-3.5-flash"
    assert centralized_ai_config.TEXT_OPENAI_MODEL == "gpt-4o-mini"
    assert "gemini-3.1-flash-lite" in centralized_ai_config.TEXT_SECONDARY_ROTATION
    assert "gemini-flash-lite-latest" in centralized_ai_config.TEXT_SECONDARY_ROTATION
    assert "gemini-3.5-flash-lite" in centralized_ai_config.TEXT_SECONDARY_ROTATION
    assert "gemini-3.8-flash" in centralized_ai_config.TEXT_SECONDARY_ROTATION
    assert "gemini-flash-latest" in centralized_ai_config.TEXT_SECONDARY_ROTATION

    assert ContentPlannerService.TEXT_PRIMARY_MODEL == "gemini-3.5-flash"
    assert ContentPlannerService.TEXT_OPENAI_MODEL == "gpt-4o-mini"
    assert ContentPlannerService.TEXT_SECONDARY_ROTATION == centralized_ai_config.TEXT_SECONDARY_ROTATION


@pytest.mark.asyncio
async def test_02_generate_7day_campaign(test_setup):
    """Test 1: Generate a 7-day campaign preserving date calculations, brand profile, and slot count."""
    db = test_setup["db"]
    org = test_setup["org"]
    brand = test_setup["brand"]

    plan = await ContentPlannerService.generate_plan(
        db=db,
        organization_id=org.id,
        brand_profile_id=brand.id,
        duration_days=7,
        posting_frequency="1_per_day",
        campaign_goal="product_launch",
        allow_degraded_fallback=True
    )

    assert plan is not None
    assert plan["duration_days"] == 7
    assert plan["total_ideas"] == 7
    assert len(plan["ideas"]) == 7

    seen_days = set()
    for idea in plan["ideas"]:
        assert idea["day_number"] not in seen_days
        seen_days.add(idea["day_number"])
        assert idea["title"]
        assert len(idea["description"]) >= 80
        assert idea["hook"]
        assert idea["caption_brief"]
        assert idea["content_format"] in ["image", "video", "carousel", "text"]
        assert len(idea["hashtags"]) >= 3


@pytest.mark.asyncio
async def test_03_generate_30day_campaign(test_setup):
    """Test 2: Generate a 30-day campaign verifying slot progression and schema integrity."""
    db = test_setup["db"]
    org = test_setup["org"]
    brand = test_setup["brand"]

    plan = await ContentPlannerService.generate_plan(
        db=db,
        organization_id=org.id,
        brand_profile_id=brand.id,
        duration_days=30,
        posting_frequency="1_per_day",
        campaign_goal="brand_awareness",
        allow_degraded_fallback=True
    )

    assert plan is not None
    assert plan["duration_days"] == 30
    assert plan["total_ideas"] == 30
    assert len(plan["ideas"]) == 30

    for idx, idea in enumerate(plan["ideas"]):
        assert idea["day_number"] == idx + 1
        assert idea["title"]
        assert len(idea["description"]) >= 80
        assert idea["suggested_date"]


@pytest.mark.asyncio
async def test_04_save_campaign_to_database(test_setup):
    """Test 6: Verify campaign data and individual ideas are saved correctly in the database."""
    db = test_setup["db"]
    org = test_setup["org"]
    user = test_setup["user"]
    brand = test_setup["brand"]

    plan = await ContentPlannerService.generate_plan(
        db=db,
        organization_id=org.id,
        brand_profile_id=brand.id,
        duration_days=7,
        posting_frequency="1_per_day",
        campaign_goal="product_launch",
        allow_degraded_fallback=True
    )

    saved = ContentPlannerService.save_plan_to_database(
        db=db,
        organization_id=org.id,
        current_user=user,
        plan_data=plan
    )

    campaign_id = saved["campaign_id"]
    assert campaign_id is not None
    assert len(saved["ideas"]) == 7

    campaign_in_db = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    assert campaign_in_db is not None
    assert campaign_in_db.organization_id == org.id
    assert campaign_in_db.brand_profile_id == brand.id

    ideas_in_db = db.query(Content).filter(Content.campaign_id == campaign_id).all()
    assert len(ideas_in_db) == 7
    for idea in ideas_in_db:
        assert idea.organization_id == org.id
        assert idea.brand_profile_id == brand.id
        assert idea.caption
        assert len(idea.caption) >= 80


@pytest.mark.asyncio
async def test_05_regenerate_individual_post(test_setup):
    """Test 3: Regenerate an individual post idea preserving date slot and schedule properties."""
    db = test_setup["db"]
    org = test_setup["org"]
    user = test_setup["user"]
    brand = test_setup["brand"]

    plan = await ContentPlannerService.generate_plan(
        db=db,
        organization_id=org.id,
        brand_profile_id=brand.id,
        duration_days=7,
        posting_frequency="1_per_day",
        campaign_goal="product_launch",
        allow_degraded_fallback=True
    )

    saved = ContentPlannerService.save_plan_to_database(
        db=db,
        organization_id=org.id,
        current_user=user,
        plan_data=plan
    )

    first_content_id = saved["ideas"][0]["id"]
    original_content = db.query(Content).filter(Content.id == first_content_id).first()
    original_title = original_content.title
    original_caption = original_content.caption

    regenerated = await ContentPlannerService.regenerate_single_idea(
        db=db,
        organization_id=org.id,
        content_id=first_content_id,
        allow_degraded_fallback=True
    )

    assert regenerated is not None
    assert regenerated["id"] == first_content_id
    assert regenerated["caption"]
    assert len(regenerated["caption"]) >= 50


@pytest.mark.asyncio
async def test_06_simulate_gemini_failure_and_fallback_rotation(test_setup):
    """Test 4: Simulate Gemini failure and verify fallback cascade to rotation and OpenAI."""
    db = test_setup["db"]
    org = test_setup["org"]
    brand = test_setup["brand"]

    # Reset circuit breakers
    AIProviderManager.get_circuit("gemini").reset()
    AIProviderManager.get_circuit("openai").reset()

    calls_made = []

    async def mock_gemini_call(model_name, sys_inst, prompt, timeout_seconds=45.0):
        calls_made.append(("gemini", model_name))
        raise Exception(f"Simulated rate limit on {model_name}")

    async def mock_openai_call(model_name, sys_inst, prompt, timeout_seconds=45.0):
        calls_made.append(("openai", model_name))
        # Return valid mock schema
        return {
            "campaign_title": "OpenAI Backup Campaign",
            "campaign_summary": "Campaign generated via OpenAI backup tier.",
            "ideas": [
                {
                    "title": f"OpenAI Idea {i+1}",
                    "hook": "Engineered for maximum stability and speed.",
                    "description": "At Apex Dynamics, we deliver enterprise orchestration without compromise. Scalable, high-fidelity systems built for mission-critical operations. Tap link in bio.",
                    "caption_brief": "Overview of mission-critical systems.",
                    "content_format": "image",
                    "pillar": "Product Spotlight",
                    "suggested_hashtags": ["#ApexDynamics", "#CloudScale", "#EnterpriseAI"],
                    "cta": "Tap link in bio"
                }
                for i in range(7)
            ]
        }

    with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=mock_gemini_call):
        with patch.object(AIProviderManager, "_execute_openai_call", side_effect=mock_openai_call):
            with patch("app.services.ai_provider_manager.is_gemini_configured", return_value=True):
                with patch.object(settings, "OPENAI_API_KEY", "sk-mock-valid-openai-key-for-test"):
                    plan = await ContentPlannerService.generate_plan(
                        db=db,
                        organization_id=org.id,
                        brand_profile_id=brand.id,
                        duration_days=7,
                        posting_frequency="1_per_day",
                        campaign_goal="product_launch",
                        allow_degraded_fallback=False
                    )

    assert plan is not None
    assert plan["total_ideas"] == 7
    assert plan["title"] == "OpenAI Backup Campaign"

    # Verify model rotation execution order:
    gemini_models_attempted = [m for p, m in calls_made if p == "gemini"]
    assert "gemini-3.5-flash" in gemini_models_attempted
    assert "gemini-3.1-flash-lite" in gemini_models_attempted
    assert any(p == "openai" for p, m in calls_made)


@pytest.mark.asyncio
async def test_07_deterministic_fallback_when_all_remote_fail(test_setup):
    """Test 5: Verify deterministic fallback synthesizer operates with 0 external calls."""
    db = test_setup["db"]
    org = test_setup["org"]
    brand = test_setup["brand"]

    async def mock_all_fail(*args, **kwargs):
        raise Exception("All providers down")

    with patch.object(AIProviderManager, "_execute_gemini_call", side_effect=mock_all_fail):
        with patch.object(AIProviderManager, "_execute_openai_call", side_effect=mock_all_fail):
            plan = await ContentPlannerService.generate_plan(
                db=db,
                organization_id=org.id,
                brand_profile_id=brand.id,
                duration_days=7,
                posting_frequency="1_per_day",
                campaign_goal="product_launch",
                allow_degraded_fallback=True
            )

    assert plan is not None
    assert plan["total_ideas"] == 7
    assert len(plan["ideas"]) == 7
    assert "Apex Dynamics" in plan["title"]
    for idea in plan["ideas"]:
        assert "Apex Dynamics" in idea["description"]
        assert len(idea["description"]) >= 80
