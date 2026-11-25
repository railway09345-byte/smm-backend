import pytest
import os
import uuid
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.unified_ai_service import UnifiedAIService
from app.core.ai_model_config import centralized_ai_config
from app.models.user import User
from app.models.organization import Organization
from app.models.brand_profile import BrandProfile
from app.models.generation_job import GenerationJob
from app.utils.enums import JobStatus, GenerationType, UserRole
from app.core.security import create_access_token


@pytest.fixture(autouse=True)
def ensure_feature_flag_reset():
    """Ensure feature flags reset to False after each test."""
    original_val = getattr(settings, "FEATURE_UNIFIED_AI_STUDIO", False)
    yield
    settings.FEATURE_UNIFIED_AI_STUDIO = original_val


@pytest.fixture
def auth_creator(db_session):
    """Seed test creator user and complete BrandProfile with valid auth headers."""
    org = Organization(name="Test Studio Org", slug=f"studio-org-{uuid.uuid4().hex[:6]}", settings={})
    db_session.add(org)
    db_session.flush()

    user = User(
        email=f"creator_{uuid.uuid4().hex[:6]}@example.com",
        hashed_password="hashed_pw",
        full_name="Creator User",
        role=UserRole.CREATOR,
        organization_id=org.id,
        is_active=True
    )
    db_session.add(user)
    db_session.flush()

    brand = BrandProfile(
        organization_id=org.id,
        name="LuxeCraft",
        industry="Lifestyle & Design",
        tone_of_voice="Sophisticated & Modern",
        cta="Explore the collection",
        brand_guidelines="Minimalist luxury aesthetics and sustainable artisan products."
    )
    db_session.add(brand)
    db_session.commit()
    db_session.refresh(user)

    token = create_access_token(subject=str(user.id))
    headers = {"Authorization": f"Bearer {token}"}
    return {"user": user, "org": org, "brand": brand, "headers": headers}


@pytest.mark.asyncio
async def test_unified_ai_service_text_direct():
    """Verify UnifiedAIService directly synthesizes complete social copy."""
    res = await UnifiedAIService.generate_text(
        prompt="Launch of our new eco-friendly sneaker line",
        brand_name="EcoStride",
        brand_tone="Inspiring",
        brand_industry="Footwear",
        brand_cta="Shop the collection now",
        target_platforms=["instagram"]
    )
    assert res is not None
    assert "caption" in res
    assert len(res["caption"]) > 0
    assert "hashtags" in res
    assert isinstance(res["hashtags"], list)
    assert res.get("provider") in ["google-gemini", "openai", "deterministic-brand-template"]


@pytest.mark.asyncio
async def test_unified_ai_service_fallback_on_llm_failure():
    """Verify UnifiedAIService falls back to deterministic brand engine when LLMs fail."""
    with patch("app.services.ai_provider_manager.AIProviderManager._execute_gemini_call", side_effect=Exception("Gemini quota 429")):
        with patch("app.services.ai_provider_manager.AIProviderManager._execute_openai_call", side_effect=Exception("OpenAI 500 error")):
            res = await UnifiedAIService.generate_text(
                prompt="Summer sale announcement",
                brand_name="GlowSpa",
                brand_tone="Luxurious",
                brand_industry="Wellness",
                brand_cta="Book today"
            )
            assert res is not None
            assert res["provider"] == "deterministic-brand-template"
            assert "GlowSpa" in res["headline"]
            assert "Book today" in res["call_to_action"]


def test_ai_studio_route_with_legacy_flag_disabled(client, auth_creator):
    """Verify legacy path executes normally when FEATURE_UNIFIED_AI_STUDIO is False."""
    settings.FEATURE_UNIFIED_AI_STUDIO = False

    payload = {
        "prompt": "Announcing our new summer menu",
        "content_type": "caption",
        "target_platforms": ["instagram"],
        "tone": "Warm & Welcoming",
        "brand_profile_id": auth_creator["brand"].id
    }

    with patch("app.services.gemini_service.GeminiService.generate_text", new_callable=AsyncMock) as mock_gemini:
        mock_gemini.return_value = {
            "headline": "Fresh Summer Flavors",
            "caption": "Experience our farm-to-table summer menu! 🥗",
            "hashtags": ["#SummerMenu", "#Foodie"],
            "call_to_action": "Reserve your table today",
            "model_name": "gemini-3.1-flash-lite",
            "tokens_used": 150
        }
        resp = client.post("/api/v1/generations/gemini/text", json=payload, headers=auth_creator["headers"])
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] in ["completed", "COMPLETED", "pending", "processing"]
        assert mock_gemini.called


def test_ai_studio_route_with_unified_flag_enabled(client, auth_creator):
    """Verify UnifiedAIService is engaged when FEATURE_UNIFIED_AI_STUDIO is True."""
    settings.FEATURE_UNIFIED_AI_STUDIO = True

    payload = {
        "prompt": "Flash sale 50 percent off everything",
        "content_type": "caption",
        "target_platforms": ["instagram"],
        "tone": "Exciting",
        "brand_profile_id": auth_creator["brand"].id
    }

    with patch("app.services.unified_ai_service.UnifiedAIService.generate_text", new_callable=AsyncMock) as mock_unified:
        mock_unified.return_value = {
            "headline": "Flash Sale Alert!",
            "caption": "Get 50% off all items this weekend only! 🔥",
            "hashtags": ["#FlashSale", "#Discounts"],
            "call_to_action": "Shop the sale",
            "model_name": "gemini-3.5-flash",
            "provider": "google-gemini",
            "tokens_used": 180,
            "generated_at": "2026-11-12T00:00:00Z"
        }
        resp = client.post("/api/v1/generations/gemini/text", json=payload, headers=auth_creator["headers"])
        assert resp.status_code == 201
        assert mock_unified.called
        data = resp.json()
        rd = data.get("result_data", {})
        assert "50% off" in rd.get("caption", "")


def test_ai_studio_route_persists_generation_job(client, auth_creator):
    """Verify GenerationJob is properly created and queryable in database."""
    settings.FEATURE_UNIFIED_AI_STUDIO = True

    payload = {
        "prompt": "Announcing our flagship store opening",
        "content_type": "caption",
        "target_platforms": ["instagram"],
        "brand_profile_id": auth_creator["brand"].id
    }

    with patch("app.services.unified_ai_service.UnifiedAIService.generate_text", new_callable=AsyncMock) as mock_unified:
        mock_unified.return_value = {
            "headline": "Our Doors Are Open!",
            "caption": "Join us at our new flagship location. 🎉",
            "hashtags": ["#GrandOpening"],
            "call_to_action": "Visit us this weekend",
            "model_name": "gemini-3.5-flash",
            "provider": "google-gemini",
            "tokens_used": 200,
            "generated_at": "2026-11-12T00:00:00Z"
        }
        resp = client.post("/api/v1/generations/gemini/text", json=payload, headers=auth_creator["headers"])
        assert resp.status_code == 201
        job_id = resp.json()["id"]

        # Verify query from /generations/jobs/{job_id}
        job_check = client.get(f"/api/v1/generations/jobs/{job_id}", headers=auth_creator["headers"])
        assert job_check.status_code == 200
        assert job_check.json()["id"] == job_id
