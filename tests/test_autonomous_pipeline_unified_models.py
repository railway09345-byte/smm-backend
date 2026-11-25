import pytest
import os
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, date, timedelta

from app.core.config import settings
from app.core.ai_model_config import centralized_ai_config
from app.db.session import SessionLocal
from app.models.user import User
from app.models.organization import Organization
from app.models.brand_profile import BrandProfile
from app.models.autonomous_pipeline import (
    BusinessSchedule,
    PipelineRun,
    ContentBrief,
    GeneratedAsset,
    AutonomousPost,
    PostVariant
)
from app.services.daily_ai_provider import DailyAIProvider, DailyCampaign, VideoScene, ProviderFailure
from app.services.pipeline_gemini_service import PipelineGeminiService, CreativeBriefSchema, PlatformCaptionsSchema
from app.services.daily_campaign_service import enqueue_daily, execute_daily_run, verify_image
from app.services.pipeline_service import regenerate_post_part
from app.services.ai_provider_manager import AIProviderManager


@pytest.fixture
def setup_pipeline_db():
    from app.db.base import Base
    db = SessionLocal()
    Base.metadata.create_all(bind=db.get_bind())

    # Create Organization 1
    org1 = db.query(Organization).filter(Organization.name == "Unified Pipeline Org 1").first()
    if not org1:
        org1 = Organization(name="Unified Pipeline Org 1", slug="unified-pipe-org-1")
        db.add(org1)
        db.commit()
        db.refresh(org1)

    # Create Organization 2 (for tenant isolation)
    org2 = db.query(Organization).filter(Organization.name == "Unified Pipeline Org 2").first()
    if not org2:
        org2 = Organization(name="Unified Pipeline Org 2", slug="unified-pipe-org-2")
        db.add(org2)
        db.commit()
        db.refresh(org2)

    # Create Brand 1
    brand1 = db.query(BrandProfile).filter(BrandProfile.organization_id == org1.id).first()
    if not brand1:
        brand1 = BrandProfile(
            organization_id=org1.id,
            name="Luminary Artisan Coffee",
            industry="Food & Beverage",
            tagline="Sip the Extraordinary",
            tone_of_voice="Warm & Connoisseur",
            target_audience="Artisan coffee enthusiasts",
            key_themes=["Direct Trade", "Micro-Roasting", "Single Origin"],
            cta="Order Fresh Beans",
            meta_info={
                "products": [
                    {"name": "Ethiopia Yirgacheffe", "price": "24", "description": "Floral notes with citrus acidity"},
                    {"name": "Sumatra Mandheling", "price": "22", "description": "Full-bodied herbal chocolate roast"}
                ]
            }
        )
        db.add(brand1)
        db.commit()
        db.refresh(brand1)

    # Create Brand 2 (under Org 2)
    brand2 = db.query(BrandProfile).filter(BrandProfile.organization_id == org2.id).first()
    if not brand2:
        brand2 = BrandProfile(
            organization_id=org2.id,
            name="Horizon Cloud Solutions",
            industry="Software & Technology",
            tagline="Seamless Enterprise Scale",
            tone_of_voice="Authoritative & Innovative",
            target_audience="Enterprise CTOs and DevOps engineers",
            cta="Schedule Technical Demo"
        )
        db.add(brand2)
        db.commit()
        db.refresh(brand2)

    # Create Business Schedule for Brand 1
    sched1 = db.query(BusinessSchedule).filter(BusinessSchedule.brand_profile_id == brand1.id).first()
    if not sched1:
        sched1 = BusinessSchedule(
            brand_profile_id=brand1.id,
            generation_time="09:00",
            posting_time="18:00",
            timezone="UTC",
            auto_approve=True,
            auto_schedule=False,
            is_active=True,
            target_platforms=["instagram", "facebook", "linkedin"],
            content_type="both",
            image_count=1,
            video_count=1
        )
        db.add(sched1)
        db.commit()
        db.refresh(sched1)

    yield {"db": db, "org1": org1, "org2": org2, "brand1": brand1, "brand2": brand2, "sched1": sched1}
    db.close()


# =========================================================================
# 1. TEXT GENERATION & RESILIENCE TESTS
# =========================================================================

@pytest.mark.asyncio
async def test_daily_ai_provider_primary_gemini_success():
    """Verify primary model gemini-3.5-flash is attempted first and returns valid DailyCampaign."""
    snapshot = {
        "name": "Luminary Coffee",
        "industry": "Food & Beverage",
        "meta_info": {"products": [{"name": "Single Origin Roast"}]}
    }
    history = []
    config = {
        "image_count": 1,
        "video_count": 1,
        "video_seconds": 30,
        "platforms": ["instagram", "facebook", "linkedin"]
    }

    mock_campaign_payload = {
        "title": "Crafted Perfection: Morning Brew",
        "theme": "Micro-Roasting Excellence",
        "objective": "Highlight artisan roasting craft",
        "selected_offering": "Single Origin Roast",
        "description": "Experience hand-selected organic coffee beans roasted with utmost care for vibrant, rich flavor.",
        "caption": "Elevate your morning routine with freshly roasted single origin coffee. Roasted daily for peak aroma.",
        "hashtags": ["#LuminaryCoffee", "#ArtisanRoast", "#SpecialtyCoffee"],
        "call_to_action": "Order your bag today",
        "target_audience": "Coffee aficionados",
        "creative_direction": "Warm morning lighting, rustic cafe aesthetic",
        "platform_captions": {
            "instagram": "Morning perfection in every cup. ✨ #LuminaryCoffee",
            "facebook": "Discover why coffee lovers choose Luminary. Order fresh beans today.",
            "linkedin": "Craftsmanship at scale: The science of precision micro-roasting."
        },
        "image_prompts": ["Warm morning sunlight streaming onto fresh espresso cup on dark wood table"],
        "video_script": [
            {"visual": "Close-up of roasting beans", "narration": "From bean to brew.", "duration_seconds": 15},
            {"visual": "Barista pouring latte art", "narration": "Experience perfection.", "duration_seconds": 15}
        ],
        "video_prompts": ["Cinematic 9:16 vertical reel of espresso extraction"]
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.is_success = True
    mock_response.json.return_value = {
        "candidates": [{
            "content": {"parts": [{"text": json.dumps(mock_campaign_payload)}]}
        }],
        "usageMetadata": {"totalTokenCount": 350}
    }

    with patch.object(DailyAIProvider, "request", AsyncMock(return_value=mock_response)):
        campaign, telemetry = await DailyAIProvider.campaign(snapshot, history, config)
        assert isinstance(campaign, DailyCampaign)
        assert campaign.title == "Crafted Perfection: Morning Brew"
        assert telemetry["provider"] == "gemini"
        assert telemetry["model"] == centralized_ai_config.TEXT_PRIMARY_MODEL
        assert telemetry["model"] == "gemini-3.5-flash"


@pytest.mark.asyncio
async def test_daily_ai_provider_secondary_rotation_on_primary_failure():
    """Verify fallback rotation through secondary models when primary fails."""
    snapshot = {
        "name": "Luminary Coffee",
        "industry": "Food & Beverage",
        "meta_info": {"products": [{"name": "Single Origin Roast"}]}
    }
    history = []
    config = {
        "image_count": 1,
        "video_count": 0,
        "video_seconds": 30,
        "platforms": ["instagram", "facebook", "linkedin"]
    }

    mock_campaign_payload = {
        "title": "Fresh Roast Revolution",
        "theme": "Quality Focus",
        "objective": "Promote direct trade",
        "selected_offering": "Single Origin Roast",
        "description": "Experience hand-selected organic coffee beans roasted with utmost care for vibrant, rich flavor.",
        "caption": "Freshly roasted single origin beans delivered right to your door. Taste the difference.",
        "hashtags": ["#LuminaryCoffee", "#CoffeeDaily"],
        "call_to_action": "Order now",
        "target_audience": "Coffee aficionados",
        "creative_direction": "Minimalist coffee visual",
        "platform_captions": {
            "instagram": "Fresh beans daily. #LuminaryCoffee",
            "facebook": "Order fresh beans today.",
            "linkedin": "Precision roasting insights."
        },
        "image_prompts": ["Fresh coffee beans in wooden scoop"],
        "video_script": [],
        "video_prompts": []
    }

    calls = []

    async def mock_request(method, url, headers=None, json=None, **kwargs):
        calls.append(url)
        # Fail the first attempt (gemini-3.5-flash) with 429 Rate Limit
        if "gemini-3.5-flash:" in url:
            raise ProviderFailure("gemini", "429: Rate limit exceeded", retryable=True)
        # Succeed on second model (gemini-3.1-flash-lite)
        resp = MagicMock()
        resp.is_success = True
        resp.json.return_value = {
            "candidates": [{
                "content": {"parts": [{"text": json_mod.dumps(mock_campaign_payload)}]}
            }],
            "usageMetadata": {"totalTokenCount": 280}
        }
        return resp

    import json as json_mod
    with patch.object(DailyAIProvider, "request", side_effect=mock_request):
        campaign, telemetry = await DailyAIProvider.campaign(snapshot, history, config)
        assert isinstance(campaign, DailyCampaign)
        assert telemetry["provider"] == "gemini"
        assert telemetry["model"] == "gemini-3.1-flash-lite"
        assert any("gemini-3.5-flash:" in c for c in calls)
        assert any("gemini-3.1-flash-lite:" in c for c in calls)


@pytest.mark.asyncio
async def test_daily_ai_provider_openai_fallback():
    """Verify fallback to OpenAI gpt-4o-mini when all Gemini models fail."""
    snapshot = {
        "name": "Luminary Coffee",
        "industry": "Food & Beverage",
        "meta_info": {"products": [{"name": "Single Origin Roast"}]}
    }
    history = []
    config = {
        "image_count": 1,
        "video_count": 0,
        "video_seconds": 30,
        "platforms": ["instagram", "facebook", "linkedin"]
    }

    mock_campaign_payload = {
        "title": "OpenAI Synthesized Campaign",
        "theme": "Resilient Roast",
        "objective": "Showcase backup generation",
        "selected_offering": "Single Origin Roast",
        "description": "Experience hand-selected organic coffee beans roasted with utmost care for vibrant, rich flavor.",
        "caption": "Artisan coffee beans delivered direct to you with exceptional flavor and rich aroma.",
        "hashtags": ["#LuminaryCoffee"],
        "call_to_action": "Order now",
        "target_audience": "Coffee aficionados",
        "creative_direction": "Studio coffee still",
        "platform_captions": {
            "instagram": "Roasted to perfection. #LuminaryCoffee",
            "facebook": "Order fresh beans today.",
            "linkedin": "Craftsmanship at scale."
        },
        "image_prompts": ["Sleek coffee bag on marble counter"],
        "video_script": [],
        "video_prompts": []
    }

    async def mock_request(method, url, headers=None, json=None, **kwargs):
        if "generativelanguage.googleapis.com" in url:
            raise ProviderFailure("gemini", "503: Unavailable", retryable=True)
        if "api.openai.com" in url:
            resp = MagicMock()
            resp.is_success = True
            resp.json.return_value = {
                "choices": [{
                    "message": {"content": json_mod.dumps(mock_campaign_payload)}
                }],
                "usage": {"total_tokens": 310}
            }
            return resp
        raise RuntimeError("Unexpected URL")

    import json as json_mod
    with patch.object(DailyAIProvider, "request", side_effect=mock_request), \
         patch.object(settings, "OPENAI_API_KEY", "sk-proj-test-openai-key-valid"):
        campaign, telemetry = await DailyAIProvider.campaign(snapshot, history, config)
        assert isinstance(campaign, DailyCampaign)
        assert campaign.title == "OpenAI Synthesized Campaign"
        assert telemetry["provider"] == "openai"
        assert telemetry["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_daily_ai_provider_deterministic_fallback():
    """Verify final deterministic brand engine synthesis when all external providers fail."""
    snapshot = {
        "name": "Luminary Coffee",
        "industry": "Food & Beverage",
        "tone_of_voice": "Connoisseur & Warm",
        "cta": "Taste Luminary Today",
        "meta_info": {
            "products": [{"name": "Ethiopia Yirgacheffe Special", "price": "24"}]
        }
    }
    history = [{"title": "Old Campaign", "caption": "Old Caption"}]
    config = {
        "image_count": 2,
        "video_count": 1,
        "video_seconds": 30,
        "platforms": ["instagram", "facebook", "linkedin", "youtube"]
    }

    async def mock_request_all_fail(method, url, **kwargs):
        raise ProviderFailure("network", "All network calls blocked in offline test", retryable=True)

    with patch.object(DailyAIProvider, "request", side_effect=mock_request_all_fail), \
         patch.object(settings, "OPENAI_API_KEY", ""):
        campaign, telemetry = await DailyAIProvider.campaign(snapshot, history, config)
        assert isinstance(campaign, DailyCampaign)
        assert telemetry["provider"] == "deterministic-brand-engine"
        assert telemetry["model"] == "brand-facts-synthesizer"
        assert len(campaign.image_prompts) == 2
        assert len(campaign.video_prompts) == 1
        assert len(campaign.video_script) == 5
        assert all(plat in campaign.platform_captions for plat in ["instagram", "facebook", "linkedin", "youtube"])
        assert "Ethiopia Yirgacheffe Special" in campaign.selected_offering


# =========================================================================
# 2. IMAGE GENERATION CASCADE & FALLBACK TESTS
# =========================================================================

@pytest.mark.asyncio
async def test_daily_ai_provider_image_cloudflare_primary():
    """Verify Cloudflare Workers AI is primary image generation engine."""
    import tempfile
    from PIL import Image

    # Create dummy PNG bytes
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        img = Image.new("RGB", (512, 512), color=(99, 102, 241))
        img.save(tf.name)
        temp_img_path = tf.name

    try:
        mock_cf_res = {
            "image_url": "https://res.cloudinary.com/demo/image/upload/sample_cf.png",
            "local_path": temp_img_path,
            "model_name": "@cf/black-forest-labs/flux-1-schnell",
            "generation_id": "cf_12345"
        }

        with patch("app.services.cloudflare_image_service.CloudflareImageService.generate_image", AsyncMock(return_value=mock_cf_res)):
            data, telemetry = await DailyAIProvider.image(
                prompt="Artisan coffee beans on rustic wood table",
                snapshot={"name": "Luminary Coffee", "organization_id": 1},
                aspect_ratio="1:1"
            )
            assert len(data) > 0
            assert telemetry["provider"] == "cloudflare-workers-ai"
            assert telemetry["model"] == "@cf/black-forest-labs/flux-1-schnell"
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)


@pytest.mark.asyncio
async def test_daily_ai_provider_image_pillow_fallback_on_cloudflare_error():
    """Verify fallback to Pillow branded graphic engine if Cloudflare and Imagen fail."""
    with patch("app.services.cloudflare_image_service.CloudflareImageService.generate_image", AsyncMock(side_effect=RuntimeError("Cloudflare 429 quota exhausted"))), \
         patch("app.services.gemini_service.GeminiService.generate_image", AsyncMock(side_effect=RuntimeError("Imagen quota exhausted"))), \
         patch("app.integrations.storage.cloudinary_storage.CloudinaryStorageProvider.is_configured", return_value=False):
        data, telemetry = await DailyAIProvider.image(
            prompt="Artisan coffee beans on rustic wood table",
            snapshot={"name": "Luminary Coffee", "organization_id": 1, "industry": "Food & Beverage"},
            aspect_ratio="1:1"
        )
        assert len(data) > 0
        assert telemetry["provider"] == "pillow-visual-engine"
        assert telemetry["model"] == "ai-brand-graphic-synthesizer"

        # Verify image dimensions
        verified_bytes, meta = verify_image(data)
        assert verified_bytes is not None
        assert meta["width"] >= 256
        assert meta["height"] >= 256


# =========================================================================
# 3. FULL PIPELINE WORKER & DATABASE PERSISTENCE TESTS
# =========================================================================

@pytest.mark.asyncio
async def test_full_autonomous_pipeline_execution(setup_pipeline_db):
    """Test full execution of execute_daily_run and database persistence."""
    import io
    from PIL import Image

    db = setup_pipeline_db["db"]
    org1 = setup_pipeline_db["org1"]
    brand1 = setup_pipeline_db["brand1"]

    # Enqueue a daily run
    today = date.today()
    # Clear any previous run for today
    db.query(PipelineRun).filter(PipelineRun.brand_profile_id == brand1.id, PipelineRun.run_date == today).delete()
    db.commit()

    run = enqueue_daily(db, org1.id, brand1.id, target_date=today)
    assert run is not None
    assert run.status == "pending"

    # Create dummy PNG bytes in memory (no disk file locks)
    buf = io.BytesIO()
    img = Image.new("RGB", (512, 512), color=(99, 102, 241))
    img.save(buf, format="PNG")
    dummy_png_bytes = buf.getvalue()

    mock_campaign_payload = DailyCampaign(
        title="Artisan Craftsmanship Unleashed",
        theme="Precision Roasting",
        objective="Drive artisan coffee sales",
        selected_offering="Ethiopia Yirgacheffe",
        description="Freshly micro-roasted artisan beans bringing vibrant notes and unparalleled smoothness to every morning.",
        caption="Start your day with extraordinary coffee. Roasted fresh daily.",
        hashtags=["#LuminaryCoffee", "#MicroRoast"],
        call_to_action="Explore our roast",
        target_audience="Artisan coffee enthusiasts",
        creative_direction="Warm studio lighting",
        platform_captions={
            "instagram": "Morning perfection in every cup. ✨ #LuminaryCoffee",
            "facebook": "Discover why coffee lovers choose Luminary.",
            "linkedin": "Craftsmanship at scale: The science of precision micro-roasting."
        },
        image_prompts=["Fresh coffee beans in wooden scoop on rustic table"],
        video_script=[],
        video_prompts=[]
    )

    async def mock_provider_campaign(*args, **kwargs):
        return mock_campaign_payload, {"provider": "gemini", "model": "gemini-3.5-flash", "usage": {}}

    async def mock_provider_image(*args, **kwargs):
        return dummy_png_bytes, {"provider": "cloudflare-workers-ai", "model": "@cf/black-forest-labs/flux-1-schnell"}

    def mock_storage(data, org_id, brand_id, run_id, asset_id, kind):
        return f"https://res.cloudinary.com/test/socialai/daily_{asset_id}.png", {"storage": "cloudinary", "public_id": f"daily_{asset_id}"}

    with patch.object(DailyAIProvider, "campaign", side_effect=mock_provider_campaign), \
         patch.object(DailyAIProvider, "image", side_effect=mock_provider_image):

        await execute_daily_run(
            run_id=run.id,
            session_factory=SessionLocal,
            provider=DailyAIProvider,
            storage=mock_storage
        )

    # Refresh and verify run with fresh session
    with SessionLocal() as verify_db:
        run_db = verify_db.get(PipelineRun, run.id)
        assert run_db.status == "completed"
        assert run_db.progress_percent == 100

        # Verify ContentBrief
        brief = verify_db.query(ContentBrief).filter(ContentBrief.run_id == run.id).first()
        assert brief is not None
        assert brief.topic == "Artisan Craftsmanship Unleashed"

        # Verify GeneratedAsset
        assets = verify_db.query(GeneratedAsset).filter(GeneratedAsset.brief_id == brief.id).all()
        assert len(assets) >= 1
        assert assets[0].status == "done"
        assert assets[0].file_path.startswith("https://res.cloudinary.com/")
        assert assets[0].metadata_json.get("provider") == "cloudflare-workers-ai"
        assert assets[0].metadata_json.get("model") == "@cf/black-forest-labs/flux-1-schnell"

        # Verify AutonomousPost
        post = verify_db.query(AutonomousPost).filter(AutonomousPost.brief_id == brief.id).first()
        assert post is not None
        assert post.organization_id == org1.id
        assert post.brand_profile_id == brand1.id

        # Verify PostVariants
        variants = verify_db.query(PostVariant).filter(PostVariant.post_id == post.id).all()
        assert len(variants) >= 3
        platform_names = [v.platform for v in variants]
        assert "instagram" in platform_names
        assert "facebook" in platform_names
        assert "linkedin" in platform_names


# =========================================================================
# 4. COMPONENT REGENERATION & PIPELINE GEMINI TESTS
# =========================================================================

@pytest.mark.asyncio
async def test_regenerate_post_part_caption_and_image(setup_pipeline_db):
    """Test regenerate_post_part for caption and image using unified services."""
    db = setup_pipeline_db["db"]
    org1 = setup_pipeline_db["org1"]
    brand1 = setup_pipeline_db["brand1"]

    # Create dummy pipeline run first
    regen_date = date.today() + timedelta(days=200)
    db.query(PipelineRun).filter(PipelineRun.brand_profile_id == brand1.id, PipelineRun.run_date == regen_date).delete()
    db.commit()

    run_obj = PipelineRun(
        brand_profile_id=brand1.id,
        organization_id=org1.id,
        run_date=regen_date,
        status="completed",
        stage="Completed",
        progress_percent=100
    )
    db.add(run_obj)
    db.commit()
    db.refresh(run_obj)

    # Create dummy post and brief linked to run_obj
    brief = ContentBrief(
        run_id=run_obj.id,
        brand_profile_id=brand1.id,
        topic="Morning Coffee Routine",
        angle="Rise and Grind",
        reason="Daily engagement",
        caption_brief="Celebrate coffee lovers",
        image_prompt="A steaming cup of coffee",
        video_prompt="Pouring coffee in slow motion",
        hashtags=["#CoffeeTime"],
        cta="Try it today",
        tone="Warm",
        target_platforms=["instagram", "facebook"],
        raw_json={
            "topic": "Morning Coffee Routine",
            "angle": "Rise and Grind",
            "reason": "Daily engagement",
            "caption_brief": "Celebrate coffee lovers",
            "image_prompt": "A steaming cup of coffee",
            "video_prompt": "Pouring coffee in slow motion",
            "hashtags": ["#CoffeeTime"],
            "cta": "Try it today",
            "tone": "Warm",
            "target_platforms": ["instagram", "facebook"]
        }
    )
    db.add(brief)
    db.flush()

    post = AutonomousPost(
        brief_id=brief.id,
        brand_profile_id=brand1.id,
        organization_id=org1.id,
        status="pending_approval"
    )
    db.add(post)
    db.flush()

    v1 = PostVariant(
        post_id=post.id,
        platform="instagram",
        caption="Initial caption",
        media_path="https://res.cloudinary.com/old.png",
        media_type="image",
        scheduled_time=datetime.utcnow()
    )
    db.add(v1)
    db.commit()

    mock_new_captions = PlatformCaptionsSchema(
        instagram="✨ Brand New Instagram Caption #CoffeeTime",
        facebook="Brand New Facebook Post",
        linkedin="Brand New LinkedIn Insight",
        youtube_title="Brand New YouTube Title",
        youtube_description="Brand New YouTube Description"
    )

    mock_new_img = {
        "status": "done",
        "file_path": "https://res.cloudinary.com/new_generated_image.png",
        "provider": "cloudflare-workers-ai",
        "model_name": "@cf/black-forest-labs/flux-1-schnell"
    }

    with patch("app.services.pipeline_gemini_service.PipelineGeminiService.generate_platform_captions", AsyncMock(return_value=mock_new_captions)), \
         patch("app.services.pipeline_gemini_service.PipelineGeminiService.generate_branded_image", AsyncMock(return_value=mock_new_img)):

        # Regenerate caption
        res_caption = await regenerate_post_part(post.id, "caption")
        assert res_caption["status"] == "success"
        db.refresh(v1)
        assert v1.caption == "✨ Brand New Instagram Caption #CoffeeTime"

        # Regenerate image
        res_image = await regenerate_post_part(post.id, "image")
        assert res_image["status"] == "success"
        db.refresh(v1)
        assert v1.media_path == "https://res.cloudinary.com/new_generated_image.png"


# =========================================================================
# 5. TENANT ISOLATION TESTS
# =========================================================================

def test_tenant_isolation_on_enqueue(setup_pipeline_db):
    """Ensure Brand belonging to Org 2 cannot be enqueued under Org 1."""
    db = setup_pipeline_db["db"]
    org1 = setup_pipeline_db["org1"]
    brand2 = setup_pipeline_db["brand2"]  # Belongs to Org 2

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        enqueue_daily(db, organization_id=org1.id, brand_id=brand2.id)
    assert exc_info.value.status_code == 404
    assert "Brand not found in your organization" in exc_info.value.detail
