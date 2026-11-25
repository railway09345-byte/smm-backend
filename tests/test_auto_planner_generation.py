import os
import sys
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from app.db.session import SessionLocal
from app.models.campaign import Campaign
from app.models.content import Content
from app.models.brand_profile import BrandProfile
from app.models.user import User
from app.models.scheduled_post import ScheduledPost
from app.utils.enums import ContentStatus, PlatformType
from app.services.content_planner_service import ContentPlannerService

async def test_scheduled_auto_generation():
    print("=== Testing Automatic AI Content Planner Media Generation Workflow ===", flush=True)
    db = SessionLocal()
    test_content_id = None
    try:
        # Find organization and brand
        brand = db.query(BrandProfile).filter(BrandProfile.organization_id == 5).first()
        if not brand:
            brand = db.query(BrandProfile).first()
        org_id = brand.organization_id if brand else 1
        user = db.query(User).filter(User.organization_id == org_id).first()

        # Find or create a test campaign
        campaign = db.query(Campaign).filter(Campaign.organization_id == org_id).first()
        if not campaign:
            campaign = Campaign(
                title="Automated Test Campaign",
                objective="Brand Awareness",
                organization_id=org_id,
                tags={"timezone": "Asia/Kolkata", "campaign_goal": "Product Launch"}
            )
            db.add(campaign)
            db.commit()
            db.refresh(campaign)

        # Create a test planned content item scheduled 2 minutes in the past so it is due now
        now = datetime.utcnow()
        scheduled_past = now - timedelta(minutes=2)
        
        test_content = Content(
            title="Automated AI Planner Video Workflow Verification",
            caption="Testing automatic scheduled AI video generation workflow without button click.",
            hashtags=["#Automation", "#SocialAI"],
            target_platforms=[PlatformType.INSTAGRAM],
            status=ContentStatus.DRAFT,
            meta_info={
                "is_planned_idea": True,
                "day_number": 1,
                "topic": "Automated AI Planner Video Workflow Verification",
                "hook": "Revolutionizing automated content creation",
                "caption_brief": "A modern showcase of hands-free social automation",
                "content_format": "video",
                "suggested_date": scheduled_past.strftime("%Y-%m-%d"),
                "suggested_time": scheduled_past.strftime("%H:%M"),
                "suggested_datetime": scheduled_past.isoformat(),
                "auto_generate_media": True,
                "plan_status": "planned",
                "generation_status": "scheduled"
            },
            organization_id=org_id,
            brand_profile_id=brand.id if brand else None,
            campaign_id=campaign.id,
            creator_id=user.id if user else 1
        )
        db.add(test_content)
        db.commit()
        db.refresh(test_content)
        test_content_id = test_content.id
        print(f"1. Created test Content #{test_content_id} scheduled for {scheduled_past.isoformat()}", flush=True)

        # Execute auto_generate_due_planned_media for this content item
        print("2. Triggering ContentPlannerService.auto_generate_due_planned_media()...", flush=True)
        processed = await ContentPlannerService.auto_generate_due_planned_media(target_content_id=test_content_id)
        print(f"3. Processed count: {processed}", flush=True)

        # Verify results in fresh database session (avoiding REPEATABLE READ snapshot isolation)
        check_db = SessionLocal()
        reloaded = check_db.query(Content).filter(Content.id == test_content_id).first()
        meta = reloaded.meta_info or {}
        gen_media = meta.get("generated_media") or {}
        img_url = gen_media.get("image", {}).get("image_url")
        vid_url = gen_media.get("video", {}).get("video_url")
        plan_status = meta.get("plan_status")
        gen_status = meta.get("generation_status")

        print("4. Verification results:", flush=True)
        print(f"   Plan Status: {plan_status}", flush=True)
        print(f"   Generation Status: {gen_status}", flush=True)
        print(f"   Image URL: {img_url}", flush=True)
        print(f"   Video URL: {vid_url}", flush=True)
        print(f"   Attached Media Asset ID: {reloaded.media_asset_id}", flush=True)

        assert processed == 1, f"Expected 1 item to be processed, got {processed}"
        assert vid_url is not None and len(vid_url) > 10, "Expected a valid generated video URL"
        assert gen_status == "completed", f"Expected 'completed', got {gen_status}"
        assert plan_status == "draft_created", f"Expected 'draft_created', got {plan_status}"
        assert reloaded.media_asset_id is not None, "Expected media_asset_id to be populated"

        # Test idempotency: calling it again should NOT generate duplicate media
        print("5. Testing idempotency on second run...", flush=True)
        second_processed = await ContentPlannerService.auto_generate_due_planned_media(target_content_id=test_content_id)
        print(f"   Second run processed count: {second_processed} (should be 0)", flush=True)
        assert second_processed == 0, f"Expected 0 on idempotent run, got {second_processed}"

        print(">>> ALL VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<", flush=True)

    finally:
        if 'check_db' in locals():
            check_db.close()
        if test_content_id:
            try:
                cleanup_db = SessionLocal()
                c_del = cleanup_db.query(Content).filter(Content.id == test_content_id).first()
                if c_del:
                    cleanup_db.delete(c_del)
                    cleanup_db.commit()
                    print(f"Cleaned up test Content #{test_content_id}", flush=True)
                cleanup_db.close()
            except Exception:
                pass
        db.close()

if __name__ == "__main__":
    asyncio.run(test_scheduled_auto_generation())
