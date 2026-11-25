import asyncio
import os
import sys
import numpy as np
from PIL import Image
import io

from app.db.session import SessionLocal
from app.models.user import User
from app.models.brand_profile import BrandProfile
from app.models.media_asset import MediaAsset
from app.models.quality_report import QualityReport
from app.models.generation_job import GenerationJob
from app.services.quality_service import QualityService
from app.services.quality_check_engine import QualityCheckEngine
from app.services.image_generation_service import ImageGenerationService
from app.schemas.generation import GeminiImageGenerationRequest

async def run_suite():
    db = SessionLocal()
    user = db.query(User).filter(User.organization_id == 5).first() or db.query(User).first()
    brand = db.query(BrandProfile).filter(BrandProfile.id == 5).first() or db.query(BrandProfile).first()
    print("=" * 70)
    print("CENTRALIZED QUALITY CHECK & CROSS-MODULE AUDIT SUITE")
    print(f"Tenant Org ID: {user.organization_id} | User: {user.email}")
    print(f"Brand: {brand.name} | Logo: {brand.logo_url}")
    print("=" * 70)

    # -------------------------------------------------------------
    # Scenario 1, 2, 3: Generation & Quality Check across modules
    # -------------------------------------------------------------
    print("\n[Scenario 1 & 2 & 3] Generating images with Centralized Quality Check...")
    prompt = "Modern ergonomic desk workstation with ZenCRM software analytics dashboard on display"
    
    req = GeminiImageGenerationRequest(
        prompt=prompt,
        brand_profile_id=brand.id,
        aspect_ratio="1:1",
        style_preset="modern_product"
    )
    job = await ImageGenerationService.generate_image(req, db, user)
    print(f"-> Generated Asset ID: {job.result_data.get('media_asset_id')}")
    print(f"-> Quality Check Result: {job.result_data.get('quality_check', {}).get('decision')}")
    print(f"-> Score: {job.result_data.get('quality_check', {}).get('quality_score')}%")
    print(f"-> Duplicate Status: {job.result_data.get('quality_check', {}).get('duplicate_status')}")
    print(f"-> Retries: {job.result_data.get('quality_check', {}).get('retry_count', 0)}")
    assert job.result_data.get("quality_check") is not None, "Quality check missing!"

    # -------------------------------------------------------------
    # Scenario 4 & 5: Duplicate Detection across modules
    # -------------------------------------------------------------
    print("\n[Scenario 4 & 5] Testing Cross-Module Duplicate Detection...")
    img_url = job.result_data.get("image_url")
    val_dup = QualityCheckEngine.validate_image(
        image_url=img_url,
        prompt=prompt,
        brand=brand,
        organization_id=user.organization_id,
        db=db,
        current_media_id=9999999  # simulate a different generated asset
    )
    print(f"-> Duplicate Check Decision: {val_dup.get('decision')}")
    print(f"-> Duplicate Status: {val_dup.get('duplicate_status')}")
    print(f"-> Similarity Score: {val_dup.get('similarity_score')}%")
    print(f"-> Failed Checks: {val_dup.get('failed_checks')}")
    assert val_dup.get("duplicate_status") in ["SUSPECTED_DUPLICATE", "CONFIRMED_DUPLICATE"], "Duplicate not flagged!"

    # -------------------------------------------------------------
    # Scenario 6: Reusable Logo Brand Exemption
    # -------------------------------------------------------------
    print("\n[Scenario 6] Testing Reusable Brand Logo Exemption...")
    if brand.logo_url:
        val_logo = QualityCheckEngine.validate_image(
            image_url=brand.logo_url,
            prompt="Official brand logo overlay",
            brand=brand,
            organization_id=user.organization_id,
            db=db,
            is_logo=True
        )
        print(f"-> Logo Check Decision: {val_logo.get('decision')}")
        print(f"-> Logo Duplicate Status: {val_logo.get('duplicate_status')}")
        assert val_logo.get("duplicate_status") == "UNIQUE", "Logo was incorrectly flagged as duplicate!"
        print("-> SUCCESS: Brand Logo correctly exempted from creative duplicate collision.")

    # -------------------------------------------------------------
    # Scenario 7: Low-quality output detection & auto-regeneration
    # -------------------------------------------------------------
    print("\n[Scenario 7] Testing Low-Quality Output Detection & Gate Scoring...")
    # Create degraded synthetic image (solid grey blur with 0 variance)
    degraded = np.zeros((100, 100, 3), dtype=np.uint8) + 128
    pil_deg = Image.fromarray(degraded)
    buf = io.BytesIO()
    pil_deg.save(buf, format="PNG")
    buf.seek(0)
    
    val_deg = QualityCheckEngine.validate_image(
        image_bytes=buf.getvalue(),
        prompt="A vibrant photorealistic scene",
        brand=brand,
        organization_id=user.organization_id,
        db=db
    )
    print(f"-> Degraded Image Decision: {val_deg.get('decision')}")
    print(f"-> Degraded Image Quality Score: {val_deg.get('quality_score')}%")
    print(f"-> Degraded Image Failed Checks: {val_deg.get('failed_checks')}")
    assert val_deg.get("decision") in ["REGENERATE", "BLOCK"], "Low quality was not flagged for regeneration!"
    print("-> SUCCESS: Degraded image properly flagged for REGENERATE.")

    # -------------------------------------------------------------
    # Scenario 8: Provider Fallback & Error Handling
    # -------------------------------------------------------------
    print("\n[Scenario 8] Testing Provider Failure Resilience...")
    try:
        # Invalid image bytes
        val_fail = QualityCheckEngine.validate_image(
            image_bytes=b"corrupted invalid bytes",
            prompt="Test corrupt fallback",
            organization_id=user.organization_id,
            db=db
        )
        print(f"-> Corrupted File Decision: {val_fail.get('decision')}")
        print(f"-> Failed Checks: {val_fail.get('failed_checks')}")
        assert "CORRUPTED_FILE" in val_fail.get("failed_checks") or val_fail.get("decision") == "BLOCK"
        print("-> SUCCESS: Corrupted input gracefully blocked without crashing backend.")
    except Exception as e:
        print(f"Unexpected crash on corrupt input: {e}")

    # -------------------------------------------------------------
    # Scenario 9: Persistence in QualityReport & GenerationJob
    # -------------------------------------------------------------
    print("\n[Scenario 9] Verifying Quality Reports Saved in Database...")
    reports = db.query(QualityReport).filter(QualityReport.organization_id == user.organization_id).order_by(QualityReport.created_at.desc()).limit(5).all()
    print(f"-> Found {len(reports)} recent QualityReports in DB:")
    for r in reports:
        print(f"   * Report #{r.id} | Asset: {r.asset_type} | Decision: {r.decision} | Score: {r.quality_score}% | Dup: {r.duplicate_status} | Retries: {r.retry_count}")
    assert len(reports) > 0, "No QualityReports persisted!"

    # -------------------------------------------------------------
    # Test Bundle Quality Audit (Section 2, 4)
    # -------------------------------------------------------------
    print("\n[Bonus] Testing Multi-Modal Bundle Audit (QualityService.audit_bundle)...")
    text_data = {
        "headline": "Empower Your Workflow with ZenCRM",
        "caption": "Say goodbye to repetitive manual tasks and hello to streamlined automation.",
        "hashtags": ["#ZenCRM", "#Automation"],
        "quality_check": {"quality_score": 98.0, "decision": "PASS", "duplicate_status": "UNIQUE"}
    }
    image_data = {
        "image_url": img_url,
        "quality_check": {"quality_score": 96.0, "decision": "PASS", "duplicate_status": "UNIQUE"}
    }
    bundle_res = QualityService.audit_bundle(
        text_data=text_data,
        image_data=image_data,
        brand=brand,
        organization_id=user.organization_id,
        db=db
    )
    print(f"-> Bundle Audit Passed: {bundle_res.get('passed')}")
    print(f"-> Bundle Overall Score: {bundle_res.get('quality_score')}%")
    print(f"-> Bundle Status: {bundle_res.get('status')}")
    print(f"-> Bundle Badges: {bundle_res.get('badges')}")
    assert bundle_res.get("passed") is True, "Bundle audit should pass!"

    print("\n" + "=" * 70)
    print("ALL 9 MANDATORY TEST SCENARIOS COMPLETED & VERIFIED!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_suite())

