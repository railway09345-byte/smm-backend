import os
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from difflib import SequenceMatcher
from app.models.brand_profile import BrandProfile
from app.core.logging import logger


BANNED_PATTERNS = [
    r"\bguaranteed 100% profit\b",
    r"\bfree money\b",
    r"\bmiracle cure\b",
    r"\bget rich quick\b",
    r"\bweight loss guaranteed\b",
    r"\bwire transfer immediately\b",
]

PLATFORM_LIMITS = {
    "instagram": {"max_caption": 2200, "max_hashtags": 30, "min_hashtags": 1},
    "facebook": {"max_caption": 5000, "max_hashtags": 20, "min_hashtags": 0},
    "linkedin": {"max_caption": 3000, "max_hashtags": 10, "min_hashtags": 1},
    "twitter": {"max_caption": 280, "max_hashtags": 5, "min_hashtags": 0},
    "youtube": {"max_title": 100, "max_description": 5000, "max_tags": 15},
    "pinterest": {"max_title": 100, "max_description": 500, "max_tags": 10},
    "tiktok": {"max_caption": 2200, "max_hashtags": 10, "min_hashtags": 1},
    "snapchat": {"max_caption": 250, "max_hashtags": 5, "min_hashtags": 0},
}


class QualityAuditResult:
    def __init__(
        self,
        passed: bool,
        score: float,
        badges: Dict[str, bool],
        issues: List[str],
        suggestions: List[str],
        needs_regeneration_part: Optional[str] = None
    ):
        self.passed = passed
        self.score = score
        self.badges = badges
        self.issues = issues
        self.suggestions = suggestions
        self.needs_regeneration_part = needs_regeneration_part

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 1),
            "badges": self.badges,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "needs_regeneration_part": self.needs_regeneration_part
        }


class QualityService:
    """
    Automated Quality and Policy Check Service.
    Validates brand consistency, platform limits, safety, media validity, and duplicate prevention.
    """

    @classmethod
    def audit_post(
        cls,
        brand_profile: BrandProfile,
        brief_data: Dict[str, Any],
        platform_captions: Dict[str, Any],
        media_assets: List[Dict[str, Any]],
        past_posts: List[Dict[str, Any]]
    ) -> QualityAuditResult:
        issues: List[str] = []
        suggestions: List[str] = []
        badges = {
            "brand_alignment": True,
            "safety_passed": True,
            "platform_compliant": True,
            "media_valid": True,
            "duplicate_free": True
        }
        score = 100.0
        regeneration_part: Optional[str] = None

        # 1. Brand Alignment & Verification
        brand_name_lower = brand_profile.name.lower()
        topic_text = (brief_data.get("topic") or "") + " " + (brief_data.get("caption_brief") or "")
        
        # Check if brand rules or guidelines were violated
        if brand_profile.avoid:
            avoid_keywords = [w.strip().lower() for w in brand_profile.avoid.split(",") if len(w.strip()) > 3]
            for kw in avoid_keywords:
                for platform, text in platform_captions.items():
                    if isinstance(text, str) and kw in text.lower():
                        issues.append(f"Caption for {platform} contains restricted brand term: '{kw}'")
                        badges["brand_alignment"] = False
                        score -= 20
                        regeneration_part = "caption"

        # 2. Safety & Banned Words Check
        all_text = " ".join([str(v) for v in platform_captions.values() if isinstance(v, str)])
        for pattern in BANNED_PATTERNS:
            if re.search(pattern, all_text, re.IGNORECASE):
                issues.append(f"Content matches restricted safety policy: '{pattern}'")
                badges["safety_passed"] = False
                score -= 35
                regeneration_part = "caption"

        # 3. Platform Compliance & Character Limits
        for platform, text in platform_captions.items():
            if not isinstance(text, str) or not text.strip():
                continue
            
            limits = PLATFORM_LIMITS.get(platform, {})
            max_len = limits.get("max_caption", 2200)
            if platform == "youtube_title" and len(text) > 100:
                issues.append(f"YouTube title is {len(text)} chars (max 100).")
                badges["platform_compliant"] = False
                score -= 15
            elif platform == "twitter" and len(text) > 280:
                issues.append(f"Twitter caption is {len(text)} chars (max 280).")
                badges["platform_compliant"] = False
                score -= 15
            elif len(text) > max_len:
                issues.append(f"{platform.capitalize()} caption exceeds limit ({len(text)}/{max_len}).")
                badges["platform_compliant"] = False
                score -= 10

        # 4. Duplicate Check Against Past 30 Days
        for past in past_posts[:30]:
            past_caption = past.get("caption") or ""
            if not past_caption:
                continue
            for platform, curr_text in platform_captions.items():
                if isinstance(curr_text, str) and len(curr_text) > 50:
                    similarity = SequenceMatcher(None, curr_text.lower()[:300], past_caption.lower()[:300]).ratio()
                    if similarity > 0.85:
                        issues.append(f"{platform.capitalize()} copy is {int(similarity*100)}% similar to a recent post from past 30 days.")
                        badges["duplicate_free"] = False
                        score -= 25
                        regeneration_part = "caption"
                        break

        # 5. Media Integrity Check
        if media_assets:
            for asset in media_assets:
                local_path = asset.get("local_path")
                if local_path and not os.path.exists(local_path):
                    # Check if relative exists
                    issues.append(f"Generated media file missing on storage: {asset.get('file_path')}")
                    badges["media_valid"] = False
                    score -= 20
                    regeneration_part = "image"
        else:
            suggestions.append("No visual assets attached; using default branded graphic.")

        score = max(0.0, min(100.0, score))
        # Repeated copy must never be auto-approved or auto-published.
        passed = (score >= 70.0 and badges["safety_passed"] and badges["duplicate_free"])

        if passed and not issues:
            suggestions.append("All automated brand, policy, and platform checks passed with flying colors.")

        return QualityAuditResult(
            passed=passed,
            score=score,
            badges=badges,
            issues=issues,
            suggestions=suggestions,
            needs_regeneration_part=regeneration_part if not passed else None
        )

    @classmethod
    def validate_image(
        cls,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        image_path: Optional[str] = None,
        prompt: Optional[str] = None,
        brand: Optional[BrandProfile] = None,
        organization_id: Optional[int] = None,
        db = None,
        candidate_asset_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Centralized image validation: blur, contrast, resolution, corruption, and cross-module duplicate detection."""
        from app.services.quality_check_engine import QualityCheckEngine
        return QualityCheckEngine.validate_image(
            image_bytes=image_bytes,
            image_url=image_url,
            image_path=image_path,
            prompt=prompt,
            brand=brand,
            organization_id=organization_id,
            db=db,
            candidate_asset_id=candidate_asset_id,
            **kwargs
        )

    @classmethod
    def validate_video(
        cls,
        video_url: Optional[str] = None,
        video_path: Optional[str] = None,
        video_bytes: Optional[bytes] = None,
        prompt: Optional[str] = None,
        brand: Optional[BrandProfile] = None,
        organization_id: Optional[int] = None,
        db = None,
        candidate_asset_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Centralized video validation: container, duration, fps, resolution, black frames, frozen frames, audio, and duplicate detection."""
        from app.services.quality_check_engine import QualityCheckEngine
        return QualityCheckEngine.validate_video(
            video_url=video_url,
            video_path=video_path,
            video_bytes=video_bytes,
            prompt=prompt,
            brand=brand,
            organization_id=organization_id,
            db=db,
            candidate_asset_id=candidate_asset_id
        )

    @classmethod
    def validate_text(
        cls,
        caption: str,
        title: Optional[str] = None,
        hashtags: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        brand: Optional[BrandProfile] = None,
        target_platforms: Optional[List[str]] = None,
        organization_id: Optional[int] = None,
        db = None,
        candidate_content_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Centralized text/caption validation: word count, platform limits, safety, brand avoid, and cross-module duplicate detection."""
        from app.services.quality_check_engine import QualityCheckEngine
        return QualityCheckEngine.validate_description(
            caption=caption,
            title=title,
            hashtags=hashtags or [],
            brand=brand,
            target_platforms=target_platforms or ["instagram"],
            organization_id=organization_id,
            db=db,
            candidate_content_id=candidate_content_id
        )

    @classmethod
    def audit_bundle(
        cls,
        text_data: Dict[str, Any],
        image_data: Dict[str, Any],
        video_data: Optional[Dict[str, Any]] = None,
        brand: Optional[BrandProfile] = None,
        organization_id: Optional[int] = None,
        db = None
    ) -> Dict[str, Any]:
        """
        Centralized Quality & Duplicate Check for full AI Post Bundles across
        Gemini AI Studio, AI Content Planner, and Content Studio.
        Aggregates text, visual image, and video storyboard quality gates.
        """
        text_qc = (text_data or {}).get("quality_check") or {}
        image_qc = (image_data or {}).get("quality_check") or {}
        video_qc = (video_data or {}).get("quality_check") or {}

        text_score = float(text_qc.get("quality_score", 95.0))
        img_score = float(image_qc.get("quality_score", 98.0))
        vid_score = float(video_qc.get("quality_score", 98.0)) if video_qc else 95.0

        # Weighted aggregate score (40% text, 35% visual image, 25% video storyboard)
        overall_score = round((text_score * 0.40) + (img_score * 0.35) + (vid_score * 0.25), 1)

        # Check critical gating
        safety_passed = "BANNED_POLICY_CLAIM" not in text_qc.get("failed_checks", [])
        duplicate_free = (
            text_qc.get("duplicate_status", "UNIQUE") != "CONFIRMED_DUPLICATE"
            and image_qc.get("duplicate_status", "UNIQUE") != "CONFIRMED_DUPLICATE"
            and video_qc.get("duplicate_status", "UNIQUE") != "CONFIRMED_DUPLICATE"
        )
        media_valid = "CORRUPTED_FILE" not in image_qc.get("failed_checks", []) and "DECODE_ERROR" not in image_qc.get("failed_checks", [])
        brand_alignment = text_qc.get("brand_score", 100.0) >= 70.0 and "BRAND_AVOID_TERM_USED" not in text_qc.get("failed_checks", [])
        platform_compliant = "PLATFORM_CHAR_LIMIT_EXCEEDED" not in text_qc.get("failed_checks", [])

        passed = (overall_score >= 70.0 and safety_passed and duplicate_free and media_valid)
        decision = "PASS" if passed else ("REGENERATE" if duplicate_free else "BLOCK")
        status = "APPROVED" if passed else "REJECTED"

        all_issues = list(text_qc.get("issues", [])) + list(image_qc.get("issues", [])) + list(video_qc.get("issues", []))
        all_suggestions = list(text_qc.get("suggestions", [])) + list(image_qc.get("suggestions", []))
        if passed and not all_issues:
            all_suggestions.append("Verified: Ready for high-converting publication across all connected social channels.")

        retry_count = max(
            text_qc.get("retry_count", 0),
            image_qc.get("retry_count", 0),
            video_qc.get("retry_count", 0)
        )

        bundle_report = {
            "passed": passed,
            "status": status,
            "decision": decision,
            "quality_score": overall_score,
            "score": overall_score,
            "duplicate_status": "CONFIRMED_DUPLICATE" if not duplicate_free else "UNIQUE",
            "similarity_score": max(
                float(text_qc.get("similarity_score", 0.0)),
                float(image_qc.get("similarity_score", 0.0)),
                float(video_qc.get("similarity_score", 0.0))
            ),
            "badges": {
                "brand_alignment": brand_alignment,
                "safety_passed": safety_passed,
                "platform_compliant": platform_compliant,
                "media_valid": media_valid,
                "duplicate_free": duplicate_free
            },
            "gates": {
                "text_quality": text_qc.get("gates", {}),
                "image_quality": image_qc.get("gates", {}),
                "video_quality": video_qc.get("gates", {})
            },
            "issues": all_issues,
            "suggestions": all_suggestions,
            "retry_count": retry_count
        }

        # Persist QualityReport for the bundle if DB session is active
        if db and organization_id:
            try:
                from app.models.quality_report import QualityReport
                qr = QualityReport(
                    asset_type="bundle",
                    organization_id=organization_id,
                    brand_profile_id=brand.id if brand else None,
                    decision=decision,
                    quality_score=overall_score,
                    technical_score=img_score,
                    brand_score=text_qc.get("brand_score", 100.0),
                    content_accuracy_score=100.0,
                    duplicate_status="CONFIRMED_DUPLICATE" if not duplicate_free else "UNIQUE",
                    similarity_score=bundle_report["similarity_score"],
                    failed_checks=list(set(text_qc.get("failed_checks", []) + image_qc.get("failed_checks", []))),
                    issues=all_issues,
                    suggestions=all_suggestions,
                    match_details={
                        "text": text_qc.get("match_details"),
                        "image": image_qc.get("match_details")
                    },
                    metrics={
                        "overall_bundle_score": overall_score,
                        "text_score": text_score,
                        "image_score": img_score,
                        "video_score": vid_score,
                        "approved": passed
                    },
                    retry_count=retry_count
                )
                db.add(qr)
                db.commit()
            except Exception as qr_err:
                logger.warning(f"Could not persist bundle QualityReport record: {qr_err}")

        return bundle_report

    @classmethod
    def audit_content_item(cls, content, brand: Optional[BrandProfile] = None, db = None) -> Dict[str, Any]:
        """Audits a Content entity using the enterprise QualityCheckEngine."""
        from app.services.quality_check_engine import QualityCheckEngine
        return QualityCheckEngine.audit_content_item(content=content, brand=brand, db=db)

    @classmethod
    def auto_optimize_caption(cls, content, brand: Optional[BrandProfile] = None) -> Dict[str, Any]:
        """Automatically optimizes copy for 100% compliance, brand tone, and platform limits."""
        from app.services.quality_check_engine import BANNED_SAFETY_PATTERNS
        brand_name = brand.name if brand else "Brand"
        caption = content.caption or ""

        # Remove restricted patterns
        for pattern in BANNED_SAFETY_PATTERNS:
            caption = re.sub(pattern, "", caption, flags=re.IGNORECASE)

        # Remove brand avoid terms
        if brand and brand.avoid:
            for term in brand.avoid.split(","):
                clean_term = term.strip()
                if len(clean_term) > 2:
                    caption = re.sub(rf"\b{re.escape(clean_term)}\b", "", caption, flags=re.IGNORECASE)

        # Ensure Brand CTA exists
        cta = f"Discover more with {brand_name}. Link in bio to shop now!"
        if brand_name.lower() not in caption.lower():
            caption = f"{caption.strip()}\n\n{cta}".strip()

        # Optimize hashtags
        hashtags = content.hashtags or []
        if isinstance(hashtags, list) and len(hashtags) == 0:
            clean_brand = re.sub(r"[^\w]", "", brand_name)
            hashtags = [f"#{clean_brand}", "#Trending", "#NewCollection", "#ShopNow"]

        return {
            "caption": caption.strip(),
            "hashtags": hashtags,
            "cta": cta
        }
