"""Brand requirements shared by AI Studio generation workflows."""
import json
from fastapi import HTTPException
from app.models.brand_profile import BrandProfile


def validate_studio_brand(brand):
    meta = brand.meta_info if brand and isinstance(brand.meta_info, dict) else {}
    missing = []
    if not brand or not (brand.name or "").strip():
        missing.append("brand name")
    if not brand or not (brand.industry or meta.get("category") or "").strip():
        missing.append("industry or business category")
    if not brand or not (brand.brand_guidelines or meta.get("description") or meta.get("business_description") or meta.get("products") or meta.get("services")):
        missing.append("business description, products, or services")
    if missing:
        raise HTTPException(status_code=400, detail="Update your Brand Profile before generating content: missing " + ", ".join(missing) + ".")
    return brand


def resolve_studio_brand(db, organization_id, brand_id=None):
    query = db.query(BrandProfile).filter(BrandProfile.organization_id == organization_id)
    if brand_id is not None:
        query = query.filter(BrandProfile.id == brand_id)
    brands = query.order_by(BrandProfile.updated_at.desc(), BrandProfile.id.desc()).limit(2).all()
    if not brands:
        raise HTTPException(status_code=400, detail="Select a Brand Profile belonging to your organization.")
    if brand_id is None and len(brands) > 1:
        raise HTTPException(status_code=400, detail="Select the Brand Profile to use for generation.")
    return validate_studio_brand(brands[0])


def studio_brand_context(brand):
    validate_studio_brand(brand)
    return json.dumps({field: getattr(brand, field, None) for field in (
        "name", "industry", "brand_guidelines", "target_audience", "tagline",
        "tone_of_voice", "color_palette", "logo_url", "cta", "avoid", "meta_info"
    )}, ensure_ascii=False, default=str)
