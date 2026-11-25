"""Check real brand-grounded scene generation without changing database records."""
import asyncio
import io
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)
from PIL import Image
import app.models
from app.db.session import SessionLocal
from app.models.brand_profile import BrandProfile
from app.services.image_generation_service import ImageGenerationService


async def main():
    with SessionLocal() as db:
        brand = db.query(BrandProfile).order_by(BrandProfile.updated_at.desc()).first()
        data = await ImageGenerationService.generate_raw_image_bytes(
            'Create a marketing visual illustrating the actual business and its services, with no invented products or offers.',
            '1:1', brand_profile=brand,
        )
    if not data:
        raise RuntimeError('No AI image provider returned an image.')
    with Image.open(io.BytesIO(data)) as result:
        result.verify()
    target = Path('.media_tmp/studio-audit-image.png')
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(data)
    print('PASS: live AI image bytes decoded and saved to', target)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as exc:
        print('FAILED:', type(exc).__name__)
        sys.exit(1)
