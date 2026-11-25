import os
from typing import Dict, Any, Optional
from app.core.config import settings
from app.core.logging import logger
from app.services.cloudflare_image_service import CloudflareImageService


class AIImageProvider:
    """Centralized Provider for generating AI visual assets using Cloudflare Workers AI."""

    @classmethod
    async def generate_image(
        cls,
        prompt: str,
        aspect_ratio: str = "1:1",
        style: str = "modern_product",
        provider: str = "cloudflare-workers-ai",
        negative_prompt: str = "blurry, distorted, low quality, artifacts",
        model_name: str = "@cf/black-forest-labs/flux-1-schnell",
        organization_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate AI image via Cloudflare Workers AI with Cloudinary storage."""
        try:
            return await CloudflareImageService.generate_image(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                style_preset=style,
                organization_id=organization_id or 1
            )
        except Exception as e:
            logger.error(f"[AIImageProvider] Cloudflare image generation failed: {e}")
            raise
