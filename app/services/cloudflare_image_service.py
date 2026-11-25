import os
import uuid
import time
import base64
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import httpx
from PIL import Image, ImageFilter
import io

from app.core.config import settings
from app.core.logging import logger
from app.models.brand_profile import BrandProfile
from app.integrations.storage.cloudinary_storage import CloudinaryStorageProvider


class CloudflareAIError(RuntimeError):
    """Raised when Cloudflare Workers AI fails to generate an image."""
    def __init__(self, message: str, status_code: int = 502, details: Optional[Dict[str, Any]] = None):
        self.status_code = status_code
        self.details = details or {}
        super().__init__(f"Cloudflare Workers AI Error: {message}")


class CloudflareImageService:
    """
    Centralized Cloudflare Workers AI Image Generation Service.
    
    Architecture Rules:
    1. Cloudflare Workers AI is the SOLE image-generation engine.
    2. Cloudinary is ONLY used as media storage and CDN delivery.
    3. Prompts are faithfully preserved with brand style and unique variation seeds.
    4. Never returns stock images, hardcoded assets, or previous generation results.
    5. If Cloudflare fails or quota is exhausted, raises transparent actionable errors.
    """

    AI_IMAGES_DIR = os.path.join(settings.LOCAL_STORAGE_DIR, "media", "ai_images")

    CLOUDFLARE_MODELS = [
        "@cf/black-forest-labs/flux-1-schnell",
        "@cf/black-forest-labs/flux-2-klein-4b",
        "@cf/bytedance/stable-diffusion-xl-lightning",
        "@cf/stabilityai/stable-diffusion-xl-base-1.0",
        "@cf/lykon/dreamshaper-8-lcm"
    ]

    DIMENSION_MAP = {
        "1:1": (1024, 1024),
        "9:16": (1080, 1920),
        "16:9": (1920, 1080),
        "4:5": (1080, 1350),
        "1.91:1": (1200, 628),
        "4:3": (1440, 1080),
        "3:4": (1080, 1440)
    }

    STYLE_ENHANCERS = {
        "modern_product": "commercial studio photography, clean background, sharp focus, 8k resolution, cinematic studio lighting",
        "photo": "hyperrealistic candid photography, natural lighting, shot on 35mm lens, high dynamic range, crisp organic details",
        "minimalist": "minimalist aesthetic, elegant soft palette, subtle textures, generous negative space, contemporary design",
        "luxury": "ultra-luxury editorial aesthetic, gold and marble accents, sleek sophisticated composition, high fashion lookbook",
        "cinematic": "cinematic movie still, dramatic lighting, anamorphic lens flare, moody color grading, 8k uhd",
        "3d_render": "octane 3D render, raytracing, vibrant colors, ultra-detailed geometry, glossy reflective textures",
        "neon": "cyberpunk synthwave aesthetic, glowing neon accents, deep contrast, reflective futuristic surfaces",
        "artistic": "contemporary digital artwork, rich painterly textures, expressive brush strokes, vibrant palette"
    }

    @classmethod
    def _ensure_directory(cls):
        os.makedirs(cls.AI_IMAGES_DIR, exist_ok=True)

    @classmethod
    def get_credentials(cls) -> Tuple[str, str, str]:
        """Resolves Cloudflare Workers AI credentials from environment and settings."""
        token = (
            os.getenv("CLOUDFLARE_API_TOKEN")
            or getattr(settings, "CLOUDFLARE_API_TOKEN", "")
            or ""
        ).strip(" \"'")
        account_id = (
            os.getenv("CLOUDFLARE_ACCOUNT_ID")
            or getattr(settings, "CLOUDFLARE_ACCOUNT_ID", "")
            or ""
        ).strip(" \"'")
        default_model = (
            os.getenv("CLOUDFLARE_IMAGE_MODEL")
            or getattr(settings, "CLOUDFLARE_IMAGE_MODEL", "@cf/black-forest-labs/flux-1-schnell")
            or "@cf/black-forest-labs/flux-1-schnell"
        ).strip(" \"'")

        if not token or not account_id:
            raise CloudflareAIError(
                "CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID is not configured in backend environment.",
                status_code=500
            )
        return token, account_id, default_model

    @classmethod
    def build_accurate_prompt(
        cls,
        prompt: str,
        brand: Optional[BrandProfile] = None,
        style_preset: str = "modern_product",
        aspect_ratio: str = "1:1",
        variation_seed: Optional[str] = None
    ) -> str:
        """
        Constructs a high-fidelity generation prompt while strictly preserving the user's
        requested product, feature, hook, visual concept, and color palette.
        NO hardcoded generic workstation or desk strings injected!
        """
        raw_prompt = prompt.strip()
        parts = [raw_prompt]

        # Extract brand details only if relevant and not already in prompt
        if brand:
            # Color palette
            palette = brand.color_palette if hasattr(brand, "color_palette") and isinstance(brand.color_palette, list) else []
            if palette and not any(str(c).lower() in raw_prompt.lower() for c in palette[:2]):
                parts.append(f"color palette accents: {', '.join(str(c) for c in palette[:2])}")

            # Visual identity guideline
            meta = brand.meta_info if hasattr(brand, "meta_info") and isinstance(brand.meta_info, dict) else {}
            visual_id = (meta.get("visual_identity") or "").strip()
            if visual_id and visual_id.lower() not in raw_prompt.lower() and len(visual_id) < 80:
                parts.append(f"style aesthetic: {visual_id}")

            # Avoid directives
            avoid = (brand.avoid if hasattr(brand, "avoid") and brand.avoid else "").strip()
            if avoid:
                parts.append(f"no text, no watermark, strictly avoid: {avoid[:100]}")
            else:
                parts.append("no text overlays, no watermark, no logo artifacts")

        # Style preset enhancer
        enhancer = cls.STYLE_ENHANCERS.get(style_preset, cls.STYLE_ENHANCERS["modern_product"])
        parts.append(enhancer)

        # Unique variation seed to guarantee non-duplicate generation
        seed_token = variation_seed or uuid.uuid4().hex[:8]
        parts.append(f"(variation seed: {seed_token})")

        return ", ".join([p for p in parts if p])

    @classmethod
    async def _call_cloudflare_model(
        cls,
        model_name: str,
        prompt: str,
        token: str,
        account_id: str,
        timeout: float = 45.0
    ) -> Optional[bytes]:
        """
        Calls a specific Cloudflare Workers AI model endpoint.
        Handles both direct image streams and base64 JSON responses.
        """
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}"
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=timeout) as client:
            # Models like flux-2-klein accept multipart form data; others accept JSON
            if "klein" in model_name.lower():
                response = await client.post(url, headers=headers, data={"prompt": prompt[:2048]})
            else:
                response = await client.post(
                    url,
                    headers={**headers, "Content-Type": "application/json"},
                    json={"prompt": prompt[:2048], "steps": 4}
                )

            logger.info(f"[CloudflareWorkersAI] {model_name} response status: {response.status_code}")

            if response.status_code == 200:
                ctype = response.headers.get("content-type", "")
                if ctype.startswith("image/"):
                    return response.content
                try:
                    data = response.json()
                    b64_img = (data.get("result") or {}).get("image") if isinstance(data.get("result"), dict) else None
                    if b64_img:
                        if b64_img.startswith("data:") and "," in b64_img:
                            b64_img = b64_img.split(",", 1)[1]
                        return base64.b64decode(b64_img)
                except Exception as parse_err:
                    logger.warning(f"[CloudflareWorkersAI] Failed to decode 200 JSON: {parse_err}")
                return None

            # Non-200 response
            err_msg = response.text[:300]
            try:
                err_data = response.json()
                if err_data.get("errors") and len(err_data["errors"]) > 0:
                    err_msg = err_data["errors"][0].get("message", err_msg)
            except Exception:
                pass

            logger.warning(f"[CloudflareWorkersAI] {model_name} error ({response.status_code}): {err_msg}")
            if response.status_code == 429 or "10,000 neurons" in err_msg.lower() or "allocation" in err_msg.lower() or "quota" in err_msg.lower():
                raise CloudflareAIError(f"Cloudflare Workers AI daily quota exhausted (429): {err_msg}", status_code=429)
            return None

    @classmethod
    def adapt_image_dimensions(cls, raw_bytes: bytes, target_width: int, target_height: int) -> bytes:
        """Crops and resizes image cleanly using Lanczos filter without distortion."""
        try:
            im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            src_w, src_h = im.size
            src_aspect = src_w / src_h
            tgt_aspect = target_width / target_height

            if abs(src_aspect - tgt_aspect) < 0.02:
                res = im.resize((target_width, target_height), Image.Resampling.LANCZOS)
            elif src_aspect > tgt_aspect:
                new_w = int(src_h * tgt_aspect)
                left = max(0, (src_w - new_w) // 2)
                res = im.crop((left, 0, left + new_w, src_h)).resize((target_width, target_height), Image.Resampling.LANCZOS)
            else:
                new_h = int(src_w / tgt_aspect)
                top = max(0, (src_h - new_h) // 2)
                res = im.crop((0, top, src_w, top + new_h)).resize((target_width, target_height), Image.Resampling.LANCZOS)

            # Micro-contrast sharpening
            res = res.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=3))
            buf = io.BytesIO()
            res.save(buf, format="JPEG", quality=92, optimize=True)
            return buf.getvalue()
        except Exception as e:
            logger.warning(f"[CloudflareImage] adapt_image_dimensions fallback: {e}")
            return raw_bytes

    @classmethod
    async def generate_image(
        cls,
        prompt: str,
        brand: Optional[BrandProfile] = None,
        aspect_ratio: str = "1:1",
        style_preset: str = "modern_product",
        organization_id: int = 1,
        variation_seed: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main centralized image generation workflow:
        1. Cloudflare Workers AI generates the image from user's exact prompt.
        2. Validates image bytes.
        3. Persists local file and uploads newly generated image to Cloudinary CDN.
        4. Returns generation ID, model, hashes, and storage URL.
        """
        cls._ensure_directory()
        token, account_id, default_model = cls.get_credentials()

        final_prompt = cls.build_accurate_prompt(
            prompt=prompt,
            brand=brand,
            style_preset=style_preset,
            aspect_ratio=aspect_ratio,
            variation_seed=variation_seed
        )

        generation_id = f"cf_gen_{uuid.uuid4().hex[:16]}"
        start_time = time.time()
        logger.info(f"[CloudflareImageService] Starting generation {generation_id} for prompt: '{prompt[:80]}'")

        # Models to try on Cloudflare Workers AI
        models_to_try = [default_model]
        # Include alternative cloudflare models in priority order
        for m in cls.CLOUDFLARE_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        img_bytes = None
        used_model = None
        last_error = None

        for model_name in models_to_try:
            try:
                logger.info(f"[CloudflareImageService] Requesting Cloudflare Workers AI model: {model_name}")
                img_bytes = await cls._call_cloudflare_model(
                    model_name=model_name,
                    prompt=final_prompt,
                    token=token,
                    account_id=account_id
                )
                if img_bytes and len(img_bytes) > 5000:
                    used_model = model_name
                    break
            except CloudflareAIError as cf_call_err:
                last_error = str(cf_call_err)
                logger.warning(f"[CloudflareImageService] {model_name} quota notice: {cf_call_err}")
                if cf_call_err.status_code == 429 or "quota" in str(cf_call_err).lower() or "allocation" in str(cf_call_err).lower() or "10,000 neurons" in str(cf_call_err).lower():
                    logger.warning("[CloudflareImageService] Account daily neuron allocation exhausted. Breaking immediately from model attempts.")
                    break
            except Exception as call_err:
                last_error = str(call_err)
                logger.warning(f"[CloudflareImageService] Error on {model_name}: {call_err}")
                if "10,000 neurons" in str(call_err).lower() or "daily free allocation" in str(call_err).lower():
                    logger.warning("[CloudflareImageService] Account daily neuron allocation exhausted. Breaking early from model attempts.")
                    break

        # If generation failed on all available Cloudflare models, raise actionable error to trigger legitimate service fallback
        if not img_bytes or len(img_bytes) < 5000:
            error_message = (
                f"Cloudflare Workers AI failed to generate image. "
                f"All Cloudflare models ({', '.join(models_to_try[:2])}) returned no image data. "
                f"Last error: {last_error or 'Daily neuron allocation exhausted or model unavailable.'}"
            )
            logger.error(f"[CloudflareImageService] {error_message}")
            error_status = 429 if ("429" in str(last_error) or "allocation" in str(last_error).lower() or "quota" in str(last_error).lower()) else 502
            raise CloudflareAIError(error_message, status_code=error_status)

        # Dimension adaptation matching requested aspect ratio
        tw, th = cls.DIMENSION_MAP.get(aspect_ratio, (1024, 1024))
        enhanced_bytes = cls.adapt_image_dimensions(img_bytes, tw, th)

        # Hashes for duplicate prevention
        sha256_hash = hashlib.sha256(enhanced_bytes).hexdigest()
        perceptual_hash = None
        try:
            import imagehash
            im = Image.open(io.BytesIO(enhanced_bytes))
            perceptual_hash = str(imagehash.dhash(im))
        except Exception:
            pass

        # Save to local storage
        filename = f"{generation_id}_{tw}x{th}.jpg"
        file_path = os.path.join(cls.AI_IMAGES_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(enhanced_bytes)

        file_size = len(enhanced_bytes)
        storage_url = f"/uploads/media/ai_images/{filename}"

        # Upload strictly the newly generated Cloudflare image to durable Cloudinary CDN Storage
        if CloudinaryStorageProvider.is_configured():
            try:
                c_res = CloudinaryStorageProvider.upload_local_file_path(
                    file_path=file_path,
                    organization_id=organization_id,
                    folder_prefix="socialai/ai_images",
                    resource_type="image"
                )
                if c_res and c_res.get("file_url"):
                    storage_url = c_res["file_url"]
                    logger.info(f"[CloudflareImageService] Stored Cloudflare image in Cloudinary: {storage_url}")
                else:
                    raise RuntimeError("Cloudinary did not return a valid delivery URL.")
            except Exception as c_err:
                logger.error(f"[CloudflareImageService] Cloudinary durable upload failed: {c_err}")
                raise CloudflareAIError(f"Cloudinary durable storage upload failed: {c_err}", status_code=502) from c_err

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"[CloudflareImageService] Generation {generation_id} succeeded in {elapsed}s using {used_model}")

        return {
            "generation_id": generation_id,
            "provider": "cloudflare-workers-ai",
            "model_name": used_model,
            "prompt": prompt,
            "final_prompt": final_prompt,
            "image_url": storage_url,
            "local_path": file_path,
            "file_name": filename,
            "file_size_bytes": file_size,
            "aspect_ratio": aspect_ratio,
            "mime_type": "image/jpeg",
            "sha256": sha256_hash,
            "perceptual_hash": perceptual_hash,
            "generation_time_seconds": elapsed,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
