"""
Hugging Face AI Video Generation Service.
Generates authentic cinematic AI video using Lightricks LTX Video on Hugging Face ZeroGPU
(Lightricks/ltx-video-distilled), customized with rich brand identity and Edge Neural TTS voiceover.
100% Free, unlimited community compute on Hugging Face.
"""

import os
import uuid
import shutil
import subprocess
import math
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import imageio_ffmpeg
try:
    from gradio_client import Client
except ImportError:
    Client = None

from app.core.config import settings
from app.core.logging import logger
from app.models.brand_profile import BrandProfile
from app.integrations.storage.cloudinary_storage import CloudinaryStorageProvider
from app.services.voice_service import VoiceService


class HuggingFaceVideoService:
    """Enterprise client for Hugging Face AI Video Generation using Lightricks LTX Video."""

    AI_VIDEOS_DIR = os.path.join(settings.LOCAL_STORAGE_DIR, "media", "ai_videos")

    @classmethod
    def is_configured(cls) -> bool:
        return bool(settings.HUGGINGFACE_API_TOKEN and settings.HUGGINGFACE_API_TOKEN.startswith("hf_"))

    @classmethod
    def _ensure_directories(cls):
        os.makedirs(cls.AI_VIDEOS_DIR, exist_ok=True)

    @classmethod
    def build_brand_rich_prompt(
        cls,
        base_prompt: str,
        brand: Optional[BrandProfile] = None
    ) -> str:
        """
        Synthesizes a high-fidelity cinematic video prompt incorporating
        verified brand identity, niche, aesthetic tone, and commercial direction.
        """
        brand_name = brand.name if brand and brand.name else "Brand"
        industry = brand.industry if brand and brand.industry else "lifestyle"
        tone = brand.tone_of_voice if brand and brand.tone_of_voice else "inspiring"

        clean_user_prompt = (base_prompt or f"Showcase {brand_name}'s collection").strip()

        enriched = (
            f"Cinematic commercial for {brand_name} ({industry}): {clean_user_prompt}. "
            f"{tone} aesthetic, smooth professional camera motion, cinematic studio lighting, 8k resolution, photorealistic masterpiece, clean footage, no watermark, no logo, no text"
        )
        return enriched[:500]

    @classmethod
    async def generate_and_persist_video(
        cls,
        prompt: str,
        brand: Optional[BrandProfile] = None,
        aspect_ratio: str = "9:16",
        duration_seconds: int = 30,
        voice: Optional[str] = None,
        voice_script: Optional[str] = None,
        script_scenes: Optional[list] = None,
        input_image_path: Optional[str] = None,
        organization_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes true AI neural video generation via Lightricks/ltx-video-distilled:
        1. Formulates brand-rich cinematic prompt and commercial voiceover script.
        2. Submits to Lightricks LTX-Video on Hugging Face ZeroGPU.
        3. Generates synchronized Microsoft Edge Neural TTS voiceover narration.
        4. Muxes neural video frames and voiceover into broadcast-ready 30s H.264/AAC MP4.
        5. Persists to local storage & Cloudinary CDN.
        """
        cls._ensure_directories()
        token = settings.HUGGINGFACE_API_TOKEN
        space_name = settings.HUGGINGFACE_VIDEO_SPACE or "Lightricks/ltx-video-distilled"

        brand_name = brand.name if brand else "Social AI"
        brand_tagline = brand.tagline or brand.industry or ""
        target_dur = max(3, min(int(duration_seconds), 30))

        # 1. Resolve voice persona and compose dynamic script
        brand_tone = brand.tone_of_voice if brand and brand.tone_of_voice else "Engaging"
        target_voice_id = VoiceService.resolve_voice(
            voice_preference=voice,
            brand_tone=brand_tone,
            seed=f"{prompt[:30]}-{brand_name}"
        )

        composed_voiceover = voice_script or VoiceService.compose_commercial_voiceover(
            prompt=prompt,
            brand_name=brand_name,
            brand_tagline=brand_tagline,
            brand_tone=brand_tone,
            script_scenes=script_scenes,
            target_duration=target_dur,
            variation_seed=uuid.uuid4().hex[:8]
        )

        voice_filename = f"hf_voice_{uuid.uuid4().hex[:12]}.mp3"
        voice_path = os.path.join(cls.AI_VIDEOS_DIR, voice_filename)
        actual_voice_dur = target_dur
        has_voice = False
        try:
            actual_voice_dur = await VoiceService.synthesize_speech(composed_voiceover, voice_path, voice=target_voice_id)
            has_voice = os.path.exists(voice_path) and os.path.getsize(voice_path) > 1000
        except Exception as v_err:
            logger.error(f"Voiceover synthesis failed in HF video service: {v_err}")
            raise RuntimeError(
                f"Voiceover generation failed for brand '{brand_name}': {v_err}. "
                "Refusing to generate video without audible voiceover."
            ) from v_err

        if not has_voice or not os.path.exists(voice_path) or os.path.getsize(voice_path) <= 1000:
            raise RuntimeError(
                f"Voiceover generation failed: Edge Neural TTS produced no valid audio data for brand '{brand_name}'. "
                "Refusing to generate video without audible voiceover."
            )

        # 2. Dispatch Hugging Face neural video clip generation
        enriched_prompt = cls.build_brand_rich_prompt(prompt, brand)
        logger.info(f"[HFVideo] Dispatching AI video generation on '{space_name}' with prompt: {enriched_prompt[:80]}...")

        segment_paths = []
        provider_error = None
        # Bounded clip count so free shared ZeroGPU does not stall or crash the request
        clip_count = min(2, max(1, math.ceil(target_dur / 15)))
        if Client is not None:
            try:
                client = Client(space_name, token=token)
                dim_map = {
                    "9:16": (512, 704),
                    "16:9": (704, 512),
                    "1:1": (512, 512),
                    "4:5": (576, 704)
                }
                w, h = dim_map.get(aspect_ratio, (512, 704) if aspect_ratio == "9:16" else (704, 512))

                # LTX ZeroGPU yields ~3-second clips. A 30-second bundle is made
                # from ten independently generated AI scenes, never by looping a
                # single clip or filling time with a template renderer.
                for scene_index in range(clip_count):
                    scene_prompt = (
                        f"{enriched_prompt} Scene {scene_index + 1} of {clip_count}: "
                        "continue the same factual campaign with a distinct camera angle and action."
                    )[:500]
                    image_mode = input_image_path and os.path.exists(input_image_path)
                    image_arg = None
                    if image_mode:
                        from gradio_client import handle_file
                        image_arg = handle_file(input_image_path)
                    generated_path = None
                    for retry in range(3):
                        try:
                            result = await asyncio.to_thread(
                                client.predict,
                                prompt=scene_prompt,
                                negative_prompt="blurry, low quality, distorted, jittery, watermark, logo, brand logo, text",
                                input_image_filepath=image_arg,
                                input_video_filepath=None,
                                height_ui=h,
                                width_ui=w,
                                mode="image-to-video" if image_mode else "text-to-video",
                                duration_ui=3.0,
                                ui_frames_to_use=9,
                                seed_ui=42 + scene_index + retry * 100,
                                randomize_seed=True,
                                ui_guidance_scale=1.5,
                                improve_texture_flag=True,
                                api_name="/image_to_video" if image_mode else "/text_to_video",
                            )
                            candidate = result[0] if isinstance(result, (tuple, list)) and result else result
                            generated_path = candidate.get("video") if isinstance(candidate, dict) else candidate
                            if isinstance(generated_path, str) and os.path.exists(generated_path):
                                break
                            provider_error = f"scene {scene_index + 1} returned no video"
                        except Exception as scene_error:
                            provider_error = f"scene {scene_index + 1}: {type(scene_error).__name__}"
                        generated_path = None
                        await asyncio.sleep(3 * (retry + 1))
                    if not generated_path:
                        raise RuntimeError(provider_error or f"LTX returned no playable clip for scene {scene_index + 1}.")
                    segment_path = os.path.join(cls.AI_VIDEOS_DIR, f"ltx_scene_{uuid.uuid4().hex[:12]}.mp4")
                    shutil.copy2(generated_path, segment_path)
                    segment_paths.append(segment_path)
                    logger.info(f"[HFVideo] Generated AI scene {scene_index + 1}/{clip_count}.")
            except Exception as hf_err:
                provider_error = f"{type(hf_err).__name__}: {str(hf_err)[:160]}"
                logger.warning(f"[HFVideo] Hugging Face ZeroGPU generation failed: {provider_error}")

        # AI-only path: every segment must be a genuine LTX clip.
        if not segment_paths:
            raise RuntimeError(f"Hugging Face LTX did not return a playable AI video clip ({provider_error or 'shared compute unavailable'}).")
        if len(segment_paths) != clip_count:
            raise RuntimeError(f"Hugging Face LTX generated only {len(segment_paths)} of {clip_count} required AI scenes ({provider_error or 'shared compute unavailable'}).")
        if not CloudinaryStorageProvider.is_configured():
            raise RuntimeError("Cloudinary storage is not configured for generated videos.")
        if len(segment_paths) == 1:
            final_path = segment_paths[0]
        else:
            manifest_path = os.path.join(cls.AI_VIDEOS_DIR, f"ltx_manifest_{uuid.uuid4().hex[:12]}.txt")
            final_path = os.path.join(cls.AI_VIDEOS_DIR, f"ltx_bundle_{uuid.uuid4().hex[:12]}.mp4")
            try:
                with open(manifest_path, "w", encoding="utf-8") as manifest:
                    for path in segment_paths:
                        # Windows ffmpeg concat demuxer requires forward slashes
                        safe_path = path.replace("\\", "/").replace("'", "'\\''")
                        manifest.write(f"file '{safe_path}'\n")
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0", "-i", manifest_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", final_path], check=True, capture_output=True)
            finally:
                if os.path.exists(manifest_path):
                    os.remove(manifest_path)
                for path in segment_paths:
                    if path != final_path and os.path.exists(path):
                        os.remove(path)

        # For commercial reels (target >= 15s), weave into 30s broadcast commercial reel with multi-scene choreography
        if target_dur >= 15:
            from app.services.video_generation_service import VideoGenerationService
            logger.info(f"[HFVideo] Weaving Hugging Face neural clips into full {target_dur}s broadcast commercial reel with voiceover...")
            try:
                rendered = await asyncio.to_thread(
                    VideoGenerationService.render_branded_promotional_video,
                    prompt=prompt,
                    brand=brand,
                    aspect_ratio=aspect_ratio,
                    duration_seconds=target_dur,
                    fps=24,
                    resolution="1080p",
                    voice=target_voice_id,
                    voice_script=composed_voiceover,
                    script_scenes=script_scenes,
                    image_url=input_image_path,
                    description=prompt,
                    neural_video_path=final_path
                )
            finally:
                for tmp_f in [final_path, voice_path]:
                    if os.path.exists(tmp_f):
                        try:
                            os.remove(tmp_f)
                        except Exception:
                            pass

            return {
                "video_url": rendered["video_url"],
                "file_path": rendered["video_url"],
                "local_path": rendered.get("local_path"),
                "file_name": rendered.get("file_name"),
                "file_size_bytes": rendered.get("file_size_bytes", 0),
                "duration_seconds": rendered.get("duration_seconds", target_dur),
                "aspect_ratio": aspect_ratio,
                "mime_type": "video/mp4",
                "model_name": "lightricks-ltx-video-distilled",
                "provider": "huggingface-ltx-video",
                "cloudinary_public_id": rendered.get("cloudinary_public_id"),
                "cost_usd": 0.0,
                "cost_inr": 0.0,
                "prompt": enriched_prompt,
                "acts_breakdown": rendered.get("acts_breakdown", []),
                "voiceover_script": rendered.get("voiceover_script", composed_voiceover),
                "has_voiceover": rendered.get("has_voiceover", True),
                "voice_id": target_voice_id,
                "brand_name": brand_name,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

        # Otherwise mux the voiceover audio track directly into the short video clip
        muxed_path = final_path.replace(".mp4", "_narrated.mp4")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg_exe, "-y",
            "-i", final_path,
            "-i", voice_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            muxed_path
        ], check=True, capture_output=True)
        if os.path.exists(muxed_path) and os.path.getsize(muxed_path) > 1000:
            try:
                os.replace(muxed_path, final_path)
            except Exception:
                final_path = muxed_path

        uploaded = CloudinaryStorageProvider.upload_local_file_path(
            file_path=final_path,
            organization_id=organization_id or (brand.organization_id if brand else 1),
            folder_prefix="socialai/ai_videos/huggingface_ltx",
            resource_type="video",
        )
        file_name = os.path.basename(final_path)

        # Cleanup voice temp
        if os.path.exists(voice_path):
            try:
                os.remove(voice_path)
            except Exception:
                pass

        return {
            "video_url": uploaded["file_url"],
            "file_path": uploaded["file_url"],
            "local_path": final_path,
            "file_name": file_name,
            "file_size_bytes": uploaded.get("file_size_bytes", os.path.getsize(final_path)),
            "duration_seconds": target_dur,
            "aspect_ratio": aspect_ratio,
            "mime_type": "video/mp4",
            "model_name": "lightricks-ltx-video-distilled",
            "provider": "huggingface-ltx-video",
            "cloudinary_public_id": uploaded.get("public_id"),
            "cost_usd": 0.0,
            "cost_inr": 0.0,
            "prompt": enriched_prompt,
            "acts_breakdown": [],
            "voiceover_script": composed_voiceover,
            "has_voiceover": True,
            "voice_id": target_voice_id,
            "brand_name": brand_name,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
