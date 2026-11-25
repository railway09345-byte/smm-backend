import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings

class AIModelConfig(BaseModel):
    """Centralized verified model identifiers and runtime routing parameters."""
    
    # Text Generation
    # Hierarchy: Primary Gemini -> Secondary Gemini Lite -> External OpenAI Backup -> Brand Fallback
    TEXT_PRIMARY_MODEL: str = "gemini-3.5-flash"
    TEXT_SECONDARY_ROTATION: List[str] = [
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-flash-latest"
    ]
    TEXT_SECONDARY_MODEL: str = "gemini-3.1-flash-lite"
    TEXT_BACKUP_MODEL: str = "gemini-flash-lite-latest"
    TEXT_OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Visual Art / Image Generation
    # Hierarchy: Cloudflare SDXL Lightning -> Gemini Imagen 3 -> Deterministic SVG Badge
    IMAGE_PRIMARY_PROVIDER: str = "cloudflare"
    IMAGE_PRIMARY_MODEL: str = "@cf/bytedance/stable-diffusion-xl-lightning"
    IMAGE_BACKUP_PROVIDER: str = "google-gemini-image"
    IMAGE_BACKUP_MODEL: str = "imagen-3.0-generate-002"
    
    # Video Generation
    # Hierarchy: Multi-Scene Cinema Director -> Lightricks LTX / HF ZeroGPU -> Fallback Reel
    VIDEO_PRIMARY_ENGINE: str = "multi-scene-cinema-director"
    VIDEO_SECONDARY_ENGINE: str = "huggingface-ltx-video"
    VIDEO_EXTERNAL_ENGINE: str = "ltx-video"
    
    # Neural Voiceovers
    # Hierarchy: Microsoft Edge Neural TTS
    VOICE_DEFAULT_MALE: str = "en-US-ChristopherNeural"
    VOICE_DEFAULT_FEMALE: str = "en-US-JennyNeural"
    VOICE_DEFAULT_BRITISH: str = "en-GB-RyanNeural"
    VOICE_DEFAULT_INDIAN: str = "en-IN-PrabhatNeural"
    
    # Timeouts & Limits
    TEXT_TIMEOUT_SECONDS: float = 30.0
    IMAGE_TIMEOUT_SECONDS: float = 45.0
    VIDEO_TIMEOUT_SECONDS: float = 840.0 # 14 minutes soft limit


# Global Singleton Instance
centralized_ai_config = AIModelConfig(
    TEXT_PRIMARY_MODEL=getattr(settings, "GEMINI_TEXT_MODEL", "gemini-3.5-flash") or "gemini-3.5-flash",
    IMAGE_BACKUP_MODEL=getattr(settings, "GEMINI_IMAGE_MODEL", "imagen-3.0-generate-002") or "imagen-3.0-generate-002"
)

