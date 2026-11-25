from PIL import TiffImagePlugin
import os
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Social AI Platform"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    @field_validator("DEBUG", mode="before")
    def parse_debug_mode(cls, value):
        """Accept common deployment labels such as DEBUG=release/prod."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod", "false", "0", "no", "off"}:
                return False
            if normalized in {"development", "dev", "true", "1", "yes", "on"}:
                return True
        return value

    # Security & JWT
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 day

    @field_validator("SECRET_KEY", mode="before")
    def validate_secret_key(cls, v: str | None) -> str:
        raw_val = (v or "").strip()
        env_mode = (os.environ.get("ENVIRONMENT") or "development").strip().lower()
        known_insecure = {
            "temporary-super-secret-key-change-in-production-1234567890",
            "changeme",
            "secret",
            "your-secret-key",
            "super-secret-key",
            "default-secret",
            "1234567890",
            "password"
        }
        if env_mode in {"production", "prod"}:
            if not raw_val or raw_val in known_insecure or len(raw_val) < 32:
                raise ValueError(
                    "FATAL SECURITY CONFIGURATION ERROR: A strong, cryptographically secure SECRET_KEY "
                    "(minimum 32 characters) must be configured in production environment. "
                    "Refusing to start with insecure, missing, or default placeholder secret."
                )
        if not raw_val:
            raw_val = "dev-only-local-secret-key-minimum-32-chars-for-safety-0987654321"
        return raw_val

    # MySQL Database Connection
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_DB: str = "social_ai_platform"
    DATABASE_URL: str | None = None

    @field_validator("DATABASE_URL", mode="before")
    def assemble_db_connection(cls, v: str | None, values) -> str:
        from urllib.parse import quote_plus, unquote

        def _normalize_and_encode_url(raw: str) -> str:
            url = raw.strip()
            # Railway and cloud providers inject mysql://, SQLAlchemy requires mysql+pymysql://
            if url.startswith("mysql://"):
                url = url.replace("mysql://", "mysql+pymysql://", 1)
            elif url.startswith("mysql2://"):
                url = url.replace("mysql2://", "mysql+pymysql://", 1)
            elif url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+psycopg2://", 1)
            elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
                url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

            if "://" not in url:
                return url

            scheme, rest = url.split("://", 1)
            path = ""
            if "/" in rest:
                netloc, path = rest.split("/", 1)
            else:
                netloc = rest

            # If userinfo (user:password) exists before host, ensure password & username are properly URL-encoded
            if "@" in netloc:
                userinfo, hostport = netloc.rsplit("@", 1)
                if ":" in userinfo:
                    user, pwd = userinfo.split(":", 1)
                    user = quote_plus(unquote(user))
                    pwd = quote_plus(unquote(pwd))
                    netloc = f"{user}:{pwd}@{hostport}"
                else:
                    user = quote_plus(unquote(userinfo))
                    netloc = f"{user}@{hostport}"

            return f"{scheme}://{netloc}/{path}"

        # Prioritize explicit .env value (v) over ambient system variables to prevent stale overrides
        raw_url = (
            v
            or os.environ.get("DATABASE_URL")
            or os.environ.get("MYSQL_URL")
            or os.environ.get("MYSQL_PRIVATE_URL")
            or os.environ.get("MYSQLPUBLICURL")
            or os.environ.get("MYSQL_PUBLIC_URL")
        )
        if isinstance(raw_url, str) and raw_url.strip():
            return _normalize_and_encode_url(raw_url)

        # Build from explicit components if full URL is omitted
        host = values.data.get('MYSQL_HOST') or os.environ.get("MYSQL_HOST") or os.environ.get("MYSQLHOST") or '127.0.0.1'
        port = int(values.data.get('MYSQL_PORT') or os.environ.get("MYSQL_PORT") or os.environ.get("MYSQLPORT") or 3306)
        user = values.data.get('MYSQL_USER') or os.environ.get("MYSQL_USER") or os.environ.get("MYSQLUSER") or 'root'
        password = values.data.get('MYSQL_PASSWORD') or os.environ.get("MYSQL_PASSWORD") or os.environ.get("MYSQLPASSWORD") or 'password'
        database = values.data.get('MYSQL_DB') or values.data.get('MYSQL_DATABASE') or os.environ.get("MYSQL_DB") or os.environ.get("MYSQLDATABASE") or 'social_ai_platform'

        encoded_user = quote_plus(unquote(str(user)))
        encoded_password = quote_plus(unquote(str(password)))
        return f"mysql+pymysql://{encoded_user}:{encoded_password}@{host}:{port}/{database}"

    # Redis & Celery
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
    ]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return []

    # Aminoid AI Unification Feature Flags (Disabled by default)
    FEATURE_UNIFIED_AI_STUDIO: bool = False
    FEATURE_UNIFIED_GEMINI_STUDIO: bool = False
    FEATURE_UNIFIED_PIPELINE: bool = False

    # AI Engine - Single Google Gemini API Key for Text, Image, and Video
    GEMINI_API_KEY: str = ""
    GEMINI_TEXT_MODEL: str = "gemini-3.5-flash"
    GEMINI_IMAGE_MODEL: str = "imagen-3.0-generate-002"
    GEMINI_VIDEO_MODEL: str = "veo-2.0-generate-001"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""  # Also used as VEO_API_KEY if VEO_API_KEY not set
    VEO_API_KEY: str = ""
    STABILITY_API_KEY: str = ""
    MAGIC_HOUR_API_KEY: str = ""

    # Cloudflare Workers AI / Flux image generation
    CLOUDFLARE_API_TOKEN: str = ""
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_IMAGE_MODEL: str = "@cf/black-forest-labs/flux-1-schnell"
    # Deactivated external image providers (Pollinations & fal.ai permanently disabled)
    FAL_KEY: str = ""
    FAL_IMAGE_MODEL: str = ""
    LTX_API_KEY: str = ""
    HUGGINGFACE_API_TOKEN: str = ""
    HUGGINGFACE_VIDEO_SPACE: str = "ByteDance/AnimateDiff-Lightning"

    # Daily marketing providers: explicit routing, no hidden paid fallbacks.
    DAILY_TEXT_MODEL: str = "gemini-3.5-flash"
    DAILY_IMAGE_PROVIDER: str = "cloudflare"
    DAILY_OPENAI_IMAGE_MODEL: str = "gpt-image-2.6-flare"
    DAILY_VIDEO_PROVIDER: str = "huggingface"
    DAILY_VIDEO_MODEL: str = "gemini-omni-1.1-flash"
    DAILY_MAX_ATTEMPTS: int = 3
    DAILY_LEASE_SECONDS: int = 1200
    DAILY_MONTHLY_ASSET_LIMIT: int = 120
    DAILY_VIDEO_SECONDS: int = 30
    DAILY_WORKER_ENABLED: bool = True
    R2_ACCOUNT_ID: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET: str = ""

    # Security & Encryption
    ENCRYPTION_KEY: str = ""
    TOKEN_ENCRYPTION_KEY: str = ""

    # Frontend URL
    FRONTEND_URL: str = "http://localhost:3000"

    # Social APIs - Meta (Instagram & Facebook)
    META_APP_ID: str = ""
    META_APP_SECRET: str = ""
    META_GRAPH_VERSION: str = "v23.0"
    META_REDIRECT_URI: str = "http://127.0.0.1:8002/api/v1/social/facebook/callback"
    META_SCOPES: str = "pages_show_list,pages_read_engagement,pages_manage_posts,instagram_basic,instagram_content_publish"
    META_WEBHOOK_VERIFY_TOKEN: str = ""
    INSTAGRAM_CLIENT_ID: str = ""
    INSTAGRAM_CLIENT_SECRET: str = ""
    FACEBOOK_CLIENT_ID: str = ""
    FACEBOOK_CLIENT_SECRET: str = ""

    # Social APIs - Google / YouTube
    YOUTUBE_CLIENT_ID: str = ""
    YOUTUBE_CLIENT_SECRET: str = ""

    # Note: Application channels are restricted strictly to Instagram, Facebook, and YouTube.

    # Storage & Cloudinary
    STORAGE_TYPE: str = "cloudinary"  # 'cloudinary', 'local', or 's3'
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: str = "social-ai-platform-assets"
    # Local disk is temporary-only for media processing. Persisted media must
    # be uploaded to Cloudinary; never use a project-level uploads directory.
    LOCAL_STORAGE_DIR: str = "./.media_tmp"

    # Cloudinary Integration (Production fallback defaults)
    CLOUDINARY_CLOUD_NAME: str = ""
    CLOUDINARY_API_KEY: str = ""
    CLOUDINARY_API_SECRET: str = ""

    # Razorpay Payments
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""
    RAZORPAY_WEBHOOK_SECRET: str = ""

    model_config = SettingsConfigDict(
        env_file=(
            os.path.join(os.path.dirname(__file__), "../../.env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow"
    )


settings = Settings()
