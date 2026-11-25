import os
import uuid
import time
from typing import Dict, Any, Tuple, Optional, List, Union
import cloudinary
import cloudinary.uploader
import cloudinary.utils
from fastapi import UploadFile, HTTPException, status
from app.core.config import settings
from app.core.logging import logger
from app.utils.enums import MediaType
from app.utils.validators import sanitize_filename

# Allowed MIME types and extensions
ALLOWED_IMAGE_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif"
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/webm": "webm",
    "video/x-msvideo": "avi",
    "video/x-matroska": "mkv"
}

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/flac": "flac"
}

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}

MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024       # 25 MB
MAX_VIDEO_SIZE_BYTES = 150 * 1024 * 1024      # 150 MB
MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024       # 50 MB


class CloudinaryStorageProvider:
    """Enterprise Cloudinary storage provider for images and videos."""

    @classmethod
    def is_configured(cls) -> bool:
        """Checks if valid Cloudinary credentials are configured."""
        return bool(
            settings.CLOUDINARY_CLOUD_NAME
            and settings.CLOUDINARY_API_KEY
            and settings.CLOUDINARY_API_SECRET
        )

    @classmethod
    def _init_cloudinary(cls):
        """Initializes Cloudinary SDK with settings credentials."""
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True
        )

    @classmethod
    def validate_file(cls, filename: str, content_type: str, file_size: int) -> Tuple[MediaType, str]:
        """
        Validates format, extension, and file size.
        Returns (MediaType, extension).
        Raises HTTPException on invalid input.
        """
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File name cannot be empty."
            )

        ext = os.path.splitext(filename)[1].lower()
        mime = (content_type or "").lower().split(";")[0].strip()

        # Check image types
        if ext in ALLOWED_IMAGE_EXTENSIONS or mime in ALLOWED_IMAGE_TYPES:
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                ext = f".{ALLOWED_IMAGE_TYPES.get(mime, 'jpg')}"
            if file_size > MAX_IMAGE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image file size exceeds the 25MB limit (provided {file_size / (1024 * 1024):.2f}MB)."
                )
            return MediaType.IMAGE, ext

        # Check video types
        if ext in ALLOWED_VIDEO_EXTENSIONS or mime in ALLOWED_VIDEO_TYPES:
            if ext not in ALLOWED_VIDEO_EXTENSIONS:
                ext = f".{ALLOWED_VIDEO_TYPES.get(mime, 'mp4')}"
            if file_size > MAX_VIDEO_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video file size exceeds the 150MB limit (provided {file_size / (1024 * 1024):.2f}MB)."
                )
            return MediaType.VIDEO, ext

        # Check audio types
        if ext in ALLOWED_AUDIO_EXTENSIONS or mime in ALLOWED_AUDIO_TYPES:
            if ext not in ALLOWED_AUDIO_EXTENSIONS:
                ext = f".{ALLOWED_AUDIO_TYPES.get(mime, 'mp3')}"
            if file_size > MAX_AUDIO_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Audio file size exceeds the 50MB limit (provided {file_size / (1024 * 1024):.2f}MB)."
                )
            return MediaType.AUDIO, ext

        # If unsupported
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{ext}' ({content_type}). "
                f"Allowed image formats: PNG, JPG, JPEG, WebP, GIF. "
                f"Allowed video formats: MP4, MOV, WebM, AVI, MKV. "
                f"Allowed audio formats: MP3, WAV, OGG, M4A, AAC, FLAC."
            )
        )

    @classmethod
    def upload_local_file_path(
        cls,
        file_path: str,
        organization_id: int = 1,
        folder_prefix: str = "socialai/ai_generated",
        resource_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Upload an existing local file directly to Cloudinary with exponential-backoff retries
        and return permanent HTTPS CDN metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        folder = f"{folder_prefix}/org_{organization_id}"
        ext = os.path.splitext(file_name)[1].lower()

        # Determine MediaType and Cloudinary resource_type
        if ext in ALLOWED_AUDIO_EXTENSIONS or resource_type == "audio":
            media_type = MediaType.AUDIO
            c_resource_type = "video"  # Cloudinary processes audio under video resource_type
            mime_type = f"audio/{ALLOWED_AUDIO_TYPES.get(ext.strip('.'), ext.strip('.'))}"
        elif ext in ALLOWED_VIDEO_EXTENSIONS or resource_type == "video":
            media_type = MediaType.VIDEO
            c_resource_type = "video"
            mime_type = f"video/{ALLOWED_VIDEO_TYPES.get(ext.strip('.'), ext.strip('.'))}"
        else:
            media_type = MediaType.IMAGE
            c_resource_type = "image"
            mime_type = f"image/{ALLOWED_IMAGE_TYPES.get(ext.strip('.'), 'png')}"

        if not cls.is_configured():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Cloudinary is not configured. Configure CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET."
            )

        cls._init_cloudinary()
        last_error = None
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                # Chunked upload for video/audio larger than 10MB to guarantee resilience
                if c_resource_type == "video" and file_size > 10 * 1024 * 1024:
                    upload_result = cloudinary.uploader.upload_large(
                        file_path,
                        folder=folder,
                        resource_type=c_resource_type,
                        chunk_size=6000000,
                        use_filename=True,
                        unique_filename=True,
                        overwrite=False
                    )
                else:
                    upload_result = cloudinary.uploader.upload(
                        file_path,
                        folder=folder,
                        resource_type=c_resource_type,
                        use_filename=True,
                        unique_filename=True,
                        overwrite=False
                    )

                delivery_url = upload_result.get("secure_url") or upload_result.get("url")
                if not delivery_url or not delivery_url.startswith("https://"):
                    if delivery_url and delivery_url.startswith("http://"):
                        delivery_url = delivery_url.replace("http://", "https://")
                    else:
                        raise RuntimeError(f"Cloudinary did not return a valid HTTPS delivery URL: {delivery_url}")

                return {
                    "file_name": file_name,
                    "file_url": delivery_url,
                    "public_id": upload_result.get("public_id"),
                    "file_size_bytes": upload_result.get("bytes", file_size),
                    "mime_type": mime_type,
                    "media_type": media_type,
                    "meta_info": {
                        "provider": "cloudinary",
                        "public_id": upload_result.get("public_id"),
                        "format": upload_result.get("format"),
                        "resource_type": upload_result.get("resource_type", c_resource_type),
                        "width": upload_result.get("width"),
                        "height": upload_result.get("height"),
                        "duration": upload_result.get("duration"),
                        "created_at": upload_result.get("created_at")
                    }
                }
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[Cloudinary] Upload attempt {attempt + 1}/{max_attempts} for {file_name} failed: {e}. "
                    f"Retrying..." if attempt < max_attempts - 1 else ""
                )
                if attempt < max_attempts - 1:
                    time.sleep(1.0 * (attempt + 1))

        logger.error(f"[Cloudinary] All {max_attempts} upload attempts failed for {file_name}: {last_error}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Durable cloud storage upload failed for {file_name}: {str(last_error)}"
        )

    @classmethod
    async def upload_file(
        cls,
        file: UploadFile,
        organization_id: int,
        folder_prefix: str = "socialai"
    ) -> Dict[str, Any]:
        """
        Reads and validates an uploaded file, then uploads to Cloudinary.
        Falls back to safe local storage simulation if Cloudinary credentials are not configured.
        """
        content = await file.read()
        file_size = len(content)

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot upload an empty file."
            )

        original_filename = sanitize_filename(file.filename or "upload.bin")
        content_type = file.content_type or "application/octet-stream"
        media_type, ext = cls.validate_file(original_filename, content_type, file_size)

        folder = f"{folder_prefix}/org_{organization_id}"
        resource_type = "video" if media_type in (MediaType.VIDEO, MediaType.AUDIO) else "image"

        # Check if Cloudinary is fully configured
        if cls.is_configured():
            cls._init_cloudinary()
            max_attempts = 3
            last_err = None
            for attempt in range(max_attempts):
                try:
                    upload_result = cloudinary.uploader.upload(
                        content,
                        folder=folder,
                        resource_type=resource_type,
                        use_filename=True,
                        unique_filename=True,
                        overwrite=False
                    )

                    delivery_url = upload_result.get("secure_url") or upload_result.get("url")
                    if delivery_url and delivery_url.startswith("http://"):
                        delivery_url = delivery_url.replace("http://", "https://")

                    return {
                        "file_name": original_filename,
                        "file_url": delivery_url,
                        "public_id": upload_result.get("public_id"),
                        "file_size_bytes": upload_result.get("bytes", file_size),
                        "mime_type": content_type or f"{resource_type}/{upload_result.get('format', ext.strip('.'))}",
                        "media_type": media_type,
                        "meta_info": {
                            "provider": "cloudinary",
                            "public_id": upload_result.get("public_id"),
                            "format": upload_result.get("format"),
                            "resource_type": upload_result.get("resource_type", resource_type),
                            "width": upload_result.get("width"),
                            "height": upload_result.get("height"),
                            "duration": upload_result.get("duration"),
                            "created_at": upload_result.get("created_at"),
                            "etag": upload_result.get("etag")
                        }
                    }
                except Exception as e:
                    last_err = e
                    logger.warning(f"Cloudinary upload attempt {attempt + 1}/{max_attempts} failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(1.0 * (attempt + 1))

            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Cloudinary upload failed after {max_attempts} attempts: {str(last_err)}"
            )

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Cloudinary is unavailable. Upload cancelled; configure valid Cloudinary credentials before scheduling media."
        )

    @classmethod
    def upload_bytes(
        cls,
        data: bytes,
        file_name: str,
        organization_id: int = 1,
        folder_prefix: str = "socialai/ai_generated",
        resource_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Upload raw in-memory bytes (e.g. from PIL, TTS, or external AI response) directly
        to Cloudinary and return durable permanent HTTPS CDN metadata.
        """
        if not data or len(data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot upload empty data bytes."
            )

        if not cls.is_configured():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Cloudinary storage is not configured."
            )

        cls._init_cloudinary()
        folder = f"{folder_prefix}/org_{organization_id}"
        ext = os.path.splitext(file_name)[1].lower()

        # Determine MediaType and Cloudinary resource_type
        if ext in ALLOWED_AUDIO_EXTENSIONS or resource_type == "audio":
            media_type = MediaType.AUDIO
            c_resource_type = "video"
            mime_type = f"audio/{ALLOWED_AUDIO_TYPES.get(ext.strip('.'), ext.strip('.'))}"
        elif ext in ALLOWED_VIDEO_EXTENSIONS or resource_type == "video":
            media_type = MediaType.VIDEO
            c_resource_type = "video"
            mime_type = f"video/{ALLOWED_VIDEO_TYPES.get(ext.strip('.'), ext.strip('.'))}"
        else:
            media_type = MediaType.IMAGE
            c_resource_type = "image"
            mime_type = f"image/{ALLOWED_IMAGE_TYPES.get(ext.strip('.'), 'png')}"

        max_attempts = 3
        last_error = None
        for attempt in range(max_attempts):
            try:
                upload_result = cloudinary.uploader.upload(
                    data,
                    folder=folder,
                    resource_type=c_resource_type,
                    use_filename=True,
                    unique_filename=True,
                    overwrite=False
                )

                delivery_url = upload_result.get("secure_url") or upload_result.get("url")
                if delivery_url and delivery_url.startswith("http://"):
                    delivery_url = delivery_url.replace("http://", "https://")

                return {
                    "file_name": file_name,
                    "file_url": delivery_url,
                    "public_id": upload_result.get("public_id"),
                    "file_size_bytes": upload_result.get("bytes", len(data)),
                    "mime_type": mime_type,
                    "media_type": media_type,
                    "meta_info": {
                        "provider": "cloudinary",
                        "public_id": upload_result.get("public_id"),
                        "format": upload_result.get("format"),
                        "resource_type": upload_result.get("resource_type", c_resource_type),
                        "width": upload_result.get("width"),
                        "height": upload_result.get("height"),
                        "duration": upload_result.get("duration"),
                        "created_at": upload_result.get("created_at")
                    }
                }
            except Exception as e:
                last_error = e
                logger.warning(f"[Cloudinary] upload_bytes attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1.0 * (attempt + 1))

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Durable cloud storage upload_bytes failed for {file_name}: {str(last_error)}"
        )

    @classmethod
    def safe_cleanup_local_file(cls, file_path: Optional[str]) -> bool:
        """Safely removes local temporary file without throwing errors if missing or locked."""
        if not file_path:
            return False
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"[Storage] Cleaned up temporary local file: {file_path}")
                return True
        except Exception as e:
            logger.warning(f"[Storage] Could not clean up temporary file {file_path}: {e}")
        return False

    @classmethod
    def upload_and_register_asset(
        cls,
        db: Any,
        file_path: Optional[str] = None,
        raw_bytes: Optional[bytes] = None,
        file_name: Optional[str] = None,
        organization_id: int = 1,
        user_id: Optional[int] = None,
        folder_prefix: str = "socialai/ai_generated",
        title: Optional[str] = None,
        description: Optional[str] = None,
        alt_text: Optional[str] = None,
        tags: Optional[list] = None,
        category_id: Optional[int] = None,
        meta_info: Optional[Dict[str, Any]] = None,
        cleanup_local: bool = False
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Durable Media Integration Method:
        1. Uploads file (or raw bytes) to Cloudinary CDN with retries.
        2. Asserts a permanent HTTPS delivery URL is returned.
        3. Persists MediaAsset record in MySQL with complete audit metadata.
        4. Cleans up temporary file if cleanup_local=True.
        5. Returns (MediaAsset, upload_data).
        """
        from app.models.media_asset import MediaAsset
        import hashlib

        if file_path:
            upload_data = cls.upload_local_file_path(
                file_path=file_path,
                organization_id=organization_id,
                folder_prefix=folder_prefix
            )
            file_bytes_for_hash = None
            if os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        file_bytes_for_hash = f.read()
                except Exception:
                    pass
            sha256 = hashlib.sha256(file_bytes_for_hash).hexdigest() if file_bytes_for_hash else None
            effective_name = file_name or os.path.basename(file_path)
        elif raw_bytes:
            effective_name = file_name or f"asset_{uuid.uuid4().hex[:10]}.bin"
            upload_data = cls.upload_bytes(
                data=raw_bytes,
                file_name=effective_name,
                organization_id=organization_id,
                folder_prefix=folder_prefix
            )
            sha256 = hashlib.sha256(raw_bytes).hexdigest()
        else:
            raise ValueError("Either file_path or raw_bytes must be provided to upload_and_register_asset.")

        combined_meta = dict(upload_data.get("meta_info") or {})
        if meta_info:
            combined_meta.update(meta_info)

        media_asset = MediaAsset(
            organization_id=organization_id,
            user_id=user_id,
            file_name=upload_data["file_name"],
            file_url=upload_data["file_url"],
            public_id=upload_data["public_id"],
            file_size_bytes=upload_data["file_size_bytes"],
            mime_type=upload_data["mime_type"],
            media_type=upload_data["media_type"],
            sha256_hash=sha256,
            meta_info=combined_meta,
            title=title or upload_data["file_name"],
            description=description,
            alt_text=alt_text,
            tags=tags or [],
            category_id=category_id,
            validation_status="APPROVED",
            quality_score=100.0
        )
        db.add(media_asset)
        db.flush()

        if cleanup_local and file_path:
            cls.safe_cleanup_local_file(file_path)

        return media_asset, upload_data

    @classmethod
    def delete_file(cls, public_id: str, media_type: MediaType) -> bool:
        """Deletes a file from Cloudinary given its public_id."""
        if not public_id:
            return False

        if cls.is_configured():
            try:
                cls._init_cloudinary()
                resource_type = "video" if media_type == MediaType.VIDEO else "image"
                res = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
                logger.info(f"Cloudinary deletion result for {public_id}: {res}")
                return res.get("result") in ["ok", "not found"]
            except Exception as e:
                logger.error(f"Failed to delete {public_id} from Cloudinary: {str(e)}")
                return False

        return True

    @classmethod
    def generate_upload_signature(cls, organization_id: int, folder_prefix: str = "socialai") -> Dict[str, Any]:
        """
        Generates a secure timestamped upload signature for direct browser uploads.
        Never returns the API Secret.
        """
        timestamp = int(time.time())
        folder = f"{folder_prefix}/org_{organization_id}"
        params_to_sign = {
            "folder": folder,
            "timestamp": timestamp
        }

        if cls.is_configured():
            signature = cloudinary.utils.api_sign_request(
                params_to_sign,
                settings.CLOUDINARY_API_SECRET
            )
            api_key = settings.CLOUDINARY_API_KEY
            cloud_name = settings.CLOUDINARY_CLOUD_NAME
        else:
            # Simulated signature for development mode
            import hashlib
            signature = hashlib.sha1(f"folder={folder}&timestamp={timestamp}".encode("utf-8")).hexdigest()
            api_key = "demo_key"
            cloud_name = settings.CLOUDINARY_CLOUD_NAME or "demo-socialai"

        return {
            "timestamp": timestamp,
            "signature": signature,
            "api_key": api_key,
            "cloud_name": cloud_name,
            "folder": folder
        }
