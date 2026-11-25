import os
import redis
from typing import Optional
from app.core.config import settings
from app.core.logging import logger

class RedisManager:
    """
    Centralized Redis connection manager with health verification,
    circuit-breaking, and resilient fallback handling.
    """
    _client: Optional[redis.Redis] = None
    _is_connected: bool = False
    _last_check: float = 0.0

    @classmethod
    def get_client(cls, timeout: float = 1.0) -> Optional[redis.Redis]:
        """
        Returns an active Redis client if Redis is available, or None if offline/unreachable.
        Uses connection caching and short socket timeouts to prevent blocking FastAPI or workers.
        """
        redis_url = getattr(settings, "REDIS_URL", None) or os.environ.get("REDIS_URL") or "redis://localhost:6379/0"
        
        # Test existing connection
        if cls._client is not None:
            try:
                cls._client.ping()
                cls._is_connected = True
                return cls._client
            except Exception:
                cls._is_connected = False
                cls._client = None

        # Attempt fresh connection
        try:
            client = redis.from_url(
                redis_url,
                socket_connect_timeout=timeout,
                socket_timeout=timeout,
                decode_responses=False
            )
            client.ping()
            cls._client = client
            cls._is_connected = True
            logger.info(f"[RedisManager] Connected successfully to Redis at {redis_url}")
            return cls._client
        except Exception as e:
            cls._is_connected = False
            cls._client = None
            logger.debug(f"[RedisManager] Redis unavailable ({e}). Falling back to in-process memory queue.")
            return None

    @classmethod
    def is_available(cls) -> bool:
        """Quick boolean check if Redis is active."""
        return cls.get_client() is not None
