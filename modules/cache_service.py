# modules/cache_service.py
"""
Redis Cache Service.
Caches LLM responses to speed up repeated/common queries.
Uses hash of (query + context_key) as cache key.
"""

import redis
import json
import hashlib
from typing import Optional
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, REDIS_CACHE_TTL


class CacheService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CacheService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print(f"[Cache] Connecting to Redis: {REDIS_HOST}:{REDIS_PORT}...")
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            # Test connection
            self.redis.ping()
            self.ttl = REDIS_CACHE_TTL
            self.prefix = "aichat:"

            info = self.redis.info("memory")
            used_mb = info.get("used_memory_human", "?")
            print(f"[Cache] ✅ Redis connected (memory: {used_mb})")
            self._initialized = True

        except redis.ConnectionError as e:
            print(f"[Cache] ❌ Redis connection failed: {e}")
            print("[Cache] ⚠️ Running without cache (no speedup)")
            self.redis = None
            self._initialized = True

        except Exception as e:
            print(f"[Cache] ❌ Redis error: {e}")
            self.redis = None
            self._initialized = True

    @property
    def is_connected(self) -> bool:
        return self.redis is not None

    def _make_key(self, query: str, context_key: str = "") -> str:
        """Generate a deterministic cache key from query + context."""
        raw = f"{query.strip().lower()}|{context_key}"
        hash_hex = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{self.prefix}resp:{hash_hex}"

    def get_cached_response(self, query: str, context_key: str = "") -> Optional[str]:
        """
        Look up a cached response.
        Returns the cached string or None if miss.
        """
        if not self.is_connected:
            return None

        key = self._make_key(query, context_key)
        try:
            cached = self.redis.get(key)
            if cached:
                print(f"[Cache] ✅ HIT: '{query[:40]}...'")
                # Bump TTL on hit (LRU-like behavior)
                self.redis.expire(key, self.ttl)
                return cached
            print(f"[Cache] MISS: '{query[:40]}...'")
            return None

        except Exception as e:
            print(f"[Cache] ⚠️ Get error: {e}")
            return None

    def set_cached_response(self, query: str, response: str, context_key: str = "", ttl: int = None) -> bool:
        """
        Cache a response.
        Only caches responses that are non-trivial (>20 chars).
        """
        if not self.is_connected:
            return False

        # Don't cache very short or error responses
        if not response or len(response) < 20:
            return False

        key = self._make_key(query, context_key)
        expire = ttl or self.ttl

        try:
            self.redis.setex(key, expire, response)
            return True

        except Exception as e:
            print(f"[Cache] ⚠️ Set error: {e}")
            return False

    def invalidate(self, query: str, context_key: str = "") -> bool:
        """Remove a specific cached response."""
        if not self.is_connected:
            return False

        key = self._make_key(query, context_key)
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            print(f"[Cache] ⚠️ Invalidate error: {e}")
            return False

    def clear_all(self) -> int:
        """Clear all AIchat cache entries."""
        if not self.is_connected:
            return 0

        try:
            pattern = f"{self.prefix}*"
            keys = list(self.redis.scan_iter(match=pattern, count=1000))
            if keys:
                count = self.redis.delete(*keys)
                print(f"[Cache] ✅ Cleared {count} entries")
                return count
            return 0

        except Exception as e:
            print(f"[Cache] ⚠️ Clear error: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.is_connected:
            return {"status": "disconnected"}

        try:
            info = self.redis.info("stats")
            memory = self.redis.info("memory")

            # Count our keys
            pattern = f"{self.prefix}*"
            our_keys = sum(1 for _ in self.redis.scan_iter(match=pattern, count=1000))

            return {
                "status": "connected",
                "cached_responses": our_keys,
                "total_hits": info.get("keyspace_hits", 0),
                "total_misses": info.get("keyspace_misses", 0),
                "memory_used": memory.get("used_memory_human", "?"),
                "ttl_seconds": self.ttl,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
cache_service = CacheService()
