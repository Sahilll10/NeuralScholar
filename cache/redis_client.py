import redis
import json
import hashlib
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-backed query response cache.
    
    Cache key design:
        neuralscholar:query:{sha256(canonical_query_json)}
        
    The canonical_query_json includes:
        - query (lowercased, stripped)
        - top_k
        - retrieval_mode
        
    This ensures cache hits only when all parameters are identical.
    Changing top_k or retrieval_mode correctly invalidates the cache.
    
    TTL: 3600 seconds (1 hour) by default. RAG answers are effectively
    immutable for a given corpus and don't need frequent invalidation.
    Invalidate manually after re-ingestion with cache.invalidate().
    
    Graceful degradation: if Redis is unavailable at startup, all cache
    operations become no-ops and the system continues without caching.
    This prevents a Redis failure from taking down the API.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600
    ):
        self.ttl = ttl
        self.available = False

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password or None,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                retry_on_timeout=False
            )
            self.client.ping()
            self.available = True
            logger.info(f"Redis cache connected: {host}:{port}/db{db}")
        except Exception as e:
            logger.warning(f"Redis unavailable — caching disabled. Reason: {e}")

    @staticmethod
    def _make_key(query: str, **kwargs) -> str:
        canonical = json.dumps(
            {"query": query.lower().strip(), **kwargs},
            sort_keys=True, ensure_ascii=True
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"neuralscholar:query:{digest}"

    def get(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        try:
            key = self._make_key(query, **kwargs)
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache HIT: {query[:60]}")
                return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache GET error: {e}")
            return None

    def set(self, query: str, response: Dict[str, Any], **kwargs):
        if not self.available:
            return
        try:
            key = self._make_key(query, **kwargs)
            self.client.setex(key, self.ttl, json.dumps(response, ensure_ascii=False))
            logger.debug(f"Cache SET: {query[:60]}")
        except Exception as e:
            logger.warning(f"Cache SET error: {e}")

    def invalidate(self, pattern: str = "neuralscholar:query:*"):
        if not self.available:
            return
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries")
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")