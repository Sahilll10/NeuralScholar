import time
import asyncio
import logging
from collections import defaultdict, deque
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter by client IP.
    
    Algorithm: For each IP, maintain a deque of request timestamps.
    On each request:
    1. Remove all timestamps older than the window period from the front.
    2. If the remaining count >= limit, reject with 429.
    3. Otherwise, append current timestamp and allow the request.
    
    This is a sliding window log algorithm — more accurate than
    fixed window (which can allow 2x burst at window boundaries)
    but uses O(requests_per_window) memory per IP.
    
    Exempt paths: /health, /docs, /redoc, /openapi.json
    """

    EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

    def __init__(self, app, requests_per_period: int = 100, period_seconds: int = 60):
        super().__init__(app)
        self.requests_per_period = requests_per_period
        self.period_seconds = period_seconds
        self.request_log: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - self.period_seconds

        async with self._lock:
            log = self.request_log[client_ip]
            while log and log[0] < cutoff:
                log.popleft()

            if len(log) >= self.requests_per_period:
                oldest = log[0]
                retry_after = int(self.period_seconds - (now - oldest)) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)}
                )

            log.append(now)

        return await call_next(request)