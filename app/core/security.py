import hmac
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

_bearer = HTTPBearer(auto_error=False)
# ponytail: in-memory per-process limiter; move to Redis when running >1 worker/replica
_hits: dict[str, deque] = defaultdict(deque)


def authorize(key: str) -> str:
    """Check API key + per-key rate limit. Raises HTTPException(401/429)."""
    if settings.keys and not any(hmac.compare_digest(key.encode(), k.encode()) for k in settings.keys):
        raise HTTPException(401, "Invalid or missing API key")
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= settings.rate_limit_per_minute:
        raise HTTPException(429, "Rate limit exceeded")
    q.append(now)
    return key


def require_api_key(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    return authorize(creds.credentials if creds else "")
