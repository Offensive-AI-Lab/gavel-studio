"""Lightweight in-process per-IP rate limiter (no external dependency).

The central server runs as a single process, so an in-memory sliding-window
counter is enough to blunt floods — specifically a storm of HF publishes that
would tie up the threadpool. Per client IP, per scope. Thread-safe and
memory-bounded.
"""
import os
import threading
import time
from collections import deque

from fastapi import HTTPException, Request, status

# Number of trusted reverse proxies in front of the app. On Render there is
# exactly ONE, and it APPENDS the real client IP to X-Forwarded-For — so the
# trustworthy IP is the entry `_TRUSTED_HOPS` from the right. Default 1.
#
# Keep this at 1 unless you genuinely add another trusted proxy (e.g. a CDN).
# Setting it too high reads further left into the header, which is exactly the
# part an attacker can forge — defeating the limiter. Set 0 to ignore
# X-Forwarded-For entirely (e.g. when there is NO proxy, like local dev).
_TRUSTED_HOPS = max(0, int(os.getenv("TRUSTED_PROXY_HOPS", "1")))


class _SlidingWindow:
    def __init__(self, max_keys: int = 20_000):
        self._hits: dict = {}          # key -> deque[timestamps]
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def _sweep(self, cutoff: float) -> None:
        # Drop keys whose most recent hit is older than the window. Called only
        # when the dict grows large, so it's amortized cheap.
        dead = [k for k, dq in self._hits.items() if not dq or dq[-1] < cutoff]
        for k in dead:
            self._hits.pop(k, None)

    def hit(self, key: str, limit: int, window: float):
        """Record a hit; return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        cutoff = now - window
        with self._lock:
            if len(self._hits) > self._max_keys:
                self._sweep(cutoff)
            dq = self._hits.get(key)
            if dq is None:
                dq = deque()
                self._hits[key] = dq
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= limit:
                return False, max(1, int(window - (now - dq[0])) + 1)
            dq.append(now)
            return True, 0

    def count(self, key: str, window: float) -> int:
        """Number of recorded hits for `key` within the window (read-only)."""
        now = time.monotonic()
        cutoff = now - window
        with self._lock:
            dq = self._hits.get(key)
            if not dq:
                return 0
            while dq and dq[0] < cutoff:
                dq.popleft()
            return len(dq)

    def add(self, key: str) -> None:
        """Record a hit without any limit check (used for failure counters)."""
        with self._lock:
            self._hits.setdefault(key, deque()).append(time.monotonic())


_limiter = _SlidingWindow()


def _client_ip(request: Request) -> str:
    """The real client IP, resilient to a forged X-Forwarded-For.

    Our trusted proxy (Render) APPENDS the connecting IP, so the trustworthy
    value is `_TRUSTED_HOPS` entries from the RIGHT. Anything further left can be
    attacker-supplied. With _TRUSTED_HOPS=0, or when the header is shorter than
    expected, we ignore X-Forwarded-For and use the direct peer address.
    """
    if _TRUSTED_HOPS > 0:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            parts = [p.strip() for p in xff.split(",") if p.strip()]
            idx = len(parts) - _TRUSTED_HOPS
            if 0 <= idx < len(parts):
                return parts[idx]
    return request.client.host if request.client else "unknown"


def rate_limit(scope: str, limit: int, window_seconds: float):
    """FastAPI dependency factory: at most `limit` requests per
    `window_seconds` per client IP for this `scope`. Raises 429 (with a
    Retry-After header) when exceeded."""
    def _dep(request: Request):
        ip = _client_ip(request)
        allowed, retry = _limiter.hit(f"{scope}:{ip}", limit, window_seconds)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests — please slow down.",
                headers={"Retry-After": str(retry)},
            )
    return _dep

