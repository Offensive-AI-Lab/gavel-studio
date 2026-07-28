"""HTTP client for the GAVEL central server.

Scope after login removal: the central server exists for exactly ONE reason —
it is the HuggingFace **write** proxy. The HF write token lives only there, so
publishing a rule / CE / rule set sends the file operations to it and it makes
the commit.

Everything else this module used to do (auth, user directory, ratings,
bookmarks) is gone: those were multi-user features that needed accounts. There
are no tokens to forward anymore — the proxy is unauthenticated, which is safe
because it is reachable only on the operator's own machine/network.

Reads never come through here. Pulling the public library from HuggingFace is
anonymous (see `services/hf_sync.py`), so Browse / Sync / Fork all work with no
central server running at all.
"""
import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

CENTRAL_SERVER_URL = os.getenv("CENTRAL_SERVER_URL", "").rstrip("/")
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# Shared keep-alive client: one TCP/TLS handshake for the process instead of one
# per request. See git history for the full tuning rationale.
try:
    import h2 as _h2  # noqa: F401  — probe only
    _HTTP2 = True
except ImportError:
    _HTTP2 = False

_client = httpx.Client(
    timeout=_TIMEOUT,
    http2=_HTTP2,
    follow_redirects=True,
    limits=httpx.Limits(
        max_keepalive_connections=10,
        max_connections=20,
        keepalive_expiry=60.0,
    ),
)

logger = logging.getLogger("central-rpc")
if not logger.handlers:
    logger.setLevel(logging.INFO)


class CentralServerError(Exception):
    """Raised when the central server is unreachable or returns an error."""

    def __init__(self, message: str, status_code: int = 500, payload: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


def is_enabled() -> bool:
    """True when a central server is configured. When False, publishing is
    unavailable — but every local feature, and library READ-sync, still work."""
    return bool(CENTRAL_SERVER_URL)


def _request(method: str, path: str, *, json: Any = None, params: Any = None) -> Any:
    if not CENTRAL_SERVER_URL:
        raise CentralServerError(
            "CENTRAL_SERVER_URL is not configured — publishing is unavailable. "
            "Everything else (including library sync) works without it.",
            status_code=503,
        )
    url = f"{CENTRAL_SERVER_URL}{path}"
    t0 = time.perf_counter()
    try:
        resp = _client.request(method, url, headers={"Content-Type": "application/json"},
                               json=json, params=params)
    except httpx.RequestError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.warning("FAIL %s %s after %.0fms: %s", method, path, elapsed, e)
        raise CentralServerError(f"Central server unreachable: {e}", status_code=502)

    elapsed = (time.perf_counter() - t0) * 1000
    tag = " SLOW" if elapsed > 200 else ""
    logger.info("%s %s -> %s in %.0fms%s", method, path, resp.status_code, elapsed, tag)

    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except Exception:
            payload = {"detail": resp.text}
        detail = payload.get("detail") if isinstance(payload, dict) else str(payload)
        raise CentralServerError(detail or f"HTTP {resp.status_code}",
                                 status_code=resp.status_code, payload=payload)

    if resp.headers.get("content-type", "").startswith("application/json"):
        return resp.json()
    return resp.text


def close() -> None:
    """Tear down the shared client. Safe to call repeatedly."""
    try:
        _client.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HuggingFace write proxy — the only remaining responsibility
# ---------------------------------------------------------------------------

def hf_head_sha() -> Optional[str]:
    """Current HEAD commit SHA of the registry repo, used as `parent_commit`
    for race detection on publish."""
    resp = _request("GET", "/hf/head-sha")
    return resp.get("sha") if isinstance(resp, dict) else None


def hf_commit(*, operations: List[Dict[str, bytes]], commit_message: str,
              parent_commit: Optional[str] = None) -> dict:
    """Commit a batch of files to HF via the central server.

    `operations` is a list of {"path": str, "content": bytes}; bytes are
    base64-encoded for JSON transport. The commit is all-or-nothing on the
    server side.
    """
    encoded = [
        {"path": op["path"], "content_b64": base64.b64encode(op["content"]).decode("ascii")}
        for op in operations
    ]
    return _request("POST", "/hf/commit",
                    json={"operations": encoded, "commit_message": commit_message,
                          "parent_commit": parent_commit})
