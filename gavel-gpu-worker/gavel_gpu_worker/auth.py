"""Bearer-token auth for every worker endpoint except /health.

The token rides inside TLS (the worker is meant to sit behind an HTTPS
terminator — RunPod proxy, a tunnel, ALB, Caddy…), so it's never exposed on the
wire. Compared in constant time to avoid a timing oracle.
"""
import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import config

_bearer = HTTPBearer(auto_error=False)


def require_token(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> None:
    expected = config.WORKER_TOKEN
    if not expected:
        # No token configured — refuse rather than run wide open.
        raise HTTPException(status_code=503, detail="Worker has no WORKER_TOKEN configured.")
    if creds is None or not hmac.compare_digest(creds.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing worker token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
