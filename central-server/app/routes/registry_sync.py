"""Control-plane REST + WebSocket endpoints.

  POST /api/v1/webhook   HF's doorbell → verify the secret over the RAW body,
                         then just trigger the watcher (which reconciles to HEAD).
  GET  /api/v1/versions  the persisted { commit, global_signature, namespaces }
                         map — lightweight, ETag-cacheable; clients poll/reconnect.
  WS   /api/v1/ws        authenticated socket; receives {"event":"version_update"}.
"""
import logging

from fastapi import (
    APIRouter, HTTPException, Request, Response,
    WebSocket, WebSocketDisconnect,
)

from ..services import control_plane as cp
from ..services.source_provider import WebhookRejected

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["registry-sync"])


@router.post("/webhook")
async def webhook(request: Request):
    """HF webhook receiver. Verifies authenticity over the RAW bytes + headers,
    then uses the event purely as a trigger — the watcher does the authoritative
    HEAD reconcile. Returns fast so HF's delivery isn't held open."""
    raw = await request.body()
    try:
        cp.PROVIDER.verify_and_normalize_webhook(request.headers, raw)
    except WebhookRejected:
        raise HTTPException(status_code=401, detail="webhook verification failed")
    cp.WATCHER.trigger()
    return {"ok": True}


@router.get("/versions")
def versions(request: Request, response: Response):
    """The current persisted version map. ETag = global_signature, so a client
    whose `If-None-Match` matches gets a cheap 304."""
    state = cp.WATCHER.current_versions
    etag = 'W/"%s"' % (state.get("global_signature") or "none")
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "no-cache"
    return state


@router.websocket("/ws")
async def ws(websocket: WebSocket):
    """PUBLIC notification socket.

    The {"event":"version_update"} signal is non-sensitive: the registry and
    GET /versions are already public, so being told "go re-check /versions"
    reveals nothing. (Nothing on this server is authenticated anyway — see the
    module docstring in app/main.py.) The only gate is a connection cap, to
    bound resource use on an open endpoint."""
    if not await cp.WS_MANAGER.connect(websocket):   # capacity check inside
        return
    try:
        while True:
            await websocket.receive_text()  # keepalive pings; content ignored
    except WebSocketDisconnect:
        pass
    finally:
        cp.WS_MANAGER.disconnect(websocket)
