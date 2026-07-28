"""Central server entrypoint.

Scope: this server exists for ONE reason — it is the HuggingFace **write**
proxy. The HF write token lives only here, so a local backend that wants to
publish a rule / CE / rule set sends the file operations here and this server
performs the commit. It also runs the registry control plane (the watcher that
notices a new registry commit and pushes `version_update` to connected
backends).

There is no login anywhere in GAVEL Studio, so this server has no users,
no auth, no ratings and no bookmarks — those were multi-user features. Every
endpoint is unauthenticated, which is safe because the deployment model is
"one operator, own machine/network". Do NOT expose this on the open internet
with a real HF_TOKEN configured.

Run locally:
    cd central-server
    pip install -r requirements.txt
    cp .env.example .env  # fill in DATABASE_URL, HF_TOKEN
    uvicorn app.main:app --reload --port 8001

In production (Render):
    Render runs `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
"""
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import hf, registry_sync
from .services import control_plane
from .utils.schema import init_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage process-wide startup/shutdown. Replaces the deprecated
    @app.on_event handlers — everything before `yield` runs at startup, the
    `finally` block at shutdown."""
    # --- startup ---
    try:
        init_schema()
    except Exception as e:
        print(f"Schema init error: {e}")
    # Bind the running loop so the watcher thread can push to WS clients, then
    # start the SourceWatcher (Subject). No-op when ENABLE_CONTROL_PLANE=0.
    control_plane.start(asyncio.get_running_loop())
    try:
        yield
    finally:
        # --- shutdown ---
        control_plane.stop()


app = FastAPI(title="GAVEL Central Server", version="0.1.0", lifespan=lifespan)

# --- Hard request-size cap -------------------------------------------------
# A single huge body (e.g. a giant /hf/commit) could balloon memory and stall
# the one worker for everyone. Reject anything over the limit up front, before
# it's read into memory. base64 inflates ~33%, so 16 MB comfortably covers a
# real publish while blocking abuse. Configurable via MAX_REQUEST_MB.
_MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_MB", "16")) * 1024 * 1024


@app.middleware("http")
async def _limit_request_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > _MAX_REQUEST_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request body too large."})
        except ValueError:
            pass
    return await call_next(request)


# --- Security headers ------------------------------------------------------
# This server is internet-facing over real HTTPS (Render/:443), so unlike the
# local http backend, HSTS is appropriate here. The rest are cheap hardening on
# a JSON API.
@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    # Tell browsers to only ever reach this host over HTTPS for the next 2 years.
    response.headers.setdefault(
        "Strict-Transport-Security", "max-age=63072000; includeSubDomains")
    return response


# --- CORS ------------------------------------------------------------------
# Default is NOT "*". We always allow localhost on any port (so local backends
# and dev frontends keep working) but block the rest of the world. Add real
# production web origins via ALLOWED_ORIGINS (comma-separated). Set
# ALLOWED_ORIGINS="*" only to deliberately open it to everyone.
_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
_cors = dict(allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
if _origins_env == "*":
    _cors["allow_origins"] = ["*"]
else:
    _cors["allow_origins"] = [o.strip() for o in _origins_env.split(",") if o.strip()]
    _cors["allow_origin_regex"] = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
app.add_middleware(CORSMiddleware, **_cors)

app.include_router(hf.router)
app.include_router(registry_sync.router)


@app.get("/")
def root():
    return {"service": "gavel-central-server", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}
