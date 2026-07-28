import os
import sys
import logging
import threading
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

# Make console output encoding-proof. On Windows the default console codec is the
# system locale (e.g. cp1255 on a Hebrew install), which can't encode characters
# like the "✓" we print in status lines — that raises UnicodeEncodeError and
# crashes the writing thread (e.g. embedding_utils' auto-index log). Force UTF-8
# with errors="replace" so a stray glyph degrades to "?" instead of throwing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load config from backend/.env as early as possible, so every module sees the
# same values regardless of import order. This is the SINGLE config file for
# both native and Docker runs. override=False (the default) means anything the
# environment already provides wins — e.g. Docker's compose `environment:` block
# (DB_HOST=postgres, the mounted SSH key) overrides the file's native defaults.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from utils.DButils import init_database

# Suppress non-actionable upstream warnings emitted by transformers during model
# load — they refer to architectural details that don't affect inference correctness.
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
# Reduce HF transfer warnings if HF_TOKEN isn't set
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --- Import Routers ---
from routes.user import router as user_router
from routes.dashboard import router as dashboard_router
from routes.rules import router as rules_router
from routes.cognitive import router as ce_router
from routes.models import router as models_router
from routes.classifiers import router as classifiers_router
from routes.ai_pipeline import router as ai_router
from routes.library import router as library_router
from routes.evaluation import router as evaluation_router
from routes.realtime import router as realtime_router
from routes.pipeline_runs import router as pipeline_runs_router

# Suppress noisy polling endpoints from uvicorn access logs
class _SuppressPollingFilter(logging.Filter):
    _SUPPRESSED = ("/training-status",)
    def filter(self, record):
        msg = record.getMessage()
        return not any(s in msg for s in self._SUPPRESSED)

logging.getLogger("uvicorn.access").addFilter(_SuppressPollingFilter())


# Drop the cosmetic Windows Proactor "Exception in callback
# _ProactorBasePipeTransport._call_connection_lost" / ConnectionResetError
# spam. Triggered when a client (browser tab, HMR ping, health poll) closes
# a socket while uvicorn is tearing down the same connection — the request
# itself completed fine, the warning is just asyncio's exception handler
# logging the racy shutdown(). See cpython#74953.
class _SuppressProactorConnLostFilter(logging.Filter):
    def filter(self, record):
        if record.exc_info and record.exc_info[0] is ConnectionResetError:
            return False
        if "_ProactorBasePipeTransport._call_connection_lost" in record.getMessage():
            return False
        return True

logging.getLogger("asyncio").addFilter(_SuppressProactorConnLostFilter())

# Boot-phase timing. Every synchronous step before HTTP serving starts
# gets a `[boot] ... in X.Xs` line so it's obvious which phase to attack
# when startup feels slow. The cheap warm-restart path (init_database
# fast-skips via schema-version sentinel) should land at <0.5s total.
import time as _boot_time
_boot_t0 = _boot_time.perf_counter()


def _boot_step(name: str, fn):
    """Run a startup step, log elapsed, swallow failures (so one
    component's hiccup doesn't take down the API). Mirrors
    `_warm_one` further down but without the readiness flag —
    these steps run BEFORE the FastAPI app object exists."""
    t0 = _boot_time.perf_counter()
    try:
        fn()
    except Exception as e:
        elapsed = _boot_time.perf_counter() - t0
        print(f"[boot] {name} FAILED in {elapsed:.2f}s: {e}")
        return
    elapsed = _boot_time.perf_counter() - t0
    print(f"[boot] {name} in {elapsed:.2f}s")


# Initialize local database. Users, ratings, bookmarks, and the HF token
# now live on the central server (central-server/) — see services/central_server.py.
# Synchronous: every route assumes the schema exists. Fast-skips when
# the live schema is already at the expected version (see DButils.SCHEMA_VERSION).
_boot_step("db init", init_database)

# ---------------------------------------------------------------------------
# There is NO app-wide auth dependency.
#
# This used to be a default-deny gate that rejected any request without a valid
# bearer token. GAVEL Studio has no login: it is a localhost, one-operator
# application, so every request is attributed to the single local user seeded by
# init_database(). Routes that need that id take `Depends(get_current_user)`,
# which resolves to a constant (see utils/auth.py).
#
# Safe ONLY because the backend binds to 127.0.0.1 and serves one human.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Process-wide startup/shutdown (modern replacement for @app.on_event)."""
    # --- startup: start the registry-sync subscriber — real-time public-library
    # sync driven by the central server's version_update notifications, with the
    # WS reconnect + safety poll as the backstop. Non-fatal if it can't start. ---
    subscriber = None
    try:
        from services.registry_sync.wiring import build_subscriber
        subscriber = build_subscriber()
        if subscriber is not None:
            await subscriber.start()
            print("[registry] subscriber started")
    except Exception as e:
        print(f"[registry] subscriber start failed (non-fatal): {e}")
    try:
        yield
    finally:
        # --- shutdown: stop the subscriber, then end warm realtime sessions. ---
        if subscriber is not None:
            try:
                await subscriber.stop()
            except Exception:
                pass
        _shutdown_end_realtime_sessions()


app = FastAPI(lifespan=lifespan)

_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
# A wildcard origin combined with credentials is forbidden by the CORS spec and a
# real foot-gun, so refuse to start that way rather than silently fall back.
if "*" in _allowed_origins:
    raise RuntimeError(
        "ALLOWED_ORIGINS must be an explicit allowlist, not '*'. This API can "
        "read and delete the operator's local models, rule sets and trained "
        "weights, so a wildcard origin would let any website drive it."
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    # There are no credentials of any kind (no login, no cookies, no bearer
    # token), so nothing needs to travel cross-site. Keeping this False also
    # removes the CORS spec's wildcard-origin foot-gun entirely.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _security_headers(request, call_next):
    """Defense-in-depth response headers. The backend is a JSON/zip API (it never
    serves HTML), so a full CSP belongs on whatever serves the SPA document — but
    these cheap headers still harden the API surface itself."""
    response = await call_next(request)
    # Never let a browser MIME-sniff an API response into something executable.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    # API responses are not meant to be framed.
    response.headers.setdefault("X-Frame-Options", "DENY")
    # Don't leak full URLs (which can carry ids) to third parties via Referer.
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response

# Per-component readiness state, exposed via /health for the frontend.
# Each flag flips to True once the corresponding background warmup step
# completes; clients can use this to gate feature-specific UI affordances.
_warmup_state = {
    "db": False,
    "embeddings": False,   # sentence_transformers + MiniLM weights
    "torch": False,        # required by training, evaluation, realtime
    "transformers": False, # required by all model load / inference paths
    "llm": False,          # required by AI rule / CE / test-set generation
}
import time as _time
_warmup_timings: dict = {}


def _warm_one(name: str, fn):
    """Run a single warmup step, record timing, flip the state flag. Failures are
    logged but never raise — a missing optional lib must not take down the API."""
    t0 = _time.perf_counter()
    try:
        fn()
        _warmup_state[name] = True
        elapsed = _time.perf_counter() - t0
        _warmup_timings[name] = round(elapsed, 2)
        print(f"[warmup] {name} ready in {elapsed:.2f}s")
    except Exception as e:
        _warmup_timings[name] = None
        print(f"[!] {name} warmup failed: {e}")


def _warmup_all():
    """Sequentially pre-load heavy modules required by feature endpoints.

    Order: library search (embeddings) → training/eval/realtime (torch +
    transformers) → AI generation (litellm). Sequential execution avoids peak
    RAM spikes and import-time global-state races between torch and
    transformers on Windows. If a request arrives before warmup completes for
    its component, the lazy import path remains intact — request latency in
    that case matches the pre-warmup baseline.
    """
    # 1. Embeddings — backs the library search endpoints.
    def _load_embeddings():
        from utils.embedding_utils import EmbeddingManager
        EmbeddingManager.get_instance()
    _warm_one("embeddings", _load_embeddings)

    # 2. torch — required by training, evaluation, and realtime classification.
    def _load_torch():
        import torch  # noqa: F401
    _warm_one("torch", _load_torch)

    # 3. transformers — AutoTokenizer / AutoModelForCausalLM are used by
    # classifier_engine.utils_train and evaluation.inference.
    def _load_transformers():
        from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
    _warm_one("transformers", _load_transformers)

    # 4. litellm — AI rule generation, CE chat, and test-set generation.
    def _load_litellm():
        import litellm  # noqa: F401
    _warm_one("llm", _load_litellm)

    total = sum(t for t in _warmup_timings.values() if t)
    print(f"[warmup] all components ready (total {total:.2f}s)")


# DB initialization completed synchronously above.
_warmup_state["db"] = True


# Early crash recovery — clear phantom "Running…"/"Generating…" markers ASAP.
#
# A crash mid-calibration/evaluation leaves a 'calibration_running'/
# 'evaluation_running' marker row; the full recovery that deletes it only runs
# AFTER the ~30s model warmup (it imports transformers and can't race the warmup
# thread). That left a long window where the UI showed a DEAD run as if it were
# still in progress. This early pass runs ONLY the DB-only strategies (no
# transformers import), so it can fire immediately after the DB is ready and the
# phantom marker disappears within ~a second of restart. It's idempotent with the
# full post-warmup recovery below.
def _run_early_recovery():
    try:
        from utils.crash_recovery import run_early_recovery
        run_early_recovery()
    except Exception as e:
        print(f"[recovery] early recovery failed: {e}")


threading.Thread(target=_run_early_recovery, daemon=True, name="early-recovery").start()


# huggingface_hub preload — moved off the synchronous boot path.
#
# The original concern was a Python import race: sentence_transformers
# (in the warmup thread) and services.hf_sync (in the library-sync
# thread) both touch huggingface_hub at first-import time; if they hit
# it simultaneously Python's import system can hand one of them a
# partially-initialised package ("cannot import name 'XetConnectionInfo'
# from ..._xet"). The original fix was to import on the main thread
# before any daemon started — correct but ~400ms-1s of synchronous boot.
#
# Replacement strategy: a dedicated bootstrap thread imports hf_hub
# FIRST (still single-threaded), then sets `_hf_hub_ready` so the
# warmup and library-sync threads can wait on it. End result:
#   * synchronous boot is ~0.4s faster (no longer waits for hf_hub)
#   * race-prevention preserved (hf_hub imports still serialized)
#   * Embeddings warmup + library-sync both block on _hf_hub_ready
#     before touching hf_hub themselves
_hf_hub_ready = threading.Event()


def _preload_huggingface_hub():
    t0 = _boot_time.perf_counter()
    try:
        import huggingface_hub  # noqa: F401
        import huggingface_hub.utils  # noqa: F401
        import huggingface_hub.hf_api  # noqa: F401
    except Exception as e:
        print(f"[!] huggingface_hub preload failed: {e}")
    finally:
        # Always set the event — even on failure — so downstream
        # threads don't deadlock waiting for a load that never happens.
        # The downstream import paths will hit the same error themselves
        # and surface it in context.
        _hf_hub_ready.set()
        elapsed = _boot_time.perf_counter() - t0
        print(f"[bg] huggingface_hub preload in {elapsed:.2f}s")


threading.Thread(target=_preload_huggingface_hub, daemon=True).start()


# Crash recovery used to run synchronously here. Moved to a daemon
# thread because it can be slow on a DB with many in-flight pipelines
# and isn't a hard prerequisite for HTTP serving — no route needs the
# recovered state in its handshake. Routes that DO need it
# (training, evaluation, real-time) gate on _warmup_state and the
# user only ever hits them after the workspace page has loaded, by
# which point recovery has long since completed.
def _run_recovery():
    try:
        from utils.crash_recovery import run_all_recovery
        run_all_recovery()
    except Exception as e:
        # Never raise from a daemon thread — failed recovery must
        # not take down the API; broken pipelines just stay broken
        # until the user manually re-runs them.
        print(f"[recovery] failed: {e}")


# Heavy module warmup runs on a daemon thread so HTTP serving begins
# immediately. The thread waits for _hf_hub_ready before touching
# sentence_transformers (which transitively imports huggingface_hub),
# so the no-race guarantee survives the move off the main thread.
def _warmup_after_hf_hub():
    # Bounded wait: 30s should be plenty; if the preload thread is
    # somehow stuck we fall through and try anyway rather than block
    # warmup forever.
    _hf_hub_ready.wait(timeout=30.0)
    _warmup_all()
    # Crash recovery runs AFTER warmup, not in parallel. Recovery
    # transitively imports `transformers` (via classifier_engine.trainer),
    # and racing with the warmup thread's `from transformers import
    # AutoTokenizer` produced a partial-init failure on startup.
    _run_recovery()


threading.Thread(target=_warmup_after_hf_hub, daemon=True).start()

# Report total synchronous-boot time. Everything below this print runs
# either lazily on first request or on daemon threads — uvicorn can
# already accept HTTP from this point onward.
_total_boot = _boot_time.perf_counter() - _boot_t0
print(f"[boot] synchronous phase done in {_total_boot:.2f}s — accepting requests")


# Pre-fetch the HF library on a background thread so the user's first login
# doesn't pay the full HF round-trip. Sync is idempotent + manifest_hash
# short-circuits, so the login-time call inside Login.jsx degrades to a near
# no-op once this lands. Spawned only after init_database() ran (the
# _warmup_state["db"] flip above) — the user explicitly asked us not to start
# this until the tables are confirmed ready, since sync_library writes into
# rules / cognitive_elements / categories.
def _bootstrap_library_sync():
    # Wait for huggingface_hub to finish loading before any import in
    # this thread reaches it — same race-prevention guarantee the old
    # synchronous main-thread preload provided. 30s bounded.
    _hf_hub_ready.wait(timeout=30.0)
    try:
        from services.hf_sync import sync_library, pull_all_aux_datasets
        result = sync_library()
        if result.errors:
            print(f"[library-sync] startup sync had errors: {result.errors}")
        else:
            print(
                f"[library-sync] startup sync done — "
                f"changed={result.changed} "
                f"ces_added={result.ces_added} "
                f"rules_added={result.rules_added} "
                f"categories_synced={result.categories_synced}"
            )

        # Auxiliary datasets (CE-level calibration, rule-level calibration,
        # rule-level evaluation) are pulled here in the SAME background
        # thread so login + every other route stays unblocked. Calibration
        # / Evaluation routes use lazy `ensure_*` helpers that fetch any
        # records this background pull hasn't reached yet, so a request
        # arriving mid-warmup just pays a few hundred ms of HF latency
        # for the records it actually needs.
        try:
            aux = pull_all_aux_datasets()
            print(f"[library-sync] aux datasets warmed: {aux}")
        except Exception as aux_err:
            # Same daemon-thread contract as the sync block above: never
            # let an aux pull failure take down the whole bootstrap.
            print(f"[library-sync] aux dataset warmup failed: {aux_err}")
    except Exception as e:
        # Never raise from a daemon thread — a failed startup sync is
        # acceptable degradation; the login-time sync acts as a fallback.
        print(f"[library-sync] startup sync crashed: {e}")


threading.Thread(target=_bootstrap_library_sync, daemon=True, name="library-sync-bootstrap").start()


# Boot-time orphan recovery, through the compute interface — each provider cleans
# up only its own leaks (the SLURM provider scancels orphaned realtime jobs +
# sweeps abandoned job dirs from a prior backend that died mid-run; local/remote
# are no-ops). Daemon thread, best-effort.
def _bootstrap_orphan_recovery():
    try:
        from services import compute
        for p in compute.all_providers():
            try:
                summary = p.recover_orphans()
                if summary:
                    print(f"[orphan-recovery] {p.name}: {summary}")
            except Exception as e:
                print(f"[orphan-recovery] {p.name} failed: {e}")
    except Exception as e:
        print(f"[orphan-recovery] failed (non-fatal): {e}")


threading.Thread(target=_bootstrap_orphan_recovery, daemon=True, name="orphan-recovery").start()


def _shutdown_end_realtime_sessions():
    """Graceful backend stop (Ctrl+C / SIGTERM) → end every warm realtime session
    NOW (free the GPU instead of waiting for its idle-timeout), across all
    providers. A hard kill can't run this; that's covered by recover_orphans() on
    the next boot."""
    try:
        from services import compute
        for p in compute.all_providers():
            try:
                n = p.end_all_realtime()
                if n:
                    print(f"[realtime] shutdown — ended {n} active session(s) on {p.name}")
            except Exception:
                pass
    except Exception as e:
        print(f"[realtime] shutdown cleanup failed: {e}")


@app.get("/")
def read_root():
    return RedirectResponse(url=os.getenv("FRONTEND_URL", "http://localhost:5173"))


@app.get("/health")
def health():
    """Liveness and readiness probe.

    `ready` reflects authentication-readiness (DB only); login is permitted as
    soon as the database is reachable. `components` reports per-module warmup
    status, allowing clients to gate feature-specific affordances on the
    relevant module being loaded. `timings` exposes observed cold-start
    durations per component, in seconds.
    """
    return {
        "status": "ok",
        "ready": _warmup_state["db"],
        "components": _warmup_state,
        "timings": _warmup_timings,
    }

# --- Register Routers ---
app.include_router(user_router, prefix="/user", tags=["User"])
app.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])
app.include_router(rules_router, prefix="/rules", tags=["Rules"])
app.include_router(ce_router, prefix="/cognitive", tags=["Cognitive Elements"])

app.include_router(models_router, prefix="/models", tags=["Models"])
app.include_router(classifiers_router, prefix="/classifiers", tags=["Classifiers"])

app.include_router(ai_router, prefix="/ai", tags=["AI Pipeline"])
app.include_router(library_router, prefix="/library", tags=["Library"])
app.include_router(evaluation_router, prefix="/evaluation", tags=["Evaluation"])
app.include_router(realtime_router, prefix="/realtime", tags=["Realtime"])
from routes.compute import router as compute_router
app.include_router(compute_router, prefix="/compute", tags=["Compute"])
# Ratings were a multi-user community feature backed by the central server.
# With no accounts there is nobody to attribute a rating to, so the router is
# gone (see routes/ratings.py deletion).
app.include_router(pipeline_runs_router, prefix="/pipeline-runs", tags=["Pipeline Runs"])
from routes.guardrail_folders import router as guardrail_folders_router
app.include_router(guardrail_folders_router, prefix="/guardrail-folders", tags=["Guardrail Folders"])
