"""Worker configuration (all from env) + the engine-version handshake.

The version is a hash of the bundled classifier_engine files that determine the
logits (RNN, dialogue tokenizer, inference core). The backend computes the SAME
hash of ITS classifier_engine and refuses a worker whose hash differs — so a
worker running stale code can never silently produce drifted results.
"""
import hashlib
import os
import sys
from pathlib import Path

# Bearer token every request must present. REQUIRED: while unset, every endpoint
# except /health returns 503 (the worker refuses to run wide open). Put the same
# value in the backend's GPU_WORKER_TOKEN.
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "").strip()

# Where the GAVEL engine code lives (compute_jobs/ + classifier_engine/ + utils/).
# The Docker image bakes it at /opt/gavel/code; for `pip` runs it's resolved
# relative to this package's install (see _default_code_dir).
CODE_DIR = os.getenv("GAVEL_CODE_DIR", "").strip()

# Scratch root for per-job/session directories (payloads, weights, results). Each
# batch job's dir is deleted right after the backend fetches its result/model; the
# sweeper purges anything left behind (see GAVEL_JOB_RETENTION).
JOBS_DIR = os.getenv("GAVEL_JOBS_DIR", os.path.join(os.path.expanduser("~"), "gavel_worker_jobs"))

# Safety-net retention for a FINISHED batch job's scratch dir. The backend deletes
# each job right after it pulls the result, so this only matters if a client died
# mid-run or the worker restarted and lost its in-memory registry — the sweeper
# then purges any job dir older than this so a limited-disk box can't fill up.
JOB_RETENTION_S = int(os.getenv("GAVEL_JOB_RETENTION", "1800"))

# Shared Hugging Face model cache. A worker is a persistent box, so the model is
# downloaded ONCE and reused across every job/session — never per-job (that would
# re-download multi-GB weights for each train/infer/realtime run). Set this to a
# mounted volume in Docker so the cache survives container restarts. When unset we
# leave HF_HOME untouched, so the job inherits the user's standard
# ~/.cache/huggingface and reuses any models already downloaded there.
HF_CACHE_DIR = os.getenv("GAVEL_HF_CACHE", "").strip()

HOST = os.getenv("WORKER_HOST", "0.0.0.0")
PORT = int(os.getenv("WORKER_PORT", "8000"))
DEVICE = os.getenv("WORKER_DEVICE", "auto")  # auto | cuda | cpu

# Realtime warm-session idle timeout handled inside realtime_job.py; the worker
# also frees the session if the client stops pinging for this long.
SESSION_IDLE_TIMEOUT_S = int(os.getenv("GAVEL_SESSION_IDLE_TIMEOUT", "900"))

# How long /session/{id}/analyze waits for the warm job to answer one request.
ANALYZE_TIMEOUT_S = int(os.getenv("GAVEL_ANALYZE_TIMEOUT", "180"))


def _default_code_dir() -> str:
    """`pip install`'d worker: the engine code is expected next to the package as
    a sibling 'gavel_code' dir, or at ~/gavel_code."""
    here = Path(__file__).resolve().parent.parent
    for cand in (here / "gavel_code", Path(os.path.expanduser("~/gavel_code"))):
        if (cand / "compute_jobs" / "infer_job.py").is_file():
            return str(cand)
    return str(here / "gavel_code")


def code_dir() -> str:
    return CODE_DIR or _default_code_dir()


def job_script(name: str) -> str:
    """Absolute path to a compute job script (infer_job.py / train_job.py /
    realtime_job.py) — shared by all GPU backends, lives in compute_jobs/."""
    return os.path.join(code_dir(), "compute_jobs", name)


_ENGINE_FILES = [
    "classifier_engine/inference_core.py",
    "classifier_engine/RNN.py",
    "classifier_engine/dialogue_dataset.py",
    "classifier_engine/utils_train.py",
]


def engine_version() -> str:
    """Stable hash of the logit-determining engine files. Empty string if the
    code can't be found (worker misconfigured)."""
    h = hashlib.sha256()
    root = Path(code_dir())
    found = False
    for rel in _ENGINE_FILES:
        p = root / rel
        try:
            # Normalize CRLF -> LF before hashing so the worker agrees with the
            # backend regardless of platform/line endings (mirrors
            # backend_engine_version in remote_worker.py).
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
            found = True
        except OSError:
            h.update(b"\x00MISSING\x00")
    return h.hexdigest()[:16] if found else ""


def worker_env() -> dict:
    """Environment for the job subprocesses: make the engine importable and route
    the HF cache to the SHARED worker cache (so models are downloaded once and
    reused across jobs/sessions). If GAVEL_HF_CACHE is unset, HF_HOME is left
    alone and the job uses the standard ~/.cache/huggingface."""
    env = dict(os.environ)
    cd = code_dir()
    env["PYTHONPATH"] = cd + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if HF_CACHE_DIR:
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
        env["HF_HOME"] = HF_CACHE_DIR
        env["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
        env["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
    return env
