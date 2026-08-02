"""RemoteWorkerProvider — talks HTTPS to a gavel-gpu-worker the user spun up on
RunPod / AWS / Colab / any box. The worker runs the SAME engine code, so logits
are identical; this side is just an HTTP client + the engine-version handshake.

Config (env):
    GPU_WORKER_URL    https://...      (plain http rejected except localhost dev)
    GPU_WORKER_TOKEN  <bearer>
"""
import gzip
import hashlib
import io
import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

from ..base import (
    Accelerator, Capabilities, ComputeError, ComputeProvider, InferenceSpec,
    JobState, RealtimeSession, RealtimeSpec, TrainingJob, TrainingSpec, TrainingStatus,
)

logger = logging.getLogger("compute.remote_worker")

_URL = os.getenv("GPU_WORKER_URL", "").strip().rstrip("/")
_TOKEN = os.getenv("GPU_WORKER_TOKEN", "").strip()
_POLL = float(os.getenv("GPU_WORKER_POLL", "5"))
_HTTP_TIMEOUT = float(os.getenv("GPU_WORKER_HTTP_TIMEOUT", "120"))
# Connect must fail FAST so an unreachable worker (laptop asleep, IP changed, no
# internet) fails over in seconds instead of waiting out the full read timeout.
_CONNECT_TIMEOUT = float(os.getenv("GPU_WORKER_CONNECT_TIMEOUT", "10"))
# Hard wall-clock cap on a single inference poll loop: if the worker keeps saying
# "running" past this without finishing (wedged GPU kernel, deadlock), give up and
# let the dispatcher retry locally rather than block the request thread forever.
_INFER_DEADLINE = float(os.getenv("GPU_WORKER_INFER_DEADLINE", "3600"))

# Files whose contents determine the logits — hashed identically here and in the
# worker (gavel_gpu_worker/config.py). A mismatch means the worker runs stale
# code, so we refuse it rather than serve drifted results.
_ENGINE_FILES = [
    "inference_core.py", "RNN.py", "dialogue_dataset.py", "utils_train.py",
]


def backend_engine_version() -> str:
    root = Path(__file__).resolve().parents[3] / "classifier_engine"
    h = hashlib.sha256()
    ok = False
    for name in _ENGINE_FILES:
        try:
            # Normalize CRLF -> LF before hashing so a Windows backend (autocrlf)
            # and a Linux worker agree on identical code (line endings don't
            # change the logits, so they must not change the version).
            h.update((root / name).read_bytes().replace(b"\r\n", b"\n"))
            ok = True
        except OSError:
            h.update(b"\x00MISSING\x00")
    return h.hexdigest()[:16] if ok else ""


def _is_secure(url: str) -> bool:
    if url.startswith("https://"):
        return True
    # Allow plain http only for localhost dev.
    return url.startswith("http://localhost") or url.startswith("http://127.0.0.1")


class RemoteWorkerProvider(ComputeProvider):
    name = "remote_worker"

    def __init__(self):
        self.url = _URL
        self.token = _TOKEN
        self._caps_cache = None
        self._configured = bool(self.url)
        self._secure = _is_secure(self.url) if self.url else False
        if self.url and not self._secure:
            logger.warning("GPU_WORKER_URL must be https:// (got %r) — the remote "
                           "worker is disabled. Put it behind TLS.", self.url)
        import httpx  # local import so the module loads even if httpx is missing
        # Short connect timeout (fast failover) + generous read timeout (a GPU
        # forward pass can be slow). Per-call timeout= kwargs still override this.
        _timeout = httpx.Timeout(_HTTP_TIMEOUT, connect=_CONNECT_TIMEOUT)
        self._client = httpx.Client(timeout=_timeout) if (self._configured and self._secure) else None

    # -- http helpers ----------------------------------------------------
    def _h(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _get(self, path: str, **kw):
        return self._client.get(self.url + path, headers=self._h(), **kw)

    def _post(self, path: str, **kw):
        return self._client.post(self.url + path, headers=self._h(), **kw)

    def _cleanup_job(self, kind: str, job_id: str) -> None:
        """Best-effort: tell the worker to delete a finished job's scratch dir so the
        per-run data (uploaded weights, inlined datasets, results) doesn't accumulate
        on the GPU box. Failures are ignored — the worker's sweeper purges leftovers."""
        try:
            self._client.delete(self.url + f"/{kind}/{job_id}", headers=self._h(), timeout=15)
        except Exception:
            pass

    def _hf_token_for(self, model_ref: Optional[str]) -> Optional[str]:
        """The worker has no DB, so resolve a gated-model token here and ship it."""
        try:
            from utils.model_hf_token import resolve_model_hf_token
            return resolve_model_hf_token(model_ref) if model_ref else None
        except Exception:
            return None

    # -- discovery -------------------------------------------------------
    def capabilities(self) -> Capabilities:
        caps = self._caps_cache or {}
        acc = caps.get("accelerator", "remote")
        try:
            accel = Accelerator(acc)
        except ValueError:
            accel = Accelerator.REMOTE
        supports = caps.get("supports", ["training", "inference", "realtime"])
        return Capabilities(
            name=self.name, accelerator=accel, is_local=False,
            supports_training="training" in supports,
            supports_inference="inference" in supports,
            supports_realtime="realtime" in supports,
            max_realtime_sessions=int(caps.get("max_realtime_sessions", 1)),
            detail=caps.get("detail", "Remote GPU worker"),
            code_version=caps.get("engine_version"),
        )

    def is_available(self) -> bool:
        if not (self._configured and self._secure and self._client):
            return False
        try:
            r = self._get("/capabilities", timeout=10)
            if r.status_code != 200:
                return False
            caps = r.json()
        except Exception:
            return False
        # Engine-version handshake: refuse a worker on different engine code.
        worker_ver = caps.get("engine_version") or ""
        mine = backend_engine_version()
        if worker_ver and mine and worker_ver != mine:
            logger.warning("Remote worker engine version %s != backend %s — refusing "
                           "it (logits would drift). Re-stage + rebuild the worker.",
                           worker_ver, mine)
            return False
        self._caps_cache = caps
        return True

    # -- inference -------------------------------------------------------
    def run_inference(self, spec: InferenceSpec, on_phase: Optional[Callable] = None,
                      on_submit: Optional[Callable] = None) -> List[dict]:
        import numpy as np

        def _phase(t):
            if on_phase:
                try:
                    on_phase(t)
                except Exception:
                    pass

        payload = {
            "classifier_id": spec.classifier_id,
            "model_hf_path": spec.model_hf_path,
            "classifier_meta": spec.classifier_meta,
            "dialogues": spec.dialogues,
            "max_length": spec.max_length,
            "window_stride": spec.window_stride,
            "hf_token": self._hf_token_for(spec.model_hf_path),
        }
        try:
            _phase("Uploading to the GPU worker…")
            # Gzip the spec: it inlines the full eval corpus (the neutral split is
            # several MB of JSON), so the raw multipart can exceed hosted-proxy
            # body limits (RunPod's ~40 MB) and get truncated -> a 400/disconnect.
            # Compressing it ~8x keeps the request near the (proven-good) rnn-only
            # size. The worker gunzips `spec_gz`; it still accepts plain `spec` too.
            spec_gz = gzip.compress(json.dumps(payload).encode("utf-8"))
            with open(spec.rnn_path, "rb") as fh:
                r = self._post("/infer",
                               files={"rnn": ("trained_rnn.pth", fh, "application/octet-stream"),
                                      "spec_gz": ("spec.json.gz", spec_gz, "application/gzip")})
            r.raise_for_status()
            job_id = r.json()["job_id"]
            if on_submit:
                try:
                    on_submit({"worker_job_id": job_id})
                except Exception:
                    pass

            # Poll until done, bounded by a hard wall-clock deadline so a wedged
            # worker can't block this thread forever.
            deadline = time.monotonic() + _INFER_DEADLINE
            while True:
                time.sleep(_POLL)
                if time.monotonic() > deadline:
                    raise ComputeError(
                        f"Worker inference exceeded {_INFER_DEADLINE:.0f}s without "
                        f"finishing (job {job_id}); giving up.", retryable_local=True)
                s = self._get(f"/infer/{job_id}")
                s.raise_for_status()
                st = s.json()
                state = st.get("state")
                if st.get("detail"):
                    _phase(f"GPU worker: {st['detail']}")
                if state == "done":
                    break
                if state in ("error", "cancelled"):
                    raise ComputeError(f"Worker inference {state}: {st.get('error')}",
                                       retryable_local=True)

            res = self._get(f"/infer/{job_id}/result")
            res.raise_for_status()
            data = res.json()
            self._cleanup_job("infer", job_id)   # free the job's scratch on the box
        except ComputeError:
            raise
        except Exception as e:
            # Network / worker outage → let the dispatcher retry locally.
            raise ComputeError(f"Remote worker inference failed: {e}", retryable_local=True)

        return [{"logits": np.array(r["logits"]), "metadata": r.get("metadata", {})}
                for r in data.get("results", [])]

    # -- training --------------------------------------------------------
    def submit_training(self, spec: TrainingSpec) -> TrainingJob:
        payload = {
            "classifier_id": spec.classifier_id,
            "user_id": spec.user_id,
            "model_hf_path": spec.model_hf_path,
            "labels": spec.labels,
            "config": spec.training_config,
            "dataset_files": spec.dataset_files,
            "calibration_entries": spec.calibration_entries,
            "hf_token": self._hf_token_for(spec.model_hf_path),
        }
        try:
            r = self._post("/train", json=payload, timeout=300)
            r.raise_for_status()
            job_id = r.json()["job_id"]
        except Exception as e:
            raise ComputeError(f"Remote worker training submit failed: {e}", retryable_local=True)
        return TrainingJob(provider=self.name, classifier_id=spec.classifier_id, id=job_id)

    def poll_training(self, job: TrainingJob) -> TrainingStatus:
        try:
            r = self._get(f"/train/{job.id}")
            r.raise_for_status()
            st = r.json()
        except Exception as e:
            # Couldn't reach the worker. Report it as RUNNING-but-UNREACHABLE so a
            # single transient blip doesn't kill the job, but the caller's deadline
            # can fail it if the worker stays gone.
            return TrainingStatus(state=JobState.RUNNING, detail=f"poll failed: {e}",
                                  reachable=False)
        state_map = {"queued": JobState.QUEUED, "running": JobState.RUNNING,
                     "done": JobState.DONE, "error": JobState.ERROR,
                     "cancelled": JobState.CANCELLED}
        return TrainingStatus(
            state=state_map.get(st.get("state"), JobState.RUNNING),
            phase=st.get("detail"), detail=st.get("detail"), error=st.get("error"),
        )

    def fetch_trained_model(self, job: TrainingJob, dest_dir: str) -> None:
        try:
            r = self._get(f"/train/{job.id}/model", timeout=300)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                os.makedirs(dest_dir, exist_ok=True)
                for name in ("trained_rnn.pth", "classifier_meta.json"):
                    if name in zf.namelist():
                        with zf.open(name) as src, open(os.path.join(dest_dir, name), "wb") as out:
                            out.write(src.read())
        except Exception as e:
            raise ComputeError(f"Fetching trained model from worker failed: {e}")
        self._cleanup_job("train", job.id)   # free the job's scratch on the box

    def cancel_training(self, job: TrainingJob) -> None:
        try:
            self._post(f"/train/{job.id}/cancel", timeout=15)
        except Exception:
            pass

    # -- realtime --------------------------------------------------------
    def start_realtime(self, spec: RealtimeSpec) -> RealtimeSession:
        payload = {
            "classifier_id": spec.classifier_id,
            "model_hf_path": spec.model_hf_path,
            "classifier_meta": spec.classifier_meta,
            "thresholds": spec.thresholds,
            "hf_token": self._hf_token_for(spec.model_hf_path),
        }
        try:
            with open(spec.rnn_path, "rb") as fh:
                r = self._post("/session/start", data={"spec": json.dumps(payload)},
                               files={"rnn": ("trained_rnn.pth", fh, "application/octet-stream")})
            if r.status_code == 409:
                raise ComputeError("GPU worker busy — a session/job is already running.")
            r.raise_for_status()
            sid = r.json()["session_id"]
        except ComputeError:
            raise
        except Exception as e:
            raise ComputeError(f"Remote worker session start failed: {e}", retryable_local=True)
        return RealtimeSession(provider=self.name, classifier_id=spec.classifier_id, id=sid)

    def realtime_status(self, session: RealtimeSession) -> str:
        try:
            r = self._get(f"/session/{session.id}/status", timeout=15)
            r.raise_for_status()
            return r.json().get("status", "dead")
        except Exception:
            return "dead"

    def realtime_analyze(self, session: RealtimeSession, payload: dict) -> dict:
        try:
            r = self._post(f"/session/{session.id}/analyze", json=payload, timeout=_HTTP_TIMEOUT)
            r.raise_for_status()
            return r.json().get("result") or {}
        except Exception as e:
            raise ComputeError(f"Remote realtime analyze failed: {e}")

    def realtime_keepalive(self, session: RealtimeSession) -> bool:
        try:
            r = self._post(f"/session/{session.id}/keepalive", timeout=15)
            return r.status_code == 200
        except Exception:
            return False

    def end_realtime(self, session: RealtimeSession) -> None:
        try:
            self._post(f"/session/{session.id}/end", timeout=30)
        except Exception:
            pass
