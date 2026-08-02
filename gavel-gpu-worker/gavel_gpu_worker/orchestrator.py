"""Runs the GAVEL job scripts locally and tracks their state.

The worker is a thin orchestration layer: instead of the backend running a
job in-process, the worker writes the job dir on local disk
and runs the SAME script as a subprocess. Inference/training are batch jobs;
realtime is a warm subprocess we feed via the request/response file protocol.

A single GPU can only do one heavy thing at a time, so a 1-slot semaphore
serializes batch jobs and is HELD for the lifetime of a warm realtime session.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from io import BytesIO
from typing import Optional

from . import config


def _now() -> float:
    return time.time()


class BatchJob:
    def __init__(self, job_id: str, kind: str, job_dir: str):
        self.id = job_id
        self.kind = kind            # "infer" | "train"
        self.dir = job_dir
        self.state = "queued"       # queued | running | done | error | cancelled
        self.error: Optional[str] = None
        self.elapsed: Optional[float] = None
        self.proc: Optional[subprocess.Popen] = None
        self.created = _now()


class Session:
    def __init__(self, session_id: str, job_dir: str):
        self.id = session_id
        self.dir = job_dir
        self.proc: Optional[subprocess.Popen] = None
        self.last_ping = _now()
        self.created = _now()
        self._req_seq = 0
        self._lock = threading.Lock()


class Orchestrator:
    def __init__(self):
        os.makedirs(config.JOBS_DIR, exist_ok=True)
        self.jobs: dict = {}
        self.sessions: dict = {}
        self._gpu = threading.BoundedSemaphore(1)   # 1 GPU slot
        self._reg_lock = threading.Lock()
        self._start_sweeper()

    # -- helpers ---------------------------------------------------------
    def _new_dir(self, prefix: str) -> tuple:
        jid = f"{prefix}_{uuid.uuid4().hex[:12]}"
        d = os.path.join(config.JOBS_DIR, jid)
        os.makedirs(d, exist_ok=True)
        return jid, d

    def _job_env(self, job_dir: str) -> dict:
        # HF cache is shared across jobs (set in worker_env via GAVEL_HF_CACHE, or
        # the user's default ~/.cache/huggingface) — NOT per-job, so multi-GB model
        # weights are downloaded once and reused by every train/infer/realtime run.
        env = config.worker_env()
        env["JOB_DIR"] = job_dir
        return env

    # -- batch (infer / train) ------------------------------------------
    def submit_batch(self, kind: str, payload: dict, rnn_bytes: Optional[bytes],
                     dataset_files: Optional[dict] = None) -> str:
        script = {"infer": "infer_job.py", "train": "train_job.py"}[kind]
        jid, d = self._new_dir(kind)
        with open(os.path.join(d, "job_payload.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)
        if rnn_bytes is not None:
            with open(os.path.join(d, "trained_rnn.pth"), "wb") as f:
                f.write(rnn_bytes)
        if dataset_files:
            ds = os.path.join(d, "dataset")
            os.makedirs(ds, exist_ok=True)
            for name, conv in dataset_files.items():
                with open(os.path.join(ds, name), "w", encoding="utf-8") as f:
                    json.dump(conv, f)

        job = BatchJob(jid, kind, d)
        with self._reg_lock:
            self.jobs[jid] = job
        threading.Thread(target=self._run_batch, args=(job, script), daemon=True,
                         name=f"gavel-{kind}-{jid}").start()
        return jid

    def _run_batch(self, job: BatchJob, script: str):
        # Queue behind any other GPU work (another batch job or a warm session).
        self._gpu.acquire()
        try:
            if job.state == "cancelled":
                return
            job.state = "running"
            t0 = _now()
            log = open(os.path.join(job.dir, "worker.log"), "w", encoding="utf-8")
            job.proc = subprocess.Popen(
                [sys.executable, "-u", config.job_script(script),
                 "--job-dir", job.dir, "--device", config.DEVICE],
                env=self._job_env(job.dir), stdout=log, stderr=subprocess.STDOUT,
            )
            rc = job.proc.wait()
            log.close()
            job.elapsed = round(_now() - t0, 1)
            status = self._read_status(job.dir)
            if job.state == "cancelled":
                return
            if status.get("status") == "success" and rc == 0:
                job.state = "done"
            else:
                job.state = "error"
                job.error = status.get("error") or f"job exited {rc}"
        except Exception as e:
            job.state = "error"
            job.error = str(e)
        finally:
            self._gpu.release()

    def _read_status(self, job_dir: str) -> dict:
        try:
            with open(os.path.join(job_dir, "status.json")) as f:
                return json.load(f)
        except Exception:
            return {}

    def get_batch(self, job_id: str) -> Optional[dict]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        out = {"job_id": job.id, "kind": job.kind, "state": job.state,
               "error": job.error, "elapsed_s": job.elapsed}
        # Surface the on-disk job status/phase if present (e.g. training progress).
        st = self._read_status(job.dir)
        if st:
            out["detail"] = st.get("phase") or st.get("status")
        return out

    def batch_result_path(self, job_id: str) -> Optional[str]:
        job = self.jobs.get(job_id)
        if not job or job.state != "done":
            return None
        p = os.path.join(job.dir, "logits.json")
        return p if os.path.isfile(p) else None

    def batch_model_zip(self, job_id: str) -> Optional[bytes]:
        job = self.jobs.get(job_id)
        if not job or job.state != "done":
            return None
        buf = BytesIO()
        wrote = False
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in ("trained_rnn.pth", "classifier_meta.json"):
                p = os.path.join(job.dir, name)
                if os.path.isfile(p):
                    zf.write(p, name)
                    wrote = True
        if not wrote:
            return None
        buf.seek(0)
        return buf.getvalue()

    def cancel_batch(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        job.state = "cancelled"
        if job.proc and job.proc.poll() is None:
            try:
                job.proc.terminate()
            except Exception:
                pass
        return True

    def cleanup_batch(self, job_id: str):
        job = self.jobs.pop(job_id, None)
        if job:
            shutil.rmtree(job.dir, ignore_errors=True)

    # -- realtime (warm session) ----------------------------------------
    def start_session(self, payload: dict, rnn_bytes: bytes) -> Optional[str]:
        # A warm session HOLDS the GPU; refuse if it's busy rather than OOM.
        if not self._gpu.acquire(blocking=False):
            return None
        try:
            sid, d = self._new_dir("rt")
            os.makedirs(os.path.join(d, "requests"), exist_ok=True)
            os.makedirs(os.path.join(d, "responses"), exist_ok=True)
            with open(os.path.join(d, "job_payload.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f)
            with open(os.path.join(d, "trained_rnn.pth"), "wb") as f:
                f.write(rnn_bytes)
            sess = Session(sid, d)
            log = open(os.path.join(d, "worker.log"), "w", encoding="utf-8")
            sess.proc = subprocess.Popen(
                [sys.executable, "-u", config.job_script("realtime_job.py"),
                 "--job-dir", d, "--device", config.DEVICE],
                env=self._job_env(d), stdout=log, stderr=subprocess.STDOUT,
            )
            with self._reg_lock:
                self.sessions[sid] = sess
            return sid
        except Exception:
            self._gpu.release()
            raise

    def session_status(self, session_id: str) -> str:
        sess = self.sessions.get(session_id)
        if not sess:
            return "dead"
        if sess.proc and sess.proc.poll() is not None:
            return "dead"
        st = self._read_status(sess.dir).get("status")
        return st or "loading"

    def keepalive(self, session_id: str) -> bool:
        sess = self.sessions.get(session_id)
        if not sess:
            return False
        sess.last_ping = _now()
        try:
            ka = os.path.join(sess.dir, "keepalive")
            with open(ka, "w") as f:
                f.write(str(_now()))
        except OSError:
            pass
        return True

    def analyze(self, session_id: str, req: dict) -> dict:
        sess = self.sessions.get(session_id)
        if not sess:
            raise KeyError("no such session")
        if self.session_status(session_id) != "ready":
            raise RuntimeError("session not ready")
        self.keepalive(session_id)
        with sess._lock:
            sess._req_seq += 1
            rid = f"{int(_now()*1000)}_{sess._req_seq}"
        req = dict(req)
        req["id"] = rid
        req_dir = os.path.join(sess.dir, "requests")
        resp_dir = os.path.join(sess.dir, "responses")
        tmp = os.path.join(req_dir, rid + ".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(req, f)
        os.replace(tmp, os.path.join(req_dir, rid + ".json"))

        deadline = _now() + config.ANALYZE_TIMEOUT_S
        resp_path = os.path.join(resp_dir, rid + ".json")
        while _now() < deadline:
            if os.path.isfile(resp_path):
                try:
                    with open(resp_path) as f:
                        out = json.load(f)
                finally:
                    try:
                        os.remove(resp_path)
                    except OSError:
                        pass
                if not out.get("ok"):
                    raise RuntimeError(out.get("error") or "analyze failed")
                return out.get("result") or {}
            if sess.proc and sess.proc.poll() is not None:
                raise RuntimeError("realtime session died")
            time.sleep(0.1)
        raise TimeoutError("analyze timed out")

    def end_session(self, session_id: str) -> bool:
        sess = self.sessions.pop(session_id, None)
        if not sess:
            return False
        self._stop_session(sess)
        return True

    def _stop_session(self, sess: Session):
        try:
            open(os.path.join(sess.dir, "stop"), "w").close()
        except OSError:
            pass
        if sess.proc and sess.proc.poll() is None:
            try:
                sess.proc.wait(timeout=15)
            except Exception:
                try:
                    sess.proc.terminate()
                except Exception:
                    pass
        try:
            self._gpu.release()
        except ValueError:
            pass  # already released
        shutil.rmtree(sess.dir, ignore_errors=True)

    # -- idle sweeper ----------------------------------------------------
    def _start_sweeper(self):
        threading.Thread(target=self._sweep_loop, daemon=True, name="gavel-rt-sweeper").start()

    def _sweep_loop(self):
        while True:
            time.sleep(30)
            try:
                self._sweep_once()
            except Exception:
                pass

    def _sweep_once(self):
        cutoff = _now() - config.SESSION_IDLE_TIMEOUT_S
        dead = []
        with self._reg_lock:
            for sid, sess in list(self.sessions.items()):
                if sess.last_ping < cutoff or (sess.proc and sess.proc.poll() is not None):
                    dead.append(sid)
        for sid in dead:
            self.end_session(sid)
        self._sweep_job_dirs()

    def _sweep_job_dirs(self):
        # Safety net for per-job scratch. The backend deletes each batch job right
        # after fetching its result/model, so this only catches leftovers: a client
        # that died mid-run, or dirs from a previous worker process (restart wipes
        # the in-memory registry but not the disk). Purge anything past the retention
        # window so a limited-disk box can't fill up over many runs.
        retention = config.JOB_RETENTION_S
        now = _now()
        # 1) finished in-memory jobs the backend never cleaned up.
        stale = []
        with self._reg_lock:
            for jid, job in list(self.jobs.items()):
                if job.state in ("done", "error", "cancelled") and now - job.created > retention:
                    stale.append(jid)
        for jid in stale:
            self.cleanup_batch(jid)
        # 2) orphaned dirs on disk not tracked by any live job/session.
        try:
            entries = os.listdir(config.JOBS_DIR)
        except OSError:
            return
        with self._reg_lock:
            active = {j.dir for j in self.jobs.values()} | {s.dir for s in self.sessions.values()}
        for name in entries:
            p = os.path.join(config.JOBS_DIR, name)
            if p in active or not os.path.isdir(p):
                continue
            try:
                if now - os.path.getmtime(p) > retention:
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass

    def active_session_count(self) -> int:
        return len(self.sessions)


# Process-wide singleton.
ORCH = Orchestrator()
