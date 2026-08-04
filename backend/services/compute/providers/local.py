"""LocalProvider — this machine's own accelerator (CUDA → MPS → CPU). The
always-present floor every other provider falls back to.

Inference and realtime run in-process. TRAINING does not: it is spawned as a
child process running `compute_jobs/train_job.py`, exactly the script the remote
GPU worker runs, driven through the same submit/poll/fetch/cancel shape as
`RemoteWorkerProvider`. Two reasons:

  * A wedged trainer (a hung CUDA kernel, an OOM-thrashing DataLoader) used to
    freeze the whole API — it ran inside the server process. Now it can only
    take down its own child.
  * The child is DB-FREE. Everything it needs is staged into a job directory and
    everything it produces comes back out of that directory; the parent does
    every database write. A second SQLite writer against a database that may
    live on NFS is a real corruption risk (see utils/sqlite_db.py).

The job handle the caller persists is `{job_dir, pid, started_at}`. It is durable
on purpose: a training child SURVIVES an API restart (uvicorn --reload restarts
the server constantly), so boot-time crash recovery re-reads the handle and
re-adopts a run that is still going instead of failing it.
"""
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from ..base import (
    Accelerator, Capabilities, ComputeError, ComputeProvider, InferenceSpec,
    JobState, TrainingJob, TrainingSpec, TrainingStatus,
)
from ..phases import phase_label

logger = logging.getLogger("compute.local")

# Sub-directory of the guardrail workdir that holds one training run's staged
# inputs and raw outputs. Nested inside the workdir (not in a temp dir) so the
# existing delete/orphan sweeps clean it up for free.
JOB_DIRNAME = "local_train_job"

# The training child's own process GROUP is killed on cancel: train_job.py forks
# DataLoader workers, and signalling only the group leader orphans them.
_CANCEL_GRACE_S = float(os.getenv("GAVEL_LOCAL_CANCEL_GRACE", "8"))

# pid -> (Popen, job_dir) for children we spawned in THIS process. Two jobs:
# `_reap` waits them on every terminal outcome (an unwaited child stays a
# zombie, and a zombie still answers kill(pid, 0), so it would read as "still
# training" forever), and an entry here is *proof* of ownership that doesn't
# depend on /proc — or on the job directory still existing.
_procs: dict = {}
_procs_lock = threading.Lock()

# How long a terminal poll waits for a child that has already written its result
# to actually leave the process table. status.json is written before the child's
# interpreter exits, so the wait is normally microseconds.
_REAP_GRACE_S = float(os.getenv("GAVEL_LOCAL_REAP_GRACE", "5"))


def _backend_root() -> str:
    # services/compute/providers/local.py -> backend/
    return str(Path(__file__).resolve().parents[3])


def _train_script() -> str:
    return os.path.join(_backend_root(), "compute_jobs", "train_job.py")


def _pid_alive(pid: int) -> bool:
    """True when `pid` is a live, non-zombie process."""
    if not pid or pid <= 0:
        return False
    with _procs_lock:
        proc = (_procs.get(pid) or (None, None))[0]
    if proc is not None:
        # We own it: poll() both answers authoritatively and reaps.
        return proc.poll() is None
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return False
    # A child of a DEAD parent is re-parented to init and reaped there, but a
    # child of a live parent that never wait()ed lingers as a zombie. /proc tells
    # them apart; where it doesn't exist (macOS/Windows) kill(0) is all we have.
    try:
        with open(f"/proc/{pid}/stat") as f:
            # "<pid> (comm) <state> ..." — comm can contain spaces/parens, so
            # split after the LAST ')'.
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def _reap(pid: int) -> None:
    """Wait() our child for `pid` and drop its registry entry.

    Every TERMINAL outcome — success, failure, cancel — goes through here. An
    unwaited child stays a zombie for the life of the API process, and a zombie
    still answers kill(pid, 0), so it would read as "still training" forever.
    Waiting is bounded so a poll request can't hang on a teardown that never
    finishes (a CUDA context that refuses to close); in that case a daemon
    thread finishes the join instead.
    """
    with _procs_lock:
        entry = _procs.pop(pid, None)
    if entry is None:
        return
    proc = entry[0]
    try:
        proc.wait(timeout=_REAP_GRACE_S)
    except Exception:
        threading.Thread(target=_wait_quietly, args=(proc,),
                         name=f"gavel-reap-{pid}", daemon=True).start()


def _wait_quietly(proc) -> None:
    try:
        proc.wait()
    except Exception:
        pass


def _pid_owns_job(pid: int, job_dir: str) -> bool:
    """The PID + job-dir handshake.

    Liveness alone is not enough to claim a run: PIDs get reused, and after a
    reboot some unrelated process very plausibly holds the number we recorded.
    Killing or adopting that would be a serious bug, so we additionally require
    the process to actually be OUR training child — its command line has to name
    both train_job.py and this job directory. Where /proc isn't readable we fall
    back to "alive AND the job dir still exists", which is the best a portable
    check can do.

    That whole disk handshake is the FALLBACK. If we spawned the child ourselves
    and still hold its Popen, that settles it without touching the filesystem —
    see below.
    """
    if not pid or pid <= 0:
        return False
    with _procs_lock:
        mine = _procs.get(pid)
    if mine is not None and mine[1] == job_dir:
        # We spawned it ourselves and still hold the Popen: conclusive, so it is
        # consulted BEFORE anything on disk. A terminal poll deletes the job dir
        # while the child may still be winding down, and gating on the directory
        # first would turn cancel into a no-op on exactly the process we most
        # need to kill. It also avoids a /proc race — for a few microseconds
        # after execve the kernel has not filled in the new argv yet, so cmdline
        # reads empty.
        return mine[0].poll() is None
    if not job_dir or not os.path.isdir(job_dir):
        return False
    if not _pid_alive(pid):
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().decode("utf-8", "replace")
    except OSError:
        return True  # no /proc — alive + job dir present is our best evidence
    return "train_job.py" in cmdline and job_dir in cmdline


class LocalProvider(ComputeProvider):
    name = "local"

    def _device_type(self) -> str:
        try:
            from utils.device import get_torch_device
            return get_torch_device().type  # "cuda" | "mps" | "cpu"
        except Exception:
            return "cpu"

    def capabilities(self) -> Capabilities:
        dev = self._device_type()
        acc = {"cuda": Accelerator.CUDA, "mps": Accelerator.MPS}.get(dev, Accelerator.CPU)
        detail = {
            "cuda": "Local CUDA GPU",
            "mps": "Local Apple GPU (MPS)",
            "cpu": "Local CPU (no GPU — slow)",
        }.get(dev, "Local")
        return Capabilities(
            name=self.name, accelerator=acc, is_local=True,
            supports_training=True, supports_inference=True, supports_realtime=True,
            max_realtime_sessions=1, detail=detail,
        )

    def is_available(self) -> bool:
        return True  # local torch is always there (CPU at worst)

    # --- inference (calibration + evaluation) ---
    def run_inference(self, spec: InferenceSpec, on_phase: Optional[Callable] = None,
                      on_submit: Optional[Callable] = None) -> List[dict]:
        from evaluation.inference import run_inference_on_dialogues

        if on_phase:
            on_phase(f"Running inference locally on {len(spec.dialogues)} conversations…")

        # inference_core emits ("inference", "Dialogue 120/500") periodically;
        # relay it to the live phase line (same wording as the old local path).
        def _infer_progress(stage, detail=""):
            if on_phase and detail:
                if stage and stage != "inference":
                    on_phase(f"Running inference locally — {detail.lower()} ({stage})")
                else:
                    on_phase(f"Running inference locally — {detail.lower()}")

        return run_inference_on_dialogues(
            spec.classifier_id, spec.dialogues, progress_callback=_infer_progress,
        )

    # --- training (spawned child) ---------------------------------------
    def _hf_token_for(self, model_ref: Optional[str]) -> Optional[str]:
        """The child has no DB, so resolve a gated-model token here and ship it
        in the payload (train_job.py exports it as HF_TOKEN)."""
        try:
            from utils.model_hf_token import resolve_model_hf_token
            return resolve_model_hf_token(model_ref) if model_ref else None
        except Exception:
            return None

    def _child_env(self) -> dict:
        env = os.environ.copy()
        # The tokenizers Rust threadpool deadlocks after fork; train_job.py forks
        # DataLoader workers, so this is not optional.
        env["TOKENIZERS_PARALLELISM"] = "false"
        # classifier_engine/ lives under backend/ here, not in the worker's
        # ~/gavel_code — put backend/ on the child's import path.
        root = _backend_root()
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root + (os.pathsep + prev if prev else "")
        return env

    def submit_training(self, spec: TrainingSpec) -> TrainingJob:
        from classifier_engine.trainer import classifier_workdir

        # Resolve the workdir from the guardrail row rather than from
        # spec.user_id: every reader (evaluation, realtime, the export bundle)
        # resolves it the same way, and a failover re-submit may not carry a
        # trustworthy user_id.
        work_dir = spec.output_dir or classifier_workdir(spec.classifier_id)
        # Retraining replaces the previous artifacts in place — same folder name,
        # fresh contents — matching what the in-process trainer always did, so a
        # guardrail's directory never moves across retrains.
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        job_dir = os.path.join(work_dir, JOB_DIRNAME)
        dataset_dir = os.path.join(job_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        payload = {
            "classifier_id": spec.classifier_id,
            "user_id": spec.user_id,
            "model_hf_path": spec.model_hf_path,
            "labels": spec.labels,
            "config": spec.training_config,
            "hf_token": self._hf_token_for(spec.model_hf_path),
        }
        with open(os.path.join(job_dir, "job_payload.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)
        for fname, conversations in (spec.dataset_files or {}).items():
            with open(os.path.join(dataset_dir, fname), "w", encoding="utf-8") as f:
                json.dump(conversations, f)
        if spec.calibration_entries:
            with open(os.path.join(job_dir, "calibration_input.json"), "w", encoding="utf-8") as f:
                json.dump(spec.calibration_entries, f)

        # Device is decided HERE: utils.device's detection (and its thread-local
        # force_cpu) lives in this process and does not cross a process boundary.
        device = self._device_type()
        cmd = [sys.executable, "-u", _train_script(), "--job-dir", job_dir, "--device", device]

        # start_new_session puts the child in its own process GROUP so cancel can
        # take out its DataLoader workers too, and so a Ctrl+C in the terminal
        # running uvicorn doesn't reach into a training run.
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True

        # The child's stdout/stderr go to a file in the job dir: it outlives this
        # process, so there is nobody left to drain a pipe (and a full pipe would
        # wedge the trainer).
        log_path = os.path.join(job_dir, "train_job.log")
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    cmd, cwd=_backend_root(), env=self._child_env(),
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    **kwargs,
                )
        except Exception as e:
            raise ComputeError(f"Could not start the local training process: {e}")

        with _procs_lock:
            _procs[proc.pid] = (proc, job_dir)
        logger.info("[local] training pid=%s classifier=%s device=%s job_dir=%s",
                    proc.pid, spec.classifier_id, device, job_dir)
        return TrainingJob(
            provider=self.name, classifier_id=spec.classifier_id, id=str(proc.pid),
            raw={"job_dir": job_dir, "pid": proc.pid, "started_at": time.time(),
                 "device": device},
        )

    # -- job-dir readers --------------------------------------------------
    @staticmethod
    def _read_json(job_dir: str, name: str):
        try:
            with open(os.path.join(job_dir, name), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _handle(job: TrainingJob) -> tuple:
        raw = job.raw or {}
        pid = raw.get("pid")
        try:
            pid = int(pid) if pid is not None else int(job.id)
        except (TypeError, ValueError):
            pid = 0
        return raw.get("job_dir") or "", pid

    def poll_training(self, job: TrainingJob) -> TrainingStatus:
        job_dir, pid = self._handle(job)
        if not job_dir or not os.path.isdir(job_dir):
            # Terminal, but the child may still be running (somebody swept the
            # workdir out from under it). Only reap one that has already exited:
            # while it lives the registry entry is the sole proof of ownership
            # cancel_training has left, and dropping it would strand the child.
            if not _pid_alive(pid):
                _reap(pid)
            return TrainingStatus(state=JobState.ERROR,
                                  error="The training job directory is gone.")

        def _terminal(status):
            if status.get("status") == "success":
                # Everything is read out of the job dir BEFORE reaping; the dir
                # itself stays until fetch_trained_model has the weights out
                # (crash recovery also re-reads it to adopt a run that finished
                # while the server was down).
                done = TrainingStatus(state=JobState.DONE, phase="complete",
                                      detail="Training complete",
                                      log=self._read_json(job_dir, "training_log.json"))
                _reap(pid)
                return done
            msg = status.get("error") or "Training failed."
            _reap(pid)
            self._discard_job_dir(job_dir)
            return TrainingStatus(state=JobState.ERROR, phase="failed",
                                  error=msg, detail=msg)

        # status.json is written before the child exits, so read it FIRST: a run
        # that just finished must not be mistaken for one that died.
        status = self._read_json(job_dir, "status.json")
        if isinstance(status, dict) and status.get("status"):
            return _terminal(status)

        if not _pid_alive(pid):
            # Re-read once — the child may have written status.json in the window
            # between our read and its exit.
            status = self._read_json(job_dir, "status.json")
            if isinstance(status, dict) and status.get("status"):
                return _terminal(status)
            tail = self._log_tail(job_dir)   # read the clue out before deleting
            _reap(pid)
            self._discard_job_dir(job_dir)
            return TrainingStatus(
                state=JobState.ERROR, phase="failed",
                error="The training process stopped without producing a result."
                      + (f" Last output: {tail}" if tail else ""))

        prog = self._read_json(job_dir, "progress.json") or {}
        stage = prog.get("stage") or "init"
        return TrainingStatus(
            state=JobState.RUNNING,
            phase=phase_label(stage),
            detail=prog.get("detail") or "Preparing…",
            progress=prog.get("progress"),
            log=self._read_json(job_dir, "training_log.json"),
        )

    @staticmethod
    def _discard_job_dir(job_dir: str) -> None:
        """Drop a job directory nothing will read again.

        The staged datasets and the multi-GB extracted representations are dead
        weight the moment a run reaches a terminal state. The success path drops
        them in fetch_trained_model, once the weights are safely out; a FAILED
        run has nothing to collect, so it is dropped here instead of being left
        behind in the guardrail's workdir until the next retrain."""
        shutil.rmtree(job_dir, ignore_errors=True)

    @staticmethod
    def _log_tail(job_dir: str, limit: int = 400) -> str:
        """Last bytes of the child's stdout — the only clue we have when it died
        hard enough (OOM-killer, segfault) to never write status.json."""
        try:
            with open(os.path.join(job_dir, "train_job.log"), "rb") as f:
                try:
                    f.seek(-limit * 4, os.SEEK_END)
                except OSError:
                    f.seek(0)
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return ""
        return " ".join(text.split())[-limit:]

    def fetch_trained_model(self, job: TrainingJob, dest_dir: str) -> None:
        job_dir, _ = self._handle(job)
        os.makedirs(dest_dir, exist_ok=True)
        missing = []
        for name in ("trained_rnn.pth", "classifier_meta.json"):
            src = os.path.join(job_dir, name)
            if not os.path.isfile(src):
                missing.append(name)
                continue
            dst = os.path.join(dest_dir, name)
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            try:
                os.replace(src, dst)   # same filesystem: atomic, no copy
            except OSError:
                shutil.move(src, dst)  # dest_dir was overridden onto another volume
        if missing:
            raise ComputeError(
                f"Training finished but produced no {' / '.join(missing)}.")
        # The weights are out; the rest of the job dir is dead weight.
        self._discard_job_dir(job_dir)

    def cancel_training(self, job: TrainingJob) -> None:
        """Kill the training child's whole process group (SIGTERM, then SIGKILL
        after a grace period). Best-effort and idempotent."""
        job_dir, pid = self._handle(job)
        # Never signal a PID we cannot positively identify as our own training
        # child — after a reboot that number belongs to somebody else.
        if not _pid_owns_job(pid, job_dir):
            return
        try:
            pgid = os.getpgid(pid) if hasattr(os, "getpgid") else None
        except OSError:
            pgid = None

        def _signal(sig):
            if pgid is not None and hasattr(os, "killpg"):
                os.killpg(pgid, sig)
            else:
                os.kill(pid, sig)

        try:
            _signal(getattr(signal, "SIGTERM", signal.SIGINT))
        except OSError:
            return
        deadline = time.monotonic() + _CANCEL_GRACE_S
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                break
            time.sleep(0.2)
        if _pid_alive(pid):
            try:
                _signal(getattr(signal, "SIGKILL", signal.SIGTERM))
            except OSError:
                pass
        _reap(pid)   # so no zombie is left behind
        logger.info("[local] cancelled training pid=%s job_dir=%s", pid, job_dir)

    # -- crash recovery ---------------------------------------------------
    def training_is_alive(self, job: TrainingJob) -> bool:
        """Is this handle's run STILL GOING right now?

        Boot-time recovery uses this to decide between adopting a training child
        that outlived the API process and writing the guardrail off as
        interrupted. Requires the full PID + job-dir handshake, so a stale handle
        whose PID has been recycled reads as dead (the safe answer: the run is
        failed, and nothing of somebody else's gets killed).
        """
        job_dir, pid = self._handle(job)
        return _pid_owns_job(pid, job_dir)
