"""Compute provider interface + the provider-agnostic data contracts.

A `ComputeProvider` hides HOW a GPU workload runs (in-process torch,
or HTTPS to a remote worker) behind a uniform set of operations. The DTOs below
("what the GPU needs") are the stable contract a new provider implements — they
deliberately mirror the fields the existing local/remote code already passes,
so wrapping those paths is a faithful adapter, not a rewrite.

Nothing here imports torch / transformers / paramiko — it's pure definitions, so
importing it is cheap and side-effect free.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Workload(str, Enum):
    """The GPU-bound workloads we dispatch. Realtime is a warm interactive
    session; the others are batch."""
    TRAINING = "training"
    INFERENCE = "inference"   # used by BOTH calibration and evaluation
    REALTIME = "realtime"


class Accelerator(str, Enum):
    CUDA = "cuda"
    MPS = "mps"      # Apple Silicon
    CPU = "cpu"
    REMOTE = "remote"  # the provider runs the GPU elsewhere; local accelerator is N/A


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class ComputeError(Exception):
    """A compute-provider failure. `retryable_local` signals the dispatcher that
    falling back to the LocalProvider is sensible (e.g. the remote was
    unreachable), versus a deterministic failure that local would hit too."""

    def __init__(self, message: str, *, retryable_local: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable_local = retryable_local


# ---------------------------------------------------------------------------
# Capabilities — what a provider can do (so dispatch can fall back per-workload)
# ---------------------------------------------------------------------------

@dataclass
class Capabilities:
    name: str
    accelerator: Accelerator
    is_local: bool
    supports_training: bool = True
    supports_inference: bool = True
    supports_realtime: bool = True
    # Max warm realtime sessions the provider can hold at once (GPU-memory bound).
    # 0 means "no realtime". Providers report their practical limits.
    max_realtime_sessions: int = 1
    # Free-text detail for the UI status line ("RunPod worker", "Mistral on CUDA").
    detail: Optional[str] = None
    # The provider's classifier_engine version, for skew detection on remote
    # workers (None for in-process providers, which are always in sync).
    code_version: Optional[str] = None

    def supports(self, workload: "Workload") -> bool:
        return {
            Workload.TRAINING: self.supports_training,
            Workload.INFERENCE: self.supports_inference,
            Workload.REALTIME: self.supports_realtime,
        }[workload]


# ---------------------------------------------------------------------------
# Specs — "what the GPU needs" for each workload (the new-provider contract)
# ---------------------------------------------------------------------------

@dataclass
class TrainingSpec:
    classifier_id: int
    user_id: int
    model_hf_path: str            # base LLM (HF repo id or local path)
    labels: Dict[str, int]        # CE name -> output index
    training_config: dict
    dataset_files: Dict[str, Any]  # {filename: conversations} excitation data
    calibration_entries: Optional[list] = None
    # The provider returns the trained artifacts (trained_rnn.pth +
    # classifier_meta.json) by writing them into this dir on fetch.
    output_dir: Optional[str] = None


@dataclass
class InferenceSpec:
    """Windowed LLM-readout + RNN inference. Used identically by calibration and
    evaluation; the metric/threshold math runs on the backend afterwards."""
    classifier_id: int
    model_hf_path: str
    classifier_meta: dict
    dialogues: List[dict]         # [{conversation|text, metadata}, ...]
    rnn_path: str                 # local path to the trained .pth to run
    max_length: Optional[int] = None
    window_stride: int = 0        # 0 => non-overlapping (stride = window size)


@dataclass
class RealtimeSpec:
    classifier_id: int
    model_hf_path: str
    classifier_meta: dict
    rnn_path: str
    thresholds: Optional[dict] = None
    idle_timeout_s: Optional[int] = None


# ---------------------------------------------------------------------------
# Handles + status returned by providers
# ---------------------------------------------------------------------------

@dataclass
class TrainingJob:
    """Opaque handle to a submitted training run. `raw` holds provider-specific
    bookkeeping (e.g. a remote job id)."""
    provider: str
    classifier_id: int
    id: str
    raw: dict = field(default_factory=dict)


@dataclass
class TrainingStatus:
    state: JobState
    phase: Optional[str] = None        # user-facing label ("Epoch 3 of 10")
    detail: Optional[str] = None
    progress: Optional[float] = None   # 0..1 if known
    log: Optional[list] = None         # structured per-epoch log if available
    error: Optional[str] = None
    reachable: bool = True             # False => this poll could NOT contact the
    #                                    worker (network/outage), so the state is
    #                                    unknown rather than authoritatively
    #                                    "running". Callers combine it with a
    #                                    deadline to fail a job whose worker vanished.


@dataclass
class RealtimeSession:
    provider: str
    classifier_id: int
    id: str
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------

PhaseCallback = Optional[Callable[[str], None]]


class ComputeProvider(ABC):
    """One way to run GAVEL's GPU workloads. Providers may raise `ComputeError`;
    the dispatcher handles fallback-to-local per `retryable_local`."""

    name: str = "base"

    # --- discovery ---------------------------------------------------------
    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap check: configured AND reachable right now. Must not raise."""

    # --- training (async batch) -------------------------------------------
    def submit_training(self, spec: TrainingSpec) -> TrainingJob:
        raise NotImplementedError

    def poll_training(self, job: TrainingJob) -> TrainingStatus:
        raise NotImplementedError

    def fetch_trained_model(self, job: TrainingJob, dest_dir: str) -> None:
        """Write trained_rnn.pth + classifier_meta.json into dest_dir."""
        raise NotImplementedError

    def cancel_training(self, job: TrainingJob) -> None:
        raise NotImplementedError

    # --- inference (calibration + evaluation) -----------------------------
    def run_inference(self, spec: InferenceSpec, on_phase: PhaseCallback = None,
                      on_submit: Optional[Callable[[dict], None]] = None) -> List[dict]:
        """Blocking. Returns per-dialogue results carrying logits + metadata.

        `on_phase(text)` surfaces the live stage to the UI. `on_submit(info)` is
        called once a remote job is submitted (info carries provider job ids) so
        the caller can persist it for crash recovery — in-process providers never
        call it. Raise `ComputeError(retryable_local=True)` when the failure is a
        provider-availability issue the LocalProvider could ride out."""
        raise NotImplementedError

    # --- realtime (warm session) ------------------------------------------
    def start_realtime(self, spec: RealtimeSpec) -> RealtimeSession:
        raise NotImplementedError

    def realtime_status(self, session: RealtimeSession) -> str:
        """queued | loading | ready | dead | stopped"""
        raise NotImplementedError

    def realtime_analyze(self, session: RealtimeSession, payload: dict) -> dict:
        raise NotImplementedError

    def realtime_keepalive(self, session: RealtimeSession) -> bool:
        raise NotImplementedError

    def end_realtime(self, session: RealtimeSession) -> None:
        raise NotImplementedError

    # --- housekeeping ------------------------------------------------------
    def recover_orphans(self) -> dict:
        """Boot-time cleanup of anything this provider may have leaked (orphan
        GPU jobs / warm sessions). Best-effort; returns a small summary."""
        return {}

    def end_all_realtime(self) -> int:
        """Shutdown-time cleanup: end every warm realtime session this provider
        holds (frees GPUs immediately instead of waiting for idle-timeout).
        Returns the count ended. Best-effort; default 0."""
        return 0

    def cancel_inference(self, pointer: dict) -> None:
        """Cancel an in-flight inference (calibration/evaluation) job described by
        `pointer` (e.g. a remote job id). A provider ignores a
        pointer that isn't its own. Best-effort; default no-op."""
        return None
