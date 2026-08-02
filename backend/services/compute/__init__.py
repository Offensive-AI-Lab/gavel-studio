"""Pluggable GPU compute backends for GAVEL.

Every GPU-bound workload (training, calibration/evaluation inference, and the
warm realtime session) is expressed against a single `ComputeProvider`
interface. The active provider is resolved from the environment by
`registry.get_provider(...)`, so the same backend runs unchanged whether the GPU
is local (CUDA/MPS/CPU) or a remote
`gavel-gpu-worker` the user spun up on RunPod / AWS / Colab / any box.

Public surface:
    from services import compute
    provider = compute.get_provider(compute.Workload.TRAINING)
    compute.status()            # for the UI "which GPU am I on" indicator
"""
from .base import (  # noqa: F401
    Accelerator,
    Capabilities,
    ComputeError,
    ComputeProvider,
    InferenceSpec,
    JobState,
    RealtimeSession,
    RealtimeSpec,
    TrainingJob,
    TrainingSpec,
    TrainingStatus,
    Workload,
)
from .registry import (  # noqa: F401
    get_provider, local_provider, status, all_providers,
    failover_providers, provider_by_name,
)
