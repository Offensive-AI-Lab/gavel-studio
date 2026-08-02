# utils/device.py
# Centralized PyTorch device selection: CUDA > MPS (Apple Silicon) > CPU.
import contextlib
import os
import threading

import torch

# Thread-local "force CPU" flag. The inference failover ladder's last tier wraps
# its call in force_cpu() so a GPU-tier crash (e.g. CUDA OOM) can be retried on
# CPU in-process, without changing any inner function signatures. The env var
# GAVEL_FORCE_CPU=1 forces it process-wide (handy for testing / GPU-less boxes).
_force = threading.local()


def _cpu_forced() -> bool:
    if getattr(_force, "on", False):
        return True
    return os.getenv("GAVEL_FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "on")


@contextlib.contextmanager
def force_cpu():
    """Within this block, device selection reports CPU even if a GPU is present."""
    prev = getattr(_force, "on", False)
    _force.on = True
    try:
        yield
    finally:
        _force.on = prev


def get_torch_device() -> torch.device:
    """Return the best available accelerator: CUDA > MPS > CPU (or CPU if forced)."""
    if _cpu_forced():
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_device_cache():
    """Free cached memory on the active accelerator."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_llm_device_map() -> "str | dict":
    """Return an appropriate device_map for HuggingFace model loading.

    - CUDA: "auto" (uses accelerate for multi-GPU)
    - MPS:  map everything to the MPS device
    - CPU:  "cpu"  (also when CPU is forced)
    """
    if _cpu_forced():
        return "cpu"
    if torch.cuda.is_available():
        return "auto"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return {"": "mps"}
    return "cpu"
