"""Registry sync — the backend's link to the public library registry.

The backend ingests the public library into its DB via
`services.hf_sync.sync_library`. This package adds:
  * the read-side port (`RegistryReader` + the HuggingFace adapter), and
  * the freshness trigger: a periodic poller that probes HF for changes and
    drives the sidebar's "Updates available" badge.

Entry point: `build_poller()` (started in the backend's lifespan).
"""
from .reader import (
    HuggingFaceReader,
    RegistryNotFound,
    RegistryReader,
    RegistryReadError,
    build_reader,
)
from .poller import RegistryUpdatePoller
from .wiring import build_poller

__all__ = [
    "RegistryUpdatePoller", "build_poller",
    # read-side port + adapters
    "RegistryReader", "HuggingFaceReader", "build_reader",
    "RegistryReadError", "RegistryNotFound",
]
