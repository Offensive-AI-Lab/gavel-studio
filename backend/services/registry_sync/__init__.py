"""Registry sync — the backend's link to the public library registry.

The backend ingests the public library into its DB via
`services.library_sync.sync_library`. This package adds:
  * the read-side port (`RegistryReader` + the GitHub adapter for the
    gavel-rules repository), and
  * the freshness trigger: a periodic poller that probes the repo's HEAD
    commit and drives the sidebar's "Updates available" badge.

Entry point: `build_poller()` (started in the backend's lifespan).
"""
from .reader import (
    GitHubReader,
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
    "RegistryReader", "GitHubReader", "build_reader",
    "RegistryReadError", "RegistryNotFound",
]
