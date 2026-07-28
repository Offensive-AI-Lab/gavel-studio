"""RegistryReader — the backend's read-side port for the public library.

The backend reads the registry (manifest, record files, neutral corpus) through
THIS interface, never a storage SDK directly. So the storage backend is a
swappable detail behind one adapter — zero changes to the sync logic
(Open/Closed + Dependency Inversion).

Bulk downloads still stream straight from the storage CDN to this backend;
nothing is proxied through, or stored on, the central server.
"""
from __future__ import annotations

import abc
import json
import os
from typing import Optional


class RegistryReadError(Exception):
    """A registry read failed at the transport level."""


class RegistryNotFound(RegistryReadError):
    """The requested path does not exist in the registry at this revision."""


class RegistryReader(abc.ABC):
    """Read-only access to the public library, addressed by repo-relative path
    (e.g. 'manifest.json', 'public_rules/<id>.json',
    'neutral/<category>/conversations.json'). An adapter implements two methods;
    `fetch_json` is provided for free."""

    name: str = "registry"

    @abc.abstractmethod
    def head_version(self) -> str:
        """Current version token of the whole registry (a commit SHA, etc.)."""

    @abc.abstractmethod
    def fetch_bytes(self, path: str, *, revision: Optional[str] = None) -> bytes:
        """Raw bytes of one file. Raise RegistryNotFound if absent,
        RegistryReadError on any other transport failure."""

    # ---- convenience (built on fetch_bytes; adapters needn't override) ----
    def fetch_json(self, path: str, *, revision: Optional[str] = None) -> dict:
        return json.loads(self.fetch_bytes(path, revision=revision).decode("utf-8"))


# --------------------------------------------------------------------------- #
# The concrete reader. Bytes stream straight from the storage CDN to this
# backend (not via the central server). Reads are anonymous (public repo).
# --------------------------------------------------------------------------- #
class HuggingFaceReader(RegistryReader):
    name = "huggingface"

    def __init__(self, repo_id: str, repo_type: str = "dataset",
                 *, token: Optional[str] = None):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self._token = token            # None = anonymous public read

    def head_version(self) -> str:
        from huggingface_hub import HfApi
        try:
            return HfApi(token=self._token).repo_info(
                repo_id=self.repo_id, repo_type=self.repo_type).sha
        except Exception as e:
            raise RegistryReadError(f"HEAD read failed: {e}") from e

    def fetch_bytes(self, path: str, *, revision: Optional[str] = None) -> bytes:
        from huggingface_hub import hf_hub_download
        try:
            local = hf_hub_download(
                repo_id=self.repo_id, repo_type=self.repo_type,
                filename=path, revision=revision, token=self._token)
        except Exception as e:
            if _is_not_found(e):
                raise RegistryNotFound(path) from e
            raise RegistryReadError(f"reading '{path}' failed: {e}") from e
        with open(local, "rb") as f:
            return f.read()


def _is_not_found(exc: Exception) -> bool:
    msg = str(exc)
    return ("404" in msg or "EntryNotFound" in type(exc).__name__
            or "not found" in msg.lower())


# --------------------------------------------------------------------------- #
# The active reader. To move to a different storage backend, return a different
# adapter here — one line, one place.
# --------------------------------------------------------------------------- #
def build_reader() -> RegistryReader:
    """The registry reader the backend uses.

    The repo ref is resolved at build time (not import time) with the same
    HF_REPO_ID / HF_REPO_TYPE overrides as hf_sync.REPO_ID — reads and writes
    must always target the SAME registry. (Not imported from hf_sync because
    hf_sync imports this module.)"""
    return HuggingFaceReader(
        os.getenv("HF_REPO_ID", "GavelPublicData/public-library"),
        os.getenv("HF_REPO_TYPE", "dataset"),
    )
