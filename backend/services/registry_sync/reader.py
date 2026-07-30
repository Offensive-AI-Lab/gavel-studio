"""RegistryReader — the backend's read-side port for the public library.

The backend reads the registry (index.json, rule/CE dataset files) through
THIS interface, never a storage SDK directly. So the storage backend is a
swappable detail behind one adapter — zero changes to the sync logic
(Open/Closed + Dependency Inversion).

The active adapter targets the gavel-rules GitHub repository. Reads are
anonymous once the repo is public; while it is private, GITHUB_TOKEN in
backend/.env authenticates them. This adapter READS ONLY — the studio never
writes to the repo (contributions go by pull request, outside the studio).
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
    (e.g. 'index.json', 'ces/<name>/excitation.json',
    'rules/<name>/tests/positive.json'). An adapter implements two methods;
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
# The concrete reader: GitHub. head_version() is the branch HEAD commit SHA
# (one cheap API call — the freshness probe the poller runs every few
# minutes); file fetches pin to a revision so one sync run reads a
# consistent snapshot even if the branch advances mid-sync.
# --------------------------------------------------------------------------- #
class GitHubReader(RegistryReader):
    name = "github"

    API = "https://api.github.com"

    def __init__(self, repo: str, ref: str = "main", *, token: Optional[str] = None):
        self.repo = repo            # "owner/name"
        self.ref = ref              # branch, tag, or commit SHA
        self._token = token         # None = anonymous (public repo)

    def _headers(self, accept: str) -> dict:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "gavel-studio-sync",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def head_version(self) -> str:
        import requests
        try:
            resp = requests.get(
                f"{self.API}/repos/{self.repo}/commits/{self.ref}",
                headers=self._headers("application/vnd.github+json"),
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["sha"]
        except Exception as e:
            raise RegistryReadError(f"HEAD read failed: {e}") from e

    def fetch_bytes(self, path: str, *, revision: Optional[str] = None) -> bytes:
        import requests
        ref = revision or self.ref
        try:
            resp = requests.get(
                f"{self.API}/repos/{self.repo}/contents/{path}",
                params={"ref": ref},
                headers=self._headers("application/vnd.github.raw"),
                timeout=120,
            )
        except Exception as e:
            raise RegistryReadError(f"reading '{path}' failed: {e}") from e
        if resp.status_code == 404:
            # GitHub also answers 404 for a private repo without (valid)
            # credentials — surface that hint, it is the #1 setup mistake.
            raise RegistryNotFound(
                f"{path} (404 — file missing at {ref}, or the repo is "
                f"private and GITHUB_TOKEN is absent/invalid)"
            )
        if resp.status_code != 200:
            raise RegistryReadError(
                f"reading '{path}' failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        return resp.content


# --------------------------------------------------------------------------- #
# The active reader. To move to a different storage backend, return a different
# adapter here — one line, one place.
# --------------------------------------------------------------------------- #
def build_reader() -> RegistryReader:
    """The registry reader the backend uses.

    Env (all optional once the repo is public):
      GAVEL_RULES_REPO  owner/name of the registry repo
      GAVEL_RULES_REF   branch/tag/SHA to sync from (default main)
      GITHUB_TOKEN      read access while the repo is private; also lifts the
                        anonymous API rate limit (60/h -> 5000/h)
    """
    token = (os.getenv("GITHUB_TOKEN") or "").strip() or None
    return GitHubReader(
        os.getenv("GAVEL_RULES_REPO", "Offensive-AI-Lab/gavel-rules").strip(),
        (os.getenv("GAVEL_RULES_REF") or "main").strip() or "main",
        token=token,
    )
