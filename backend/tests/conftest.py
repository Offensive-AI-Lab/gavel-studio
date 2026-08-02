"""Shared root-level pytest config.

Does the sys.path bootstrap so backend modules are importable from any test,
and installs the registry network kill-switch. Fixtures that need a database
(TestClient, test_user, snapshot/restore, etc.) live in
tests/integration/conftest.py — that way unit tests never trigger any
database connection just by being collected.
"""
import os
import sys

# Ensure backend/ root is on sys.path so `from main import app`,
# `from utils.sqlite_db import ...` etc. resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Registry network kill-switch.
#
# Tests must NEVER touch the real gavel-rules GitHub registry. But main.py
# spawns a daemon thread at import time that runs a startup library sync, the
# registry poller probes HEAD on a timer, and several routes lazily fetch
# datasets. All of that goes through ONE seam — the RegistryReader singleton
# in services.library_sync — so replacing it with a dead adapter here makes
# every registry access fail fast and offline (sync_library degrades to
# `errors=["registry unreachable: ..."]`, check_for_updates to
# `checked=False`, ensure_* to False — all handled paths).
#
# Tests that exercise the sync logic itself install their own in-memory
# FakeReader over this (tests/unit/test_library_sync.py).
# ---------------------------------------------------------------------------
import services.library_sync as _library_sync  # noqa: E402


class _DeadRegistryReader:
    name = "dead-test-reader"

    def head_version(self):
        raise RuntimeError("registry network access is disabled in tests")

    def fetch_bytes(self, path, *, revision=None):
        raise RuntimeError("registry network access is disabled in tests")

    def fetch_json(self, path, *, revision=None):
        raise RuntimeError("registry network access is disabled in tests")


_library_sync._READER = _DeadRegistryReader()
