"""Pytest config for the central-server suite.

Puts the central-server root on sys.path so `from app...` resolves when pytest
runs from this directory, and pins the couple of env knobs the tests assert
against.

There is no auth on this server (login was removed project-wide), so there is
no JWT secret to seed. These are pure-logic tests: none touch a database — the
pool is built lazily on first query, and the endpoints exercised here
(/health, /, the middleware chain, /hf/commit with HfApi mocked) never issue
one.
"""
import os
import sys
from pathlib import Path

# 1. import path -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 2. deterministic secrets / config, set BEFORE app import -----------------
# Default to exactly one trusted proxy hop (the production default) so the
# X-Forwarded-For tests assert against the real behaviour. Individual tests
# monkeypatch rate_limit._TRUSTED_HOPS when they need a different value.
os.environ.setdefault("TRUSTED_PROXY_HOPS", "1")
# Keep the control-plane background watcher OFF during tests — no thread, no DB,
# no HF calls on app startup. Tests drive the watcher / routes directly instead.
os.environ.setdefault("ENABLE_CONTROL_PLANE", "0")
