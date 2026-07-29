"""Shared root-level pytest config.

Only does sys.path bootstrap so backend modules are importable from any test.
Fixtures that need a database (TestClient, test_user, snapshot/restore, etc.)
live in tests/integration/conftest.py — that way unit tests never trigger any
database connection just by being collected.
"""
import os
import sys

# Ensure backend/ root is on sys.path so `from main import app`,
# `from utils.sqlite_db import ...` etc. resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
