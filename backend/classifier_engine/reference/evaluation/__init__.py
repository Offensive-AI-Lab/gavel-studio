"""Evaluation utilities for GAVEL.

LOCAL MODIFICATION (kept minimal — re-sync with care):
  The upstream __init__.py eagerly re-exports symbols from every
  submodule, which transitively imports torch + transformers at
  package-load time. Our integration tests + adapter access these
  submodules directly (`from gavel.evaluation.calibration import
  calibrate`), so we don't need the re-exports. Removing them keeps
  the eager import surface small enough that the test suite can load
  the package in isolation.

  When re-syncing from a newer upstream commit, preserve this thin
  __init__.py — the upstream version is the heavy one.
"""
