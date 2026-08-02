"""Preprocessing utilities for GAVEL.

LOCAL MODIFICATION: package-level re-exports removed for the same
reason as evaluation/__init__.py — they pull transformers eagerly via
gavel.training.utils, which makes the test suite unable to import
classifier_engine.reference without a fully-initialised LLM stack. Callers
import from submodules directly (`from gavel.preprocessing.utils import
build_assistant_windows`).
"""
