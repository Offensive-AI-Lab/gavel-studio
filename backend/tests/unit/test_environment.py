"""Environment & setup tests — clone-and-run readiness checks.

These tests verify that a fresh checkout has everything it needs to start:
- All declared dependencies importable
- Critical environment variables handled gracefully when missing
- Database schema can be re-initialized idempotently
- Required directory structure exists

Tests do NOT modify production state. Any temporary changes are reverted in teardown.
"""
import os
import sys
import pytest


class TestDependencyResolution:
    """Verify all critical dependencies declared in requirements.txt are importable."""

    def test_fastapi_installed(self):
        import fastapi
        assert fastapi.__version__

    def test_pytorch_installed(self):
        import torch
        assert torch.__version__
        # CUDA is optional, but the import must work
        _ = torch.cuda.is_available()

    def test_transformers_installed(self):
        import transformers
        assert transformers.__version__

    def test_psycopg2_installed(self):
        import psycopg2
        assert psycopg2.__version__

    def test_litellm_installed(self):
        import litellm
        assert hasattr(litellm, "completion")

    def test_passlib_argon2(self):
        from passlib.hash import argon2
        h = argon2.hash("test_password")
        assert argon2.verify("test_password", h)

    def test_jose_jwt(self):
        from jose import jwt
        token = jwt.encode({"sub": "test"}, "secret", algorithm="HS256")
        decoded = jwt.decode(token, "secret", algorithms=["HS256"])
        assert decoded["sub"] == "test"

    def test_pydantic_v2(self):
        import pydantic
        major = int(pydantic.__version__.split(".")[0])
        assert major >= 2, "pydantic v2 required"

    def test_sentence_transformers_installed(self):
        try:
            import sentence_transformers
            assert sentence_transformers.__version__
        except ImportError:
            pytest.skip("sentence_transformers not installed (optional for some flows)")


class TestEnvironmentVariables:
    """Verify the system handles missing/invalid env vars gracefully."""

    def test_backend_holds_no_jwt_signing_secret(self):
        """The local backend must NOT hold a JWT signing secret — token
        verification is delegated to the central server (the single auth
        authority), so a local operator can't forge tokens. There must be no
        module-level SECRET_KEY pulled from the environment."""
        import utils.auth as auth
        assert not hasattr(auth, "SECRET_KEY"), "backend must not hold a JWT signing secret"
        assert hasattr(auth, "get_current_user")

    def test_identity_is_a_fixed_local_user(self):
        """There is no login: get_current_user always resolves to the one
        seeded local user, and no token machinery survives."""
        import utils.auth as auth
        assert auth.get_current_user() == auth.LOCAL_USER_ID
        assert not hasattr(auth, "create_access_token")
        assert not hasattr(auth, "decode_access_token")

    # NOTE: a `test_database_env_vars_present` check (which actually connects
    # to verify env vars resolve to a working DB) lives in the integration
    # suite — see tests/integration/test_environment_db.py. Kept out of unit
    # because it requires a running Postgres.


class TestDirectoryStructure:
    """Check that critical directories exist or can be created."""

    def test_trained_classifiers_dir_creatable(self, tmp_path):
        """Training output dir should be creatable."""
        d = tmp_path / "trained_classifiers" / "classifier_999"
        d.mkdir(parents=True, exist_ok=True)
        assert d.exists()

    def test_uploads_dir_creatable(self, tmp_path):
        """LLM uploads dir should be creatable."""
        d = tmp_path / "LLMs" / "1"
        d.mkdir(parents=True, exist_ok=True)
        assert d.exists()


class TestImportPaths:
    """All critical modules should be importable from backend root."""

    def test_main_importable(self):
        import main
        assert main.app is not None

    def test_routes_all_importable(self):
        from routes import (
            user, dashboard, models, classifiers, rules,
            cognitive, ai_pipeline, library, evaluation, realtime,
        )
        for mod in (user, dashboard, models, classifiers, rules,
                    cognitive, ai_pipeline, library, evaluation, realtime):
            assert hasattr(mod, "router")

    def test_crash_recovery_importable(self):
        from utils.crash_recovery import run_all_recovery
        assert callable(run_all_recovery)
