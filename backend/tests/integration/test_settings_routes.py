"""Integration tests for GET/PUT /settings/openai-key.

Proves the router is mounted under the /settings prefix and that the two calls
the "set your key" prompt makes behave over HTTP. The writer is pointed at
tmp_path in every test that saves — the real backend/.env is never touched.
"""
import pytest

import routes.settings as settings
from utils import openai_key


@pytest.fixture
def sandboxed_env(tmp_path, monkeypatch):
    """.env under tmp_path, no provider round-trip, restored OPENAI_API_KEY."""
    env = tmp_path / ".env"
    monkeypatch.setattr(openai_key, "ENV_PATH", env)
    monkeypatch.setattr(openai_key, "ENV_EXAMPLE_PATH", tmp_path / ".env.example")
    monkeypatch.setattr(settings, "_provider_rejects", lambda key: False)
    monkeypatch.setenv(openai_key.ENV_VAR, "")
    return env


class TestOpenAiKeyStatus:
    def test_reports_missing(self, client, monkeypatch):
        monkeypatch.delenv(openai_key.ENV_VAR, raising=False)
        res = client.get("/settings/openai-key")
        assert res.status_code == 200
        assert res.json() == {"configured": False}

    def test_reports_configured_and_never_returns_the_key(self, client, monkeypatch):
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-do-not-leak")
        res = client.get("/settings/openai-key")
        assert res.status_code == 200
        assert res.json() == {"configured": True}
        assert "sk-do-not-leak" not in res.text

    def test_no_auth_is_allowed(self, client):
        res = client.get("/settings/openai-key")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


class TestSaveOpenAiKey:
    def test_saves_and_takes_effect_immediately(self, client, sandboxed_env):
        res = client.put("/settings/openai-key", json={"api_key": "sk-live"})
        assert res.status_code == 200
        assert res.json() == {"configured": True}
        assert sandboxed_env.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-live\n"
        # No restart: the very next request already sees the key.
        assert client.get("/settings/openai-key").json() == {"configured": True}

    def test_blank_key_400(self, client, sandboxed_env):
        res = client.put("/settings/openai-key", json={"api_key": "   "})
        assert res.status_code == 400
        assert isinstance(res.json()["detail"], str)
        assert not sandboxed_env.exists()

    def test_missing_field_400(self, client, sandboxed_env):
        # Validated in the handler, not by pydantic: a schema-level constraint
        # would put the submitted key in the 422 body's `input` field.
        res = client.put("/settings/openai-key", json={})
        assert res.status_code == 400
        assert isinstance(res.json()["detail"], str)
        assert not sandboxed_env.exists()

    def test_oversized_key_is_refused_without_echoing_it(self, client, sandboxed_env):
        secret = "sk-" + ("x" * 4000)
        res = client.put("/settings/openai-key", json={"api_key": secret})
        assert res.status_code == 400
        assert secret not in res.text
        assert not sandboxed_env.exists()

    def test_refused_key_400_and_not_stored(self, client, sandboxed_env, monkeypatch):
        monkeypatch.setattr(settings, "_provider_rejects", lambda key: True)
        res = client.put("/settings/openai-key", json={"api_key": "sk-wrong"})
        assert res.status_code == 400
        assert isinstance(res.json()["detail"], str)
        assert not sandboxed_env.exists()
        assert client.get("/settings/openai-key").json() == {"configured": False}
