"""Unit tests for the local-LLM setting (utils/local_llm.py), the combined
"is some LLM available" gate (utils/llm_access.py), and the /settings/local-llm
routes.

Every test that writes points `local_llm.ENV_PATH` at tmp_path — the real
backend/.env is never opened, read or written here. Mirrors
tests/unit/test_openai_key.py's fixture shape; the atomic-write edge cases
(temp files, permissions, concurrency) live in that file since both modules
now share the same writer (utils/env_file.py) and there is nothing
local_llm-specific about them.
"""
import os

import pytest
from fastapi import HTTPException

from utils import llm_access, local_llm, openai_key
import routes.settings as settings


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    example = tmp_path / ".env.example"
    monkeypatch.setattr(local_llm, "ENV_PATH", env)
    monkeypatch.setattr(local_llm, "ENV_EXAMPLE_PATH", example)
    monkeypatch.setenv(local_llm.ENV_VAR, "")
    return env


@pytest.fixture
def no_openai_key(monkeypatch):
    """Isolate llm_access tests from whatever OPENAI_API_KEY is set in the
    real environment running the suite."""
    monkeypatch.setenv(openai_key.ENV_VAR, "")


# ---------------------------------------------------------------------------
# is_configured / get_model / save_model / clear_model
# ---------------------------------------------------------------------------

class TestLocalLlm:
    def test_unset_is_not_configured(self, monkeypatch):
        monkeypatch.delenv(local_llm.ENV_VAR, raising=False)
        assert local_llm.is_configured() is False
        assert local_llm.get_model() == ""

    @pytest.mark.parametrize("value", ["", "   ", "\n"])
    def test_blank_is_not_configured(self, monkeypatch, value):
        monkeypatch.setenv(local_llm.ENV_VAR, value)
        assert local_llm.is_configured() is False

    def test_present_is_configured(self, monkeypatch):
        monkeypatch.setenv(local_llm.ENV_VAR, "HuggingFaceTB/SmolLM2-135M-Instruct")
        assert local_llm.is_configured() is True
        assert local_llm.get_model() == "HuggingFaceTB/SmolLM2-135M-Instruct"

    def test_save_writes_env_and_applies(self, env_file):
        local_llm.save_model("  org/tiny-model  ")
        assert os.environ[local_llm.ENV_VAR] == "org/tiny-model"
        assert env_file.read_text(encoding="utf-8") == "GAVEL_LOCAL_LLM_MODEL=org/tiny-model\n"
        assert local_llm.is_configured() is True

    def test_save_leaves_other_settings_alone(self, env_file):
        env_file.write_text("OPENAI_API_KEY=sk-old\nDB_PATH=db/gavel.sqlite3\n", encoding="utf-8")
        local_llm.save_model("org/tiny-model")
        assert env_file.read_text(encoding="utf-8") == (
            "OPENAI_API_KEY=sk-old\nDB_PATH=db/gavel.sqlite3\nGAVEL_LOCAL_LLM_MODEL=org/tiny-model\n"
        )

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_blank_is_rejected(self, env_file, value):
        with pytest.raises(ValueError):
            local_llm.save_model(value)
        assert not env_file.exists()

    def test_multiline_paste_is_rejected(self, env_file):
        with pytest.raises(ValueError):
            local_llm.save_model("org/model\nOPENAI_API_KEY=sk-injected")
        assert not env_file.exists()

    def test_clear_removes_it(self, env_file):
        local_llm.save_model("org/tiny-model")
        local_llm.clear_model()
        assert local_llm.is_configured() is False
        assert os.environ.get(local_llm.ENV_VAR, "") == ""
        assert env_file.read_text(encoding="utf-8") == "GAVEL_LOCAL_LLM_MODEL=\n"


# ---------------------------------------------------------------------------
# llm_access — either credential satisfies the gate
# ---------------------------------------------------------------------------

class TestLlmAccess:
    def test_neither_configured_raises_the_contract_error(self, monkeypatch, no_openai_key):
        monkeypatch.delenv(local_llm.ENV_VAR, raising=False)
        assert llm_access.is_configured() is False
        with pytest.raises(HTTPException) as exc:
            llm_access.require_llm()
        assert exc.value.status_code == 503
        # Reuses the OpenAI-key contract's code on purpose — see the module
        # docstring — so the existing frontend "unlock AI" banner still fires.
        assert exc.value.detail["code"] == openai_key.MISSING_CODE

    def test_openai_key_alone_satisfies_it(self, monkeypatch, no_openai_key):
        monkeypatch.delenv(local_llm.ENV_VAR, raising=False)
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-test")
        assert llm_access.is_configured() is True
        assert llm_access.require_llm() is None

    def test_local_model_alone_satisfies_it(self, monkeypatch, no_openai_key):
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        assert llm_access.is_configured() is True
        assert llm_access.require_llm() is None

    def test_both_configured_satisfies_it(self, monkeypatch):
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-test")
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        assert llm_access.is_configured() is True


# ---------------------------------------------------------------------------
# GET / PUT / DELETE /settings/local-llm
# ---------------------------------------------------------------------------

class TestLocalLlmRoutes:
    def test_status_reports_missing(self, monkeypatch):
        monkeypatch.delenv(local_llm.ENV_VAR, raising=False)
        assert settings.get_local_llm_status() == {"configured": False, "model": ""}

    def test_status_reports_the_configured_model(self, monkeypatch):
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        assert settings.get_local_llm_status() == {"configured": True, "model": "org/tiny-model"}

    def test_put_saves_and_applies(self, env_file):
        body = settings.put_local_llm(settings.LocalLlmRequest(model="org/tiny-model"))
        assert body == {"configured": True, "model": "org/tiny-model"}
        assert os.environ[local_llm.ENV_VAR] == "org/tiny-model"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_put_rejects_blank_with_400(self, env_file, value):
        with pytest.raises(HTTPException) as exc:
            settings.put_local_llm(settings.LocalLlmRequest(model=value))
        assert exc.value.status_code == 400
        assert not env_file.exists()

    def test_put_rejects_an_over_long_path(self, env_file):
        pasted = "org/" + "x" * settings._MAX_MODEL_LENGTH
        with pytest.raises(HTTPException) as exc:
            settings.put_local_llm(settings.LocalLlmRequest(model=pasted))
        assert exc.value.status_code == 400
        assert not env_file.exists()

    def test_put_treats_a_non_string_as_no_model(self, env_file):
        with pytest.raises(HTTPException) as exc:
            settings.put_local_llm(settings.LocalLlmRequest(model={"model": "org/tiny-model"}))
        assert exc.value.status_code == 400

    def test_delete_clears_it(self, env_file):
        settings.put_local_llm(settings.LocalLlmRequest(model="org/tiny-model"))
        body = settings.delete_local_llm()
        assert body == {"configured": False, "model": ""}
        assert local_llm.is_configured() is False


# ---------------------------------------------------------------------------
# get_active_provider / set_active_provider — the "both configured" tiebreak
# ---------------------------------------------------------------------------

class TestActiveProvider:
    def test_defaults_to_openai(self, monkeypatch):
        monkeypatch.delenv(local_llm.PROVIDER_ENV_VAR, raising=False)
        assert local_llm.get_active_provider() == "openai"

    def test_an_invalid_stored_value_falls_back_to_openai(self, monkeypatch):
        monkeypatch.setenv(local_llm.PROVIDER_ENV_VAR, "garbage")
        assert local_llm.get_active_provider() == "openai"

    def test_set_and_read_back(self, env_file):
        local_llm.set_active_provider("local")
        assert local_llm.get_active_provider() == "local"
        assert os.environ[local_llm.PROVIDER_ENV_VAR] == "local"

    def test_rejects_anything_else(self):
        with pytest.raises(ValueError):
            local_llm.set_active_provider("anthropic")


# ---------------------------------------------------------------------------
# GET / PUT /settings/ai-provider
# ---------------------------------------------------------------------------

class TestAiProviderRoute:
    def test_status_reports_both_credentials_and_the_preference(self, monkeypatch):
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-test")
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        monkeypatch.setenv(local_llm.PROVIDER_ENV_VAR, "local")
        assert settings.get_ai_provider_status() == {
            "openai_configured": True,
            "local_configured": True,
            "local_model": "org/tiny-model",
            "active_provider": "local",
        }

    def test_put_switches_to_local_when_configured(self, env_file, monkeypatch):
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        body = settings.put_ai_provider(settings.AiProviderRequest(provider="local"))
        assert body == {"active_provider": "local"}
        assert local_llm.get_active_provider() == "local"

    def test_put_rejects_openai_when_no_key_set(self, monkeypatch, no_openai_key):
        with pytest.raises(HTTPException) as exc:
            settings.put_ai_provider(settings.AiProviderRequest(provider="openai"))
        assert exc.value.status_code == 400

    def test_put_rejects_local_when_no_model_set(self, monkeypatch):
        monkeypatch.delenv(local_llm.ENV_VAR, raising=False)
        with pytest.raises(HTTPException) as exc:
            settings.put_ai_provider(settings.AiProviderRequest(provider="local"))
        assert exc.value.status_code == 400

    def test_put_rejects_an_unknown_provider(self):
        with pytest.raises(HTTPException) as exc:
            settings.put_ai_provider(settings.AiProviderRequest(provider="anthropic"))
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# GET /settings/local-llm/status, POST /settings/local-llm/warmup
# ---------------------------------------------------------------------------

class TestLocalLlmLoadRoutes:
    def test_status_reports_not_configured_with_nothing_set(self, monkeypatch):
        monkeypatch.delenv(local_llm.ENV_VAR, raising=False)
        assert settings.get_local_llm_load_status() == {"state": "not_configured", "detail": None}

    def test_status_delegates_to_llm_client_for_the_configured_model(self, monkeypatch):
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        from gavel_pipeline import llm_client
        monkeypatch.setattr(llm_client, "_status", {"org/tiny-model": {"state": "loading", "detail": None}})
        assert settings.get_local_llm_load_status() == {"state": "loading", "detail": None}

    def test_warmup_requires_a_configured_model(self, monkeypatch):
        monkeypatch.delenv(local_llm.ENV_VAR, raising=False)
        with pytest.raises(HTTPException) as exc:
            settings.post_local_llm_warmup()
        assert exc.value.status_code == 400

    def test_warmup_starts_loading_the_configured_model(self, monkeypatch):
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        from gavel_pipeline import llm_client
        monkeypatch.setattr(llm_client, "start_warmup", lambda model_id: "loading")
        assert settings.post_local_llm_warmup() == {"state": "loading"}
