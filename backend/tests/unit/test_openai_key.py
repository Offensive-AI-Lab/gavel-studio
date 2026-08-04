"""Unit tests for the OpenAI key: detection, the contract error, the auth-error
classifier, the .env writer, and the two settings routes.

Every test that writes points `openai_key.ENV_PATH` at tmp_path — the real
backend/.env is never opened, read or written here.
"""
import os
import sys
import types

import pytest
from fastapi import HTTPException

from utils import openai_key
import routes.settings as settings


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Isolate the writer: .env and .env.example both live under tmp_path, and
    the process env var is restored after the test."""
    env = tmp_path / ".env"
    example = tmp_path / ".env.example"
    monkeypatch.setattr(openai_key, "ENV_PATH", env)
    monkeypatch.setattr(openai_key, "ENV_EXAMPLE_PATH", example)
    monkeypatch.setenv(openai_key.ENV_VAR, "")
    return env


# ---------------------------------------------------------------------------
# is_configured / require_key — the contract error
# ---------------------------------------------------------------------------

class TestRequireKey:
    def test_unset_is_not_configured(self, monkeypatch):
        monkeypatch.delenv(openai_key.ENV_VAR, raising=False)
        assert openai_key.is_configured() is False

    @pytest.mark.parametrize("value", ["", "   ", "\n"])
    def test_blank_is_not_configured(self, monkeypatch, value):
        monkeypatch.setenv(openai_key.ENV_VAR, value)
        assert openai_key.is_configured() is False

    def test_present_is_configured(self, monkeypatch):
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-test")
        assert openai_key.is_configured() is True

    def test_require_key_raises_the_contract_error(self, monkeypatch):
        monkeypatch.delenv(openai_key.ENV_VAR, raising=False)
        with pytest.raises(HTTPException) as exc:
            openai_key.require_key()
        assert exc.value.status_code == 503
        assert exc.value.detail["code"] == "OPENAI_KEY_MISSING"
        # The frontend keys on `code`; the message is for the human reading it.
        assert isinstance(exc.value.detail["message"], str)
        assert exc.value.detail["message"].strip()

    def test_require_key_is_silent_when_configured(self, monkeypatch):
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-test")
        assert openai_key.require_key() is None


# ---------------------------------------------------------------------------
# is_auth_error — "the provider refused the credential", not "something broke"
# ---------------------------------------------------------------------------

class AuthenticationError(Exception):
    """Stands in for litellm/openai's AuthenticationError (matched by name, so
    the classifier never has to import litellm just to check)."""


class _StatusError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class TestIsAuthError:
    def test_authentication_error_type(self):
        assert openai_key.is_auth_error(AuthenticationError("nope")) is True

    def test_subclass_of_authentication_error(self):
        class Wrapped(AuthenticationError):
            pass
        assert openai_key.is_auth_error(Wrapped("nope")) is True

    def test_401_status(self):
        assert openai_key.is_auth_error(_StatusError("boom", 401)) is True

    @pytest.mark.parametrize("message", [
        "litellm.AuthenticationError: OpenAIException - Missing credentials",
        "Error code: 401 - Incorrect API key provided",
        "invalid_api_key",
        "OPENAI_API_KEY is not configured",
    ])
    def test_rewrapped_message_still_classified(self, message):
        # Several call paths re-wrap the provider error in a plain RuntimeError
        # before a route sees it, so the text has to be readable too.
        assert openai_key.is_auth_error(RuntimeError(message)) is True

    def test_plain_error_string_is_accepted(self):
        # Helpers that report {"success": False, "error": "..."} instead of raising.
        assert openai_key.is_auth_error("LLM error: AuthenticationError - no api key") is True

    @pytest.mark.parametrize("exc", [
        None,
        RuntimeError("Rate limit reached for gpt-4.1"),
        TimeoutError("request timed out"),
        _StatusError("service unavailable", 503),
        ValueError("model gpt-9 does not exist"),
    ])
    def test_other_failures_are_not_auth_errors(self, exc):
        assert openai_key.is_auth_error(exc) is False


# ---------------------------------------------------------------------------
# save_key — writes backend/.env, keeps everything else, applies immediately
# ---------------------------------------------------------------------------

class TestSaveKey:
    def test_creates_env_from_example(self, env_file, monkeypatch):
        openai_key.ENV_EXAMPLE_PATH.write_text(
            "# Copy to .env\n"
            "# GITHUB_TOKEN=\n"
            "OPENAI_API_KEY=\n"
            "# DB_PATH=db/gavel.sqlite3\n",
            encoding="utf-8",
        )
        assert not env_file.exists()

        openai_key.save_key("sk-new")

        assert env_file.read_text(encoding="utf-8") == (
            "# Copy to .env\n"
            "# GITHUB_TOKEN=\n"
            "OPENAI_API_KEY=sk-new\n"
            "# DB_PATH=db/gavel.sqlite3\n"
        )
        # The example itself is a template — it must not be written to.
        assert "sk-new" not in openai_key.ENV_EXAMPLE_PATH.read_text(encoding="utf-8")

    def test_creates_env_when_nothing_exists(self, env_file):
        openai_key.save_key("sk-solo")
        assert env_file.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-solo\n"

    def test_replaces_existing_line_and_leaves_the_rest_alone(self, env_file):
        env_file.write_text(
            "# my notes\n"
            "GAVEL_RULES_REF=main\n"
            "OPENAI_API_KEY=sk-old\n"
            "GPU_WORKER_URL=https://box\n"
            "\n"
            "# trailing comment\n",
            encoding="utf-8",
        )

        openai_key.save_key("sk-fresh")

        assert env_file.read_text(encoding="utf-8") == (
            "# my notes\n"
            "GAVEL_RULES_REF=main\n"
            "OPENAI_API_KEY=sk-fresh\n"
            "GPU_WORKER_URL=https://box\n"
            "\n"
            "# trailing comment\n"
        )

    def test_appends_when_the_key_is_absent(self, env_file):
        env_file.write_text("DB_PATH=db/gavel.sqlite3\n", encoding="utf-8")
        openai_key.save_key("sk-appended")
        assert env_file.read_text(encoding="utf-8") == (
            "DB_PATH=db/gavel.sqlite3\n"
            "OPENAI_API_KEY=sk-appended\n"
        )

    def test_commented_out_line_is_not_treated_as_the_setting(self, env_file):
        # Same convention as run.sh's set_env: a commented line is not a value.
        env_file.write_text("# OPENAI_API_KEY=sk-disabled\n", encoding="utf-8")
        openai_key.save_key("sk-real")
        assert env_file.read_text(encoding="utf-8") == (
            "# OPENAI_API_KEY=sk-disabled\n"
            "OPENAI_API_KEY=sk-real\n"
        )

    def test_writes_only_the_first_match(self, env_file):
        env_file.write_text(
            "OPENAI_API_KEY=one\nOTHER=x\nOPENAI_API_KEY=two\n", encoding="utf-8"
        )
        openai_key.save_key("sk-only")
        assert env_file.read_text(encoding="utf-8") == (
            "OPENAI_API_KEY=sk-only\nOTHER=x\nOPENAI_API_KEY=two\n"
        )

    def test_applies_to_the_running_process(self, env_file):
        openai_key.save_key("  sk-trimmed  ")
        assert os.environ[openai_key.ENV_VAR] == "sk-trimmed"
        assert openai_key.is_configured() is True
        assert env_file.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-trimmed\n"

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_blank_is_rejected_and_changes_nothing(self, env_file, value):
        env_file.write_text("OPENAI_API_KEY=sk-old\n", encoding="utf-8")
        with pytest.raises(ValueError):
            openai_key.save_key(value)
        assert env_file.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-old\n"
        assert os.environ[openai_key.ENV_VAR] == ""

    def test_a_multiline_paste_cannot_write_a_second_setting(self, env_file):
        with pytest.raises(ValueError):
            openai_key.save_key("sk-good\nDB_PATH=/elsewhere.sqlite3")
        assert not env_file.exists()
        assert os.environ[openai_key.ENV_VAR] == ""

    def test_leaves_no_temp_file_behind(self, env_file, tmp_path):
        openai_key.save_key("sk-clean")
        assert [p.name for p in tmp_path.iterdir()] == [".env"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits only")
    def test_file_is_owner_only(self, env_file):
        openai_key.save_key("sk-secret")
        assert (env_file.stat().st_mode & 0o777) == 0o600

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits only")
    def test_an_existing_file_keeps_the_permissions_it_had(self, env_file):
        # os.replace swaps in a new inode. If the write reset the mode, the
        # documented Docker run (root in the container, ./backend bind-mounted)
        # would hand the operator back a .env their own account can't read.
        env_file.write_text("OPENAI_API_KEY=sk-old\n", encoding="utf-8")
        os.chmod(env_file, 0o640)
        openai_key.save_key("sk-secret")
        assert (env_file.stat().st_mode & 0o777) == 0o640

    @pytest.mark.skipif(os.name == "nt", reason="POSIX ownership only")
    def test_an_existing_file_keeps_its_owner(self, env_file, monkeypatch):
        env_file.write_text("OPENAI_API_KEY=sk-old\n", encoding="utf-8")
        before = env_file.stat()
        calls = []
        real_chown = os.chown
        monkeypatch.setattr(
            os, "chown", lambda p, uid, gid: calls.append((uid, gid)) or real_chown(p, uid, gid)
        )

        openai_key.save_key("sk-secret")

        assert calls == [(before.st_uid, before.st_gid)]
        after = env_file.stat()
        assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX ownership only")
    def test_an_unprivileged_chown_failure_does_not_fail_the_save(self, env_file, monkeypatch):
        # Restoring another user's ownership needs privilege we may not have.
        # Losing that is not a reason to refuse to store the key.
        env_file.write_text("OPENAI_API_KEY=sk-old\n", encoding="utf-8")

        def denied(*_args):
            raise PermissionError("not owner")

        monkeypatch.setattr(os, "chown", denied)
        openai_key.save_key("sk-secret")
        assert env_file.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-secret\n"

    def test_a_new_file_does_not_inherit_anything(self, env_file, monkeypatch):
        # Nothing to preserve when we create the file, so no chown attempt and
        # the owner-only default stands (asserted in test_file_is_owner_only).
        calls = []
        monkeypatch.setattr(os, "chown", lambda *a: calls.append(a))
        openai_key.save_key("sk-brand-new")
        assert calls == []

    def test_the_temp_file_is_unique_per_call(self, env_file, monkeypatch):
        # Sync routes run in a threadpool: two saves can be inside the writer at
        # once, and a pid-keyed temp name would have them writing the same file.
        seen = []
        real_replace = os.replace
        monkeypatch.setattr(
            os, "replace", lambda src, dst: seen.append(src) or real_replace(src, dst)
        )
        openai_key.save_key("sk-one")
        openai_key.save_key("sk-two")
        assert len(set(seen)) == 2

    def test_concurrent_saves_all_succeed(self, env_file):
        import threading

        env_file.write_text("KEEP=me\nOPENAI_API_KEY=sk-old\n", encoding="utf-8")
        start = threading.Barrier(8)
        failures = []

        def save(i):
            start.wait()
            try:
                openai_key.save_key(f"sk-{i}")
            except Exception as e:  # noqa: BLE001 - the assertion is "none of these"
                failures.append(e)

        threads = [threading.Thread(target=save, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert failures == []
        text = env_file.read_text(encoding="utf-8")
        assert text.startswith("KEEP=me\nOPENAI_API_KEY=sk-")
        assert [p.name for p in env_file.parent.iterdir()] == [".env"]


# ---------------------------------------------------------------------------
# GET / PUT /settings/openai-key
# ---------------------------------------------------------------------------

class TestSettingsRoutes:
    def test_status_reports_missing(self, monkeypatch):
        monkeypatch.delenv(openai_key.ENV_VAR, raising=False)
        assert settings.get_openai_key_status() == {"configured": False}

    def test_status_reports_configured_without_leaking_the_key(self, monkeypatch):
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-secret")
        body = settings.get_openai_key_status()
        assert body == {"configured": True}
        assert "sk-secret" not in str(body)

    def test_put_saves_and_applies(self, env_file, monkeypatch):
        monkeypatch.setattr(settings, "_provider_rejects", lambda key: False)
        body = settings.put_openai_key(settings.OpenAiKeyRequest(api_key="sk-put"))
        assert body == {"configured": True}
        assert os.environ[openai_key.ENV_VAR] == "sk-put"
        assert env_file.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-put\n"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_put_rejects_blank_with_400(self, env_file, monkeypatch, value):
        monkeypatch.setattr(
            settings, "_provider_rejects",
            lambda key: pytest.fail("must not call the provider for a blank key"),
        )
        with pytest.raises(HTTPException) as exc:
            settings.put_openai_key(settings.OpenAiKeyRequest(api_key=value))
        assert exc.value.status_code == 400
        assert isinstance(exc.value.detail, str)
        assert not env_file.exists()

    def test_put_rejects_a_key_the_provider_refuses(self, env_file, monkeypatch):
        monkeypatch.setattr(settings, "_provider_rejects", lambda key: True)
        with pytest.raises(HTTPException) as exc:
            settings.put_openai_key(settings.OpenAiKeyRequest(api_key="sk-bad"))
        assert exc.value.status_code == 400
        assert isinstance(exc.value.detail, str)
        # A refused key is not stored, so the next request still reports missing.
        assert not env_file.exists()
        assert os.environ[openai_key.ENV_VAR] == ""

    def test_put_rejects_an_over_long_paste_without_quoting_it(self, env_file, monkeypatch):
        # Length is enforced here, not by the schema: a pydantic 422 would send
        # the whole `input` back, and the input is the operator's key.
        monkeypatch.setattr(
            settings, "_provider_rejects",
            lambda key: pytest.fail("must not call the provider for an over-long value"),
        )
        pasted = "sk-" + "x" * settings._MAX_KEY_LENGTH
        with pytest.raises(HTTPException) as exc:
            settings.put_openai_key(settings.OpenAiKeyRequest(api_key=pasted))
        assert exc.value.status_code == 400
        assert pasted not in str(exc.value.detail)
        assert "x" * 20 not in str(exc.value.detail)
        assert not env_file.exists()

    def test_put_accepts_a_key_at_the_limit(self, env_file, monkeypatch):
        monkeypatch.setattr(settings, "_provider_rejects", lambda key: False)
        key = "s" * settings._MAX_KEY_LENGTH
        assert settings.put_openai_key(settings.OpenAiKeyRequest(api_key=key)) == {"configured": True}

    def test_put_treats_a_non_string_as_no_key(self, env_file, monkeypatch):
        monkeypatch.setattr(
            settings, "_provider_rejects",
            lambda key: pytest.fail("must not call the provider for a non-string"),
        )
        with pytest.raises(HTTPException) as exc:
            settings.put_openai_key(settings.OpenAiKeyRequest(api_key={"api_key": "sk-hidden"}))
        assert exc.value.status_code == 400
        assert "sk-hidden" not in str(exc.value.detail)

    def test_no_rejection_repeats_the_key(self, env_file, monkeypatch):
        monkeypatch.setattr(settings, "_provider_rejects", lambda key: True)
        secret = "sk-never-echo-me"
        with pytest.raises(HTTPException) as exc:
            settings.put_openai_key(settings.OpenAiKeyRequest(api_key=secret))
        assert secret not in str(exc.value.detail)

    def test_put_reports_an_unwritable_env_file(self, env_file, monkeypatch):
        monkeypatch.setattr(settings, "_provider_rejects", lambda key: False)

        def boom(value):
            raise OSError("read-only file system")

        monkeypatch.setattr(openai_key, "save_key", boom)
        with pytest.raises(HTTPException) as exc:
            settings.put_openai_key(settings.OpenAiKeyRequest(api_key="sk-put"))
        assert exc.value.status_code == 500


class TestProviderCheck:
    def _fake_litellm(self, monkeypatch, error):
        module = types.ModuleType("litellm")

        def completion(**kwargs):
            if error:
                raise error
            return {"ok": True}

        module.completion = completion
        monkeypatch.setitem(sys.modules, "litellm", module)

    def test_accepts_a_key_the_provider_honours(self, monkeypatch):
        self._fake_litellm(monkeypatch, None)
        assert settings._provider_rejects("sk-good") is False

    def test_rejects_only_on_refusal(self, monkeypatch):
        self._fake_litellm(monkeypatch, AuthenticationError("Missing credentials"))
        assert settings._provider_rejects("sk-bad") is True

    def test_an_outage_does_not_block_saving(self, monkeypatch):
        # No network / rate limit / provider down is not the operator's fault:
        # a key they say is good must still be storable.
        self._fake_litellm(monkeypatch, ConnectionError("name resolution failed"))
        assert settings._provider_rejects("sk-unknown") is False
