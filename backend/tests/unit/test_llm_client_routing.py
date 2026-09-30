"""Unit tests for gavel_pipeline/llm_client.py's provider routing
(`_use_local`) and load-status tracking (`get_status` / `start_warmup`).
Does not exercise real model loading or generation — that needs a real model
on a GPU, covered by manual smoke testing, not this suite.
"""
import threading

import pytest

from gavel_pipeline import llm_client
from utils import local_llm, openai_key


class _SyncThread:
    """Stand-in for threading.Thread that runs the target immediately and
    synchronously, so warmup tests don't need to poll/sleep for a real
    background thread to finish."""
    def __init__(self, target=None, daemon=None):
        self._target = target

    def start(self):
        self._target()


@pytest.fixture(autouse=True)
def clean_llm_client_state(monkeypatch):
    """Every test gets a fresh _loaded/_status — these are module-level
    caches that would otherwise leak between tests (and between this file
    and real usage)."""
    monkeypatch.setattr(llm_client, "_loaded", {})
    monkeypatch.setattr(llm_client, "_status", {})


@pytest.fixture
def no_openai_key(monkeypatch):
    monkeypatch.setenv(openai_key.ENV_VAR, "")


@pytest.fixture
def no_local_model(monkeypatch):
    monkeypatch.delenv(local_llm.ENV_VAR, raising=False)


class TestUseLocal:
    def test_neither_configured(self, no_openai_key, no_local_model):
        assert llm_client._use_local() is False

    def test_only_openai_configured(self, no_local_model, monkeypatch):
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-test")
        assert llm_client._use_local() is False

    def test_only_local_configured(self, no_openai_key, monkeypatch):
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        assert llm_client._use_local() is True

    def test_both_configured_defaults_to_openai(self, monkeypatch):
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-test")
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        monkeypatch.delenv(local_llm.PROVIDER_ENV_VAR, raising=False)
        assert llm_client._use_local() is False

    def test_both_configured_explicit_local_preference(self, monkeypatch):
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-test")
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        monkeypatch.setenv(local_llm.PROVIDER_ENV_VAR, "local")
        assert llm_client._use_local() is True

    def test_both_configured_explicit_openai_preference(self, monkeypatch):
        monkeypatch.setenv(openai_key.ENV_VAR, "sk-test")
        monkeypatch.setenv(local_llm.ENV_VAR, "org/tiny-model")
        monkeypatch.setenv(local_llm.PROVIDER_ENV_VAR, "openai")
        assert llm_client._use_local() is False


# ---------------------------------------------------------------------------
# JSON mode on a local model — extraction + regenerate-until-valid
# ---------------------------------------------------------------------------

class TestExtractJsonObject:
    def test_bare_object(self):
        assert llm_client._extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_object_with_prose_around_it(self):
        text = 'Here is the config:\n```json\n{"a": [1, 2]}\n```\nHope that helps!'
        assert llm_client._extract_json_object(text) == {"a": [1, 2]}

    def test_trailing_junk_after_the_object_is_ignored(self):
        assert llm_client._extract_json_object('{"a": 1} and then some prose') == {"a": 1}

    def test_missing_comma_is_none(self):
        # The reported failure: "Expecting ',' delimiter".
        assert llm_client._extract_json_object('{"a": 1 "b": 2}') is None

    def test_a_broken_outer_object_never_yields_a_nested_fragment(self):
        # Outer object is missing a comma; the inner object is valid on its
        # own. Returning the inner one would silently hand back a fragment.
        broken = '{"reasoning": {"why": "x"} "groups": {"g": ["ce"]}}'
        assert llm_client._extract_json_object(broken) is None

    def test_no_object_at_all_is_none(self):
        assert llm_client._extract_json_object("sorry, I can't do that") is None


class TestGenerateValidJson:
    def _gen(self, outputs):
        calls = []

        def generate(temp):
            calls.append(temp)
            return outputs[min(len(calls), len(outputs)) - 1]
        return generate, calls

    def test_valid_first_try_makes_one_call(self):
        gen, calls = self._gen(['{"ok": true}'])
        assert llm_client._generate_valid_json(gen, 0.7) == '{"ok": true}'
        assert calls == [0.7]

    def test_regenerates_at_a_lower_temperature_until_valid(self):
        gen, calls = self._gen(['{"a": 1 "b": 2}', '{"a": 1, "b": 2}'])
        assert llm_client._generate_valid_json(gen, 0.7) == '{"a": 1, "b": 2}'
        assert calls == [0.7, 0.3]

    def test_returns_only_the_object_not_the_surrounding_prose(self):
        gen, _ = self._gen(['Sure!\n```json\n{"a": 1}\n```'])
        assert llm_client._generate_valid_json(gen, 0.2) == '{"a": 1}'

    def test_gives_up_after_the_attempt_limit_and_returns_the_raw_text(self):
        gen, calls = self._gen(["not json at all"])
        assert llm_client._generate_valid_json(gen, 0.7) == "not json at all"
        assert len(calls) == llm_client._JSON_ATTEMPTS


# ---------------------------------------------------------------------------
# OpenAI path — must behave exactly as before local-LLM support existed
# ---------------------------------------------------------------------------

class TestOpenAiPathUnchanged:
    def test_messages_and_kwargs_reach_litellm_untouched(self, monkeypatch):
        import sys
        import types

        captured = {}
        fake = types.ModuleType("litellm")
        fake.completion = lambda **kw: captured.update(kw) or "resp"
        monkeypatch.setitem(sys.modules, "litellm", fake)
        monkeypatch.setattr(llm_client, "_use_local", lambda: False)

        # Shapes Mistral would reject (system mid-list, greeting first, two
        # user turns in a row) — OpenAI accepts them, so they must pass through.
        msgs = [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "system", "content": "late system"},
        ]
        out = llm_client.complete(
            messages=msgs, model="gpt-4.1", temperature=0.3,
            response_format={"type": "json_object"}, local_max_tokens=8192,
        )

        assert out == "resp"
        assert captured["messages"] == msgs
        assert captured["model"] == "gpt-4.1"
        assert captured["response_format"] == {"type": "json_object"}
        # The local-only token budget must never reach OpenAI.
        assert "local_max_tokens" not in captured
        assert "max_tokens" not in captured


# ---------------------------------------------------------------------------
# get_status / start_warmup — the loading indicator's backing state
# ---------------------------------------------------------------------------

class TestLoadStatus:
    def test_unattempted_model_is_not_loaded(self):
        assert llm_client.get_status("org/tiny-model") == {"state": "not_loaded", "detail": None}

    def test_warmup_reports_loading_then_ready(self, monkeypatch):
        monkeypatch.setattr(llm_client.threading, "Thread", _SyncThread)

        class FakeModel:
            def parameters(self):
                import torch
                yield torch.zeros(1)

        monkeypatch.setattr(
            "classifier_engine.utils_train.load_model_and_tokenizer",
            lambda model_id: (FakeModel(), object()),
        )

        state = llm_client.start_warmup("org/tiny-model")
        # _SyncThread runs the load synchronously, so by the time start_warmup
        # returns the load already finished — state reflects the END result.
        assert state == "loading"
        assert llm_client.get_status("org/tiny-model") == {
            "state": "ready", "detail": None, "device": "cpu",
        }

    def test_warmup_is_a_noop_once_already_loaded(self, monkeypatch):
        llm_client._loaded["org/tiny-model"] = (object(), object(), "cpu")
        called = []
        monkeypatch.setattr(llm_client.threading, "Thread", lambda **kw: called.append(kw) or _SyncThread(**kw))
        assert llm_client.start_warmup("org/tiny-model") == "ready"
        assert called == []

    def test_warmup_does_not_double_start_while_already_loading(self, monkeypatch):
        llm_client._status["org/tiny-model"] = {"state": "loading", "detail": None}
        called = []
        monkeypatch.setattr(llm_client.threading, "Thread", lambda **kw: called.append(kw) or _SyncThread(**kw))
        assert llm_client.start_warmup("org/tiny-model") == "loading"
        assert called == []

    def test_a_load_failure_is_recorded_as_an_error_status(self, monkeypatch):
        monkeypatch.setattr(llm_client.threading, "Thread", _SyncThread)
        monkeypatch.setattr(
            "classifier_engine.utils_train.load_model_and_tokenizer",
            lambda model_id: (_ for _ in ()).throw(RuntimeError("no such repo")),
        )
        llm_client.start_warmup("org/broken-model")
        assert llm_client.get_status("org/broken-model") == {"state": "error", "detail": "no such repo"}

    def test_progress_is_captured_while_loading(self, monkeypatch):
        # A tqdm bar ticking during the load should show up as `progress`
        # before the load finishes (get_status is read mid-load here, not
        # after — unlike the other tests, this doesn't use _SyncThread).
        llm_client._status["org/tiny-model"] = {"state": "loading", "detail": None}
        with llm_client._track_progress("org/tiny-model"):
            import tqdm
            bar = tqdm.tqdm(total=4, desc="Fetching 4 files")
            bar.update(1)
            bar.update(1)
        assert llm_client.get_status("org/tiny-model")["progress"] == {
            "desc": "Fetching 4 files", "n": 2, "total": 4, "percent": 50,
        }

    def test_progress_is_ignored_once_no_longer_loading(self, monkeypatch):
        # A stray/late tqdm tick after the load moved on (e.g. a bar from a
        # previous attempt still flushing) must not resurrect stale progress.
        llm_client._status["org/tiny-model"] = {"state": "ready", "detail": None, "device": "cpu"}
        with llm_client._track_progress("org/tiny-model"):
            import tqdm
            tqdm.tqdm(total=4, desc="late").update(1)
        assert "progress" not in llm_client.get_status("org/tiny-model")


# ---------------------------------------------------------------------------
# _normalize_messages — adapting OpenAI-shaped message lists to the strict
# alternation some local chat templates (Mistral's) enforce
# ---------------------------------------------------------------------------

class TestNormalizeMessages:
    def test_already_alternating_is_unchanged(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        assert llm_client._normalize_messages(msgs) == msgs

    def test_a_trailing_system_message_merges_into_the_preceding_user_turn(self):
        # The want_json branch in _build_prompt appends a system instruction
        # after the real conversation — this is exactly that shape.
        msgs = [
            {"role": "user", "content": "generate a rule"},
            {"role": "system", "content": "Respond with only valid JSON."},
        ]
        assert llm_client._normalize_messages(msgs) == [
            {"role": "user", "content": "generate a rule\n\nRespond with only valid JSON."},
        ]

    def test_consecutive_user_messages_are_merged_not_dropped(self):
        # rule_generator.call_thinking_model turns a leading "system" into
        # "user" for reasoning-model compatibility, which can land right
        # next to an existing "user" turn — reproduces the reported bug.
        msgs = [
            {"role": "user", "content": "You are a rule-writing assistant."},
            {"role": "user", "content": "Here is the scenario: ..."},
        ]
        assert llm_client._normalize_messages(msgs) == [
            {"role": "user", "content": "You are a rule-writing assistant.\n\nHere is the scenario: ..."},
        ]

    def test_a_system_message_after_the_first_position_is_downgraded_to_user(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "mid-conversation instruction"},
            {"role": "assistant", "content": "ok"},
        ]
        assert llm_client._normalize_messages(msgs) == [
            {"role": "user", "content": "hi\n\nmid-conversation instruction"},
            {"role": "assistant", "content": "ok"},
        ]

    def test_only_the_first_message_may_stay_system(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "system", "content": "also-first-ish but not index 0"},
        ]
        result = llm_client._normalize_messages(msgs)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_a_leading_scripted_greeting_before_any_user_turn_is_dropped(self):
        # Reproduces ai_pipeline.py's scenario/CE chat seed shape:
        # [system, assistant: <static greeting>] — sent to the LLM as soon
        # as the operator's first message arrives. A strict template
        # requires the turn right after system to be "user"; there's no
        # user turn for the greeting to attach to, so it's dropped rather
        # than raising.
        msgs = [
            {"role": "system", "content": "You are a scenario-writing assistant."},
            {"role": "assistant", "content": "Hi! Describe the problematic AI behavior…"},
            {"role": "user", "content": "Detect fake tech-support pressure."},
        ]
        assert llm_client._normalize_messages(msgs) == [
            {"role": "system", "content": "You are a scenario-writing assistant."},
            {"role": "user", "content": "Detect fake tech-support pressure."},
        ]

    def test_an_assistant_turn_after_the_first_user_message_is_kept(self):
        msgs = [
            {"role": "system", "content": "You are a scenario-writing assistant."},
            {"role": "assistant", "content": "Hi! Describe the problematic AI behavior…"},
            {"role": "user", "content": "Detect fake tech-support pressure."},
            {"role": "assistant", "content": "Got it, any more detail on the pressure tactic?"},
            {"role": "user", "content": "Urgency and a fake countdown."},
        ]
        assert llm_client._normalize_messages(msgs) == [
            {"role": "system", "content": "You are a scenario-writing assistant."},
            {"role": "user", "content": "Detect fake tech-support pressure."},
            {"role": "assistant", "content": "Got it, any more detail on the pressure tactic?"},
            {"role": "user", "content": "Urgency and a fake countdown."},
        ]

    def test_a_leading_greeting_with_no_system_message_is_also_dropped(self):
        msgs = [
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "hello"},
        ]
        assert llm_client._normalize_messages(msgs) == [
            {"role": "user", "content": "hello"},
        ]
