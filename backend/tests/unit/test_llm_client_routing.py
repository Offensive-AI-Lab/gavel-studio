"""Unit tests for gavel_pipeline/llm_client.py's provider routing
(`_use_local`) and load-status tracking (`get_status` / `start_warmup`).
Does not exercise real model loading or generation — that needs a real model
on a GPU, covered by manual smoke testing, not this suite.
"""
import json
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
# System-prompt placement — Mistral-style templates glue it to the LAST user turn
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Renders messages the way a given template family would, as text."""
    def __init__(self, style, name="fake/model"):
        self.style, self.name_or_path, self.chat_template = style, name, "x"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        if self.style == "raises":
            raise ValueError("template does not support the system role")
        system = next((m["content"] for m in msgs if m["role"] == "system"), None)
        rest = [m for m in msgs if m["role"] != "system"]
        out = []
        for i, m in enumerate(rest):
            last_user = m["role"] == "user" and i == len(rest) - 1
            if self.style == "mistral" and last_user and system:
                out.append(f"[INST] {system}\n\n{m['content']} [/INST]")  # system next to NEWEST user text
            elif m["role"] == "user":
                out.append(f"[INST] {m['content']} [/INST]")
            else:
                out.append(m["content"])
        head = f"<<SYS>>{system}<</SYS>>" if (self.style == "native" and system) else ""
        return head + "".join(out)


class TestTemplateKeepsSystemFirst:
    def test_native_system_at_the_top(self):
        assert llm_client._template_keeps_system_first(_FakeTokenizer("native", "a")) is True

    def test_mistral_style_glues_system_to_the_last_user_turn(self):
        assert llm_client._template_keeps_system_first(_FakeTokenizer("mistral", "b")) is False

    def test_a_template_that_drops_the_system_role_counts_as_not_keeping_it(self):
        assert llm_client._template_keeps_system_first(_FakeTokenizer("drops", "c")) is False

    def test_a_template_that_raises_counts_as_not_keeping_it(self):
        assert llm_client._template_keeps_system_first(_FakeTokenizer("raises", "d")) is False


class TestFoldSystem:
    def test_folds_into_the_first_user_turn(self):
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
        assert llm_client._normalize_messages(msgs, fold_system=True) == [
            {"role": "user", "content": "SYS\n\nu1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]

    def test_default_leaves_the_system_message_alone(self):
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "u1"}]
        assert llm_client._normalize_messages(msgs) == msgs

    def test_a_lone_system_message_becomes_a_user_turn(self):
        assert llm_client._normalize_messages([{"role": "system", "content": "SYS"}], fold_system=True) == [
            {"role": "user", "content": "SYS"},
        ]

    def test_no_system_message_is_a_noop(self):
        msgs = [{"role": "user", "content": "u1"}]
        assert llm_client._normalize_messages(msgs, fold_system=True) == msgs


class TestBuildPromptKeepsTheInstructionsAtTheStart:
    """The reported bug: on a Mistral-style template every scenario-chat turn
    re-greeted, because the system prompt ('...greet the user!') was rendered
    right next to the user's newest message."""

    HISTORY = [
        {"role": "system", "content": "SYS: ... Now, greet the user!"},
        {"role": "assistant", "content": "scripted greeting"},
        {"role": "user", "content": "my scenario"},
        {"role": "assistant", "content": "a clarifying question"},
        {"role": "user", "content": "my answer"},
    ]

    def test_mistral_style_puts_the_system_prompt_before_the_conversation(self):
        text = llm_client._build_prompt(_FakeTokenizer("mistral", "m1"), self.HISTORY, want_json=False)
        assert text.index("greet the user") < text.index("my scenario")
        # ...and NOT next to the newest message
        assert "greet the user!\n\nmy answer" not in text
        assert text.rstrip().endswith("my answer [/INST]")

    def test_native_templates_still_get_a_real_system_message(self):
        text = llm_client._build_prompt(_FakeTokenizer("native", "n1"), self.HISTORY, want_json=False)
        assert text.startswith("<<SYS>>SYS: ... Now, greet the user!<</SYS>>")


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
        feedbacks = self.feedbacks = []

        def generate(temp, feedback=None):
            calls.append(temp)
            feedbacks.append(feedback)
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


class TestCloseUnbalancedJson:
    """A local model's most common JSON slip is forgetting a closing brace."""

    @pytest.mark.parametrize("broken,expected", [
        ('{"a": {"b": [1, 2]}', {"a": {"b": [1, 2]}}),           # missing final }
        ('{"a": [1, {"b": 2}', {"a": [1, {"b": 2}]}),            # missing ] and }
        ('{"a": "hello', {"a": "hello"}),                        # cut mid-string
        ('{"a": [1, 2,', {"a": [1, 2]}),                         # dangling comma
        ('{"a": 1, "b":', {"a": 1, "b": None}),                  # dangling colon
        ('{"a": "he said \\"hi\\" and', {"a": 'he said "hi" and'}),  # escaped quote
    ])
    def test_repairs_truncation(self, broken, expected):
        assert json.loads(llm_client._close_unbalanced_json(broken)) == expected

    @pytest.mark.parametrize("not_a_truncation", [
        '{"a": 1}',                 # already balanced: nothing to close
        '{"a": [1}',                # mismatched closer: not a simple truncation
        '{"a": 1 "b": 2}',          # error in the MIDDLE: out of scope
        "no json at all",
    ])
    def test_leaves_everything_else_alone(self, not_a_truncation):
        assert llm_client._close_unbalanced_json(not_a_truncation) is None

    def test_extraction_uses_it_for_a_fenced_answer_missing_its_last_brace(self):
        text = 'Here you go:\n```json\n{"rule_name": "r", "groups": {"g": ["a"]}\n```'
        assert llm_client._extract_json_object(text) == {"rule_name": "r", "groups": {"g": ["a"]}}

    def test_extraction_still_never_returns_a_nested_fragment(self):
        # outer object is broken in the middle (not repairable) -> None, not the inner dict
        assert llm_client._extract_json_object('{"x": {"y": 1} "z": 2}') is None


class TestJsonErrorHint:
    def test_names_the_parse_problem_and_position(self):
        assert "line 1" in llm_client._json_error_hint('{"a": 1 "b": 2}')

    def test_no_object(self):
        assert llm_client._json_error_hint("sorry") == "no JSON object found"


class TestDescribeJsonError:
    """The correction retry used to quote only 'Invalid control character at:
    line 22 column 121'; the model could not locate that in a 5,000-character
    answer and returned the SAME text twice. Locating + explaining it is the fix."""

    # The real pass-6 slip: a string closed with  '}  instead of  "}
    BAD = ('{\n  "a": "fine",\n  "b": "Example benign response 2: '
           "'I see the updates tab. Which updates should I focus on?'}\n  ,\"c\": 1\n}")

    def _err(self, blob):
        with pytest.raises(json.JSONDecodeError) as e:
            json.loads(blob)
        return e.value

    def test_quotes_the_offending_line_and_names_the_usual_cause(self):
        text = llm_client.describe_json_error(self.BAD, self._err(self.BAD))
        assert "line 3" in text
        assert "focus on?'}" in text                    # shows WHERE
        assert "not closed with a double quote" in text  # says WHY
        assert " at at " not in text

    @pytest.mark.parametrize("blob,expected", [
        ('{"a": 1,}', "trailing comma"),
        ('{"a": 1 "b": 2}', "comma is missing"),
        ('{"a": }', "value is missing"),
    ])
    def test_other_common_slips_get_a_plain_hint(self, blob, expected):
        assert expected in llm_client.describe_json_error(blob, self._err(blob))

    def test_works_for_errors_without_a_position(self):
        assert llm_client.describe_json_error("x", ValueError("boom")) == "boom"

    def test_regeneration_feedback_now_carries_the_location(self):
        hint = llm_client._json_error_hint(self.BAD)
        assert "line 3" in hint and "double quote" in hint




class TestRawControlCharsInsideStrings:
    """Reported from the UI: CE calibration died with
    `Invalid control character at: line 2 column 298` — the model wrote a
    multi-sentence string value across lines (a raw newline inside the string),
    failed all 3 regeneration attempts the same way, and the raw text reached
    the caller's strict json.loads."""

    RAW = '{\n  "scenario_instructions": "First sentence.\nSecond sentence.\tTabbed.",\n  "n": 2\n}'

    def test_strict_json_really_does_reject_it(self):
        with pytest.raises(json.JSONDecodeError, match="control character"):
            json.loads(self.RAW)

    def test_extraction_accepts_it_and_keeps_the_text(self):
        obj = llm_client._extract_json_object(self.RAW)
        assert obj == {"scenario_instructions": "First sentence.\nSecond sentence.\tTabbed.", "n": 2}

    def test_the_output_the_caller_gets_is_strictly_valid_json(self):
        out = llm_client._generate_valid_json(lambda temp, feedback=None: self.RAW, 0.7)
        assert json.loads(out)["n"] == 2                    # the caller's json.loads works
        assert "\n" in json.loads(out)["scenario_instructions"]   # text preserved, now escaped

    def test_no_model_regeneration_is_spent_on_it(self):
        calls = []

        def generate(temp, feedback=None):
            calls.append(feedback)
            return self.RAW

        llm_client._generate_valid_json(generate, 0.7)
        assert calls == [None]                              # one generation, no retry

    def test_a_fenced_answer_with_a_raw_newline_is_accepted_too(self):
        assert llm_client._extract_json_object("```json\n" + self.RAW + "\n```")["n"] == 2

    def test_an_unclosed_string_is_still_rejected_not_swallowed(self):
        # strict=False must not turn a broken quote into a "valid" object
        unclosed = '{"a": "he said \'x\'}\n  ,"b": 1}'
        assert llm_client._extract_json_object(unclosed) is None

    def test_the_retry_hint_names_the_real_problem_not_the_tolerated_one(self):
        both = '{"a": "line one\nline two", "b": 1 "c": 2}'     # raw newline AND a missing comma
        hint = llm_client._json_error_hint(both)
        # the REAL remaining problem comes first; the tolerated line break is only a footnote
        assert "delimiter" in hint
        assert hint.index("delimiter") < hint.index("control character")

    def test_an_unclosed_string_still_gets_its_located_clue(self):
        # the pass-6 slip: closed with  '}  instead of  "}  -> the strict error (earlier
        # than where the lenient parse gives up) must still reach the model
        bad = ('{\n  "a": "fine",\n  "b": "Example 2: \'Which updates should I focus on?\'}\n'
               '  ,"c": 1\n}')
        hint = llm_client._json_error_hint(bad)
        assert "line 3" in hint and "not closed with a double quote" in hint


class TestRetryTemperatures:
    """Reported: all regeneration attempts failed the same way and the raw text
    reached the caller's json.loads (a 500). The last try is now greedy."""

    def test_four_attempts_and_the_last_one_is_greedy(self):
        temps = []

        def generate(temp, feedback=None):
            temps.append(temp)
            return "never valid"

        llm_client._generate_valid_json(generate, 0.7)
        assert llm_client._JSON_ATTEMPTS == 4
        assert temps == [0.7, 0.3, 0.3, 0.0]

    def test_a_greedy_final_attempt_can_still_rescue_it(self):
        def generate(temp, feedback=None):
            return '{"ok": true}' if temp == 0.0 else "no json here at all"

        assert llm_client._generate_valid_json(generate, 0.7) == '{"ok": true}'

    def test_already_cool_temperatures_are_never_raised(self):
        temps = []

        def generate(temp, feedback=None):
            temps.append(temp)
            return "never valid"

        llm_client._generate_valid_json(generate, 0.1)
        assert temps == [0.1, 0.1, 0.1, 0.0]


class TestRegenerationShowsTheModelItsMistake:
    def test_the_retry_receives_the_bad_output_and_the_parse_error(self):
        seen = []

        def generate(temp, feedback=None):
            seen.append(feedback)
            return '{"a": 1 "b": 2}' if feedback is None else '{"a": 1, "b": 2}'

        assert llm_client._generate_valid_json(generate, 0.7) == '{"a": 1, "b": 2}'
        assert seen[0] is None
        bad, hint = seen[1]
        assert bad == '{"a": 1 "b": 2}' and "Expecting" in hint

    def test_a_repairable_answer_needs_no_second_model_call(self):
        calls = []

        def generate(temp, feedback=None):
            calls.append(feedback)
            return '{"a": {"b": 1}'          # missing the final brace

        assert json.loads(llm_client._generate_valid_json(generate, 0.7)) == {"a": {"b": 1}}
        assert len(calls) == 1


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
