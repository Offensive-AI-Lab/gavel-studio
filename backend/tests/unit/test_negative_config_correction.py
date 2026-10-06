"""Local-model correction of the negative-set config (ai_pipeline.build_negative_config).

Reproduces the full-scale pass failure: the model wrote ONE field and then
`... (The remaining fields follow the same structure as the provided example)`,
which is not JSON, so the rule's whole default test set failed with
"Negative config LLM output was not valid JSON".
"""
import json

import pytest

import routes.ai_pipeline as ap

LAZY = (
    "## REASONING\n1. thinking\n\n```json\n{\n"
    '  "scenario_instructions": "benign tutorial",\n\n'
    "  ... (The remaining fields follow the same structure as the provided example)\n}\n```"
)
FULL = (
    "## REASONING\n1. thinking again\n\n```json\n"
    + json.dumps({"scenario_instructions": "benign tutorial", "necessary_labels": {"a": 1}})
    + "\n```"
)


class _Resp:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class FakeClient:
    def __init__(self, replies, local=True):
        self.replies, self.calls, self._local = list(replies), [], local

    def using_local(self):
        return self._local

    def complete(self, **kw):
        self.calls.append(kw)
        return _Resp(self.replies.pop(0))

    def extract_json(self, text):
        import gavel_pipeline.llm_client as lc
        return lc.extract_json(text)


def _run(monkeypatch, replies, local=True):
    client = FakeClient(replies, local)
    monkeypatch.setattr(ap, "_get_llm_client", lambda: client)
    return client


class TestSplit:
    def test_fenced(self):
        reasoning, text = ap._split_negative_config(FULL)
        assert reasoning.startswith("## REASONING") and json.loads(text)["necessary_labels"] == {"a": 1}

    def test_no_fence_means_the_whole_body_is_the_json(self):
        assert ap._split_negative_config('{"a": 1}') == ("", '{"a": 1}')


class TestFeedback:
    def test_names_the_error(self):
        assert "line 4 column 3" in ap._neg_config_feedback("line 4 column 3", "x")

    def test_calls_out_a_placeholder(self):
        assert "placeholder" in ap._neg_config_feedback("e", LAZY)
        assert "placeholder" not in ap._neg_config_feedback("e", FULL)


class TestBuildNegativeConfig:
    POS = {"scenario_instructions": "do the thing"}

    def test_a_lazy_answer_is_corrected_by_a_second_call(self, monkeypatch):
        client = _run(monkeypatch, [LAZY, FULL])
        cfg, reasoning = ap.build_negative_config(self.POS)
        assert cfg["necessary_labels"] == {"a": 1}
        assert "thinking again" in reasoning
        assert len(client.calls) == 2
        retry = client.calls[1]["messages"]
        assert [m["role"] for m in retry] == ["user", "assistant", "user"]
        assert retry[1]["content"] == LAZY and "placeholder" in retry[2]["content"]

    def test_a_valid_first_answer_makes_no_extra_call(self, monkeypatch):
        client = _run(monkeypatch, [FULL])
        ap.build_negative_config(self.POS)
        assert len(client.calls) == 1

    def test_a_retry_missing_its_closing_brace_is_still_accepted(self, monkeypatch):
        missing = "```json\n" + json.dumps({"scenario_instructions": "s", "k": {"v": 1}})[:-1] + "\n```"
        _run(monkeypatch, [LAZY, missing])
        cfg, _ = ap.build_negative_config(self.POS)
        assert cfg["k"] == {"v": 1}

    def test_gives_up_after_the_attempt_limit_with_the_original_error_type(self, monkeypatch):
        client = _run(monkeypatch, [LAZY] * 10)
        with pytest.raises(RuntimeError, match="not valid JSON"):
            ap.build_negative_config(self.POS)
        assert len(client.calls) == 1 + ap._NEG_CONFIG_FIX_ATTEMPTS

    def test_openai_never_retries(self, monkeypatch):
        client = _run(monkeypatch, [LAZY, FULL], local=False)
        with pytest.raises(RuntimeError, match="not valid JSON"):
            ap.build_negative_config(self.POS)
        assert len(client.calls) == 1        # unchanged behavior: no second call


class TestLocatedFeedbackAndFreeParsing:
    # Reported: both retries returned byte-identical output because the feedback
    # was just 'Invalid control character at: line 22 column 121'.
    UNCLOSED = (
        '## REASONING\n1. x\n\n```json\n{\n  "scenario_instructions": "ok",\n'
        "  \"b\": \"Example 2: 'Which updates should I focus on?'}\n  ,\"c\": 1\n}\n```"
    )

    def test_feedback_shows_the_offending_line_and_asks_for_a_minimal_fix(self):
        try:
            json.loads(ap._split_negative_config(self.UNCLOSED)[1])
        except json.JSONDecodeError as e:
            text = ap._neg_config_feedback(e, self.UNCLOSED)
        assert "focus on?'}" in text and "double quote" in text
        assert "change ONLY what is wrong" in text

    def test_a_missing_closing_brace_costs_no_model_call(self, monkeypatch):
        no_brace = "```json\n" + json.dumps({"scenario_instructions": "s", "k": {"v": 1}})[:-1] + "\n```"
        client = _run(monkeypatch, [no_brace])
        cfg, _ = ap.build_negative_config({"scenario_instructions": "x"})
        assert cfg["k"] == {"v": 1}
        assert len(client.calls) == 1                  # the repair was free

    def test_three_attempts(self):
        assert ap._NEG_CONFIG_FIX_ATTEMPTS == 3
