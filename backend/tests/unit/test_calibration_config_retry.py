"""The CE calibration config call is retried with FRESH calls when the model's
answer is still not valid JSON (reported: e2e build failed with HTTP 500
"Calibration data generation failed: Expecting ',' delimiter")."""
import json

import pytest

import routes.ai_pipeline as ap

GOOD = {"scenario_instructions": "x", "dialogue_controls": {}}


class _Client:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def complete(self, **kw):
        self.calls += 1
        content = self.replies.pop(0)
        msg = type("M", (), {"content": content})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(ap, "execute_query_dict", lambda *a, **k: [{"ce_id": 1, "name": "ce", "definition": "d"}])
    monkeypatch.setattr(ap, "_load_ces_from_db", lambda: {})
    monkeypatch.setattr(ap, "_load_rules_from_db", lambda: {})
    monkeypatch.setattr(ap, "_load_prompt", lambda name: "{ce_name}{ce_definition}{all_ces}{all_rules}{current_rule}")

    def use(replies):
        client = _Client(replies)
        monkeypatch.setattr(ap, "_get_llm_client", lambda: client)
        return client
    return use


def test_a_good_first_answer_costs_one_call(patched):
    client = patched([json.dumps(GOOD)])
    assert ap._generate_ce_calibration_config(1) == GOOD
    assert client.calls == 1


def test_broken_json_is_asked_again_and_the_fresh_answer_is_used(patched):
    client = patched(['{"a": 1 "b": 2}', "```json\n" + json.dumps(GOOD) + "\n```"])
    assert ap._generate_ce_calibration_config(1) == GOOD
    assert client.calls == 2


def test_it_gives_up_with_the_json_error_after_the_attempt_limit(patched):
    client = patched(['{"a": 1 "b": 2}'] * 10)
    with pytest.raises(json.JSONDecodeError):
        ap._generate_ce_calibration_config(1)
    assert client.calls == ap._CAL_CONFIG_ATTEMPTS
