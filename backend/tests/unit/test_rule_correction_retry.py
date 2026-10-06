"""Unit tests for the local-model corrective retry in rule generation
(routes/ai_pipeline.py: _correct_rule_with_feedback & friends).

Reproduces the reported failure: the model's `groups` named CEs
(`tech_support`, `urgency_pressure`) that its own `new_ces` never defined
(it defined `tech_support_impersonation`), which surfaced as a 500
"Rule '...' references unknown CE(s)".
"""
import json

import pytest

import routes.ai_pipeline as ap

CES = {"making_threat": {}, "being_sycophantic": {}}


def _rule(group_members, new_ces):
    return {
        "rule_name": "r", "description": "d",
        "groups": {"g": group_members},
        "condition": "all of g",
        "new_ces": new_ces,
    }


def _validate(rule, ces):
    """Same shape of check as rule_generator.validate_rule's CE-reference part."""
    used = {m for ms in rule.get("groups", {}).values() for m in ms}
    avail = set(ces) | set((rule.get("new_ces") or {}).keys())
    unknown = used - avail
    return [f"Unknown CEs referenced: {unknown}"] if unknown else []


class FakeRG:
    """Stands in for gavel_pipeline.rule_generator."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def call_thinking_model(self, messages):
        self.calls.append(messages)
        if not self.replies:
            return None, "boom"
        r = self.replies.pop(0)
        return (r, None) if not isinstance(r, Exception) else (None, str(r))

    def extract_json_from_response(self, text):
        try:
            return json.loads(text), None
        except ValueError as e:
            return None, str(e)

    def validate_rule(self, rule, ces):
        return _validate(rule, ces)


BAD = _rule(["tech_support", "making_threat"], {"tech_support_impersonation": {"definition": "x"}})
GOOD = _rule(["tech_support", "making_threat"], {"tech_support": {"definition": "x"}})


class TestCorrectionRetry:
    def test_a_fixed_answer_replaces_the_broken_one(self):
        rg = FakeRG([json.dumps(GOOD)])
        data, _, issues = ap._correct_rule_with_feedback(
            rg, "PROMPT", "RESP", BAD, _validate(BAD, CES), CES)
        assert data == GOOD and issues == []
        assert len(rg.calls) == 1

    def test_the_model_is_shown_its_own_json_and_the_exact_problem(self):
        rg = FakeRG([json.dumps(GOOD)])
        ap._correct_rule_with_feedback(rg, "PROMPT", "RESP", BAD, _validate(BAD, CES), CES)
        msgs = rg.calls[0]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[0]["content"] == "PROMPT"
        assert json.loads(msgs[1]["content"]) == BAD
        feedback = msgs[2]["content"]
        assert "tech_support" in feedback                      # the unknown name
        assert "tech_support_impersonation" in feedback        # what it did define
        assert "COMPLETE corrected JSON" in feedback

    def test_stops_early_once_valid(self):
        rg = FakeRG([json.dumps(GOOD), json.dumps(BAD)])
        ap._correct_rule_with_feedback(rg, "P", "R", BAD, _validate(BAD, CES), CES)
        assert len(rg.calls) == 1

    def test_gives_up_after_the_attempt_limit_and_keeps_the_original(self):
        rg = FakeRG([json.dumps(BAD)] * 5)
        data, _, issues = ap._correct_rule_with_feedback(
            rg, "P", "R", BAD, _validate(BAD, CES), CES)
        assert len(rg.calls) == ap._RULE_FIX_ATTEMPTS
        assert data == BAD and issues            # nothing better was found

    def test_an_unparseable_retry_does_not_lose_the_original(self):
        rg = FakeRG(["not json", "still not json"])
        data, _, issues = ap._correct_rule_with_feedback(
            rg, "P", "R", BAD, _validate(BAD, CES), CES)
        assert data == BAD and issues

    def test_a_model_error_on_retry_keeps_the_original(self):
        rg = FakeRG([RuntimeError("gpu fell over")])
        data, _, issues = ap._correct_rule_with_feedback(
            rg, "P", "R", BAD, _validate(BAD, CES), CES)
        assert data == BAD and issues

    def test_a_retry_that_only_partly_helps_is_still_kept(self):
        worse_first = _rule(["a", "b", "c"], {})            # 3 unknown
        partly = _rule(["a", "making_threat"], {})          # 1 unknown
        rg = FakeRG([json.dumps(partly), json.dumps(worse_first)])
        data, _, issues = ap._correct_rule_with_feedback(
            rg, "P", "R", worse_first, _validate(worse_first, CES), CES)
        assert data == partly and len(issues) == 1

    def test_no_issues_means_no_model_call(self):
        rg = FakeRG([json.dumps(GOOD)])
        ap._correct_rule_with_feedback(rg, "P", "R", GOOD, [], CES)
        assert rg.calls == []


class TestMapRuleCategories:
    """The extraction must behave exactly as the old inline block did."""
    CATS = [{"id": 1, "name": "Security"}, {"id": 2, "name": "Fairness"}]

    @pytest.mark.parametrize("assigned,new_cat,expected", [
        ([1, 2], None, ["Security", "Fairness"]),
        (["1"], None, ["Security"]),                                   # digit string = id
        (["Custom Name"], None, ["Custom Name"]),                      # a name is kept
        ([99], None, []),                                              # unknown id dropped
        ([1], {"name": "Brand New"}, ["Security", "Brand New"]),
        ([], {"name": "Only New"}, ["Only New"]),
        ([], {"description": "no name"}, []),
        ([], None, []),
    ])
    def test_mapping(self, assigned, new_cat, expected):
        rd = {"assigned_categories": assigned, "new_category": new_cat}
        assert ap._map_rule_categories(rd, self.CATS) == expected

    def test_missing_keys_are_fine(self):
        assert ap._map_rule_categories({}, self.CATS) == []


class TestRuleFixFeedback:
    def test_lists_every_issue_and_the_defined_ces(self):
        text = ap._rule_fix_feedback(["Unknown CEs referenced: {'x'}", "Bad group name 'A'"],
                                     {"new_ces": {"foo": {}, "bar": {}}})
        assert "Unknown CEs referenced" in text and "Bad group name" in text
        assert "bar, foo" in text

    def test_tolerates_new_ces_that_is_not_a_dict(self):
        text = ap._rule_fix_feedback(["Unknown CEs referenced: {'x'}"], {"new_ces": ["not", "a", "dict"]})
        assert "(none)" in text


class TestOnlyTheLocalProviderRetries:
    """The OpenAI path must not change: no retry call is ever made for it."""

    def _run(self, monkeypatch, local):
        rg = FakeRG([json.dumps(GOOD)])
        bad_text = "```json\n" + json.dumps(BAD) + "\n```"

        # Route the first model call to the bad answer, retries to FakeRG.
        first = {"done": False}

        def call_thinking_model(messages):
            if not first["done"]:
                first["done"] = True
                return bad_text, None
            return rg.call_thinking_model(messages)

        rg_proxy = type("RG", (), {})()
        rg_proxy.call_thinking_model = call_thinking_model
        rg_proxy.extract_json_from_response = lambda t: (
            (lambda m: (json.loads(m), None))(t.split("```json\n")[1].split("\n```")[0])
            if "```json" in t else rg.extract_json_from_response(t))
        rg_proxy.validate_rule = rg.validate_rule
        rg_proxy.format_ces_for_prompt = lambda d: ""
        rg_proxy.format_rules_for_prompt = lambda d: ""

        monkeypatch.setattr(ap, "fetch_ces_dict", lambda: CES)
        monkeypatch.setattr(ap, "fetch_rules_dict", lambda: {})
        monkeypatch.setattr(ap, "fetch_categories_dict", lambda: [])
        monkeypatch.setattr(ap, "_load_prompt", lambda name: (
            "{scenario_description}{available_ces}{existing_rules}{current_categories}"))
        monkeypatch.setattr(ap, "_rule_generator", lambda: rg_proxy)
        monkeypatch.setattr(ap, "_get_llm_client", lambda: type("C", (), {"using_local": staticmethod(lambda: local)})())
        return ap._generate_rule_from_scenario("scenario"), rg

    def test_local_provider_gets_a_corrected_rule(self, monkeypatch):
        result, rg = self._run(monkeypatch, local=True)
        assert result["success"] is True
        assert result["validation_issues"] == []
        assert result["rule_data"]["new_ces"] == GOOD["new_ces"]
        assert len(rg.calls) == 1

    def test_openai_provider_never_retries(self, monkeypatch):
        result, rg = self._run(monkeypatch, local=False)
        assert result["success"] is True
        assert result["validation_issues"]            # reported, exactly as before
        assert rg.calls == []                         # no extra model call


class TestConditionFeedback:
    """Reported failure: the retry fired twice but the model kept writing a bare
    group name with no quantifier — quoting the parser error wasn't enough."""

    ISSUE = "condition does not parse: expected quantifier, got 'remote_access_software_solicitation'"
    RULE = {"groups": {"context": ["a"], "tactics": ["b", "c"], "asks": ["d"]}, "new_ces": {}}

    def test_restates_the_grammar(self):
        text = ap._rule_fix_feedback([self.ISSUE], self.RULE)
        assert "all of <group>" in text and "1 of <group>" in text
        assert "bare group name" in text and "`not` is forbidden" in text

    def test_gives_a_valid_example_built_from_the_rules_own_groups(self):
        text = ap._rule_fix_feedback([self.ISSUE], self.RULE)
        assert "all of context and 1 of tactics and 1 of asks" in text

    def test_unknown_ce_guidance_only_appears_when_relevant(self):
        cond_only = ap._rule_fix_feedback([self.ISSUE], self.RULE)
        assert "Available Cognitive Elements" not in cond_only
        unknown_only = ap._rule_fix_feedback(["Unknown CEs referenced: {'x'}"], self.RULE)
        assert "Available Cognitive Elements" in unknown_only
        assert "valid condition is" not in unknown_only

    def test_tolerates_missing_or_malformed_groups(self):
        assert "valid condition is" not in ap._rule_fix_feedback([self.ISSUE], {"groups": "nope"})
        assert "valid condition is" not in ap._rule_fix_feedback([self.ISSUE], {})


def test_three_correction_attempts_are_allowed():
    assert ap._RULE_FIX_ATTEMPTS == 3


class TestRuleParseFailureIsRepairedForFree:
    """A first answer that only lacks its final brace used to go to the LLM
    'repair' call, whose own output then had the same slip (stress trial 2)."""

    def _run(self, monkeypatch, local, response):
        repair_calls = []
        rg = type("RG", (), {})()
        rg.call_thinking_model = lambda msgs: (response, None)

        def strict(text):
            try:
                return json.loads(text), None
            except ValueError as e:
                return None, str(e)
        rg.extract_json_from_response = strict
        rg.validate_rule = lambda rule, ces: []
        rg.format_ces_for_prompt = lambda d: ""
        rg.format_rules_for_prompt = lambda d: ""
        monkeypatch.setattr(ap, "fetch_ces_dict", lambda: CES)
        monkeypatch.setattr(ap, "fetch_rules_dict", lambda: {})
        monkeypatch.setattr(ap, "fetch_categories_dict", lambda: [])
        monkeypatch.setattr(ap, "_load_prompt", lambda n: "{scenario_description}{available_ces}{existing_rules}{current_categories}")
        monkeypatch.setattr(ap, "_rule_generator", lambda: rg)
        monkeypatch.setattr(ap, "_repair_rule_json",
                            lambda raw, ces: repair_calls.append(1) or (None, "repair failed"))
        import gavel_pipeline.llm_client as lc
        fake = type("C", (), {"using_local": staticmethod(lambda: local),
                              "extract_json": staticmethod(lc.extract_json)})()
        monkeypatch.setattr(ap, "_get_llm_client", lambda: fake)
        return ap._generate_rule_from_scenario("s"), repair_calls

    NO_BRACE = "Here:\n```json\n" + json.dumps({"rule_name": "r", "groups": {"g": ["a"]}})[:-1] + "\n```"

    def test_local_missing_brace_is_fixed_without_the_llm_repair_call(self, monkeypatch):
        result, repair_calls = self._run(monkeypatch, True, self.NO_BRACE)
        assert result["success"] and result["rule_data"]["rule_name"] == "r"
        assert repair_calls == []

    def test_an_unrelated_json_snippet_is_not_mistaken_for_the_rule(self, monkeypatch):
        other = "```json\n" + json.dumps({"unrelated": 1})[:-1] + "\n```"
        result, repair_calls = self._run(monkeypatch, True, other)
        assert repair_calls == [1] and result["success"] is False

    def test_openai_path_still_goes_straight_to_the_existing_repair(self, monkeypatch):
        result, repair_calls = self._run(monkeypatch, False, self.NO_BRACE)
        assert repair_calls == [1] and result["success"] is False
