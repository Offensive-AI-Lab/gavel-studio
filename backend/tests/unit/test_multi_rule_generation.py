"""Unit tests for multi-rule generation (routes/ai_pipeline.py).

One scenario used to produce exactly one rule. A small first step now decides
whether the scenario names SEPARATE misuses; one rule (the normal case) takes
the original path untouched, several are generated one after the other. These
tests cover the splitter, the orchestration, CE reuse between rules and the
/ai/generate-pipeline endpoint (DB layer faked — nothing touches a real
database).
"""
import json
import re

import pytest
from fastapi import HTTPException

import routes.ai_pipeline as ap

CES = {"making_threat": {}, "urgency_pressure": {}, "healthcare": {}, "banking": {}}


def _rule(name, groups, condition, **extra):
    return {"rule_name": name, "description": f"{name} desc",
            "groups": groups, "condition": condition, **extra}


R1 = _rule("doctor_impersonation_advice", {"ctx": ["healthcare"], "act": ["fake_doctor_claim", "making_threat"]},
           "all of ctx and 1 of act",
           new_ces={"fake_doctor_claim": {"definition": "claims to be a physician",
                                          "examples": [{"input": "I am a doctor", "output": "YES"}]}})
R2 = _rule("bank_card_theft", {"ctx": ["banking"], "act": ["card_number_request", "urgency_pressure"]},
           "all of ctx and 1 of act",
           new_ces={"card_number_request": {"definition": "asks for card numbers"}})


class _Client:
    """Stands in for the llm_client module (what the splitter and the pair check use)."""

    def __init__(self, reply=None, error=None, local=False, relation="INDEPENDENT", pair_reply=None,
                 pair_error=None, relations=None):
        self.reply, self.error, self.local, self.calls = reply, error, local, []
        self.relation, self.pair_reply, self.pair_error = relation, pair_reply, pair_error
        self.relations = relations or {}        # (part_a, part_b) -> "SAME_SCHEME" / "INDEPENDENT"
        self.pair_calls = []

    @staticmethod
    def _resp(content):
        msg = type("M", (), {"content": content})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

    def complete(self, **kw):
        self.calls.append(kw)
        prompt = kw["messages"][0]["content"]
        if "Description they came from:" in prompt:            # a pair check
            head = prompt.split("Examples")[0]
            part_a = re.search(r"^A: (.*)$", head, re.M).group(1)
            part_b = re.search(r"^B: (.*)$", head, re.M).group(1)
            self.pair_calls.append((part_a, part_b))
            if self.pair_error:
                raise self.pair_error
            if self.pair_reply is not None:
                return self._resp(self.pair_reply)
            relation = self.relations.get((part_a, part_b), self.relation)
            return self._resp(json.dumps({"a_aim": "x", "b_aim": "y", "relation": relation}))
        if self.error:
            raise self.error
        return self._resp(self.reply)

    def using_local(self):
        return self.local


def _splitter(monkeypatch, reply=None, error=None, **kw):
    client = _Client(reply=reply, error=error, **kw)
    monkeypatch.setattr(ap, "_get_llm_client", lambda: client)
    return client


SCEN = "two separate misuses: a fake doctor and a fake bank employee"


class TestSplitScenario:
    def test_two_separate_misuses_become_two_scenarios(self, monkeypatch):
        _splitter(monkeypatch, json.dumps({"misuses": ["A fake doctor gives advice.", "A fake banker asks for cards."]}))
        assert ap._split_scenario(SCEN) == ["A fake doctor gives advice.", "A fake banker asks for cards."]

    def test_one_misuse_returns_the_original_text_untouched(self, monkeypatch):
        _splitter(monkeypatch, json.dumps({"misuses": ["a REWORDED version"]}))
        assert ap._split_scenario(SCEN) == [SCEN]

    def test_the_prompt_carries_the_scenario(self, monkeypatch):
        client = _splitter(monkeypatch, json.dumps({"misuses": ["x"]}))
        ap._split_scenario(SCEN)
        sent = client.calls[0]["messages"][0]["content"]
        assert SCEN in sent
        assert client.calls[0]["temperature"] == 0
        assert client.calls[0]["response_format"] == {"type": "json_object"}

    @pytest.mark.parametrize("reply", [
        "not json", "[]", "{}", '{"misuses": "a string"}', '{"misuses": []}',
        '{"misuses": [1, 2, null]}', '{"misuses": ["", "  "]}', "",
    ])
    def test_any_odd_answer_means_one_rule_as_before(self, monkeypatch, reply):
        _splitter(monkeypatch, reply)
        assert ap._split_scenario(SCEN) == [SCEN]

    def test_a_model_error_means_one_rule_as_before(self, monkeypatch):
        _splitter(monkeypatch, error=RuntimeError("model unavailable"))
        assert ap._split_scenario(SCEN) == [SCEN]

    def test_a_missing_prompt_file_means_one_rule_as_before(self, monkeypatch):
        _splitter(monkeypatch, json.dumps({"misuses": ["a", "b"]}))

        def boom(name):
            raise FileNotFoundError(name)

        monkeypatch.setattr(ap, "_load_prompt", boom)
        assert ap._split_scenario(SCEN) == [SCEN]

    def test_duplicates_are_dropped_and_a_single_leftover_is_one_rule(self, monkeypatch):
        _splitter(monkeypatch, json.dumps({"misuses": ["Same thing", "same thing", "  "]}))
        assert ap._split_scenario(SCEN) == [SCEN]

    def test_there_is_no_limit_on_the_number_of_rules(self, monkeypatch):
        parts = [f"misuse {i}" for i in range(9)]
        _splitter(monkeypatch, json.dumps({"misuses": parts}))
        assert ap._split_scenario(SCEN) == parts

    def test_the_real_prompt_file_formats_cleanly(self):
        template = ap._load_prompt("scenario_splitter.md")
        text = template.format(scenario_description="X {braces} here")
        assert "X {braces} here" in text
        assert '{"goals": [' in text and '"misuses": [' in text   # doubled braces became real ones


class TestPairwiseCheck:
    """Parts are compared two at a time; parts that are one scheme are merged back."""

    def _three(self):
        return json.dumps({"misuses": ["part A", "part B", "part C"]})

    def test_independent_parts_stay_separate(self, monkeypatch):
        client = _splitter(monkeypatch, self._three(), relation="INDEPENDENT")
        assert ap._split_scenario(SCEN) == ["part A", "part B", "part C"]
        assert len(client.pair_calls) == 3                     # A-B, A-C, B-C

    def test_parts_of_one_scheme_are_merged_into_one_rule(self, monkeypatch):
        _splitter(monkeypatch, self._three(), relation="SAME_SCHEME")
        assert ap._split_scenario(SCEN) == [SCEN]

    def test_a_partial_merge_leaves_the_right_number_of_rules(self, monkeypatch):
        client = _splitter(monkeypatch, self._three(), relations={("part A", "part B"): "SAME_SCHEME"})
        assert ap._split_scenario(SCEN) == ["part A part B", "part C"]
        assert len(client.pair_calls) == 3

    def test_all_pairs_are_compared(self, monkeypatch):
        client = _splitter(monkeypatch, self._three(), relation="SAME_SCHEME")
        assert ap._split_scenario(SCEN) == [SCEN]
        assert client.pair_calls == [("part A", "part B"), ("part A", "part C"), ("part B", "part C")]

    def test_one_wrong_same_answer_cannot_chain_unrelated_parts(self, monkeypatch):
        # A~B and A~C said "same", but B and C are independent: C must NOT join A+B
        _splitter(monkeypatch, self._three(), relations={("part A", "part B"): "SAME_SCHEME",
                                                         ("part A", "part C"): "SAME_SCHEME"})
        assert ap._split_scenario(SCEN) == ["part A part B", "part C"]

    def test_the_original_order_is_kept(self, monkeypatch):
        _splitter(monkeypatch, self._three(), relations={("part A", "part C"): "SAME_SCHEME"})
        assert ap._split_scenario(SCEN) == ["part A part C", "part B"]

    def test_the_relation_is_case_insensitive(self, monkeypatch):
        _splitter(monkeypatch, self._three(), relation=" independent ")
        assert len(ap._split_scenario(SCEN)) == 3

    @pytest.mark.parametrize("pair_reply", ["not json", "[]", "{}", '{"relation": 3}',
                                            '{"relation": "MAYBE"}', '{"relation": ""}', ""])
    def test_anything_but_a_clear_independent_merges(self, monkeypatch, pair_reply):
        _splitter(monkeypatch, json.dumps({"misuses": ["part A", "part B"]}), pair_reply=pair_reply)
        assert ap._split_scenario(SCEN) == [SCEN]

    def test_an_error_in_a_pair_check_merges(self, monkeypatch):
        _splitter(monkeypatch, json.dumps({"misuses": ["part A", "part B"]}), pair_error=RuntimeError("down"))
        assert ap._split_scenario(SCEN) == [SCEN]

    def test_the_check_sees_the_description_and_both_parts(self, monkeypatch):
        client = _splitter(monkeypatch, json.dumps({"misuses": ["part A", "part B"]}))
        ap._split_scenario(SCEN)
        sent = client.calls[1]["messages"][0]["content"]
        assert SCEN in sent and "A: part A" in sent and "B: part B" in sent
        assert client.calls[1]["temperature"] == 0
        assert client.calls[1]["response_format"] == {"type": "json_object"}

    def test_a_single_part_is_never_checked(self, monkeypatch):
        client = _splitter(monkeypatch, json.dumps({"misuses": ["only one"]}))
        assert ap._split_scenario(SCEN) == [SCEN]
        assert client.pair_calls == [] and len(client.calls) == 1

    def test_many_parts_are_all_compared_and_all_kept(self, monkeypatch):
        parts = [f"misuse {i}" for i in range(7)]
        client = _splitter(monkeypatch, json.dumps({"misuses": parts}))
        assert ap._split_scenario(SCEN) == parts
        assert len(client.pair_calls) == 21                    # 7 choose 2

    def test_a_missing_pair_prompt_merges(self, monkeypatch):
        _splitter(monkeypatch, json.dumps({"misuses": ["part A", "part B"]}))
        real = ap._load_prompt

        def load(name):
            if name == "scenario_pair_check.md":
                raise FileNotFoundError(name)
            return real(name)

        monkeypatch.setattr(ap, "_load_prompt", load)
        assert ap._split_scenario(SCEN) == [SCEN]

    def test_the_real_pair_prompt_formats_cleanly(self):
        text = ap._load_prompt("scenario_pair_check.md").format(
            scenario_description="X {braces}", part_a="first part", part_b="second part")
        assert "X {braces}" in text and "A: first part" in text and "B: second part" in text
        assert '"relation": "INDEPENDENT"' in text            # the doubled braces became real ones


class TestCrossRuleIssues:
    def test_distinct_rules_have_no_issues(self):
        assert ap._cross_rule_issues([R1, R2]) == []

    def test_duplicate_names_are_reported(self):
        issues = ap._cross_rule_issues([R1, dict(R2, rule_name=R1["rule_name"])])
        assert any("both named" in i for i in issues)

    def test_identical_logic_is_reported(self):
        issues = ap._cross_rule_issues([R1, dict(R1, rule_name="another_name")])
        assert any("identical logic" in i for i in issues)

    def test_malformed_logic_is_not_this_functions_business(self):
        assert ap._cross_rule_issues([R1, {"rule_name": "x", "groups": "nope", "condition": None}]) == []


class TestCesForReuse:
    def test_shape_matches_the_library_and_junk_is_dropped(self):
        out = ap._ces_for_reuse({
            "a": {"definition": "d", "examples": [{"input": "i", "output": "YES"}, {"input": "no output"}, "x"]},
            "b": {"examples": None},
            "c": "not a dict",
        })
        assert out == {"a": {"definition": "d", "examples": [{"input": "i", "output": "YES"}]},
                       "b": {"definition": "", "examples": []}}

    def test_nothing_in_nothing_out(self):
        assert ap._ces_for_reuse(None) == {} and ap._ces_for_reuse({}) == {}


def _one(rule, issues=(), success=True, error=None):
    if not success:
        return {"success": False, "error": error or "boom", "reasoning": "r"}
    return {"success": True, "rule_data": rule, "validation_issues": list(issues), "reasoning": f"reasoning {rule['rule_name']}"}


class TestGenerateRuleFromScenario:
    def _patch(self, monkeypatch, parts, outcomes):
        calls = []
        outcomes = list(outcomes)

        def one(scenario, extra_ces=None):
            calls.append((scenario, dict(extra_ces or {})))
            return outcomes.pop(0)

        monkeypatch.setattr(ap, "_split_scenario", lambda s: parts)
        monkeypatch.setattr(ap, "_generate_one_rule", one)
        return calls

    def test_one_rule_takes_the_original_path_with_the_original_scenario(self, monkeypatch):
        calls = self._patch(monkeypatch, [SCEN], [_one(json.loads(json.dumps(R1)))])
        res = ap._generate_rule_from_scenario(SCEN)
        assert calls == [(SCEN, {})]                      # no extra_ces, scenario untouched
        assert res["success"] and res["rule_data"]["rule_name"] == R1["rule_name"]
        assert res["rules_data"] == [res["rule_data"]]
        assert list(res["new_ces"]) == ["fake_doctor_claim"]
        assert "scenario" not in res["rule_data"]         # single rule: the user's scenario is used

    def test_a_single_rule_failure_is_returned_as_is(self, monkeypatch):
        self._patch(monkeypatch, [SCEN], [_one(None, success=False, error="no json")])
        res = ap._generate_rule_from_scenario(SCEN)
        assert res == {"success": False, "error": "no json", "reasoning": "r"}

    def test_two_misuses_are_generated_one_after_the_other(self, monkeypatch):
        calls = self._patch(monkeypatch, ["doctor part", "bank part"],
                            [_one(json.loads(json.dumps(R1))), _one(json.loads(json.dumps(R2)))])
        res = ap._generate_rule_from_scenario(SCEN)
        assert [c[0] for c in calls] == ["doctor part", "bank part"]
        assert res["success"] and [r["rule_name"] for r in res["rules_data"]] == [R1["rule_name"], R2["rule_name"]]
        assert res["rule_data"] is res["rules_data"][0]
        assert [r["scenario"] for r in res["rules_data"]] == ["doctor part", "bank part"]
        assert sorted(res["new_ces"]) == ["card_number_request", "fake_doctor_claim"]
        assert res["validation_issues"] == []

    def test_a_later_rule_is_offered_the_new_ces_of_the_earlier_ones(self, monkeypatch):
        calls = self._patch(monkeypatch, ["a", "b"],
                            [_one(json.loads(json.dumps(R1))), _one(json.loads(json.dumps(R2)))])
        ap._generate_rule_from_scenario(SCEN)
        assert calls[0][1] == {}
        assert list(calls[1][1]) == ["fake_doctor_claim"]
        assert calls[1][1]["fake_doctor_claim"]["definition"] == "claims to be a physician"

    def test_a_ce_defined_twice_is_kept_once_first_definition_wins(self, monkeypatch):
        r2 = json.loads(json.dumps(R2))
        r2["new_ces"]["fake_doctor_claim"] = {"definition": "SECOND"}
        self._patch(monkeypatch, ["a", "b"], [_one(json.loads(json.dumps(R1))), _one(r2)])
        res = ap._generate_rule_from_scenario(SCEN)
        assert res["new_ces"]["fake_doctor_claim"]["definition"] == "claims to be a physician"

    def test_a_rule_that_cannot_be_generated_fails_the_whole_scenario(self, monkeypatch):
        self._patch(monkeypatch, ["a", "b"],
                    [_one(json.loads(json.dumps(R1))), _one(None, success=False, error="bad"),
                     _one(None, success=False, error="bad again")])
        res = ap._generate_rule_from_scenario(SCEN)
        assert res["success"] is False and "Rule 2 of 2" in res["error"] and "bad" in res["error"]

    def test_a_rule_with_remaining_problems_gets_another_try_and_the_better_answer_wins(self, monkeypatch):
        worse = json.loads(json.dumps(R2))
        better = json.loads(json.dumps(R2))
        calls = self._patch(monkeypatch, ["a", "b"], [
            _one(json.loads(json.dumps(R1))),
            _one(worse, issues=["Unknown CEs referenced: {'x'}", "bad condition"]),
            _one(better, issues=["Unknown CEs referenced: {'x'}"]),
        ])
        res = ap._generate_rule_from_scenario(SCEN)
        assert len(calls) == 3
        assert res["validation_issues"] == ["[rule 'bank_card_theft'] Unknown CEs referenced: {'x'}"]

    def test_a_clean_rule_is_not_regenerated(self, monkeypatch):
        calls = self._patch(monkeypatch, ["a", "b"],
                            [_one(json.loads(json.dumps(R1))), _one(json.loads(json.dumps(R2)))])
        ap._generate_rule_from_scenario(SCEN)
        assert len(calls) == 2

    def test_issues_are_tagged_with_their_rule_and_cross_rule_problems_added(self, monkeypatch):
        dup = json.loads(json.dumps(R2))
        dup["rule_name"] = R1["rule_name"]
        self._patch(monkeypatch, ["a", "b"], [
            _one(json.loads(json.dumps(R1)), issues=["Bad group"]),
            _one(json.loads(json.dumps(R1)), issues=["Bad group"]),   # the second try is no better
            _one(dup),
        ])
        res = ap._generate_rule_from_scenario(SCEN)
        assert "[rule 'doctor_impersonation_advice'] Bad group" in res["validation_issues"]
        assert any("both named" in i for i in res["validation_issues"])


class FakeRG:
    """Stands in for gavel_pipeline.rule_generator (single-rule generator)."""

    def __init__(self, reply):
        self.reply, self.prompts = reply, []
        self.format_ces_for_prompt = self._fmt

    def _fmt(self, ces):
        return "|".join(f"{n}:{(d or {}).get('definition', '')}" for n, d in ces.items())

    def format_rules_for_prompt(self, rules):
        return ""

    def call_thinking_model(self, messages):
        self.prompts.append(messages[0]["content"])
        return self.reply, None

    def extract_json_from_response(self, text):
        return json.loads(text), None

    def validate_rule(self, rule, ces):
        used = {m for ms in (rule.get("groups") or {}).values() for m in ms}
        unknown = used - set(ces) - set(rule.get("new_ces") or {})
        return [f"Unknown CEs referenced: {unknown}"] if unknown else []


class TestGenerateOneRuleWithReusedCes:
    def _run(self, monkeypatch, rule, extra):
        rg = FakeRG(json.dumps(rule))
        monkeypatch.setattr(ap, "fetch_ces_dict", lambda: CES)
        monkeypatch.setattr(ap, "fetch_rules_dict", lambda: {})
        monkeypatch.setattr(ap, "fetch_categories_dict", lambda: [])
        monkeypatch.setattr(ap, "_load_prompt", lambda n: "{scenario_description}#{available_ces}#"
                                                         "{existing_rules}#{current_categories}")
        monkeypatch.setattr(ap, "_rule_generator", lambda: rg)
        monkeypatch.setattr(ap, "_get_llm_client", lambda: _Client(local=False))
        return ap._generate_one_rule("scenario", extra_ces=extra), rg

    def test_earlier_ces_are_shown_to_the_model_as_available(self, monkeypatch):
        _, rg = self._run(monkeypatch, R2, {"fake_doctor_claim": {"definition": "claims to be a physician", "examples": []}})
        assert "fake_doctor_claim:claims to be a physician" in rg.prompts[0]

    def test_a_rule_may_use_them_without_defining_them_again(self, monkeypatch):
        rule = _rule("uses_earlier_ce", {"ctx": ["banking"], "act": ["fake_doctor_claim"]}, "all of ctx and 1 of act")
        res, _ = self._run(monkeypatch, rule, {"fake_doctor_claim": {"definition": "d", "examples": []}})
        assert res["validation_issues"] == []

    def test_without_extra_ces_the_same_reference_is_still_unknown(self, monkeypatch):
        rule = _rule("uses_earlier_ce", {"ctx": ["banking"], "act": ["fake_doctor_claim"]}, "all of ctx and 1 of act")
        res, _ = self._run(monkeypatch, rule, None)
        assert res["validation_issues"] and "fake_doctor_claim" in res["validation_issues"][0]


class _Cursor:
    """Fakes the DB layer used by /ai/generate-pipeline."""

    def __init__(self, upsert_fail_on=None):
        self.next_ce, self.next_rule = 100, 500
        self.created_ces, self.upserts, self.deleted, self.upsert_fail_on = [], [], [], upsert_fail_on

    def create_ce(self, **kw):
        self.next_ce += 1
        self.created_ces.append(kw["name"])
        return {"ce_id": self.next_ce, "is_new": True}

    def upsert_rule(self, payload, mark_pending=False):
        if self.upsert_fail_on is not None and len(self.upserts) == self.upsert_fail_on:
            raise ValueError("boom while saving the rule")
        self.next_rule += 1
        self.upserts.append(payload)
        return self.next_rule

    def execute_query(self, sql, params=()):
        self.deleted.append((sql, params))

    def execute_query_dict(self, sql, params=()):
        return []


def _patch_endpoint(monkeypatch, db, result):
    monkeypatch.setattr(ap.llm_access, "require_llm", lambda: None)
    monkeypatch.setattr(ap, "_generate_rule_from_scenario", lambda scenario: result)
    monkeypatch.setattr(ap, "create_ce", db.create_ce)
    monkeypatch.setattr(ap, "get_excitation_dataset", lambda ce_id: None)
    monkeypatch.setattr(ap, "get_calibration_dataset", lambda ce_id: None)
    monkeypatch.setattr(ap, "upsert_category", lambda *a, **k: None)
    monkeypatch.setattr(ap, "fetch_rules_dict", lambda: {})
    monkeypatch.setattr(ap, "upsert_rule_with_links", db.upsert_rule)
    monkeypatch.setattr(ap, "execute_query", db.execute_query)
    monkeypatch.setattr(ap, "execute_query_dict", db.execute_query_dict)


def _generated(*rules, scenarios=False):
    rules = [json.loads(json.dumps(r)) for r in rules]
    pool = {}
    for r in rules:
        r["categories"] = []
        for name, details in (r.get("new_ces") or {}).items():
            pool.setdefault(name, details)
        if scenarios:
            r["scenario"] = f"{r['rule_name']} scenario"
    return {"success": True, "rule_data": rules[0], "rules_data": rules, "new_ces": pool,
            "validation_issues": [], "reasoning": "r"}


def _single_result_without_rules_data(rule):
    """The shape the generator returned before multi-rule existed."""
    rule = json.loads(json.dumps(rule))
    rule["categories"] = []
    return {"success": True, "rule_data": rule, "validation_issues": [], "reasoning": "r"}


def _call(scenario="doctor and bank scenario"):
    return ap.generate_gavel_pipeline(ap.PipelineRequest(scenario=scenario, user_id=1), _=1)


class TestGeneratePipelineEndpoint:
    def test_two_rules_are_saved_and_the_shared_ces_created_once(self, monkeypatch):
        db = _Cursor()
        _patch_endpoint(monkeypatch, db, _generated(R1, R2, scenarios=True))
        resp = _call()
        assert db.created_ces == ["fake_doctor_claim", "card_number_request"]
        assert [u["rule_name"] for u in db.upserts] == ["doctor_impersonation_advice", "bank_card_theft"]
        assert [r.name for r in resp.rules] == ["doctor_impersonation_advice", "bank_card_theft"]
        assert [r.rule_id for r in resp.rules] == [501, 502]
        assert len(resp.new_ces) == 2

    def test_the_first_rule_is_mirrored_at_the_top_level(self, monkeypatch):
        db = _Cursor()
        _patch_endpoint(monkeypatch, db, _generated(R1, R2, scenarios=True))
        resp = _call()
        assert resp.rule_id == 501 and resp.name == "doctor_impersonation_advice"
        assert resp.groups == R1["groups"] and resp.condition == R1["condition"]

    def test_each_rule_lists_only_the_new_ces_it_uses(self, monkeypatch):
        db = _Cursor()
        _patch_endpoint(monkeypatch, db, _generated(R1, R2, scenarios=True))
        resp = _call()
        assert resp.rules[0].new_ce_names == ["fake_doctor_claim"]
        assert resp.rules[1].new_ce_names == ["card_number_request"]

    def test_a_later_rule_that_reuses_an_earlier_new_ce_lists_it_too(self, monkeypatch):
        r2 = _rule("bank_card_theft", {"ctx": ["banking"], "act": ["fake_doctor_claim"]}, "all of ctx and 1 of act")
        db = _Cursor()
        _patch_endpoint(monkeypatch, db, _generated(R1, r2, scenarios=True))
        resp = _call()
        assert db.created_ces == ["fake_doctor_claim", "card_number_request"][:1]
        assert resp.rules[1].new_ce_names == ["fake_doctor_claim"]

    def test_several_rules_each_carry_their_own_scenario(self, monkeypatch):
        db = _Cursor()
        _patch_endpoint(monkeypatch, db, _generated(R1, R2, scenarios=True))
        resp = _call()
        assert [r.scenario for r in resp.rules] == ["doctor_impersonation_advice scenario",
                                                    "bank_card_theft scenario"]

    def test_a_single_rule_has_no_own_scenario_so_the_users_one_is_used(self, monkeypatch):
        db = _Cursor()
        _patch_endpoint(monkeypatch, db, _generated(R1))
        resp = _call()
        assert len(resp.rules) == 1 and resp.rules[0].scenario is None
        assert resp.rule_id == resp.rules[0].rule_id == 501
        assert db.created_ces == ["fake_doctor_claim"]

    def test_a_generator_result_without_the_new_keys_still_works(self, monkeypatch):
        """Old shape (no rules_data / new_ces): exactly the single-rule behavior."""
        db = _Cursor()
        _patch_endpoint(monkeypatch, db, _single_result_without_rules_data(R1))
        resp = _call()
        assert len(resp.rules) == 1 and resp.rule_id == 501
        assert db.created_ces == ["fake_doctor_claim"]

    def test_a_failure_saving_rule_two_removes_rule_one_and_the_ces(self, monkeypatch):
        db = _Cursor(upsert_fail_on=1)
        _patch_endpoint(monkeypatch, db, _generated(R1, R2, scenarios=True))
        with pytest.raises(HTTPException) as exc:
            _call()
        assert exc.value.status_code == 500
        rule_deletes = [p for sql, p in db.deleted if "DELETE FROM rules" in sql]
        assert rule_deletes == [(501,)]
        # only a pending local draft may be removed by the rollback
        assert any("is_local_draft = TRUE AND is_ready = FALSE" in sql for sql, _ in db.deleted)
        assert sum("DELETE FROM cognitive_elements" in sql for sql, _ in db.deleted) == 2

    def test_a_failure_saving_a_single_rule_removes_nothing_it_did_not_save(self, monkeypatch):
        db = _Cursor(upsert_fail_on=0)
        _patch_endpoint(monkeypatch, db, _generated(R1))
        with pytest.raises(HTTPException):
            _call()
        assert not [p for sql, p in db.deleted if "DELETE FROM rules" in sql]


class TestDiscardManyRules:
    def test_every_given_rule_is_considered(self, monkeypatch):
        import asyncio
        seen = []

        def qd(sql, params=()):
            if "FROM rules" in sql:
                seen.append(params[0])
                return [{"is_local_draft": True}]
            return []

        deleted = []
        monkeypatch.setattr(ap, "execute_query_dict", qd)
        monkeypatch.setattr(ap, "execute_query", lambda sql, params=(): deleted.append(params))
        out = asyncio.run(ap.discard_pipeline_resources(
            ap.DiscardPipelineRequest(ce_ids=[], rule_id=7, rule_ids=[7, 8, 9])))
        assert seen == [7, 8, 9]             # rule_id repeated in rule_ids is handled once
        assert out["deleted_rule"] == 3
        assert deleted == [(7,), (8,), (9,)]

    def test_a_published_rule_is_skipped_not_deleted(self, monkeypatch):
        import asyncio
        monkeypatch.setattr(ap, "execute_query_dict", lambda sql, params=(): [{"is_local_draft": False}])
        deleted = []
        monkeypatch.setattr(ap, "execute_query", lambda sql, params=(): deleted.append(params))
        out = asyncio.run(ap.discard_pipeline_resources(
            ap.DiscardPipelineRequest(ce_ids=[], rule_ids=[3])))
        assert out["deleted_rule"] == 0 and out["skipped_published"] == 1 and deleted == []

    def test_the_old_single_rule_call_still_works(self, monkeypatch):
        import asyncio
        monkeypatch.setattr(ap, "execute_query_dict", lambda sql, params=(): [{"is_local_draft": True}])
        monkeypatch.setattr(ap, "execute_query", lambda sql, params=(): None)
        out = asyncio.run(ap.discard_pipeline_resources(ap.DiscardPipelineRequest(ce_ids=[], rule_id=5)))
        assert out["deleted_rule"] == 1
