"""Pure unit tests for the v2 rule structural fingerprint in
``sql_scripts.junction_scripts``.

``compute_rule_fingerprint_v2(ce_groups, condition)`` canonicalizes a rule's
(groups, condition) logic so that two rules with the SAME firing semantics
fingerprint identically regardless of surface spelling:

  * group NAMES don't matter (canonical rename to g0..gN ordered by member
    tuple) — the v2 analogue of "fallback group numbers don't matter";
  * and/or children are commutative (sorted serialization);
  * groups never referenced by the condition (e.g. the migrated 'supporting'
    bucket) are excluded entirely — they don't affect firing;
  * condition-less / unparseable logic falls back to a membership-only
    fingerprint (dedup must never raise).

The two DB-scanning finders (`find_existing_rule_by_fingerprint_v2`,
`find_existing_rule_setup_by_fingerprint_v2`) are exercised with a
monkeypatched ``execute_query_dict`` — no database is touched.
"""
import pytest

import sql_scripts.junction_scripts as js
from sql_scripts.junction_scripts import (
    compute_rule_fingerprint_v2,
    find_existing_rule_by_fingerprint_v2,
    find_existing_rule_setup_by_fingerprint_v2,
)


# ===========================================================================
# compute_rule_fingerprint_v2 — canonicalization properties
# ===========================================================================

class TestGroupNameIndependence:
    def test_renamed_groups_fingerprint_identically(self):
        a = compute_rule_fingerprint_v2(
            {"hook": ["A"], "action": ["B", "C"]}, "all of hook and 1 of action")
        b = compute_rule_fingerprint_v2(
            {"x": ["A"], "y": ["B", "C"]}, "all of x and 1 of y")
        assert a == b

    def test_swapped_group_names_same_structure(self):
        # Same partition, same selectors — only the labels swapped.
        a = compute_rule_fingerprint_v2(
            {"g1": ["A"], "g2": ["B"]}, "all of g1 and 1 of g2")
        b = compute_rule_fingerprint_v2(
            {"g2": ["A"], "g1": ["B"]}, "all of g2 and 1 of g1")
        assert a == b

    def test_member_order_within_group_irrelevant(self):
        a = compute_rule_fingerprint_v2({"g": ["B", "A"]}, "all of g")
        b = compute_rule_fingerprint_v2({"g": ["A", "B"]}, "all of g")
        assert a == b

    def test_duplicate_members_deduped(self):
        a = compute_rule_fingerprint_v2({"g": ["A", "A", "B"]}, "all of g")
        b = compute_rule_fingerprint_v2({"g": ["A", "B"]}, "all of g")
        assert a == b


class TestCommutativity:
    def test_and_children_commute(self):
        a = compute_rule_fingerprint_v2(
            {"p": ["A"], "q": ["B"]}, "all of p and all of q")
        b = compute_rule_fingerprint_v2(
            {"p": ["A"], "q": ["B"]}, "all of q and all of p")
        assert a == b

    def test_or_children_commute(self):
        a = compute_rule_fingerprint_v2(
            {"p": ["A"], "q": ["B"]}, "all of p or all of q")
        b = compute_rule_fingerprint_v2(
            {"p": ["A"], "q": ["B"]}, "all of q or all of p")
        assert a == b

    def test_and_vs_or_differ(self):
        groups = {"p": ["A"], "q": ["B"]}
        a = compute_rule_fingerprint_v2(groups, "all of p and all of q")
        o = compute_rule_fingerprint_v2(groups, "all of p or all of q")
        assert a != o

    def test_nesting_structure_preserved(self):
        # (p and q) or r  !=  p and (q or r)
        groups = {"p": ["A"], "q": ["B"], "r": ["C"]}
        a = compute_rule_fingerprint_v2(groups, "(all of p and all of q) or all of r")
        b = compute_rule_fingerprint_v2(groups, "all of p and (all of q or all of r)")
        assert a != b


class TestDeadGroupExclusion:
    def test_unreferenced_group_does_not_affect_fingerprint(self):
        # The migrated 'supporting' bucket: present in ce_groups, absent from
        # the condition — must not change the fingerprint.
        with_dead = compute_rule_fingerprint_v2(
            {"required": ["A", "B"], "supporting": ["Z"]}, "all of required")
        without = compute_rule_fingerprint_v2(
            {"required": ["A", "B"]}, "all of required")
        assert with_dead == without

    def test_different_dead_groups_still_equal(self):
        a = compute_rule_fingerprint_v2(
            {"g": ["A"], "supporting": ["X"]}, "all of g")
        b = compute_rule_fingerprint_v2(
            {"g": ["A"], "extra_stuff": ["Y", "Z"]}, "all of g")
        assert a == b

    def test_moving_a_member_into_the_condition_changes_it(self):
        # Same membership union, but Z participates in firing in one of them.
        dead = compute_rule_fingerprint_v2(
            {"g": ["A"], "supporting": ["Z"]}, "all of g")
        live = compute_rule_fingerprint_v2(
            {"g": ["A"], "z": ["Z"]}, "all of g and all of z")
        assert dead != live


class TestQuantifierSensitivity:
    def test_n_of_value_matters(self):
        groups = {"g": ["A", "B", "C"]}
        one = compute_rule_fingerprint_v2(groups, "1 of g")
        two = compute_rule_fingerprint_v2(groups, "2 of g")
        assert one != two

    def test_all_vs_k(self):
        groups = {"g": ["A", "B", "C"]}
        all_of = compute_rule_fingerprint_v2(groups, "all of g")
        two = compute_rule_fingerprint_v2(groups, "2 of g")
        assert all_of != two

    def test_not_wrapper_matters(self):
        groups = {"g": ["A"], "h": ["B"]}
        plain = compute_rule_fingerprint_v2(groups, "all of g and all of h")
        negated = compute_rule_fingerprint_v2(groups, "all of g and not all of h")
        assert plain != negated

    def test_different_membership_differs(self):
        a = compute_rule_fingerprint_v2({"g": ["A"]}, "all of g")
        b = compute_rule_fingerprint_v2({"g": ["B"]}, "all of g")
        assert a != b

    def test_partition_matters(self):
        # {A,B} as one 1-of group vs two singleton 1-of groups joined by AND
        # are different firing semantics.
        one_group = compute_rule_fingerprint_v2({"g": ["A", "B"]}, "1 of g")
        two_groups = compute_rule_fingerprint_v2(
            {"g1": ["A"], "g2": ["B"]}, "1 of g1 and 1 of g2")
        assert one_group != two_groups


class TestMembershipFallback:
    def test_condition_none_uses_membership_fingerprint(self):
        fp = compute_rule_fingerprint_v2({"g": ["B", "A"]}, None)
        assert fp.startswith("M:")
        assert fp == compute_rule_fingerprint_v2({"other": ["A", "B"]}, "")

    def test_membership_fallback_ignores_grouping(self):
        # With no condition there is no logic — only the member union counts.
        a = compute_rule_fingerprint_v2({"g1": ["A"], "g2": ["B"]}, None)
        b = compute_rule_fingerprint_v2({"g": ["A", "B"]}, "")
        assert a == b

    def test_unparseable_condition_falls_back_not_raises(self):
        fp = compute_rule_fingerprint_v2({"g": ["A"]}, "all of of nonsense ((")
        assert fp.startswith("M:")

    def test_empty_everything_is_stable(self):
        assert compute_rule_fingerprint_v2({}, None) == compute_rule_fingerprint_v2(None, "")

    def test_membership_differs_from_conditioned(self):
        bare = compute_rule_fingerprint_v2({"g": ["A"]}, None)
        cond = compute_rule_fingerprint_v2({"g": ["A"]}, "all of g")
        assert bare != cond


class TestDeterminism:
    def test_same_input_same_output(self):
        groups = {"hook": ["A", "B"], "action": ["C"]}
        cond = "1 of hook and all of action"
        assert compute_rule_fingerprint_v2(dict(groups), cond) == \
            compute_rule_fingerprint_v2(dict(groups), cond)

    def test_returns_str(self):
        assert isinstance(compute_rule_fingerprint_v2({"g": ["A"]}, "all of g"), str)


# ===========================================================================
# The DB-scanning finders (execute_query_dict monkeypatched)
# ===========================================================================

def _rule_row(rule_id, name, groups, condition):
    return {"rule_id": rule_id, "name": name, "ce_groups": groups, "condition": condition}


def _setup_row(setup_id, custom_name, groups, condition):
    return {"setup_id": setup_id, "custom_name": custom_name,
            "ce_groups": groups, "condition": condition}


class TestFindExistingRule:
    def test_finds_structural_duplicate_under_different_names(self, monkeypatch):
        rows = [
            _rule_row(1, "other", {"g": ["X"]}, "all of g"),
            _rule_row(2, "match", {"named_differently": ["A", "B"]}, "1 of named_differently"),
        ]
        monkeypatch.setattr(js, "execute_query_dict", lambda q, p=None: rows)
        fp = compute_rule_fingerprint_v2({"anything": ["A", "B"]}, "1 of anything")
        hit = find_existing_rule_by_fingerprint_v2(fp)
        assert hit is not None and hit["rule_id"] == 2

    def test_returns_none_when_no_match(self, monkeypatch):
        rows = [_rule_row(1, "r", {"g": ["X"]}, "all of g")]
        monkeypatch.setattr(js, "execute_query_dict", lambda q, p=None: rows)
        fp = compute_rule_fingerprint_v2({"g": ["Y"]}, "all of g")
        assert find_existing_rule_by_fingerprint_v2(fp) is None

    def test_exclude_name_threaded_into_query(self, monkeypatch):
        captured = {}

        def fake(q, p=None):
            captured["params"] = p
            return []

        monkeypatch.setattr(js, "execute_query_dict", fake)
        find_existing_rule_by_fingerprint_v2("fp", exclude_name="myself")
        assert captured["params"] == ("myself", "myself")

    def test_rows_without_members_never_match(self, monkeypatch):
        # Fresh/no-logic rules (empty groups) carry no logic and must not
        # dedup against each other.
        rows = [_rule_row(1, "empty", {}, None), _rule_row(2, "empty2", {"g": []}, None)]
        monkeypatch.setattr(js, "execute_query_dict", lambda q, p=None: rows)
        fp = compute_rule_fingerprint_v2({}, None)
        assert find_existing_rule_by_fingerprint_v2(fp) is None

    def test_none_rows_guarded(self, monkeypatch):
        monkeypatch.setattr(js, "execute_query_dict", lambda q, p=None: None)
        assert find_existing_rule_by_fingerprint_v2("fp") is None


class TestFindExistingSetup:
    def test_finds_sibling_with_same_logic(self, monkeypatch):
        rows = [
            _setup_row(10, "sib", {"grp": ["A"]}, "all of grp"),
        ]
        monkeypatch.setattr(js, "execute_query_dict", lambda q, p=None: rows)
        fp = compute_rule_fingerprint_v2({"renamed": ["A"]}, "all of renamed")
        hit = find_existing_rule_setup_by_fingerprint_v2(5, fp)
        assert hit is not None and hit["setup_id"] == 10

    def test_classifier_and_exclusion_threaded_into_query(self, monkeypatch):
        captured = {}

        def fake(q, p=None):
            captured["params"] = p
            return []

        monkeypatch.setattr(js, "execute_query_dict", fake)
        find_existing_rule_setup_by_fingerprint_v2(7, "fp", exclude_setup_id=42)
        assert captured["params"] == (7, 42, 42)

    def test_no_match_returns_none(self, monkeypatch):
        rows = [_setup_row(10, "s", {"g": ["B"]}, "all of g")]
        monkeypatch.setattr(js, "execute_query_dict", lambda q, p=None: rows)
        fp = compute_rule_fingerprint_v2({"g": ["A"]}, "all of g")
        assert find_existing_rule_setup_by_fingerprint_v2(1, fp) is None

    def test_memberless_setups_skipped(self, monkeypatch):
        # A fresh custom setup (ce_groups = {}) must never be reported.
        rows = [_setup_row(10, "fresh", {}, None)]
        monkeypatch.setattr(js, "execute_query_dict", lambda q, p=None: rows)
        fp = compute_rule_fingerprint_v2({}, None)
        assert find_existing_rule_setup_by_fingerprint_v2(1, fp) is None
