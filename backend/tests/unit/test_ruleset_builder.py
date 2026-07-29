"""Pure unit tests for ``evaluation.ruleset_builder.build_unified_ruleset``.

``build_unified_ruleset`` does all DB access through ``execute_query_dict``,
which it imports *lazily inside the function body* from ``utils.sqlite_db``:

    from utils.sqlite_db import execute_query_dict

Because the symbol is re-imported on every call (it is not bound at module
import time on ``evaluation.ruleset_builder``), the only reliable patch point
is the attribute on the ``utils.sqlite_db`` module object itself — that is
what the ``from utils.sqlite_db import ...`` statement resolves against. We
replace it with a small scripted fake that returns canned rows; no database is
ever touched.

The function issues TWO kinds of query against the (single) faked function:

  1. The snapshot probe:
       SELECT trained_rule_setup_ids FROM classifiers WHERE classifier_id = %s
     -> returns a one-row list like ``[{"trained_rule_setup_ids": [...]}]``.
  2. The ruleset SELECT (rule_setup JOIN setup_ce_link JOIN cognitive_elements)
     -> returns the per-(setup_id, ce) link rows.

Our fake dispatches on the SQL text so a single ``execute_query_dict`` stub can
serve both. ``classifier_id`` selection logic (trained snapshot vs live
fallback) and role mapping (necessary -> all_required, fallback -> any_of,
sufficient -> supporting) are exercised below.
"""
import pytest

import utils.sqlite_db as pg
from evaluation.ruleset_builder import build_unified_ruleset


# ---------------------------------------------------------------------------
# Fake DB plumbing
# ---------------------------------------------------------------------------

def _row(setup_id, rule_name, is_active, ce_name, role, fallback_group,
         predicate=None):
    """Build one ruleset-SELECT result row matching the column projection."""
    return {
        "setup_id": setup_id,
        "rule_name": rule_name,
        "is_active": is_active,
        "predicate": predicate,
        "ce_name": ce_name,
        "role": role,
        "fallback_group": fallback_group,
    }


def _install(monkeypatch, *, snapshot, ruleset_rows, capture=None):
    """Patch ``execute_query_dict`` with a fake that dispatches on SQL text.

    ``snapshot``      -> what the classifiers probe returns (a list of rows, or
                         None to simulate a missing classifier row).
    ``ruleset_rows``  -> either a single list returned for every ruleset
                         SELECT, OR a dict keyed by the WHERE-clause flavour:
                            "snapshot" for the ``setup_id = ANY(%s)`` query,
                            "live"     for the plain ``classifier_id`` query.
                         This lets a test return rows for the live query while
                         returning [] for the snapshot query (orphan fallback).
    ``capture``       -> optional list recording (query, params) per call.
    """
    def fake(query, params):
        if capture is not None:
            capture.append((query, params))
        if "trained_rule_setup_ids" in query:
            return snapshot
        # ruleset SELECT
        if isinstance(ruleset_rows, dict):
            if "ANY(%s)" in query:
                return ruleset_rows.get("snapshot", [])
            return ruleset_rows.get("live", [])
        return ruleset_rows

    monkeypatch.setattr(pg, "execute_query_dict", fake)
    return capture


# ===========================================================================
# Live-fallback branch (no trained snapshot)
# ===========================================================================

class TestLiveFallback:
    def test_no_snapshot_falls_back_to_live_rule_setup(self, monkeypatch):
        # classifiers row exists but trained_rule_setup_ids is NULL -> live.
        rows = [
            _row(1, "Bribery", True, "CE_A", "necessary", 0),
        ]
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=rows)
        result = build_unified_ruleset(99)
        assert result == {
            "Bribery": {
                "all_required": ["CE_A"],
                "any_of": [],
                "supporting": [],
                "enabled": True,
            }
        }

    def test_missing_classifier_row_treated_as_no_snapshot(self, monkeypatch):
        # The probe returns [] (``or []`` guard) -> trained_ids = [] -> live.
        rows = [_row(7, "R", True, "CE_X", "necessary", 0)]
        _install(monkeypatch, snapshot=[], ruleset_rows=rows)
        result = build_unified_ruleset(5)
        assert "R" in result

    def test_probe_returns_none_is_guarded(self, monkeypatch):
        # ``execute_query_dict(...) or []`` handles a None probe result.
        rows = [_row(7, "R", True, "CE_X", "necessary", 0)]
        _install(monkeypatch, snapshot=None, ruleset_rows=rows)
        result = build_unified_ruleset(5)
        assert "R" in result

    def test_empty_trained_ids_list_falls_back_to_live(self, monkeypatch):
        # trained_rule_setup_ids = [] (empty list, not NULL) -> still live.
        rows = [_row(2, "Live", True, "CE_Y", "necessary", 0)]
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": []}],
                 ruleset_rows=rows)
        assert "Live" in build_unified_ruleset(1)

    def test_live_query_uses_plain_classifier_predicate(self, monkeypatch):
        # When there is no snapshot, only the live WHERE-clause query should be
        # issued for the ruleset (no ANY(%s) variant).
        cap = []
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[_row(1, "R", True, "CE", "necessary", 0)],
                 capture=cap)
        build_unified_ruleset(42)
        ruleset_queries = [q for q, _ in cap if "setup_ce_link" in q]
        assert len(ruleset_queries) == 1
        assert "ANY(%s)" not in ruleset_queries[0]
        # classifier_id is threaded through the params.
        ruleset_params = [p for q, p in cap if "setup_ce_link" in q][0]
        assert ruleset_params == (42,)

    def test_empty_ruleset_rows_yields_empty_dict(self, monkeypatch):
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[])
        assert build_unified_ruleset(1) == {}

    def test_ruleset_query_returns_none_is_guarded(self, monkeypatch):
        # ``or []`` guards a None ruleset result too.
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=None)
        assert build_unified_ruleset(1) == {}


# ===========================================================================
# Trained-snapshot branch
# ===========================================================================

class TestTrainedSnapshot:
    def test_snapshot_query_used_when_ids_present(self, monkeypatch):
        cap = []
        _install(
            monkeypatch,
            snapshot=[{"trained_rule_setup_ids": [10, 11]}],
            ruleset_rows={
                "snapshot": [_row(10, "Snap", True, "CE_S", "necessary", 0)],
                "live": [_row(99, "LIVE_SHOULD_NOT_APPEAR", True, "CE_L",
                              "necessary", 0)],
            },
            capture=cap,
        )
        result = build_unified_ruleset(3)
        # Only the snapshot rows are used; live is ignored entirely.
        assert "Snap" in result
        assert "LIVE_SHOULD_NOT_APPEAR" not in result
        # The ANY(%s) query was issued with (classifier_id, trained_ids).
        ruleset_calls = [(q, p) for q, p in cap if "setup_ce_link" in q]
        assert len(ruleset_calls) == 1
        q, p = ruleset_calls[0]
        assert "ANY(%s)" in q
        assert p == (3, [10, 11])

    def test_orphaned_snapshot_falls_back_to_live(self, monkeypatch):
        # Snapshot ids point at rows that no longer exist -> snapshot query
        # returns [], so the live query is issued and its rows are used.
        cap = []
        _install(
            monkeypatch,
            snapshot=[{"trained_rule_setup_ids": [777]}],
            ruleset_rows={
                "snapshot": [],
                "live": [_row(1, "Recovered", True, "CE_R", "necessary", 0)],
            },
            capture=cap,
        )
        result = build_unified_ruleset(8)
        assert "Recovered" in result
        # Both ruleset queries were attempted: snapshot first, then live.
        ruleset_calls = [q for q, _ in cap if "setup_ce_link" in q]
        assert len(ruleset_calls) == 2
        assert "ANY(%s)" in ruleset_calls[0]
        assert "ANY(%s)" not in ruleset_calls[1]

    def test_snapshot_with_rows_does_not_issue_live_query(self, monkeypatch):
        cap = []
        _install(
            monkeypatch,
            snapshot=[{"trained_rule_setup_ids": [10]}],
            ruleset_rows={
                "snapshot": [_row(10, "Snap", True, "CE", "necessary", 0)],
                "live": [],
            },
            capture=cap,
        )
        build_unified_ruleset(3)
        ruleset_calls = [q for q, _ in cap if "setup_ce_link" in q]
        # Exactly one ruleset query (the snapshot one); no live fallback.
        assert len(ruleset_calls) == 1
        assert "ANY(%s)" in ruleset_calls[0]


# ===========================================================================
# Role mapping: necessary / fallback / sufficient
# ===========================================================================

class TestRoleMapping:
    def _build_single_rule(self, monkeypatch, rows):
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=rows)
        return build_unified_ruleset(1)

    def test_necessary_maps_to_all_required(self, monkeypatch):
        rows = [
            _row(1, "R", True, "CE_A", "necessary", 0),
            _row(1, "R", True, "CE_B", "necessary", 0),
        ]
        result = self._build_single_rule(monkeypatch, rows)
        assert result["R"]["all_required"] == ["CE_A", "CE_B"]
        assert result["R"]["any_of"] == []
        assert result["R"]["supporting"] == []

    def test_sufficient_maps_to_supporting(self, monkeypatch):
        rows = [_row(1, "R", True, "CE_D", "sufficient", 0)]
        result = self._build_single_rule(monkeypatch, rows)
        assert result["R"]["supporting"] == ["CE_D"]
        assert result["R"]["all_required"] == []

    def test_fallback_groups_become_any_of_lists(self, monkeypatch):
        # Two distinct fallback groups -> two inner lists in any_of, ordered
        # by group number (sorted ascending).
        rows = [
            _row(1, "R", True, "CE_C", "fallback", 2),
            _row(1, "R", True, "CE_A", "fallback", 1),
            _row(1, "R", True, "CE_B", "fallback", 1),
        ]
        result = self._build_single_rule(monkeypatch, rows)
        assert result["R"]["any_of"] == [["CE_A", "CE_B"], ["CE_C"]]

    def test_role_is_case_insensitive(self, monkeypatch):
        rows = [
            _row(1, "R", True, "CE_A", "NECESSARY", 0),
            _row(1, "R", True, "CE_B", "Sufficient", 0),
            _row(1, "R", True, "CE_C", "FALLBACK", 1),
        ]
        result = self._build_single_rule(monkeypatch, rows)
        assert result["R"]["all_required"] == ["CE_A"]
        assert result["R"]["supporting"] == ["CE_B"]
        assert result["R"]["any_of"] == [["CE_C"]]

    def test_null_role_defaults_to_necessary(self, monkeypatch):
        # ``(row["role"] or "necessary")`` -> None becomes necessary.
        rows = [_row(1, "R", True, "CE_A", None, 0)]
        result = self._build_single_rule(monkeypatch, rows)
        assert result["R"]["all_required"] == ["CE_A"]

    def test_unknown_role_treated_as_necessary(self, monkeypatch):
        rows = [_row(1, "R", True, "CE_A", "bogus", 0)]
        result = self._build_single_rule(monkeypatch, rows)
        assert result["R"]["all_required"] == ["CE_A"]

    def test_null_fallback_group_defaults_to_zero(self, monkeypatch):
        # Two fallback CEs with NULL group both land in group 0 -> one any_of.
        rows = [
            _row(1, "R", True, "CE_A", "fallback", None),
            _row(1, "R", True, "CE_B", "fallback", None),
        ]
        result = self._build_single_rule(monkeypatch, rows)
        assert result["R"]["any_of"] == [["CE_A", "CE_B"]]

    def test_mixed_roles_full_rule(self, monkeypatch):
        rows = [
            _row(1, "Complex", True, "N1", "necessary", 0),
            _row(1, "Complex", True, "S1", "sufficient", 0),
            _row(1, "Complex", True, "F1", "fallback", 1),
            _row(1, "Complex", True, "F2", "fallback", 1),
            _row(1, "Complex", True, "F3", "fallback", 2),
        ]
        result = self._build_single_rule(monkeypatch, rows)
        assert result["Complex"] == {
            "all_required": ["N1"],
            "any_of": [["F1", "F2"], ["F3"]],
            "supporting": ["S1"],
            "enabled": True,
        }


# ===========================================================================
# Misc semantics: enabled flag, naming, multiple rules
# ===========================================================================

class TestMiscSemantics:
    def test_is_active_false_sets_enabled_false(self, monkeypatch):
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[_row(1, "R", False, "CE", "necessary", 0)])
        result = build_unified_ruleset(1)
        assert result["R"]["enabled"] is False

    def test_is_active_truthy_coerced_to_bool(self, monkeypatch):
        # ``bool(row["is_active"])`` -> any truthy value becomes True.
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[_row(1, "R", 1, "CE", "necessary", 0)])
        result = build_unified_ruleset(1)
        assert result["R"]["enabled"] is True

    def test_null_rule_name_falls_back_to_rule_id_label(self, monkeypatch):
        # name = rule["name"] or f"rule_{sid}" -> None name -> "rule_<sid>".
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[_row(55, None, True, "CE", "necessary", 0)])
        result = build_unified_ruleset(1)
        assert "rule_55" in result

    def test_multiple_rules_keyed_by_name(self, monkeypatch):
        rows = [
            _row(1, "Alpha", True, "CE_A", "necessary", 0),
            _row(2, "Beta", True, "CE_B", "sufficient", 0),
        ]
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=rows)
        result = build_unified_ruleset(1)
        assert set(result.keys()) == {"Alpha", "Beta"}
        assert result["Alpha"]["all_required"] == ["CE_A"]
        assert result["Beta"]["supporting"] == ["CE_B"]

    def test_rows_for_same_setup_id_are_grouped(self, monkeypatch):
        # Multiple link rows for one setup_id collapse into one rule entry,
        # with the first row establishing name/enabled.
        rows = [
            _row(1, "Same", True, "CE_A", "necessary", 0),
            _row(1, "Same", True, "CE_B", "necessary", 0),
            _row(1, "Same", True, "CE_C", "sufficient", 0),
        ]
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=rows)
        result = build_unified_ruleset(1)
        assert len(result) == 1
        assert result["Same"]["all_required"] == ["CE_A", "CE_B"]
        assert result["Same"]["supporting"] == ["CE_C"]


# ===========================================================================
# CE-name sanitization — must match the trained classifier's labels-dict keys
# (classifier_engine.trainer._sanitize_label), otherwise required CEs are
# silently dropped and rules look "missing required CEs" though they triggered.
# ===========================================================================

class TestCeNameSanitization:
    def _build(self, monkeypatch, rows):
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=rows)
        return build_unified_ruleset(1)

    def test_spaces_become_underscores(self, monkeypatch):
        # "provide or give" -> "provide_or_give" (matches labels-dict key).
        rows = [_row(1, "R", True, "provide or give", "necessary", 0)]
        result = self._build(monkeypatch, rows)
        assert result["R"]["all_required"] == ["provide_or_give"]

    def test_punctuation_sanitized_in_all_roles(self, monkeypatch):
        rows = [
            _row(1, "R", True, "Tax Evasion!", "necessary", 0),
            _row(1, "R", True, "bribe (cash)", "fallback", 1),
            _row(1, "R", True, "side-channel", "sufficient", 0),
        ]
        result = self._build(monkeypatch, rows)
        assert result["R"]["all_required"] == ["Tax_Evasion"]
        # hyphen is a \w-safe char in the regex, so it's preserved.
        assert result["R"]["supporting"] == ["side-channel"]
        # space + "(" both map to "_", so two underscores; trailing ")" stripped.
        assert result["R"]["any_of"] == [["bribe__cash"]]

    def test_already_safe_names_unchanged(self, monkeypatch):
        # The common case: simple names are a no-op, so existing rulesets are
        # unaffected.
        rows = [
            _row(1, "R", True, "go", "necessary", 0),
            _row(1, "R", True, "tax", "necessary", 0),
            _row(1, "R", True, "provide_or_give", "necessary", 0),
        ]
        result = self._build(monkeypatch, rows)
        assert result["R"]["all_required"] == ["go", "tax", "provide_or_give"]
