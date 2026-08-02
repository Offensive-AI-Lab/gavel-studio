"""Pure unit tests for ``evaluation.ruleset_builder.build_unified_ruleset``
under the v2 (ce_groups, condition) rule model.

``build_unified_ruleset`` does all DB access through ``execute_query_dict``,
which it imports *lazily inside the function body* from ``utils.sqlite_db``:

    from utils.sqlite_db import execute_query_dict

Because the symbol is re-imported on every call, the only reliable patch point
is the attribute on the ``utils.sqlite_db`` module object itself. We replace
it with a small scripted fake that returns canned rows; no database is
touched.

The function issues TWO kinds of query against the (single) faked function:

  1. The snapshot probe:
       SELECT trained_rule_setup_ids FROM classifiers WHERE classifier_id = %s
  2. The ruleset SELECT over rule_setup (LEFT JOIN rules for the name) —
     the logic now lives on the setup row itself (ce_groups + condition);
     the membership junction is no longer consulted.

Covered: snapshot-vs-live selection (unchanged semantics), the new unified
shape {groups, condition, enabled}, member-name sanitization INSIDE groups
(group names pass through), and the omission of logic-less setups.
"""
import pytest

import utils.sqlite_db as pg
from evaluation.ruleset_builder import build_unified_ruleset


# ---------------------------------------------------------------------------
# Fake DB plumbing
# ---------------------------------------------------------------------------

def _row(setup_id, rule_name, is_active, ce_groups, condition):
    """Build one ruleset-SELECT result row matching the column projection."""
    return {
        "setup_id": setup_id,
        "rule_name": rule_name,
        "is_active": is_active,
        "ce_groups": ce_groups,
        "condition": condition,
    }


def _install(monkeypatch, *, snapshot, ruleset_rows, capture=None):
    """Patch ``execute_query_dict`` with a fake that dispatches on SQL text.

    ``snapshot``      -> what the classifiers probe returns (a list of rows, or
                         None to simulate a missing classifier row).
    ``ruleset_rows``  -> either a single list returned for every ruleset
                         SELECT, OR a dict keyed by the WHERE-clause flavour:
                            "snapshot" for the ``setup_id = ANY(%s)`` query,
                            "live"     for the plain ``classifier_id`` query.
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


_G = {"required": ["CE_A"]}
_C = "all of required"


# ===========================================================================
# Live-fallback branch (no trained snapshot)
# ===========================================================================

class TestLiveFallback:
    def test_no_snapshot_falls_back_to_live_rule_setup(self, monkeypatch):
        rows = [_row(1, "Bribery", True, _G, _C)]
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=rows)
        result = build_unified_ruleset(99)
        assert result == {
            "Bribery": {
                "groups": {"required": ["CE_A"]},
                "condition": "all of required",
                "enabled": True,
            }
        }

    def test_missing_classifier_row_treated_as_no_snapshot(self, monkeypatch):
        rows = [_row(7, "R", True, _G, _C)]
        _install(monkeypatch, snapshot=[], ruleset_rows=rows)
        assert "R" in build_unified_ruleset(5)

    def test_probe_returns_none_is_guarded(self, monkeypatch):
        rows = [_row(7, "R", True, _G, _C)]
        _install(monkeypatch, snapshot=None, ruleset_rows=rows)
        assert "R" in build_unified_ruleset(5)

    def test_empty_trained_ids_list_falls_back_to_live(self, monkeypatch):
        rows = [_row(2, "Live", True, _G, _C)]
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": []}],
                 ruleset_rows=rows)
        assert "Live" in build_unified_ruleset(1)

    def test_live_query_uses_plain_classifier_predicate(self, monkeypatch):
        cap = []
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[_row(1, "R", True, _G, _C)],
                 capture=cap)
        build_unified_ruleset(42)
        ruleset_queries = [q for q, _ in cap if "FROM rule_setup" in q]
        assert len(ruleset_queries) == 1
        assert "ANY(%s)" not in ruleset_queries[0]
        ruleset_params = [p for q, p in cap if "FROM rule_setup" in q][0]
        assert ruleset_params == (42,)

    def test_ruleset_query_reads_logic_columns_not_junction(self, monkeypatch):
        cap = []
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[_row(1, "R", True, _G, _C)],
                 capture=cap)
        build_unified_ruleset(1)
        q = [q for q, _ in cap if "FROM rule_setup" in q][0]
        assert "ce_groups" in q and "condition" in q
        assert "setup_ce_link" not in q

    def test_empty_ruleset_rows_yields_empty_dict(self, monkeypatch):
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[])
        assert build_unified_ruleset(1) == {}

    def test_ruleset_query_returns_none_is_guarded(self, monkeypatch):
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
                "snapshot": [_row(10, "Snap", True, _G, _C)],
                "live": [_row(99, "LIVE_SHOULD_NOT_APPEAR", True, _G, _C)],
            },
            capture=cap,
        )
        result = build_unified_ruleset(3)
        assert "Snap" in result
        assert "LIVE_SHOULD_NOT_APPEAR" not in result
        ruleset_calls = [(q, p) for q, p in cap if "FROM rule_setup" in q]
        assert len(ruleset_calls) == 1
        q, p = ruleset_calls[0]
        assert "ANY(%s)" in q
        assert p == (3, [10, 11])

    def test_orphaned_snapshot_falls_back_to_live(self, monkeypatch):
        cap = []
        _install(
            monkeypatch,
            snapshot=[{"trained_rule_setup_ids": [777]}],
            ruleset_rows={
                "snapshot": [],
                "live": [_row(1, "Recovered", True, _G, _C)],
            },
            capture=cap,
        )
        result = build_unified_ruleset(8)
        assert "Recovered" in result
        ruleset_calls = [q for q, _ in cap if "FROM rule_setup" in q]
        assert len(ruleset_calls) == 2
        assert "ANY(%s)" in ruleset_calls[0]
        assert "ANY(%s)" not in ruleset_calls[1]

    def test_snapshot_with_rows_does_not_issue_live_query(self, monkeypatch):
        cap = []
        _install(
            monkeypatch,
            snapshot=[{"trained_rule_setup_ids": [10]}],
            ruleset_rows={
                "snapshot": [_row(10, "Snap", True, _G, _C)],
                "live": [],
            },
            capture=cap,
        )
        build_unified_ruleset(3)
        ruleset_calls = [q for q, _ in cap if "FROM rule_setup" in q]
        assert len(ruleset_calls) == 1
        assert "ANY(%s)" in ruleset_calls[0]


# ===========================================================================
# The unified shape: groups + condition pass through; logic-less omitted
# ===========================================================================

class TestUnifiedShape:
    def _build(self, monkeypatch, rows):
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=rows)
        return build_unified_ruleset(1)

    def test_multi_group_rule_passes_groups_and_condition_through(self, monkeypatch):
        groups = {"hook": ["CE_A"], "action": ["CE_B", "CE_C"]}
        cond = "all of hook and 1 of action"
        result = self._build(monkeypatch, [_row(1, "R", True, groups, cond)])
        assert result["R"]["groups"] == groups
        assert result["R"]["condition"] == cond

    def test_group_names_pass_through_unsanitized(self, monkeypatch):
        # Group names only ever meet the condition parser — never the model —
        # so they must NOT be touched by CE-name sanitization.
        groups = {"my_group_2": ["CE_A"]}
        result = self._build(monkeypatch, [_row(1, "R", True, groups, "all of my_group_2")])
        assert list(result["R"]["groups"].keys()) == ["my_group_2"]

    def test_setup_without_groups_is_omitted(self, monkeypatch):
        rows = [
            _row(1, "Empty", True, {}, "all of g"),
            _row(2, "Kept", True, _G, _C),
        ]
        result = self._build(monkeypatch, rows)
        assert "Empty" not in result
        assert "Kept" in result

    def test_setup_without_condition_is_omitted(self, monkeypatch):
        # A migrated supporting-only rule has groups but no condition — no
        # firing semantics, so it must not appear in the unified ruleset.
        rows = [
            _row(1, "SupportingOnly", True, {"supporting": ["CE_S"]}, None),
            _row(2, "Kept", True, _G, _C),
        ]
        result = self._build(monkeypatch, rows)
        assert "SupportingOnly" not in result
        assert "Kept" in result

    def test_blank_condition_is_omitted(self, monkeypatch):
        rows = [_row(1, "Blank", True, _G, "   ")]
        assert self._build(monkeypatch, rows) == {}

    def test_none_groups_guarded(self, monkeypatch):
        rows = [_row(1, "NoneGroups", True, None, _C)]
        assert self._build(monkeypatch, rows) == {}


# ===========================================================================
# Misc semantics: enabled flag, naming, multiple rules
# ===========================================================================

class TestMiscSemantics:
    def test_is_active_false_sets_enabled_false(self, monkeypatch):
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[_row(1, "R", False, _G, _C)])
        result = build_unified_ruleset(1)
        assert result["R"]["enabled"] is False

    def test_is_active_truthy_coerced_to_bool(self, monkeypatch):
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[_row(1, "R", 1, _G, _C)])
        result = build_unified_ruleset(1)
        assert result["R"]["enabled"] is True

    def test_null_rule_name_falls_back_to_rule_id_label(self, monkeypatch):
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=[_row(55, None, True, _G, _C)])
        result = build_unified_ruleset(1)
        assert "rule_55" in result

    def test_multiple_rules_keyed_by_name(self, monkeypatch):
        rows = [
            _row(1, "Alpha", True, {"g": ["CE_A"]}, "all of g"),
            _row(2, "Beta", True, {"g": ["CE_B"]}, "1 of g"),
        ]
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=rows)
        result = build_unified_ruleset(1)
        assert set(result.keys()) == {"Alpha", "Beta"}
        assert result["Alpha"]["groups"] == {"g": ["CE_A"]}
        assert result["Beta"]["condition"] == "1 of g"


# ===========================================================================
# CE-name sanitization — must match the trained classifier's labels-dict keys
# (classifier_engine.trainer._sanitize_label), otherwise member CEs are
# silently dropped and rules look "missing required CEs" though they triggered.
# Sanitization applies INSIDE groups only; group names pass through.
# ===========================================================================

class TestCeNameSanitization:
    def _build(self, monkeypatch, rows):
        _install(monkeypatch, snapshot=[{"trained_rule_setup_ids": None}],
                 ruleset_rows=rows)
        return build_unified_ruleset(1)

    def test_spaces_become_underscores(self, monkeypatch):
        rows = [_row(1, "R", True, {"g": ["provide or give"]}, "all of g")]
        result = self._build(monkeypatch, rows)
        assert result["R"]["groups"] == {"g": ["provide_or_give"]}

    def test_punctuation_sanitized_in_every_group(self, monkeypatch):
        groups = {
            "req": ["Tax Evasion!"],
            "opt": ["bribe (cash)"],
            "extra": ["side-channel"],
        }
        rows = [_row(1, "R", True, groups, "all of req and 1 of opt and 1 of extra")]
        result = self._build(monkeypatch, rows)
        assert result["R"]["groups"]["req"] == ["Tax_Evasion"]
        # space + "(" both map to "_", so two underscores; trailing ")" stripped.
        assert result["R"]["groups"]["opt"] == ["bribe__cash"]
        # hyphen is a \w-safe char in the regex, so it's preserved.
        assert result["R"]["groups"]["extra"] == ["side-channel"]

    def test_already_safe_names_unchanged(self, monkeypatch):
        groups = {"g": ["go", "tax", "provide_or_give"]}
        rows = [_row(1, "R", True, groups, "all of g")]
        result = self._build(monkeypatch, rows)
        assert result["R"]["groups"] == {"g": ["go", "tax", "provide_or_give"]}
