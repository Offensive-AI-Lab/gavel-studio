"""Pure unit tests for the policy-drift / retraining helpers in
`sql_scripts.model_scripts` under the v2 (ce_groups, condition) rule model.

There is NO database in this environment, so every DB seam is monkeypatched:
`execute_query` / `execute_query_dict` are imported INTO model_scripts at
module load, so we patch them on the `model_scripts` module namespace. The
draft-creation path also calls `upsert_rule_with_links`, imported at call time
from `gavel_pipeline.db_access`, so we patch it there.

The real `compute_rule_fingerprint_v2` (a pure, DB-free helper) is left in
place — the fingerprint functions are deterministic and exercising the real
one keeps the fingerprint-equivalence assertions honest.

Functions covered:
  * compute_classifier_policy_fingerprint_v2
  * reconcile_classifier_status
  * create_draft_rule_from_bookmarked  (v2 editor-shape signature)
  * the derived display predicate      (utils.rule_condition.render_predicate)
"""
import hashlib

import pytest

from sql_scripts import model_scripts
from sql_scripts.junction_scripts import compute_rule_fingerprint_v2
from utils.rule_condition import parse, render_predicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DictDB:
    """Records calls to execute_query_dict and replays a queued list of
    return values (one per call). Lets a test assert on the SQL/params seen."""

    def __init__(self, returns):
        self._returns = list(returns)
        self.calls = []

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        if self._returns:
            return self._returns.pop(0)
        return []


class _RecordingWriter:
    """Records calls to execute_query (the write/UPDATE seam)."""

    def __init__(self):
        self.calls = []

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        return None


def _setup_row(groups, condition):
    return {"ce_groups": groups, "condition": condition}


# ===========================================================================
# compute_classifier_policy_fingerprint_v2
# ===========================================================================


class TestComputeFingerprint:
    def test_no_active_setups_returns_empty_string(self, monkeypatch):
        db = _DictDB([[]])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        assert model_scripts.compute_classifier_policy_fingerprint_v2(1) == ""
        assert len(db.calls) == 1
        assert db.calls[0][1] == (1,)

    def test_none_rows_treated_as_empty(self, monkeypatch):
        monkeypatch.setattr(
            model_scripts, "execute_query_dict", lambda *a, **k: None
        )
        assert model_scripts.compute_classifier_policy_fingerprint_v2(99) == ""

    def test_setups_without_groups_contribute_nothing(self, monkeypatch):
        # A fresh custom setup (ce_groups {} / None) carries no logic.
        rows = [_setup_row({}, None), _setup_row(None, "all of g")]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        assert model_scripts.compute_classifier_policy_fingerprint_v2(1) == ""

    def test_single_rule_matches_sha256_of_v2_fingerprint(self, monkeypatch):
        rows = [_setup_row({"required": ["A"]}, "all of required")]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        result = model_scripts.compute_classifier_policy_fingerprint_v2(1)

        rule_fp = compute_rule_fingerprint_v2({"required": ["A"]}, "all of required")
        expected = hashlib.sha256(rule_fp.encode("utf-8")).hexdigest()
        assert result == expected
        assert len(result) == 64

    def test_fingerprint_independent_of_rule_order(self, monkeypatch):
        a = [_setup_row({"g": ["A"]}, "all of g"), _setup_row({"g": ["B"]}, "all of g")]
        b = list(reversed(a))
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a_, **k: a)
        fp_a = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a_, **k: b)
        fp_b = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        assert fp_a == fp_b

    def test_fingerprint_independent_of_group_names(self, monkeypatch):
        # Renaming a setup's groups (same structure) must NOT drift the policy.
        a = [_setup_row({"hook": ["A"], "act": ["B", "C"]}, "all of hook and 1 of act")]
        b = [_setup_row({"x": ["A"], "y": ["B", "C"]}, "all of x and 1 of y")]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a_, **k: a)
        fp_a = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a_, **k: b)
        fp_b = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        assert fp_a == fp_b

    def test_different_composition_yields_different_fingerprint(self, monkeypatch):
        a = [_setup_row({"g": ["A"]}, "all of g")]
        b = [_setup_row({"g": ["B"]}, "all of g")]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a_, **k: a)
        fp_a = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a_, **k: b)
        fp_b = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        assert fp_a != fp_b

    def test_quantifier_change_registers_as_drift(self, monkeypatch):
        a = [_setup_row({"g": ["A", "B", "C"]}, "1 of g")]
        b = [_setup_row({"g": ["A", "B", "C"]}, "2 of g")]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a_, **k: a)
        fp_a = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a_, **k: b)
        fp_b = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        assert fp_a != fp_b


# ===========================================================================
# reconcile_classifier_status
# ===========================================================================


class TestReconcileStatus:
    def test_missing_classifier_returns_untrained(self, monkeypatch):
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: [])
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == "untrained"
        assert writer.calls == []

    def test_none_rows_returns_untrained(self, monkeypatch):
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: None)
        assert model_scripts.reconcile_classifier_status(1) == "untrained"

    @pytest.mark.parametrize("status", ["untrained", "training", "error"])
    def test_passthrough_non_trainable_status(self, status, monkeypatch):
        rows = [{"status": status, "trained_policy_fingerprint": "abc"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == status
        assert writer.calls == []

    def test_no_fingerprint_snapshot_passes_through(self, monkeypatch):
        # active but no trained_policy_fingerprint (legacy model) -> unchanged.
        rows = [{"status": "active", "trained_policy_fingerprint": None}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == "active"
        assert writer.calls == []

    def test_missing_fingerprint_key_passes_through(self, monkeypatch):
        rows = [{"status": "active"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        assert model_scripts.reconcile_classifier_status(1) == "active"

    def test_no_drift_stays_active_no_writeback(self, monkeypatch):
        trained_fp = "deadbeef"
        rows = [{"status": "active", "trained_policy_fingerprint": trained_fp}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts,
            "compute_classifier_policy_fingerprint_v2",
            lambda cid: trained_fp,
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == "active"
        assert writer.calls == []

    def test_drift_flips_to_needs_retraining_and_writes_back(self, monkeypatch):
        rows = [{"status": "active", "trained_policy_fingerprint": "trained"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts,
            "compute_classifier_policy_fingerprint_v2",
            lambda cid: "current_differs",
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(7) == "needs_retraining"
        assert len(writer.calls) == 1
        sql, params = writer.calls[0]
        assert "UPDATE classifiers" in sql
        assert params == ("needs_retraining", 7)

    def test_drift_resolved_heals_back_to_active(self, monkeypatch):
        rows = [{"status": "needs_retraining", "trained_policy_fingerprint": "fp"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint_v2", lambda cid: "fp"
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(3) == "active"
        assert len(writer.calls) == 1
        assert writer.calls[0][1] == ("active", 3)

    def test_still_needs_retraining_no_redundant_writeback(self, monkeypatch):
        rows = [{"status": "needs_retraining", "trained_policy_fingerprint": "fp"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint_v2", lambda cid: "other"
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(3) == "needs_retraining"
        assert writer.calls == []


# ===========================================================================
# The derived display predicate — utils.rule_condition.render_predicate
# (replaces the retired _build_predicate_from_roles)
# ===========================================================================


class TestRenderPredicate:
    def _render(self, groups, condition):
        return render_predicate(parse(condition), groups)

    def test_single_member_selector_renders_bare_name(self):
        assert self._render({"g": ["A"]}, "all of g") == "A"

    def test_all_of_multi_member_joined_with_and(self):
        assert self._render({"g": ["A", "B"]}, "all of g") == "(A AND B)"

    def test_one_of_multi_member_joined_with_or(self):
        assert self._render({"g": ["A", "B"]}, "1 of g") == "(A OR B)"

    def test_k_of_renders_at_least_phrase(self):
        out = self._render({"g": ["A", "B", "C"]}, "2 of g")
        assert out == "at least 2 of (A, B, C)"

    def test_k_equal_to_size_renders_as_and(self):
        # 2 of a 2-member group is the same as all of it.
        assert self._render({"g": ["A", "B"]}, "2 of g") == "(A AND B)"

    def test_and_of_selectors(self):
        out = self._render({"req": ["A"], "opt": ["B", "C"]},
                           "all of req and 1 of opt")
        assert out == "A AND (B OR C)"

    def test_or_children_get_parens_inside_and(self):
        out = self._render(
            {"p": ["A"], "q": ["B"], "r": ["C"]},
            "all of p and (all of q or all of r)",
        )
        assert out == "A AND (B OR C)"

    def test_not_renders_upper_not(self):
        out = self._render({"g": ["A"]}, "not all of g")
        assert out == "NOT A"

    def test_legacy_role_shape_renders_like_the_old_builder(self):
        # necessary {A} + fallback groups {B,C} / {D} — the structure legacy
        # role links map to — must render the familiar predicate.
        groups = {"required": ["A"], "option_1": ["B", "C"], "option_2": ["D"]}
        out = self._render(groups, "all of required and 1 of option_1 and 1 of option_2")
        assert out == "A AND (B OR C) AND D"


# ===========================================================================
# create_draft_rule_from_bookmarked — v2 editor-shape signature
# ===========================================================================


class TestCreateDraftFromBookmarked:
    def _patch_upsert(self, monkeypatch, capture):
        import gavel_pipeline.db_access as dba

        def fake_upsert(rule_data, mark_pending=False):
            capture["rule_data"] = rule_data
            capture["mark_pending"] = mark_pending
            return 4242

        monkeypatch.setattr(dba, "upsert_rule_with_links", fake_upsert)

    def _patch_names(self, monkeypatch, name_rows):
        db = _DictDB([name_rows])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        return db

    def test_empty_groups_raises(self, monkeypatch):
        with pytest.raises(ValueError, match="groups cannot be empty"):
            model_scripts.create_draft_rule_from_bookmarked("R", {}, "all of g")

    def test_none_groups_raises(self, monkeypatch):
        with pytest.raises(ValueError):
            model_scripts.create_draft_rule_from_bookmarked("R", None, "all of g")

    def test_groups_with_only_empty_members_raise(self, monkeypatch):
        with pytest.raises(ValueError, match="groups cannot be empty"):
            model_scripts.create_draft_rule_from_bookmarked("R", {"g": []}, "all of g")

    def test_happy_path_returns_rule_id_and_predicate(self, monkeypatch):
        name_rows = [
            {"ce_id": 1, "name": "Alpha"},
            {"ce_id": 2, "name": "Beta"},
            {"ce_id": 3, "name": "Gamma"},
        ]
        db = self._patch_names(monkeypatch, name_rows)
        capture = {}
        self._patch_upsert(monkeypatch, capture)

        rule_id, predicate = model_scripts.create_draft_rule_from_bookmarked(
            "My Draft",
            {"required": [1], "option_1": [2, 3]},
            "all of required and 1 of option_1",
            categories=[7, 8],
            description="d",
        )

        assert rule_id == 4242
        assert predicate == "Alpha AND (Beta OR Gamma)"
        # mark_pending must be True (documented hidden-draft behavior).
        assert capture["mark_pending"] is True
        rd = capture["rule_data"]
        assert rd["rule_name"] == "My Draft"
        # ce_groups stored by NAME — the identity the condition grammar speaks.
        assert rd["ce_groups"] == {"required": ["Alpha"], "option_1": ["Beta", "Gamma"]}
        assert rd["condition"] == "all of required and 1 of option_1"
        assert rd["categories"] == [7, 8]
        assert rd["description"] == "d"
        # The ids -> names lookup was a single ANY(%s) query over the union.
        assert len(db.calls) == 1
        assert db.calls[0][1] == ([1, 2, 3],)

    def test_unknown_ce_id_raises(self, monkeypatch):
        self._patch_names(monkeypatch, [{"ce_id": 1, "name": "Alpha"}])
        capture = {}
        self._patch_upsert(monkeypatch, capture)
        with pytest.raises(ValueError, match="Unknown CE id"):
            model_scripts.create_draft_rule_from_bookmarked(
                "R", {"g": [1, 99]}, "all of g")

    def test_invalid_condition_raises(self, monkeypatch):
        self._patch_names(monkeypatch, [{"ce_id": 1, "name": "Alpha"}])
        capture = {}
        self._patch_upsert(monkeypatch, capture)
        with pytest.raises(ValueError, match="Invalid rule logic"):
            model_scripts.create_draft_rule_from_bookmarked(
                "R", {"g": [1]}, "all of undefined_group")

    def test_unsatisfiable_quantifier_raises(self, monkeypatch):
        self._patch_names(monkeypatch, [{"ce_id": 1, "name": "Alpha"}])
        capture = {}
        self._patch_upsert(monkeypatch, capture)
        with pytest.raises(ValueError, match="never be satisfied"):
            model_scripts.create_draft_rule_from_bookmarked(
                "R", {"g": [1]}, "2 of g")

    def test_categories_default_to_empty_list(self, monkeypatch):
        self._patch_names(monkeypatch, [{"ce_id": 1, "name": "Alpha"}])
        capture = {}
        self._patch_upsert(monkeypatch, capture)
        rule_id, predicate = model_scripts.create_draft_rule_from_bookmarked(
            "X", {"core": [1]}, "all of core")
        assert predicate == "Alpha"
        assert capture["rule_data"]["categories"] == []
        assert capture["rule_data"]["description"] == ""
