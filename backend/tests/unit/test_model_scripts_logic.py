"""Pure unit tests for the policy/fingerprint/editor-logic helpers in
`sql_scripts.model_scripts`, complementing `test_policy_drift.py`.

NO database or network is available in this environment, so every DB seam is
monkeypatched. `execute_query` / `execute_query_dict` are imported INTO
`model_scripts` at module load time, so we patch them on the `model_scripts`
module namespace (NOT on `utils.sqlite_db`).

The pure, DB-free helper `compute_rule_fingerprint_v2`
(`sql_scripts.junction_scripts`) is exercised for real — it is deterministic,
and using the real one keeps the equivalence/determinism assertions honest.

Functions covered here (with an emphasis on edge cases / properties NOT already
asserted in test_policy_drift.py):
  * compute_classifier_policy_fingerprint_v2 (composite hashing, malformed rows)
  * reconcile_classifier_status              (state-machine boundaries)
  * resolve_editor_logic                     (ids->names, validation, predicate)
  * _build_logic_payload                     (the read-contract shape)
"""
import hashlib

import pytest

from sql_scripts import model_scripts
from sql_scripts.junction_scripts import compute_rule_fingerprint_v2


# ---------------------------------------------------------------------------
# Small fakes for the DB seams
# ---------------------------------------------------------------------------


class _QueuedDictDB:
    """Replays a queued list of return values for execute_query_dict, one per
    call, recording (sql, params) for each invocation."""

    def __init__(self, returns):
        self._returns = list(returns)
        self.calls = []

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        if self._returns:
            return self._returns.pop(0)
        return []


class _Writer:
    """Records execute_query (write/UPDATE) invocations."""

    def __init__(self):
        self.calls = []

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        return None


# ===========================================================================
# compute_classifier_policy_fingerprint_v2  (composite hashing)
# ===========================================================================


class TestPolicyFingerprintComposite:
    def test_setup_rows_hashed_via_v2_rule_fingerprints(self, monkeypatch):
        rows = [
            {"ce_groups": {"g": ["A"]}, "condition": "all of g"},
            {"ce_groups": {"g": ["B", "C"]}, "condition": "1 of g"},
        ]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        result = model_scripts.compute_classifier_policy_fingerprint_v2(1)

        fp1 = compute_rule_fingerprint_v2({"g": ["A"]}, "all of g")
        fp2 = compute_rule_fingerprint_v2({"g": ["B", "C"]}, "1 of g")
        canonical = ";".join(sorted([fp1, fp2]))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert result == expected

    def test_same_logic_split_across_setups_changes_hash(self, monkeypatch):
        # Two groups in ONE setup vs the SAME groups split into two setups are
        # structurally different policies -> different overall fingerprint.
        single = [
            {"ce_groups": {"p": ["A"], "q": ["B"]}, "condition": "all of p and all of q"},
        ]
        split = [
            {"ce_groups": {"p": ["A"]}, "condition": "all of p"},
            {"ce_groups": {"q": ["B"]}, "condition": "all of q"},
        ]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: single)
        fp_single = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: split)
        fp_split = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        assert fp_single != fp_split

    def test_classifier_id_threaded_into_query_params(self, monkeypatch):
        db = _QueuedDictDB([[]])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        model_scripts.compute_classifier_policy_fingerprint_v2(424242)
        assert db.calls[0][1] == (424242,)

    def test_query_targets_active_setup_logic_columns(self, monkeypatch):
        # The fingerprint must read the per-setup logic copy, not the junctions.
        db = _QueuedDictDB([[]])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        model_scripts.compute_classifier_policy_fingerprint_v2(1)
        sql = db.calls[0][0]
        assert "ce_groups" in sql and "condition" in sql
        assert "rule_setup" in sql and "is_active" in sql
        assert "setup_ce_link" not in sql

    def test_result_is_hex_sha256_when_non_empty(self, monkeypatch):
        rows = [{"ce_groups": {"g": ["A"]}, "condition": "all of g"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        out = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        assert len(out) == 64
        int(out, 16)  # raises ValueError if not valid hex

    def test_condition_less_setup_still_counts_membership(self, monkeypatch):
        # A setup with members but a NULL condition contributes its
        # membership-only fingerprint (legacy no-logic rule) — it is part of
        # the trained policy, not invisible.
        rows = [{"ce_groups": {"g": ["A"]}, "condition": None}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        out = model_scripts.compute_classifier_policy_fingerprint_v2(1)
        rule_fp = compute_rule_fingerprint_v2({"g": ["A"]}, None)
        assert out == hashlib.sha256(rule_fp.encode("utf-8")).hexdigest()


# ===========================================================================
# reconcile_classifier_status  (state machine boundaries)
# ===========================================================================


class TestReconcileStateMachine:
    def test_active_no_drift_returns_active_no_write(self, monkeypatch):
        rows = [{"status": "active", "trained_policy_fingerprint": "fp"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint_v2", lambda cid: "fp"
        )
        writer = _Writer()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == "active"
        assert writer.calls == []

    def test_active_with_drift_writes_needs_retraining(self, monkeypatch):
        rows = [{"status": "active", "trained_policy_fingerprint": "trained"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint_v2", lambda cid: "live"
        )
        writer = _Writer()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(11) == "needs_retraining"
        assert len(writer.calls) == 1
        assert writer.calls[0][1] == ("needs_retraining", 11)

    def test_empty_trained_fp_string_is_falsy_passthrough(self, monkeypatch):
        rows = [{"status": "active", "trained_policy_fingerprint": ""}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)

        def _boom(cid):  # pragma: no cover - must not be called
            raise AssertionError("fingerprint should not be recomputed")

        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint_v2", _boom
        )
        writer = _Writer()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == "active"
        assert writer.calls == []

    @pytest.mark.parametrize("status", ["untrained", "training", "error", "queued"])
    def test_non_trainable_statuses_pass_through_untouched(self, status, monkeypatch):
        rows = [{"status": status, "trained_policy_fingerprint": "fp"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)

        def _boom(cid):  # pragma: no cover
            raise AssertionError("should not recompute for non-trainable status")

        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint_v2", _boom
        )
        writer = _Writer()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == status
        assert writer.calls == []

    def test_needs_retraining_heals_to_active_with_writeback(self, monkeypatch):
        rows = [{"status": "needs_retraining", "trained_policy_fingerprint": "fp"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint_v2", lambda cid: "fp"
        )
        writer = _Writer()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(5) == "active"
        assert writer.calls[0][1] == ("active", 5)

    def test_missing_or_none_rows_return_untrained(self, monkeypatch):
        for ret in ([], None):
            monkeypatch.setattr(
                model_scripts, "execute_query_dict", lambda *a, **k: ret
            )
            writer = _Writer()
            monkeypatch.setattr(model_scripts, "execute_query", writer)
            assert model_scripts.reconcile_classifier_status(1) == "untrained"
            assert writer.calls == []


# ===========================================================================
# resolve_editor_logic — the single ids->names+validate+render choke point
# ===========================================================================


class TestResolveEditorLogic:
    def _patch_names(self, monkeypatch, rows):
        db = _QueuedDictDB([rows])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        return db

    def test_translates_ids_to_names_and_renders(self, monkeypatch):
        db = self._patch_names(monkeypatch, [
            {"ce_id": 1, "name": "Alpha"},
            {"ce_id": 2, "name": "Beta"},
        ])
        ce_groups, condition, predicate, ce_ids = model_scripts.resolve_editor_logic(
            {"req": [1], "opt": [2]}, "all of req and 1 of opt")
        assert ce_groups == {"req": ["Alpha"], "opt": ["Beta"]}
        assert condition == "all of req and 1 of opt"
        assert predicate == "Alpha AND Beta"
        assert ce_ids == [1, 2]
        # Single deduped ANY(%s) lookup over the membership union.
        assert len(db.calls) == 1
        assert db.calls[0][1] == ([1, 2],)

    def test_string_ids_coerced_to_int(self, monkeypatch):
        self._patch_names(monkeypatch, [{"ce_id": 5, "name": "N"}])
        ce_groups, _, _, ce_ids = model_scripts.resolve_editor_logic(
            {"core": ["5"]}, "all of core")
        assert ce_groups == {"core": ["N"]}
        assert ce_ids == [5]

    def test_unknown_ce_id_raises(self, monkeypatch):
        self._patch_names(monkeypatch, [{"ce_id": 1, "name": "Alpha"}])
        with pytest.raises(ValueError, match="Unknown CE id"):
            model_scripts.resolve_editor_logic({"g": [1, 77]}, "all of g")

    def test_undefined_group_reference_raises(self, monkeypatch):
        self._patch_names(monkeypatch, [{"ce_id": 1, "name": "Alpha"}])
        with pytest.raises(ValueError, match="Invalid rule logic"):
            model_scripts.resolve_editor_logic({"g": [1]}, "all of ghost")

    def test_empty_group_raises(self, monkeypatch):
        self._patch_names(monkeypatch, [{"ce_id": 1, "name": "Alpha"}])
        with pytest.raises(ValueError, match="no members"):
            model_scripts.resolve_editor_logic(
                {"g": [1], "empty": []}, "all of g")

    def test_unparseable_condition_raises(self, monkeypatch):
        self._patch_names(monkeypatch, [{"ce_id": 1, "name": "Alpha"}])
        with pytest.raises(ValueError, match="Invalid rule logic"):
            model_scripts.resolve_editor_logic({"g": [1]}, "all of g and")

    def test_shared_ce_across_groups_deduped_in_lookup(self, monkeypatch):
        db = self._patch_names(monkeypatch, [{"ce_id": 3, "name": "Shared"}])
        ce_groups, _, _, ce_ids = model_scripts.resolve_editor_logic(
            {"a": [3], "b": [3]}, "all of a or all of b")
        assert ce_groups == {"a": ["Shared"], "b": ["Shared"]}
        assert ce_ids == [3]
        assert db.calls[0][1] == ([3],)


# ===========================================================================
# _build_logic_payload — the read contract for rule cards / the editor
# ===========================================================================


class TestBuildLogicPayload:
    def test_shapes_groups_with_ids_and_names(self):
        membership = [{"ce_id": 1, "name": "A"}, {"ce_id": 2, "name": "B"}]
        out = model_scripts._build_logic_payload(
            {"g": ["A", "B"]}, "all of g", "(A AND B)", membership)
        assert out == {
            "groups": {"g": [{"ce_id": 1, "name": "A"}, {"ce_id": 2, "name": "B"}]},
            "condition": "all of g",
            "predicate": "(A AND B)",
        }

    def test_member_missing_from_membership_degrades_to_none_id(self):
        out = model_scripts._build_logic_payload(
            {"g": ["Ghost"]}, "all of g", "Ghost", [])
        assert out["groups"] == {"g": [{"ce_id": None, "name": "Ghost"}]}

    def test_empty_logic_yields_empty_contract(self):
        out = model_scripts._build_logic_payload(None, None, None, [])
        assert out == {"groups": {}, "condition": "", "predicate": ""}
