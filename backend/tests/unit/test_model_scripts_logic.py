"""Pure unit tests for the policy/fingerprint logic in
`sql_scripts.model_scripts`, complementing `test_policy_drift.py`.

NO database or network is available in this environment, so every DB seam is
monkeypatched. `execute_query` / `execute_query_dict` are imported INTO
`model_scripts` at module load time, so we patch them on the `model_scripts`
module namespace (NOT on `utils.sqlite_db`). The draft-creation path imports
`upsert_rule_with_links` at call time from `gavel_pipeline.db_access`, so we
patch it there.

The pure, DB-free helper `compute_rule_fingerprint_from_links`
(`sql_scripts.junction_scripts`) is exercised for real — it is deterministic,
and using the real one keeps the equivalence/determinism assertions honest.

Functions covered here (with an emphasis on edge cases / properties NOT already
asserted in test_policy_drift.py):
  * compute_rule_fingerprint_from_links  (determinism, role/order/grouping)
  * compute_classifier_policy_fingerprint (composite-key grouping, malformed rows)
  * reconcile_classifier_status           (state-machine boundaries)
  * create_draft_rule_from_bookmarked     (role mapping, fallback ordering)
"""
import hashlib

import pytest

from sql_scripts import model_scripts
from sql_scripts.junction_scripts import compute_rule_fingerprint_from_links


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
# compute_rule_fingerprint_from_links  (the pure structural fingerprint)
# ===========================================================================


class TestRuleFingerprintFromLinks:
    def test_empty_links_is_stable_canonical_form(self):
        # Both [] and None route through `ce_links or []` to the same string.
        empty = "N:()|F:[]|S:()"
        assert compute_rule_fingerprint_from_links([]) == empty
        assert compute_rule_fingerprint_from_links(None) == empty

    def test_order_independent_over_necessary(self):
        a = compute_rule_fingerprint_from_links(
            [{"ce_id": 3, "role": "necessary"}, {"ce_id": 1, "role": "necessary"}]
        )
        b = compute_rule_fingerprint_from_links(
            [{"ce_id": 1, "role": "necessary"}, {"ce_id": 3, "role": "necessary"}]
        )
        assert a == b

    def test_deterministic_same_content_same_hash(self):
        links = [
            {"ce_id": 5, "role": "necessary"},
            {"ce_id": 9, "role": "sufficient"},
        ]
        assert compute_rule_fingerprint_from_links(
            list(links)
        ) == compute_rule_fingerprint_from_links(list(links))

    def test_role_distinguishes_fingerprint(self):
        # Same ce_id, different role -> different fingerprint.
        as_nec = compute_rule_fingerprint_from_links([{"ce_id": 1, "role": "necessary"}])
        as_suf = compute_rule_fingerprint_from_links([{"ce_id": 1, "role": "sufficient"}])
        assert as_nec != as_suf

    def test_none_ce_id_links_are_dropped(self):
        # A link with ce_id None must be ignored, leaving the canonical empty form.
        out = compute_rule_fingerprint_from_links(
            [{"ce_id": None, "role": "necessary"}]
        )
        assert out == "N:()|F:[]|S:()"

    def test_missing_role_defaults_to_necessary(self):
        no_role = compute_rule_fingerprint_from_links([{"ce_id": 7}])
        explicit = compute_rule_fingerprint_from_links(
            [{"ce_id": 7, "role": "necessary"}]
        )
        assert no_role == explicit

    def test_role_case_insensitive(self):
        upper = compute_rule_fingerprint_from_links([{"ce_id": 1, "role": "NECESSARY"}])
        lower = compute_rule_fingerprint_from_links([{"ce_id": 1, "role": "necessary"}])
        assert upper == lower

    def test_fallback_grouping_normalized_within_group(self):
        # Members within a fallback group are sorted, so member order in a group
        # doesn't matter.
        a = compute_rule_fingerprint_from_links(
            [
                {"ce_id": 2, "role": "fallback", "fallback_group": 1},
                {"ce_id": 1, "role": "fallback", "fallback_group": 1},
            ]
        )
        b = compute_rule_fingerprint_from_links(
            [
                {"ce_id": 1, "role": "fallback", "fallback_group": 1},
                {"ce_id": 2, "role": "fallback", "fallback_group": 1},
            ]
        )
        assert a == b

    def test_fallback_group_numbering_irrelevant_only_partition_matters(self):
        # The fingerprint sorts groups as tuples, so the actual group *number*
        # is irrelevant — only the partition of ce_ids into groups matters.
        groups_1_2 = compute_rule_fingerprint_from_links(
            [
                {"ce_id": 1, "role": "fallback", "fallback_group": 1},
                {"ce_id": 2, "role": "fallback", "fallback_group": 2},
            ]
        )
        groups_5_9 = compute_rule_fingerprint_from_links(
            [
                {"ce_id": 1, "role": "fallback", "fallback_group": 5},
                {"ce_id": 2, "role": "fallback", "fallback_group": 9},
            ]
        )
        assert groups_1_2 == groups_5_9

    def test_distinct_partitions_differ(self):
        # {1},{2} (two singleton groups) vs {1,2} (one group) must differ.
        two_groups = compute_rule_fingerprint_from_links(
            [
                {"ce_id": 1, "role": "fallback", "fallback_group": 1},
                {"ce_id": 2, "role": "fallback", "fallback_group": 2},
            ]
        )
        one_group = compute_rule_fingerprint_from_links(
            [
                {"ce_id": 1, "role": "fallback", "fallback_group": 1},
                {"ce_id": 2, "role": "fallback", "fallback_group": 1},
            ]
        )
        assert two_groups != one_group


# ===========================================================================
# compute_classifier_policy_fingerprint  (composite-key grouping + hashing)
# ===========================================================================


class TestPolicyFingerprintComposite:
    def test_rows_grouped_by_setup_id_then_hashed(self, monkeypatch):
        # Two distinct setups; verify the result equals the sha256 of the sorted
        # per-rule fingerprints joined by ';'.
        rows = [
            {"setup_id": 1, "ce_id": 5, "role": "necessary", "fallback_group": 0},
            {"setup_id": 2, "ce_id": 9, "role": "sufficient", "fallback_group": 0},
        ]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        result = model_scripts.compute_classifier_policy_fingerprint(1)

        fp1 = compute_rule_fingerprint_from_links(
            [{"ce_id": 5, "role": "necessary", "fallback_group": 0}]
        )
        fp2 = compute_rule_fingerprint_from_links(
            [{"ce_id": 9, "role": "sufficient", "fallback_group": 0}]
        )
        canonical = ";".join(sorted([fp1, fp2]))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert result == expected

    def test_multiple_links_in_one_setup_collected(self, monkeypatch):
        # Several rows sharing a setup_id form ONE rule with multiple CEs.
        rows = [
            {"setup_id": 7, "ce_id": 1, "role": "necessary", "fallback_group": 0},
            {"setup_id": 7, "ce_id": 2, "role": "necessary", "fallback_group": 0},
        ]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        result = model_scripts.compute_classifier_policy_fingerprint(1)

        rule_fp = compute_rule_fingerprint_from_links(
            [
                {"ce_id": 1, "role": "necessary", "fallback_group": 0},
                {"ce_id": 2, "role": "necessary", "fallback_group": 0},
            ]
        )
        expected = hashlib.sha256(rule_fp.encode("utf-8")).hexdigest()
        assert result == expected

    def test_same_content_split_across_setups_changes_hash(self, monkeypatch):
        # Two CEs in ONE setup vs the SAME two CEs split into two setups are
        # structurally different policies -> different overall fingerprint.
        single = [
            {"setup_id": 1, "ce_id": 1, "role": "necessary", "fallback_group": 0},
            {"setup_id": 1, "ce_id": 2, "role": "necessary", "fallback_group": 0},
        ]
        split = [
            {"setup_id": 1, "ce_id": 1, "role": "necessary", "fallback_group": 0},
            {"setup_id": 2, "ce_id": 2, "role": "necessary", "fallback_group": 0},
        ]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: single)
        fp_single = model_scripts.compute_classifier_policy_fingerprint(1)
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: split)
        fp_split = model_scripts.compute_classifier_policy_fingerprint(1)
        assert fp_single != fp_split

    def test_classifier_id_threaded_into_query_params(self, monkeypatch):
        db = _QueuedDictDB([[]])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        model_scripts.compute_classifier_policy_fingerprint(424242)
        assert db.calls[0][1] == (424242,)

    def test_result_is_hex_sha256_when_non_empty(self, monkeypatch):
        rows = [{"setup_id": 1, "ce_id": 5, "role": "necessary", "fallback_group": 0}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        out = model_scripts.compute_classifier_policy_fingerprint(1)
        assert len(out) == 64
        int(out, 16)  # raises ValueError if not valid hex


# ===========================================================================
# reconcile_classifier_status  (state machine boundaries)
# ===========================================================================


class TestReconcileStateMachine:
    def test_active_no_drift_returns_active_no_write(self, monkeypatch):
        rows = [{"status": "active", "trained_policy_fingerprint": "fp"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint", lambda cid: "fp"
        )
        writer = _Writer()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == "active"
        assert writer.calls == []

    def test_active_with_drift_writes_needs_retraining(self, monkeypatch):
        rows = [{"status": "active", "trained_policy_fingerprint": "trained"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint", lambda cid: "live"
        )
        writer = _Writer()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(11) == "needs_retraining"
        assert len(writer.calls) == 1
        assert writer.calls[0][1] == ("needs_retraining", 11)

    def test_empty_trained_fp_string_is_falsy_passthrough(self, monkeypatch):
        # An empty-string fingerprint is falsy -> `not trained_fp` short-circuits
        # and the status passes through WITHOUT recomputation.
        rows = [{"status": "active", "trained_policy_fingerprint": ""}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)

        def _boom(cid):  # pragma: no cover - must not be called
            raise AssertionError("fingerprint should not be recomputed")

        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint", _boom
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
            model_scripts, "compute_classifier_policy_fingerprint", _boom
        )
        writer = _Writer()
        monkeypatch.setattr(model_scripts, "execute_query", writer)
        assert model_scripts.reconcile_classifier_status(1) == status
        assert writer.calls == []

    def test_needs_retraining_heals_to_active_with_writeback(self, monkeypatch):
        rows = [{"status": "needs_retraining", "trained_policy_fingerprint": "fp"}]
        monkeypatch.setattr(model_scripts, "execute_query_dict", lambda *a, **k: rows)
        monkeypatch.setattr(
            model_scripts, "compute_classifier_policy_fingerprint", lambda cid: "fp"
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
# create_draft_rule_from_bookmarked  (role mapping + fallback ordering)
# ===========================================================================


def _patch_upsert(monkeypatch, capture, return_id=999):
    import gavel_pipeline.db_access as dba

    def fake_upsert(rule_data, mark_pending=False):
        capture["rule_data"] = rule_data
        capture["mark_pending"] = mark_pending
        return return_id

    monkeypatch.setattr(dba, "upsert_rule_with_links", fake_upsert)


class TestCreateDraftFromBookmarked:
    def test_empty_and_none_ce_roles_raise(self, monkeypatch):
        with pytest.raises(ValueError, match="ce_roles cannot be empty"):
            model_scripts.create_draft_rule_from_bookmarked("R", [])
        with pytest.raises(ValueError):
            model_scripts.create_draft_rule_from_bookmarked("R", None)

    def test_mark_pending_true_and_role_buckets(self, monkeypatch):
        ce_roles = [
            {"ce_id": 1, "role": "necessary"},
            {"ce_id": 2, "role": "sufficient"},
        ]
        name_rows = [{"ce_id": 1, "name": "A"}, {"ce_id": 2, "name": "B"}]
        monkeypatch.setattr(
            model_scripts, "execute_query_dict", lambda *a, **k: name_rows
        )
        capture = {}
        _patch_upsert(monkeypatch, capture)

        rule_id, predicate = model_scripts.create_draft_rule_from_bookmarked(
            "Draft", ce_roles
        )
        assert rule_id == 999
        assert capture["mark_pending"] is True
        rd = capture["rule_data"]
        assert rd["necessary"] == ["A"]
        assert rd["sufficient"] == ["B"]
        assert rd["fallback"] == []
        # 'sufficient' CEs are helpful-only and excluded from the firing
        # predicate, so only the necessary CE (A) appears.
        assert predicate == "A"
        assert rd["predicate"] == predicate

    def test_multiple_fallback_groups_ordered_by_group_key(self, monkeypatch):
        # Groups must be emitted in ascending group-key order regardless of the
        # order roles appear in the input list.
        ce_roles = [
            {"ce_id": 3, "role": "fallback", "fallback_group": 2},
            {"ce_id": 1, "role": "fallback", "fallback_group": 1},
            {"ce_id": 2, "role": "fallback", "fallback_group": 1},
        ]
        name_rows = [
            {"ce_id": 1, "name": "G1a"},
            {"ce_id": 2, "name": "G1b"},
            {"ce_id": 3, "name": "G2"},
        ]
        monkeypatch.setattr(
            model_scripts, "execute_query_dict", lambda *a, **k: name_rows
        )
        capture = {}
        _patch_upsert(monkeypatch, capture)

        rule_id, predicate = model_scripts.create_draft_rule_from_bookmarked(
            "D", ce_roles
        )
        rd = capture["rule_data"]
        # group 1 first (two members), then group 2.
        assert rd["fallback"] == [["G1a", "G1b"], ["G2"]]
        assert predicate == "(G1a OR G1b) AND (G2)"

    def test_duplicate_role_entries_both_kept_in_bucket(self, monkeypatch):
        # The bucket logic does not dedupe; two necessary CEs both land in the
        # necessary bucket and the predicate.
        ce_roles = [
            {"ce_id": 1, "role": "necessary"},
            {"ce_id": 2, "role": "necessary"},
        ]
        name_rows = [{"ce_id": 1, "name": "A"}, {"ce_id": 2, "name": "B"}]
        monkeypatch.setattr(
            model_scripts, "execute_query_dict", lambda *a, **k: name_rows
        )
        capture = {}
        _patch_upsert(monkeypatch, capture)
        _, predicate = model_scripts.create_draft_rule_from_bookmarked("D", ce_roles)
        assert capture["rule_data"]["necessary"] == ["A", "B"]
        assert predicate == "A AND B"

    def test_none_ce_id_excluded_from_lookup_params(self, monkeypatch):
        ce_roles = [
            {"ce_id": 1, "role": "necessary"},
            {"role": "necessary"},  # ce_id absent
        ]
        db = _QueuedDictDB([[{"ce_id": 1, "name": "A"}]])
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        capture = {}
        _patch_upsert(monkeypatch, capture)
        model_scripts.create_draft_rule_from_bookmarked("D", ce_roles)
        # Only the non-None ce_id is passed to the IN(...) lookup.
        assert db.calls[0][1] == (1,)

    def test_all_none_ce_ids_skips_db_lookup_entirely(self, monkeypatch):
        ce_roles = [{"role": "necessary"}, {"role": "sufficient"}]
        db = _QueuedDictDB([])  # would error on a popped call beyond default
        monkeypatch.setattr(model_scripts, "execute_query_dict", db)
        capture = {}
        _patch_upsert(monkeypatch, capture)
        rule_id, _ = model_scripts.create_draft_rule_from_bookmarked("D", ce_roles)
        assert db.calls == []  # no name lookup attempted
        assert rule_id == 999
        assert capture["rule_data"]["necessary"] == []
        assert capture["rule_data"]["sufficient"] == []
