"""Integration tests for the auxiliary-dataset upsert + lazy-fetch flow.

The HF push side of the contract is covered by the bootstrap script's own
runs and the Pydantic schema unit tests. This file pins down the local-DB
integration that survives the rule-calibration table merge:

  * `rule_calibration_datasets` is gone — its data lives in `test_datasets`
    (dataset_type='positive_calibration').
  * `ensure_rule_calibration` now reports presence based on a
    `test_datasets` row attached to the rule's classifier.
  * `ensure_rule_aux_for_classifier` summary buckets are accurate against
    the new storage.

We don't touch HuggingFace from these tests — rule calibration is
strictly local and never round-trips through HF anymore.
"""
import json

import pytest


def _insert_pending_ce(name: str, public_id: str | None = None) -> int:
    from utils.sqlite_db import execute_query_dict
    rows = execute_query_dict(
        """
        INSERT INTO cognitive_elements (name, definition, public_id, is_local_draft)
        VALUES (%s, %s, %s, %s) RETURNING ce_id
        """,
        (name, "test", public_id, public_id is None),
    )
    return rows[0]["ce_id"]


def _insert_pending_rule(name: str, public_id: str | None = None) -> int:
    from utils.sqlite_db import execute_query_dict
    rows = execute_query_dict(
        """
        INSERT INTO rules (name, predicate, description, categories, is_ready, public_id, is_local_draft)
        VALUES (%s, %s, %s, %s, TRUE, %s, %s) RETURNING rule_id
        """,
        (name, "A AND B", "test", [], public_id, public_id is None),
    )
    return rows[0]["rule_id"]


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestAuxTablesExist:
    def test_rule_calibration_table_was_merged_away(self, client):
        # The old `rule_calibration_datasets` table is gone after the
        # merge — its content lives in `test_datasets` instead.
        from utils.sqlite_db import execute_query_dict
        rows = execute_query_dict(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'rule_calibration_datasets'
            """
        ) or []
        assert not rows, "rule_calibration_datasets should be dropped post-merge"

    def test_test_datasets_table_present(self, client):
        from utils.sqlite_db import execute_query_dict
        rows = execute_query_dict(
            "SELECT name AS column_name FROM pragma_table_info('test_datasets')"
        ) or []
        cols = {r["column_name"] for r in rows}
        # Must at least have the columns the calibration runner reads from.
        # Test sets are rule-scoped now (v10): no classifier_id.
        assert {"dataset_id", "rule_id", "is_default", "dataset_type", "conversations", "status"}.issubset(cols)
        assert "classifier_id" not in cols

    def test_rule_evaluation_table_dropped(self, client):
        # Earlier iteration created `rule_evaluation_datasets`; the
        # decision was walked back. The migration explicitly drops the
        # table, and there should be no trace of it after init_database.
        from utils.sqlite_db import execute_query_dict
        rows = execute_query_dict(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'rule_evaluation_datasets'
            """
        ) or []
        assert not rows, "rule_evaluation_datasets should have been dropped"


# ---------------------------------------------------------------------------
# Presence helpers — ensure_rule_calibration / ensure_rule_aux_for_classifier
# ---------------------------------------------------------------------------


class TestEnsureRuleCalibrationPresence:
    def test_returns_false_when_classifier_has_no_calibration_row(self, client, test_classifier):
        from services.hf_sync import ensure_rule_calibration
        from utils.sqlite_db import execute_query_dict
        cid = test_classifier["classifier_id"]
        rule_id = _insert_pending_rule("aux_presence_no_row_rule")
        rows = execute_query_dict(
            """
            INSERT INTO rule_setup (classifier_id, rule_id, custom_name, predicate)
            VALUES (%s, %s, %s, %s) RETURNING setup_id
            """,
            (cid, rule_id, "aux_presence_no_row_setup", "A AND B"),
        )
        setup_id = rows[0]["setup_id"]
        try:
            assert ensure_rule_calibration(rule_id) is False
        finally:
            from utils.sqlite_db import execute_query
            execute_query("DELETE FROM rule_setup WHERE setup_id = %s", (setup_id,))

    def test_returns_true_when_calibration_row_exists_for_classifier(self, client, test_classifier):
        from services.hf_sync import ensure_rule_calibration
        from utils.sqlite_db import execute_query_dict, execute_query
        cid = test_classifier["classifier_id"]
        rule_id = _insert_pending_rule("aux_presence_with_row_rule")
        rs_rows = execute_query_dict(
            """
            INSERT INTO rule_setup (classifier_id, rule_id, custom_name, predicate)
            VALUES (%s, %s, %s, %s) RETURNING setup_id
            """,
            (cid, rule_id, "aux_presence_with_row_setup", "A AND B"),
        )
        setup_id = rs_rows[0]["setup_id"]
        td_rows = execute_query_dict(
            """
            INSERT INTO test_datasets
                (rule_id, dataset_type, scenario_name, conversations, status)
            VALUES (%s, 'positive_calibration', 'unit', %s::jsonb, 'ready')
            RETURNING dataset_id
            """,
            (rule_id, json.dumps([[{"role": "user", "content": "x"}]])),
        )
        ds_id = td_rows[0]["dataset_id"]
        try:
            assert ensure_rule_calibration(rule_id) is True
        finally:
            execute_query("DELETE FROM test_datasets WHERE dataset_id = %s", (ds_id,))
            execute_query("DELETE FROM rule_setup WHERE setup_id = %s", (setup_id,))


# ---------------------------------------------------------------------------
# Bulk per-classifier helper
# ---------------------------------------------------------------------------


class TestEnsureRuleAuxForClassifier:
    def test_classifier_with_no_rules_returns_zero_summary(self, client, test_classifier):
        from services.hf_sync import ensure_rule_aux_for_classifier
        from utils.sqlite_db import execute_query
        cid = test_classifier["classifier_id"]
        # Make absolutely sure no rules are linked.
        execute_query("DELETE FROM rule_setup WHERE classifier_id = %s", (cid,))
        summary = ensure_rule_aux_for_classifier(cid)
        assert summary["calibration"] == {"fetched": 0, "missing": 0, "already_present": 0}
