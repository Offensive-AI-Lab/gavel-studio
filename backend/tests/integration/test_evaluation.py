"""Tests for evaluation pipeline: calibration status, results, test datasets."""
import pytest


class TestCalibrationStatus:
    """Calibration dataset status checks."""

    def test_get_calibration_status(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/evaluation/{cid}/calibration-status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "ces" in data
        assert "all_ready" in data
        assert isinstance(data["ces"], list)

    def test_get_thresholds_before_calibration(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/evaluation/{cid}/thresholds", headers=auth_headers)
        # Should 404 if never calibrated
        assert res.status_code in (200, 404)


class TestEvaluationResults:
    """Evaluation results retrieval."""

    def test_get_results_empty(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/evaluation/{cid}/results", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "calibration" in data
        assert "evaluation" in data

    def test_get_results_history(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/evaluation/{cid}/results/history", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "results" in data
        assert isinstance(data["results"], list)


class TestPostRetrainResultsFilter:
    """Pin down the snapshot-based filter that hides pre-retrain
    evaluation_results rows.

    Setup pattern for each test:
      1. Insert a row dated `before` retrain (created_at backdated).
      2. Bump the classifier's trained_at to `now()` to simulate retrain.
      3. Insert a row dated `after` retrain.
      4. Confirm the relevant endpoint surfaces only the post-retrain row.

    The pre-retrain row is left in the DB — the filter is a read-time
    masking, not a destructive write — so the rest of the assertion is
    that it's still queryable directly even though the endpoint hides it.
    """

    @staticmethod
    def _insert_eval_row(classifier_id: int, eval_type: str, *, age_seconds: int,
                        thresholds: dict | None = None) -> int:
        """Insert an evaluation_results row backdated by `age_seconds`."""
        import json
        from utils.sqlite_db import execute_query_dict
        rows = execute_query_dict(
            """
            INSERT INTO evaluation_results (
                classifier_id, eval_type, thresholds, metrics, plots, created_at
            )
            VALUES (%s, %s, %s::jsonb, NULL, NULL, datetime('now', '-' || %s || ' seconds'))
            RETURNING eval_id
            """,
            (
                classifier_id,
                eval_type,
                json.dumps(thresholds) if thresholds is not None else None,
                str(age_seconds),
            ),
        )
        return rows[0]["eval_id"]

    @staticmethod
    def _set_trained_at(classifier_id: int, age_seconds: int) -> None:
        """Set classifiers.trained_at to (now - age_seconds). Negative
        age means future-dated, but we always pass non-negative here."""
        from utils.sqlite_db import execute_query
        execute_query(
            "UPDATE classifiers SET trained_at = datetime('now', '-' || %s || ' seconds') WHERE classifier_id = %s",
            (str(age_seconds), classifier_id),
        )

    def test_history_hides_pre_retrain_rows(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        # Pre-retrain calibration row 60 s ago
        old_id = self._insert_eval_row(cid, "calibration", age_seconds=60,
                                        thresholds={"old": 0.5})
        # Retrain happened 30 s ago
        self._set_trained_at(cid, 30)
        # Post-retrain calibration row 5 s ago
        new_id = self._insert_eval_row(cid, "calibration", age_seconds=5,
                                        thresholds={"new": 0.7})

        res = client.get(f"/evaluation/{cid}/results/history", headers=auth_headers)
        assert res.status_code == 200
        ids = {r["eval_id"] for r in res.json()["results"]}
        assert new_id in ids
        assert old_id not in ids

    def test_thresholds_endpoint_returns_only_post_retrain(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        self._insert_eval_row(cid, "calibration", age_seconds=60,
                               thresholds={"stale": 0.5})
        self._set_trained_at(cid, 30)
        self._insert_eval_row(cid, "calibration", age_seconds=5,
                               thresholds={"current": 0.7})

        res = client.get(f"/evaluation/{cid}/thresholds", headers=auth_headers)
        assert res.status_code == 200
        # The endpoint surfaces the freshly-calibrated thresholds, not the stale ones.
        assert res.json()["thresholds"] == {"current": 0.7}

    def test_thresholds_404_when_only_pre_retrain_exists(self, client, test_classifier, auth_headers):
        # Worst case: classifier has been retrained but never recalibrated.
        # The endpoint must hide the stale thresholds AND return 404, so
        # the user is forced to recalibrate against the current model.
        cid = test_classifier["classifier_id"]
        self._insert_eval_row(cid, "calibration", age_seconds=60,
                               thresholds={"stale": 0.5})
        self._set_trained_at(cid, 30)

        res = client.get(f"/evaluation/{cid}/thresholds", headers=auth_headers)
        assert res.status_code == 404

    def test_pre_retrain_row_still_in_db(self, client, test_classifier, auth_headers):
        # Filter is read-time masking, NOT destructive. The pre-retrain
        # row is still queryable by direct DB inspection — only the
        # endpoints hide it.
        from utils.sqlite_db import execute_query_dict
        cid = test_classifier["classifier_id"]
        old_id = self._insert_eval_row(cid, "calibration", age_seconds=60)
        self._set_trained_at(cid, 30)

        rows = execute_query_dict(
            "SELECT eval_id FROM evaluation_results WHERE eval_id = %s", (old_id,)
        )
        assert rows, "pre-retrain row should NOT have been deleted by the filter"

    def test_results_endpoint_returns_none_when_only_pre_retrain(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        self._insert_eval_row(cid, "calibration", age_seconds=60,
                               thresholds={"stale": 0.5})
        self._insert_eval_row(cid, "evaluation", age_seconds=60)
        self._set_trained_at(cid, 30)

        res = client.get(f"/evaluation/{cid}/results", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        # Both surfaces should be empty — no current-model results yet.
        assert data["calibration"] is None
        assert data["evaluation"] is None

    def test_never_trained_classifier_passes_filter_through(self, client, test_classifier, auth_headers):
        # Defensive: when trained_at IS NULL, the filter must NOT hide
        # rows. COALESCE to '-infinity' makes the predicate vacuously true.
        from utils.sqlite_db import execute_query
        cid = test_classifier["classifier_id"]
        execute_query(
            "UPDATE classifiers SET trained_at = NULL WHERE classifier_id = %s",
            (cid,),
        )
        eid = self._insert_eval_row(cid, "calibration", age_seconds=60,
                                     thresholds={"untrained": 0.42})

        res = client.get(f"/evaluation/{cid}/thresholds", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["thresholds"] == {"untrained": 0.42}


class TestEvaluationEndpoints:
    """Calibration and evaluation trigger tests."""

    def test_calibrate_untrained_classifier(self, client, test_classifier, auth_headers):
        """Calibration should fail on untrained classifier."""
        cid = test_classifier["classifier_id"]
        res = client.post(f"/evaluation/{cid}/calibrate", json={}, headers=auth_headers)
        # Should fail because classifier isn't trained
        assert res.status_code in (400, 500)

    def test_evaluate_untrained_classifier(self, client, test_classifier, auth_headers):
        """Evaluation should fail on untrained classifier."""
        cid = test_classifier["classifier_id"]
        res = client.post(f"/evaluation/{cid}/evaluate", json={
            "test_dataset_ids": [1],
        }, headers=auth_headers)
        assert res.status_code in (400, 500)

    def test_evaluate_no_datasets(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.post(f"/evaluation/{cid}/evaluate", json={
            "test_dataset_ids": [],
        }, headers=auth_headers)
        assert res.status_code in (400, 422)


class TestTestDatasets:
    """Test dataset listing."""

    def test_list_test_datasets(self, client, test_classifier, auth_headers):
        # Test sets are rule-scoped (v10); list by rule id. An arbitrary id
        # with no datasets still returns an empty list.
        res = client.get("/ai/test-sets/by-rule/999999", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "datasets" in data
        assert isinstance(data["datasets"], list)
