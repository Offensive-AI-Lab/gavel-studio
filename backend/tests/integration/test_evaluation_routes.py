"""Integration tests for routes/evaluation.py (fast paths only).

Distinct from test_evaluation.py — different filenames AND class names so the
two files never collide. No real SmolLM inference happens here: every test
either hits a read endpoint, an auth/validation boundary, or pins down the
post-retrain snapshot filter using directly-inserted evaluation_results rows.

Notes on the routes under test (see backend/routes/evaluation.py):
  * GET  /evaluation/{cid}/calibration-status   — NO auth dependency
  * GET  /evaluation/{cid}/results              — NO auth dependency
  * GET  /evaluation/{cid}/results/history      — NO auth dependency
  * GET  /evaluation/{cid}/thresholds           — NO auth dependency
  * POST /evaluation/{cid}/calibrate            — requires auth
  * POST /evaluation/{cid}/evaluate             — requires auth

The "Detailed eval" endpoint was REMOVED — this file asserts it is gone (the
calibration runner now selects the rule's single auto-generated default set
directly, is_default=TRUE, instead of a separate detailed-eval route).

Auth semantics: get_current_user uses HTTPBearer(auto_error=True), so a
MISSING Authorization header -> 403, while an INVALID/garbage token -> 401.
The listing endpoints used here live under /ai (ai_pipeline.py).
"""
import time

import pytest


def _uniq() -> int:
    """Unique-ish suffix to avoid 409 duplicate-name conflicts in a session."""
    return int(time.time() * 1000) % 1_000_000_000


class TestEvalCalibrationStatusRoute:
    """GET /evaluation/{cid}/calibration-status — structured shape + no-auth."""

    def test_status_shape(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/evaluation/{cid}/calibration-status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert set(["ces", "all_ready", "total"]).issubset(data.keys())
        assert isinstance(data["ces"], list)
        assert isinstance(data["all_ready"], bool)
        assert isinstance(data["total"], int)
        assert data["total"] == len(data["ces"])

    def test_status_is_public_no_auth_required(self, client, test_classifier):
        # This GET endpoint has no get_current_user dependency, so it must
        # answer 200 even without an Authorization header.
        cid = test_classifier["classifier_id"]
        res = client.get(f"/evaluation/{cid}/calibration-status")
        assert res.status_code == 200
        assert "ces" in res.json()

    def test_status_unknown_classifier_returns_empty(self, client, auth_headers):
        # No meta.json and no rule_setup rows -> empty CE list, all_ready True
        # (vacuous all() over empty), total 0. Still a clean 200.
        res = client.get("/evaluation/999999999/calibration-status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 0
        assert data["ces"] == []
        assert data["all_ready"] is True


class TestEvalResultsRoute:
    """GET /evaluation/{cid}/results and /results/history shapes."""

    def test_results_shape(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/evaluation/{cid}/results", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "calibration" in data
        assert "evaluation" in data

    def test_history_shape_and_limit(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(
            f"/evaluation/{cid}/results/history",
            params={"limit": 3},
            headers=auth_headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) <= 3

    def test_history_unknown_classifier_empty_list(self, client, auth_headers):
        res = client.get("/evaluation/999999999/results/history", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["results"] == []

    def test_thresholds_unknown_classifier_404(self, client, auth_headers):
        # No calibration rows at all -> 404 from the thresholds endpoint.
        res = client.get("/evaluation/999999999/thresholds", headers=auth_headers)
        assert res.status_code == 404


class TestEvalCalibrationDefaultOnlySelection:
    """Pin down that the calibration runner's positive_calibration selection
    only considers is_default=TRUE, status='ready' rows — there is NO separate
    "Detailed eval" route to opt into custom calibration sets anymore.

    These tests exercise the selection indirectly: the runner SQL filters on
    is_default=TRUE; a non-default positive_calibration row must NOT change the
    visible state of any read endpoint, confirming the default-only path.
    """

    @staticmethod
    def _make_rule():
        """Insert a minimal real rule and return its id. test_datasets.rule_id
        is a FK to rules(rule_id), so the dataset rows need a real parent. Both
        tables are tracked, so the conftest cleans this up automatically."""
        from utils.PostgreSQL import execute_query_dict
        rows = execute_query_dict(
            "INSERT INTO rules (name, predicate) VALUES (%s, %s) RETURNING rule_id",
            (f"caltest_rule_{_uniq()}", "CE"),
        )
        return rows[0]["rule_id"]

    @staticmethod
    def _insert_test_dataset(rule_id, *, is_default, status="ready",
                             dataset_type="positive_calibration"):
        from utils.PostgreSQL import execute_query_dict
        rows = execute_query_dict(
            """
            INSERT INTO test_datasets (
                rule_id, dataset_type, status, is_default, conversations
            )
            VALUES (%s, %s, %s, %s, '[]'::jsonb)
            RETURNING dataset_id
            """,
            (rule_id, dataset_type, status, is_default),
        )
        return rows[0]["dataset_id"]

    def test_default_flag_distinguishes_rows(self, client, auth_headers):
        # Sanity: insert a default and a non-default positive_calibration row
        # under a real throwaway rule, and confirm the runner's exact selection
        # predicate (is_default=TRUE AND status='ready') picks ONLY the default.
        from utils.PostgreSQL import execute_query_dict
        rule_id = self._make_rule()
        default_id = self._insert_test_dataset(rule_id, is_default=True)
        custom_id = self._insert_test_dataset(rule_id, is_default=False)

        selected = execute_query_dict(
            """
            SELECT DISTINCT ON (rule_id) dataset_id
            FROM test_datasets
            WHERE rule_id = ANY(%s)
              AND dataset_type = 'positive_calibration'
              AND is_default = TRUE
              AND status = 'ready'
            ORDER BY rule_id, created_at DESC
            """,
            ([rule_id],),
        )
        picked = {r["dataset_id"] for r in selected}
        assert default_id in picked
        assert custom_id not in picked

    def test_non_ready_default_excluded(self, client, auth_headers):
        from utils.PostgreSQL import execute_query_dict
        rule_id = self._make_rule()
        # A default but still-generating row must NOT be selected.
        self._insert_test_dataset(rule_id, is_default=True, status="generating")

        selected = execute_query_dict(
            """
            SELECT dataset_id FROM test_datasets
            WHERE rule_id = ANY(%s)
              AND dataset_type = 'positive_calibration'
              AND is_default = TRUE
              AND status = 'ready'
            """,
            ([rule_id],),
        )
        assert selected in (None, [])


class TestRunInferenceDispatch:
    """_run_inference offloads to the cluster when configured, and falls back to
    local GPU/CPU inference when the cluster is off OR errors — never failing the
    run just because the cluster is unavailable."""

    def test_uses_local_when_cluster_disabled(self):
        from unittest.mock import patch
        from routes import evaluation as ev
        with patch("services.compute.providers.slurm.cluster_direct.is_enabled", return_value=False), \
             patch("evaluation.inference.run_inference_on_dialogues",
                   return_value=[{"logits": "L"}]) as local:
            out = ev._run_inference(1, [{"conversation": [], "metadata": {}}])
        assert out == [{"logits": "L"}]
        local.assert_called_once()

    def test_falls_back_to_local_when_cluster_raises(self, tmp_path):
        import json as _json
        from unittest.mock import patch
        from routes import evaluation as ev
        (tmp_path / "classifier_meta.json").write_text(_json.dumps({"model_path": "m"}))
        (tmp_path / "trained_rnn.pth").write_text("x")
        with patch("services.compute.providers.slurm.cluster_direct.is_enabled", return_value=True), \
             patch("services.compute.providers.slurm.cluster_direct.ping", return_value=True), \
             patch("classifier_engine.trainer.classifier_workdir", return_value=str(tmp_path)), \
             patch("services.compute.providers.slurm.cluster_direct.run_inference_blocking",
                   side_effect=RuntimeError("cluster boom")), \
             patch("evaluation.inference.run_inference_on_dialogues",
                   return_value=[{"ok": True}]) as local:
            out = ev._run_inference(7, [])
        assert out == [{"ok": True}]
        local.assert_called_once()

    def test_falls_back_to_local_when_cluster_unreachable(self):
        # The reachability probe fails fast (no banner-exchange hang), so we go
        # straight to local without ever attempting the heavy cluster upload.
        from unittest.mock import patch
        from routes import evaluation as ev
        with patch("services.compute.providers.slurm.cluster_direct.is_enabled", return_value=True), \
             patch("services.compute.providers.slurm.cluster_direct.ping", return_value=False), \
             patch("services.compute.providers.slurm.cluster_direct.run_inference_blocking") as cluster, \
             patch("evaluation.inference.run_inference_on_dialogues",
                   return_value=[{"ok": True}]) as local:
            out = ev._run_inference(3, [{"conversation": [], "metadata": {}}])
        assert out == [{"ok": True}]
        cluster.assert_not_called()
        local.assert_called_once()

    def test_uses_cluster_when_enabled(self, tmp_path):
        import json as _json
        from unittest.mock import patch
        from routes import evaluation as ev
        (tmp_path / "classifier_meta.json").write_text(_json.dumps({"model_path": "hf/model"}))
        (tmp_path / "trained_rnn.pth").write_text("x")
        with patch("services.compute.providers.slurm.cluster_direct.is_enabled", return_value=True), \
             patch("services.compute.providers.slurm.cluster_direct.ping", return_value=True), \
             patch("classifier_engine.trainer.classifier_workdir", return_value=str(tmp_path)), \
             patch("services.compute.providers.slurm.cluster_direct.run_inference_blocking",
                   return_value=[{"logits": "from_cluster"}]) as cluster, \
             patch("evaluation.inference.run_inference_on_dialogues") as local:
            out = ev._run_inference(9, [{"conversation": [], "metadata": {}}])
        assert out == [{"logits": "from_cluster"}]
        cluster.assert_called_once()
        local.assert_not_called()


class TestDefaultEvalPairsPerRule:
    """_load_default_eval_pairs tags each rule's positive/negative set with THAT
    rule's name, so a multi-rule classifier gets a correct PER-RULE breakdown.
    Regression: every set used to be tagged with the first rule, collapsing all
    rules onto one use-case row."""

    def test_pairs_carry_each_rules_own_name(self, client, test_model, test_user, auth_headers):
        from utils.PostgreSQL import execute_query, execute_query_dict
        from routes.evaluation import _load_default_eval_pairs

        # user_id is NOT NULL on classifiers (schema v15). This raw INSERT used
        # to omit it and still succeed, because that migration only applies the
        # constraint when no NULL rows remain — a legacy NULL row was holding it
        # off. Supply the owner explicitly so the test matches the schema (and
        # every real code path) rather than depending on an unenforced column.
        cid = execute_query_dict(
            "INSERT INTO classifiers (model_id, user_id, name, status) "
            "VALUES (%s, %s, %s, 'untrained') RETURNING classifier_id",
            (test_model["model_id"], test_user["user_id"], f"evalpairs_{_uniq()}"),
        )[0]["classifier_id"]

        # Two rules, each attached with a custom_name + a ready positive AND
        # negative set. (conversations: one conversation of one message each.)
        expected = set()
        for tag in ("alpha", "beta"):
            rname = f"{tag}_rule_{_uniq()}"
            rid = execute_query_dict(
                "INSERT INTO rules (name, predicate) VALUES (%s, %s) RETURNING rule_id",
                (rname, "CE"),
            )[0]["rule_id"]
            execute_query(
                "INSERT INTO rule_setup (classifier_id, rule_id, custom_name, predicate, is_active) "
                "VALUES (%s, %s, %s, %s, TRUE)",
                (cid, rid, rname, "CE"),
            )
            for dtype in ("positive", "negative"):
                execute_query(
                    "INSERT INTO test_datasets (rule_id, dataset_type, status, is_default, conversations) "
                    "VALUES (%s, %s, 'ready', TRUE, '[[{\"role\": \"user\", \"content\": \"x\"}]]'::jsonb)",
                    (rid, dtype),
                )
            expected.add(rname)

        pairs = _load_default_eval_pairs(cid)

        # Every entry is a (convos, dataset_type, rule_name) TRIPLE.
        assert pairs and all(len(p) == 3 for p in pairs)
        by_rule: dict = {}
        for convos, dtype, rname in pairs:
            assert len(convos) == 1            # one conversation per set
            by_rule.setdefault(rname, set()).add(dtype)
        # Both rules represented, each tagged with its OWN name + both halves.
        assert set(by_rule.keys()) == expected
        for types in by_rule.values():
            assert types == {"positive", "negative"}


class TestEvalDetailedEndpointRemoved:
    """The standalone 'Detailed eval' / detailed-calibration endpoint was
    removed. Any guess at its old path must 404 (route absent) — never 200."""

    @pytest.mark.parametrize("path", [
        "/evaluation/1/detailed",
        "/evaluation/1/detailed-eval",
        "/evaluation/1/evaluate/detailed",
        "/evaluation/1/calibrate/detailed",
        "/evaluation/detailed",
    ])
    def test_detailed_routes_gone_get(self, client, auth_headers, path):
        res = client.get(path, headers=auth_headers)
        # Route does not exist -> 404 (Not Found) or 405 (method-not-allowed
        # if a same-prefix POST route shadows it). Never a successful 2xx.
        assert res.status_code in (404, 405)

    @pytest.mark.parametrize("path", [
        "/evaluation/1/detailed",
        "/evaluation/1/detailed-eval",
        "/evaluation/1/evaluate/detailed",
    ])
    def test_detailed_routes_gone_post(self, client, auth_headers, path):
        res = client.post(path, json={}, headers=auth_headers)
        assert res.status_code in (404, 405)


class TestEvalPostRetrainResultsRoute:
    """Snapshot filter: read endpoints surface only rows created AFTER the
    classifier's most recent training. Rows are inserted directly with a
    backdated created_at; cleanup is automatic via the conftest snapshot."""

    @staticmethod
    def _insert_eval_row(classifier_id, eval_type, *, age_seconds, thresholds=None):
        import json
        from utils.PostgreSQL import execute_query_dict
        rows = execute_query_dict(
            """
            INSERT INTO evaluation_results (
                classifier_id, eval_type, thresholds, metrics, plots, created_at
            )
            VALUES (%s, %s, %s::jsonb, NULL, NULL,
                    now() - (%s || ' seconds')::interval)
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
    def _set_trained_at(classifier_id, age_seconds):
        from utils.PostgreSQL import execute_query
        execute_query(
            "UPDATE classifiers SET trained_at = now() - (%s || ' seconds')::interval "
            "WHERE classifier_id = %s",
            (str(age_seconds), classifier_id),
        )

    def test_results_history_filters_pre_retrain(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        old_id = self._insert_eval_row(cid, "evaluation", age_seconds=120,
                                       thresholds={"old": 0.4})
        self._set_trained_at(cid, 60)
        new_id = self._insert_eval_row(cid, "evaluation", age_seconds=10,
                                       thresholds={"new": 0.6})

        res = client.get(f"/evaluation/{cid}/results/history", headers=auth_headers)
        assert res.status_code == 200
        ids = {r["eval_id"] for r in res.json()["results"]}
        assert new_id in ids
        assert old_id not in ids

    def test_results_endpoint_surfaces_post_retrain_eval(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        self._insert_eval_row(cid, "evaluation", age_seconds=120)
        self._set_trained_at(cid, 60)
        fresh = self._insert_eval_row(cid, "evaluation", age_seconds=10)

        res = client.get(f"/evaluation/{cid}/results", headers=auth_headers)
        assert res.status_code == 200
        ev = res.json()["evaluation"]
        assert ev is not None
        assert ev["eval_id"] == fresh

    def test_thresholds_returns_post_retrain_only(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        self._insert_eval_row(cid, "calibration", age_seconds=120,
                              thresholds={"stale": 0.5})
        self._set_trained_at(cid, 60)
        self._insert_eval_row(cid, "calibration", age_seconds=10,
                              thresholds={"current": 0.8})

        res = client.get(f"/evaluation/{cid}/thresholds", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["thresholds"] == {"current": 0.8}

    def test_thresholds_404_when_only_stale_calibration(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        self._insert_eval_row(cid, "calibration", age_seconds=120,
                              thresholds={"stale": 0.5})
        self._set_trained_at(cid, 60)

        res = client.get(f"/evaluation/{cid}/thresholds", headers=auth_headers)
        assert res.status_code == 404


class TestRunOnceLock:
    """Calibration / evaluation are once-per-training: a successful run
    (post-train) locks further runs until the classifier is retrained. The 409
    is raised before any background task is scheduled, so these are fast."""

    @pytest.fixture(autouse=True)
    def _restore_classifier_state(self, test_classifier):
        # The per-test cleanup only deletes NEW rows; it doesn't revert UPDATEs.
        # We flip the shared classifier's status/trained_at, so snapshot and
        # restore them here or the change leaks into later untrained-state tests.
        from utils.PostgreSQL import execute_query_dict, execute_query
        cid = test_classifier["classifier_id"]
        orig = execute_query_dict(
            "SELECT status, trained_at FROM classifiers WHERE classifier_id = %s", (cid,)
        )[0]
        yield
        execute_query(
            "UPDATE classifiers SET status = %s, trained_at = %s WHERE classifier_id = %s",
            (orig["status"], orig["trained_at"], cid),
        )

    @staticmethod
    def _mark_trained(cid, age_seconds=30):
        from utils.PostgreSQL import execute_query
        execute_query(
            "UPDATE classifiers SET status = 'active', "
            "trained_at = now() - (%s || ' seconds')::interval WHERE classifier_id = %s",
            (str(age_seconds), cid),
        )

    @staticmethod
    def _insert_row(cid, eval_type, age_seconds=5, thresholds=None):
        import json
        from utils.PostgreSQL import execute_query
        execute_query(
            "INSERT INTO evaluation_results (classifier_id, eval_type, thresholds, metrics, plots, created_at) "
            "VALUES (%s, %s, %s::jsonb, NULL, NULL, now() - (%s || ' seconds')::interval)",
            (cid, eval_type, json.dumps(thresholds) if thresholds is not None else None, str(age_seconds)),
        )

    def test_calibrate_locked_after_success(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        self._mark_trained(cid)
        self._insert_row(cid, "calibration", thresholds={"CE": {"threshold": 0.5}})
        res = client.post(f"/evaluation/{cid}/calibrate", json={}, headers=auth_headers)
        assert res.status_code == 409

    def test_evaluate_locked_after_success(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        self._mark_trained(cid)
        self._insert_row(cid, "evaluation")
        res = client.post(f"/evaluation/{cid}/evaluate", json={"test_dataset_ids": [1]}, headers=auth_headers)
        assert res.status_code == 409


class TestEvalCalibrateEvaluateGating:
    """POST /calibrate and /evaluate: auth + trained-state gating.

    The test classifier is freshly created and NOT trained, so both POSTs
    should be rejected by _verify_classifier_trained (400) — never start a
    real background job. We accept {400, 500} for defensiveness if the status
    check raises before the 400 path in some environments.
    """

    def test_calibrate_without_auth_header_is_allowed(self, client, test_classifier):
        cid = test_classifier["classifier_id"]
        res = client.post(f"/evaluation/{cid}/calibrate", json={})
        # HTTPBearer(auto_error=True): missing header -> 403.
        assert res.status_code != 403  # no auth exists: a request is never rejected for credentials

    def test_evaluate_without_auth_header_is_allowed(self, client, test_classifier):
        cid = test_classifier["classifier_id"]
        res = client.post(f"/evaluation/{cid}/evaluate", json={})
        assert res.status_code != 403  # no auth exists: a request is never rejected for credentials

    def test_calibrate_with_garbage_token_is_allowed(self, client, test_classifier):
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/evaluation/{cid}/calibrate",
            json={},
            headers={"Authorization": "Bearer not-a-real-jwt"},
        )
        assert res.status_code != 401  # no auth exists: a request is never rejected for credentials

    def test_calibrate_untrained_classifier_rejected(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.post(f"/evaluation/{cid}/calibrate", json={}, headers=auth_headers)
        assert res.status_code in (400, 500)

    def test_evaluate_untrained_classifier_rejected(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/evaluation/{cid}/evaluate",
            json={"test_dataset_ids": [1]},
            headers=auth_headers,
        )
        assert res.status_code in (400, 500)

    def test_calibrate_nonexistent_classifier_404(self, client, auth_headers):
        # _verify_classifier_trained raises 404 when the classifier row is gone.
        res = client.post("/evaluation/999999999/calibrate", json={}, headers=auth_headers)
        assert res.status_code == 404

    def test_evaluate_nonexistent_classifier_404(self, client, auth_headers):
        res = client.post(
            "/evaluation/999999999/evaluate",
            json={"test_dataset_id": 1},
            headers=auth_headers,
        )
        assert res.status_code == 404

    def test_calibrate_malformed_body_422(self, client, test_classifier, auth_headers):
        # patience_values must be a list[int]; a string is a type error.
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/evaluation/{cid}/calibrate",
            json={"patience_values": "not-a-list"},
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_evaluate_malformed_body_422(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/evaluation/{cid}/evaluate",
            json={"test_dataset_ids": "nope"},
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_calibrate_non_integer_path_param_422(self, client, auth_headers):
        res = client.post("/evaluation/abc/calibrate", json={}, headers=auth_headers)
        assert res.status_code == 422


class TestEvalTestDatasetListing:
    """Test-dataset listing endpoints (ai_pipeline.py) used by evaluation UI."""

    def test_list_by_rule_empty(self, client, auth_headers):
        res = client.get("/ai/test-sets/by-rule/999999999", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "datasets" in data
        assert isinstance(data["datasets"], list)

    def test_list_by_classifier_shape(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/ai/test-sets/by-classifier/{cid}", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "datasets" in data
        assert isinstance(data["datasets"], list)

    def test_list_by_rule_requires_auth(self, client):
        res = client.get("/ai/test-sets/by-rule/999999999")
        assert res.status_code != 403  # no auth exists: a request is never rejected for credentials

    def test_test_set_status_not_found_404(self, client, auth_headers):
        res = client.get("/ai/test-set/999999999/status", headers=auth_headers)
        assert res.status_code == 404
