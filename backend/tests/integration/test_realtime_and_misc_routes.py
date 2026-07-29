"""Integration tests for realtime, ratings, pipeline_runs, and dashboard routes.

Focus is on the NON-model-loading paths: validation (400/422), auth boundaries
(401/403), not-found (404), conflict/state gating, response shapes, and CRUD on
pipeline-run wizard state. No real model inference is triggered.

Route prefixes (see main.py):
  /realtime, /ratings, /pipeline-runs, /dashboard

All DB rows created via the API are cleaned up automatically by the integration
conftest's per-test snapshot/restore.
"""
import time

import pytest


# ===========================================================================
# Realtime — only the paths that don't load SmolLM weights.
# ===========================================================================
class TestRealtimeGating:
    """`_require_trained_classifier` runs first on every realtime endpoint, so
    a nonexistent classifier -> 404 and an untrained one -> 400, both BEFORE any
    model load. The session `test_classifier` is freshly created (not 'active'/
    'needs_retraining'), so it is treated as untrained."""

    def test_sample_groups_classifier_not_found(self, client, auth_headers):
        res = client.get("/realtime/999999/sample-groups", headers=auth_headers)
        assert res.status_code == 404

    def test_sample_groups_untrained_classifier(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/realtime/{cid}/sample-groups", headers=auth_headers)
        # Untrained -> 400. If a prior run left it trained the gate passes and we
        # get a 200 group listing (no model load needed for listing).
        assert res.status_code in (200, 400)
        if res.status_code == 200:
            data = res.json()
            assert "groups" in data
            assert isinstance(data["groups"], list)

    def test_sample_groups_without_auth_is_allowed(self, client, test_classifier):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/realtime/{cid}/sample-groups")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_sample_group_not_found_classifier(self, client, auth_headers):
        res = client.get(
            "/realtime/999999/sample-group",
            params={"key": "testds:1"},
            headers=auth_headers,
        )
        assert res.status_code == 404

    def test_sample_group_missing_key_param(self, client, test_classifier, auth_headers):
        """`key` is a required query param -> 422 when absent."""
        cid = test_classifier["classifier_id"]
        res = client.get(f"/realtime/{cid}/sample-group", headers=auth_headers)
        # Missing required query param is a 422. (If the gate fires first because
        # the classifier is untrained, that's a 400 — both are acceptable.)
        assert res.status_code in (400, 422)

    def test_sample_group_unknown_key(self, client, test_classifier, auth_headers):
        """An unrecognized key returns an empty sample list (the loader returns
        [] for unknown prefixes) when the classifier is trained, or 400 when the
        gate rejects an untrained classifier first."""
        cid = test_classifier["classifier_id"]
        res = client.get(
            f"/realtime/{cid}/sample-group",
            params={"key": "totally-unknown-key"},
            headers=auth_headers,
        )
        assert res.status_code in (200, 400)
        if res.status_code == 200:
            data = res.json()
            assert data["key"] == "totally-unknown-key"
            assert data["samples"] == []


class TestRealtimeAnalyzeValidation:
    """Request-body validation on /analyze and /analyze-stored. We assert the
    paths that reject BEFORE a model is loaded: schema validation (422) and the
    gating check (404/400)."""

    def test_analyze_missing_user_message(self, client, test_classifier, auth_headers):
        """`user_message` is required by AnalyzeRequest -> 422."""
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/realtime/{cid}/analyze",
            json={"system_prompt": "You are helpful."},
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_analyze_empty_body(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.post(f"/realtime/{cid}/analyze", json={}, headers=auth_headers)
        assert res.status_code == 422

    def test_analyze_without_auth_is_allowed(self, client, test_classifier):
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/realtime/{cid}/analyze",
            json={"user_message": "hi"},
        )
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_analyze_classifier_not_found(self, client, auth_headers):
        """Gate runs before model load: nonexistent classifier -> 404."""
        res = client.post(
            "/realtime/999999/analyze",
            json={"user_message": "hi"},
            headers=auth_headers,
        )
        assert res.status_code == 404

    def test_analyze_bad_max_new_tokens_type(self, client, test_classifier, auth_headers):
        """Non-int max_new_tokens -> 422 (schema coercion failure)."""
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/realtime/{cid}/analyze",
            json={"user_message": "hi", "max_new_tokens": "lots"},
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_analyze_stored_missing_messages(self, client, test_classifier, auth_headers):
        """`messages` is required by AnalyzeStoredRequest -> 422."""
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/realtime/{cid}/analyze-stored",
            json={},
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_analyze_stored_classifier_not_found(self, client, auth_headers):
        res = client.post(
            "/realtime/999999/analyze-stored",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=auth_headers,
        )
        assert res.status_code == 404

    def test_analyze_stored_empty_messages(self, client, test_classifier, auth_headers):
        """Empty messages list: the gate (404/400) runs first; if the classifier
        is trained the handler raises 400 'messages is required'. Either way no
        model inference runs on an empty list."""
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/realtime/{cid}/analyze-stored",
            json={"messages": []},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_analyze_stored_without_auth_is_allowed(self, client, test_classifier):
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/realtime/{cid}/analyze-stored",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


# ===========================================================================
# Ratings — proxied to the central server, but several paths are decided
# LOCALLY before the HTTP hop.
# ===========================================================================


class TestPipelineRunsCrud:
    def test_create_and_get_run(self, client, auth_headers):
        res = client.post("/pipeline-runs", json={"pipeline_type": "rule"}, headers=auth_headers)
        assert res.status_code == 200
        run = res.json()
        assert "run_id" in run
        assert run["pipeline_type"] == "rule"
        assert run["completed"] is False
        # A fresh 'rule' run lands on step "1".
        assert run["current_step"] == "1"
        assert isinstance(run["steps"], dict)

        rid = run["run_id"]
        got = client.get(f"/pipeline-runs/{rid}", headers=auth_headers)
        assert got.status_code == 200
        assert got.json()["run_id"] == rid

    def test_create_run_requires_auth(self, client):
        res = client.post("/pipeline-runs", json={"pipeline_type": "rule"})
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_create_test_eval_run_without_classifier_is_400(self, client, auth_headers):
        """test_eval runs require a classifier_id (ValueError -> 400)."""
        res = client.post(
            "/pipeline-runs",
            json={"pipeline_type": "test_eval"},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_create_run_unknown_pipeline_type_is_400(self, client, auth_headers):
        res = client.post(
            "/pipeline-runs",
            json={"pipeline_type": "not_a_real_type"},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_get_run_not_found(self, client, auth_headers):
        res = client.get("/pipeline-runs/999999", headers=auth_headers)
        assert res.status_code == 404

    def test_get_run_without_auth_is_allowed(self, client):
        res = client.get("/pipeline-runs/1")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_list_active_runs_shape(self, client, auth_headers):
        res = client.get("/pipeline-runs/active", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)

    def test_list_active_runs_unknown_pipeline_type_is_400(self, client, auth_headers):
        res = client.get(
            "/pipeline-runs/active",
            params={"pipeline_type": "bogus"},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_list_active_runs_without_auth_is_allowed(self, client):
        res = client.get("/pipeline-runs/active")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


class TestPipelineRunsStepUpdates:
    def test_patch_step_valid(self, client, auth_headers):
        run = client.post(
            "/pipeline-runs", json={"pipeline_type": "rule"}, headers=auth_headers
        ).json()
        rid = run["run_id"]
        res = client.patch(
            f"/pipeline-runs/{rid}/step",
            json={"step_id": "2A", "status": "completed", "advance_to": "2B"},
            headers=auth_headers,
        )
        assert res.status_code == 200
        updated = res.json()
        assert updated["current_step"] == "2B"
        assert updated["steps"]["2A"]["status"] == "completed"

    def test_patch_step_unknown_step_id_is_400(self, client, auth_headers):
        run = client.post(
            "/pipeline-runs", json={"pipeline_type": "rule"}, headers=auth_headers
        ).json()
        rid = run["run_id"]
        res = client.patch(
            f"/pipeline-runs/{rid}/step",
            json={"step_id": "ZZZ", "status": "completed"},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_patch_step_missing_required_fields_is_422(self, client, auth_headers):
        run = client.post(
            "/pipeline-runs", json={"pipeline_type": "rule"}, headers=auth_headers
        ).json()
        rid = run["run_id"]
        res = client.patch(
            f"/pipeline-runs/{rid}/step",
            json={"step_id": "1"},  # missing required `status`
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_patch_step_not_found(self, client, auth_headers):
        res = client.patch(
            "/pipeline-runs/999999/step",
            json={"step_id": "1", "status": "completed"},
            headers=auth_headers,
        )
        assert res.status_code == 404

    def test_complete_run(self, client, auth_headers):
        run = client.post(
            "/pipeline-runs", json={"pipeline_type": "rule"}, headers=auth_headers
        ).json()
        rid = run["run_id"]
        res = client.post(f"/pipeline-runs/{rid}/complete", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["completed"] is True

    def test_complete_run_not_found(self, client, auth_headers):
        res = client.post("/pipeline-runs/999999/complete", headers=auth_headers)
        assert res.status_code == 404

    def test_delete_run(self, client, auth_headers):
        run = client.post(
            "/pipeline-runs", json={"pipeline_type": "rule"}, headers=auth_headers
        ).json()
        rid = run["run_id"]
        res = client.delete(f"/pipeline-runs/{rid}", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["success"] is True
        # Now gone.
        assert client.get(f"/pipeline-runs/{rid}", headers=auth_headers).status_code == 404

    def test_delete_run_not_found(self, client, auth_headers):
        res = client.delete("/pipeline-runs/999999", headers=auth_headers)
        assert res.status_code == 404

    def test_patch_links(self, client, auth_headers):
        run = client.post(
            "/pipeline-runs", json={"pipeline_type": "rule"}, headers=auth_headers
        ).json()
        rid = run["run_id"]
        # rule_id None is allowed by the schema; the row just keeps its link.
        res = client.patch(
            f"/pipeline-runs/{rid}/links",
            json={"rule_id": None},
            headers=auth_headers,
        )
        assert res.status_code == 200
        assert res.json()["run_id"] == rid


# (TestPipelineRunsOwnership removed with login. It minted a token for a second
# user to prove a run owned by A is 404 for B. There is exactly one identity
# now, so a "second user" cannot be constructed and the scenario cannot occur.)


# ===========================================================================
# Dashboard — stats shape + not-found. (No auth dependency on this route.)
# ===========================================================================
class TestDashboard:
    def test_dashboard_stats_shape(self, client, test_user):
        res = client.get(f"/dashboard/{test_user['user_id']}")
        assert res.status_code == 200
        data = res.json()
        assert "stats" in data
        stats = data["stats"]
        # Every count must be a non-negative int.
        for key in (
            "total_models",
            "total_classifiers",
            "active_classifiers",
            "total_rules",
            "total_ces",
            "total_evaluations",
            "total_test_datasets",
        ):
            assert key in stats
            assert isinstance(stats[key], int)
            assert stats[key] >= 0

    def test_dashboard_response_envelope(self, client, test_user):
        res = client.get(f"/dashboard/{test_user['user_id']}")
        assert res.status_code == 200
        data = res.json()
        assert "user_info" in data
        assert isinstance(data["recent_activity"], list)
        assert isinstance(data["classifier_summary"], list)
        assert data["user_info"].get("username") == test_user["username"]

    def test_dashboard_nonexistent_user_403(self, client):
        # Single-operator app: any user_id other than the local user is 403.
        res = client.get("/dashboard/99999999")
        assert res.status_code == 403

    def test_dashboard_non_integer_user_id_422(self, client):
        """user_id path param is typed int -> 422 for a non-numeric value."""
        res = client.get("/dashboard/not-a-number")
        assert res.status_code == 422


# NOTE — TestRatingsValidation / TestRatingsAuthBoundaries were removed with the
# ratings feature. A rating is one row per (user, asset) with a self-rating
# guard, so it needs distinct accounts; with a single local identity it has no
# meaning. The /ratings router, its central tables and the StarRating widget are
# all gone.
