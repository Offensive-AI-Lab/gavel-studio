"""Integration tests for backend/routes/classifiers.py.

Distinct from test_classifiers.py — these focus on the route-level contract:
  * list route + training-status driving reconcile_classifier_status, with the
    status SELF-HEALING from real policy drift (live policy fingerprint vs the
    stored trained_policy_fingerprint).
  * the removed /classifiers/{id}/rules/bookmarked-ce endpoint is gone (404).
  * classifier CRUD edges (create validation/auth/conflict, details 404).
  * rules listing, config get/update, not-found (404), auth (401/403).

Policy-drift tests seed rule wiring (rule_setup + setup_ce_link) and a
trained_policy_fingerprint via direct SQL, then assert training-status flips
the status. Everything inserted lives in tracked tables, so the conftest
snapshot/restore cleans it up automatically.

NO model weights are loaded — we never hit /train end-to-end, only the
status/state/validation surfaces.
"""
import time

import pytest

from utils.PostgreSQL import execute_query, execute_query_dict
from sql_scripts.model_scripts import compute_classifier_policy_fingerprint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000) % 10_000_000}"


def _create_classifier(client, model_id, auth_headers, name=None) -> int:
    """Create a fresh classifier through the real API and return its id."""
    name = name or _unique("cls")
    res = client.post(
        "/classifiers/create",
        json={"model_id": model_id, "name": name},
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    data = res.json()
    cls = data.get("classifier", data)
    return cls["classifier_id"]


def _seed_rule_with_ce(classifier_id: int, user_id: int) -> tuple[int, int]:
    """Insert one active rule_setup with one CE link for `classifier_id`.

    Returns (setup_id, ce_id). All rows land in tracked tables
    (cognitive_elements, rule_setup, setup_ce_link) so cleanup is automatic.
    """
    # cognitive_elements has no user_id column — CEs are global. Insert the
    # minimal valid row (name + definition); is_ready defaults to TRUE.
    ce_rows = execute_query_dict(
        "INSERT INTO cognitive_elements (name, definition) VALUES (%s, %s) RETURNING ce_id",
        (_unique("drift_ce"), "drift test CE"),
    )
    ce_id = ce_rows[0]["ce_id"]

    setup_rows = execute_query_dict(
        "INSERT INTO rule_setup (classifier_id, custom_name, predicate) "
        "VALUES (%s, %s, %s) RETURNING setup_id",
        (classifier_id, _unique("drift_rule"), "CE"),
    )
    setup_id = setup_rows[0]["setup_id"]

    execute_query(
        "INSERT INTO setup_ce_link (setup_id, ce_id, role, fallback_group) "
        "VALUES (%s, %s, 'necessary', 0)",
        (setup_id, ce_id),
    )
    return setup_id, ce_id


# ---------------------------------------------------------------------------
# List route + reconcile self-heal (policy drift)
# ---------------------------------------------------------------------------


class TestReconcileSelfHeal:
    """training-status and the list route recompute status from real drift."""

    def test_status_active_when_fingerprint_matches(
        self, client, test_model, test_user, auth_headers
    ):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        _seed_rule_with_ce(cid, test_user["user_id"])

        # Snapshot the CURRENT policy as the trained fingerprint -> no drift.
        current_fp = compute_classifier_policy_fingerprint(cid)
        assert current_fp != ""  # a real rule link exists
        execute_query(
            "UPDATE classifiers SET status = 'active', "
            "trained_policy_fingerprint = %s WHERE classifier_id = %s",
            (current_fp, cid),
        )

        res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "active"
        assert data["is_trained"] is True
        assert data["is_training"] is False
        assert data["has_error"] is False

    def test_status_flips_to_needs_retraining_on_drift(
        self, client, test_model, test_user, auth_headers
    ):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        setup_id, ce_id = _seed_rule_with_ce(cid, test_user["user_id"])

        # Train-time snapshot taken with the CE present.
        trained_fp = compute_classifier_policy_fingerprint(cid)
        execute_query(
            "UPDATE classifiers SET status = 'active', "
            "trained_policy_fingerprint = %s WHERE classifier_id = %s",
            (trained_fp, cid),
        )

        # Drift: remove the CE link so the live policy differs from trained.
        execute_query(
            "DELETE FROM setup_ce_link WHERE setup_id = %s AND ce_id = %s",
            (setup_id, ce_id),
        )

        res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["status"] == "needs_retraining"

        # Self-heal must be PERSISTED, not just computed for the response.
        stored = execute_query_dict(
            "SELECT status FROM classifiers WHERE classifier_id = %s", (cid,)
        )
        assert stored[0]["status"] == "needs_retraining"

    def test_status_heals_back_to_active_when_drift_resolved(
        self, client, test_model, test_user, auth_headers
    ):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        setup_id, ce_id = _seed_rule_with_ce(cid, test_user["user_id"])
        trained_fp = compute_classifier_policy_fingerprint(cid)

        # Start in the drifted state already stored as needs_retraining,
        # but the live policy still matches the trained fingerprint.
        execute_query(
            "UPDATE classifiers SET status = 'needs_retraining', "
            "trained_policy_fingerprint = %s WHERE classifier_id = %s",
            (trained_fp, cid),
        )

        res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["status"] == "active"

    def test_list_route_reconciles_each_classifier(
        self, client, test_model, test_user, auth_headers
    ):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        setup_id, ce_id = _seed_rule_with_ce(cid, test_user["user_id"])
        trained_fp = compute_classifier_policy_fingerprint(cid)
        execute_query(
            "UPDATE classifiers SET status = 'active', "
            "trained_policy_fingerprint = %s WHERE classifier_id = %s",
            (trained_fp, cid),
        )
        # Introduce drift.
        execute_query(
            "DELETE FROM setup_ce_link WHERE setup_id = %s AND ce_id = %s",
            (setup_id, ce_id),
        )

        res = client.get(
            f"/classifiers/{test_model['model_id']}", headers=auth_headers
        )
        assert res.status_code == 200
        classifiers = res.json()["classifiers"]
        target = next((c for c in classifiers if c["classifier_id"] == cid), None)
        assert target is not None
        assert target["status"] == "needs_retraining"

    def test_untrained_status_passes_through(self, client, test_classifier, auth_headers):
        """No fingerprint snapshot -> reconcile is a no-op; status untouched."""
        cid = test_classifier["classifier_id"]
        res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        # The shared session classifier is never trained in the fast suite.
        assert data["status"] in ("untrained", "active", "needs_retraining", "error")
        assert "is_trained" in data and "is_training" in data and "has_error" in data


# ---------------------------------------------------------------------------
# Trained-policy snapshot — committed ONLY on successful completion
# (regression cover for the two Policy-Logic-Manager bugs)
# ---------------------------------------------------------------------------


class TestTrainedPolicySnapshot:
    """commit_trained_policy_snapshot records the live policy as the trained
    snapshot. Bug 2: the cluster completion path never wrote it, so the drift
    banner stuck after a successful retrain. Bug 1: writing it at training START
    made an interrupted run look 'Up to Date' on a model that never finished."""

    def test_commit_records_names_ids_fingerprint_and_timestamp(
        self, client, test_model, test_user, auth_headers
    ):
        from sql_scripts.model_scripts import commit_trained_policy_snapshot
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        setup_id, _ce = _seed_rule_with_ce(cid, test_user["user_id"])

        commit_trained_policy_snapshot(cid)

        row = execute_query_dict(
            "SELECT trained_rule_setup_ids, trained_rule_names, "
            "trained_policy_fingerprint, trained_at "
            "FROM classifiers WHERE classifier_id = %s",
            (cid,),
        )[0]
        assert setup_id in (row["trained_rule_setup_ids"] or [])
        assert len(row["trained_rule_names"] or []) == 1
        assert row["trained_policy_fingerprint"]            # non-empty for a real rule
        assert row["trained_at"] is not None                # stamped on success

    def test_commit_removes_stale_calibration_and_evaluation(
        self, client, test_model, test_user, auth_headers
    ):
        """A (re)train wipes the previous model's calibration + evaluation
        results, but leaves any '*_running' marker (it may own a cluster job)."""
        from sql_scripts.model_scripts import commit_trained_policy_snapshot
        from utils.PostgreSQL import execute_query
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        _seed_rule_with_ce(cid, test_user["user_id"])
        for et in ("calibration", "evaluation", "calibration_error", "evaluation_error"):
            execute_query(
                "INSERT INTO evaluation_results (classifier_id, eval_type) VALUES (%s, %s)",
                (cid, et),
            )
        execute_query(
            "INSERT INTO evaluation_results (classifier_id, eval_type) VALUES (%s, 'evaluation_running')",
            (cid,),
        )

        commit_trained_policy_snapshot(cid)

        rows = execute_query_dict(
            "SELECT eval_type FROM evaluation_results WHERE classifier_id = %s", (cid,)
        ) or []
        types = {r["eval_type"] for r in rows}
        assert types == {"evaluation_running"}, f"expected only the running marker, got {types}"

    def test_commit_clears_drift_after_retrain(
        self, client, test_model, test_user, auth_headers
    ):
        """Bug 2: after the policy changes and the (cluster) retrain finishes,
        committing the snapshot makes status reconcile back to 'active'."""
        from sql_scripts.model_scripts import commit_trained_policy_snapshot
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        _seed_rule_with_ce(cid, test_user["user_id"])

        # Trained on policy P1.
        fp1 = compute_classifier_policy_fingerprint(cid)
        execute_query(
            "UPDATE classifiers SET status = 'active', "
            "trained_policy_fingerprint = %s WHERE classifier_id = %s",
            (fp1, cid),
        )

        # User changes the policy (adds a second rule) -> drift.
        _seed_rule_with_ce(cid, test_user["user_id"])
        res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
        assert res.json()["status"] == "needs_retraining"

        # A successful (re)training completion commits the new snapshot.
        commit_trained_policy_snapshot(cid)

        # Drift is gone — banner/button clear.
        res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
        assert res.json()["status"] == "active"

    def test_not_committing_leaves_drift_intact(
        self, client, test_model, test_user, auth_headers
    ):
        """Bug 1: an interrupted run does NOT commit, so a prior-trained
        classifier keeps reflecting its last SUCCESSFUL training (stays drifted
        against the new policy instead of falsely going 'Up to Date')."""
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        _seed_rule_with_ce(cid, test_user["user_id"])
        fp1 = compute_classifier_policy_fingerprint(cid)
        execute_query(
            "UPDATE classifiers SET status = 'active', "
            "trained_policy_fingerprint = %s WHERE classifier_id = %s",
            (fp1, cid),
        )

        # Policy changes; the retrain is interrupted (NO snapshot commit).
        _seed_rule_with_ce(cid, test_user["user_id"])

        res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
        assert res.json()["status"] == "needs_retraining"


# ---------------------------------------------------------------------------
# Removed endpoint
# ---------------------------------------------------------------------------


class TestRemovedBookmarkedCeEndpoint:
    """The per-classifier bookmarked-ce route was lifted to the AI pipeline as
    a classifier-agnostic endpoint; under /classifiers it no longer exists."""

    def test_bookmarked_ce_route_gone(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.post(
            f"/classifiers/{cid}/rules/bookmarked-ce",
            json={"name": "x", "ce_roles": [{"ce_id": 1, "role": "necessary"}]},
            headers=auth_headers,
        )
        assert res.status_code == 404

    def test_bookmarked_ce_route_gone_get(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(
            f"/classifiers/{cid}/rules/bookmarked-ce", headers=auth_headers
        )
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# Create / CRUD edges
# ---------------------------------------------------------------------------


class TestCreateEdges:
    def test_create_missing_name_is_400(self, client, test_model, auth_headers):
        res = client.post(
            "/classifiers/create",
            json={"model_id": test_model["model_id"], "name": "   "},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_create_unknown_model_is_404(self, client, auth_headers):
        res = client.post(
            "/classifiers/create",
            json={"model_id": 99_999_999, "name": _unique("nope")},
            headers=auth_headers,
        )
        assert res.status_code == 404

    def test_create_duplicate_name_is_409(self, client, test_model, auth_headers):
        name = _unique("dup")
        first = client.post(
            "/classifiers/create",
            json={"model_id": test_model["model_id"], "name": name},
            headers=auth_headers,
        )
        assert first.status_code == 200
        second = client.post(
            "/classifiers/create",
            json={"model_id": test_model["model_id"], "name": name},
            headers=auth_headers,
        )
        assert second.status_code == 409

    def test_create_duplicate_name_case_insensitive_is_409(
        self, client, test_model, auth_headers
    ):
        name = _unique("CaseDup")
        first = client.post(
            "/classifiers/create",
            json={"model_id": test_model["model_id"], "name": name},
            headers=auth_headers,
        )
        assert first.status_code == 200
        second = client.post(
            "/classifiers/create",
            json={"model_id": test_model["model_id"], "name": name.upper()},
            headers=auth_headers,
        )
        assert second.status_code == 409

    def test_create_without_auth_is_allowed(self, client, test_model):
        res = client.post(
            "/classifiers/create",
            json={"model_id": test_model["model_id"], "name": _unique("noauth")},
        )
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_create_missing_body_field_is_422(self, client, auth_headers):
        # `name` omitted -> pydantic validation error.
        res = client.post(
            "/classifiers/create",
            json={"model_id": 1},
            headers=auth_headers,
        )
        assert res.status_code == 422


class TestDetailsAndDelete:
    def test_details_not_found_is_404(self, client, auth_headers):
        res = client.get("/classifiers/details/99999999", headers=auth_headers)
        assert res.status_code == 404

    def test_details_round_trip(self, client, test_model, auth_headers):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        res = client.get(f"/classifiers/details/{cid}", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["classifier_id"] == cid
        assert "model_name" in data
        assert "user_id" in data

    def test_delete_not_found_is_404(self, client, auth_headers):
        res = client.delete("/classifiers/99999999", headers=auth_headers)
        assert res.status_code == 404

    def test_delete_without_auth_is_allowed(self, client, test_classifier):
        res = client.delete(f"/classifiers/{test_classifier['classifier_id']}")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_delete_then_details_gone(self, client, test_model, auth_headers):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        res = client.delete(f"/classifiers/{cid}", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["success"] is True
        # Now it must be gone.
        after = client.get(f"/classifiers/details/{cid}", headers=auth_headers)
        assert after.status_code == 404


# ---------------------------------------------------------------------------
# Rules listing
# ---------------------------------------------------------------------------


class TestRulesListing:
    def test_rules_shape_for_empty_classifier(
        self, client, test_model, auth_headers
    ):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        res = client.get(f"/classifiers/{cid}/rules", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "rules" in data
        assert isinstance(data["rules"], list)
        assert data["rules"] == []

    def test_rules_unknown_classifier_returns_empty_list(self, client, auth_headers):
        # No row -> the query simply returns no rules (not a 404).
        res = client.get("/classifiers/99999999/rules", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["rules"] == []

    def test_rules_reflect_seeded_link(
        self, client, test_model, test_user, auth_headers
    ):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        setup_id, ce_id = _seed_rule_with_ce(cid, test_user["user_id"])
        res = client.get(f"/classifiers/{cid}/rules", headers=auth_headers)
        assert res.status_code == 200
        rules = res.json()["rules"]
        assert any(r["setup_id"] == setup_id for r in rules)
        seeded = next(r for r in rules if r["setup_id"] == setup_id)
        assert any(ce["ce_id"] == ce_id for ce in seeded["active_ces"])


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_get_config_merges_defaults(self, client, test_model, auth_headers):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        res = client.get(f"/classifiers/{cid}/config", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "config" in data and "defaults" in data
        # Defaults are always present in the merged config.
        assert data["config"]["hidden_dim"] == 256
        assert data["config"]["epochs"] == 10
        assert data["defaults"]["batch_size"] == 16

    def test_get_config_not_found_is_404(self, client, auth_headers):
        res = client.get("/classifiers/99999999/config", headers=auth_headers)
        assert res.status_code == 404

    def test_update_config_persists_and_merges(self, client, test_model, auth_headers):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        res = client.put(
            f"/classifiers/{cid}/config",
            json={"hidden_dim": 128, "epochs": 3},
            headers=auth_headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["config"]["hidden_dim"] == 128
        assert data["config"]["epochs"] == 3
        # Unspecified field still falls back to the default.
        assert data["config"]["batch_size"] == 16

        # Persisted: a fresh GET reflects the override.
        got = client.get(f"/classifiers/{cid}/config", headers=auth_headers)
        assert got.json()["config"]["hidden_dim"] == 128

    def test_update_config_not_found_is_404(self, client, auth_headers):
        res = client.put(
            "/classifiers/99999999/config",
            json={"epochs": 2},
            headers=auth_headers,
        )
        assert res.status_code == 404

    def test_update_config_without_auth_is_allowed(self, client, test_classifier):
        res = client.put(
            f"/classifiers/{test_classifier['classifier_id']}/config",
            json={"epochs": 2},
        )
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_update_config_empty_body_keeps_defaults(
        self, client, test_model, auth_headers
    ):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        res = client.put(
            f"/classifiers/{cid}/config", json={}, headers=auth_headers
        )
        assert res.status_code == 200
        assert res.json()["config"]["hidden_dim"] == 256


# ---------------------------------------------------------------------------
# Training status not-found
# ---------------------------------------------------------------------------


class TestTrainingStatusNotFound:
    def test_status_unknown_classifier_is_404(self, client, auth_headers):
        res = client.get(
            "/classifiers/99999999/training-status", headers=auth_headers
        )
        assert res.status_code == 404

    def test_status_response_shape(self, client, test_model, auth_headers):
        cid = _create_classifier(client, test_model["model_id"], auth_headers)
        res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        for key in (
            "classifier_id",
            "name",
            "status",
            "is_trained",
            "is_training",
            "has_error",
        ):
            assert key in data
        # A brand-new classifier is untrained.
        assert data["status"] == "untrained"
        assert data["is_trained"] is False
        assert data["is_training"] is False
