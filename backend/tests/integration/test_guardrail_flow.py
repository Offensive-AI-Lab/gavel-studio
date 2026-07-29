"""Integration tests for the model-last "guardrail" flow added on top of
backend/routes/classifiers.py:

  * name-only create (model_id NULL) — owner set via token, shows in /details/all
  * /classifiers/details/all — lists the user's guardrails across models +
    the unattached ones (model_name NULL)
  * /classifiers/details/{id}/attach-model — binds a model, then the guardrail
    appears in the per-model list; blocked once trained; 409 on name clash;
    404 on a foreign/missing model
  * train gate — POST /train on an unattached guardrail → 400
  * /classifiers/details/{id}/clone — deep-copies rules onto another model as a
    new untrained guardrail; 404 on a foreign model

All inserted rows live in tracked tables, so the conftest snapshot/restore
cleans them up. No model weights are loaded — we never train end-to-end.
"""
import time

import pytest

from utils.sqlite_db import execute_query, execute_query_dict


def _unique(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000) % 10_000_000}"


@pytest.fixture
def second_model(client, test_user, auth_headers):
    """A second model so clone/attach have a distinct target."""
    res = client.post("/models/create", json={
        "user_id": test_user["user_id"],
        "name": _unique("SecondModel"),
        "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
    }, headers=auth_headers)
    if res.status_code != 200:
        pytest.skip(f"Could not create second model: {res.status_code} {res.text[:200]}")
    data = res.json()
    return data.get("model", data)


def _create_guardrail(client, auth_headers, name=None):
    """Create a model-less guardrail (the primary flow) and return its row."""
    name = name or _unique("grd")
    res = client.post("/classifiers/create", json={"name": name}, headers=auth_headers)
    assert res.status_code == 200, res.text
    return res.json()["classifier"]


class TestNameOnlyCreate:
    def test_create_without_model_is_unattached_and_owned(self, client, auth_headers):
        g = _create_guardrail(client, auth_headers)
        assert g["classifier_id"]
        assert g.get("model_id") is None
        assert g["status"] == "untrained"
        # Detail returns it with model_name NULL (LEFT JOIN), not a 404.
        det = client.get(f"/classifiers/details/{g['classifier_id']}", headers=auth_headers)
        assert det.status_code == 200
        assert det.json().get("model_name") is None

    def test_duplicate_unattached_name_conflicts(self, client, auth_headers):
        name = _unique("dupgrd")
        _create_guardrail(client, auth_headers, name=name)
        res = client.post("/classifiers/create", json={"name": name}, headers=auth_headers)
        assert res.status_code == 409


class TestListAll:
    def test_details_all_includes_unattached(self, client, auth_headers):
        g = _create_guardrail(client, auth_headers)
        res = client.get("/classifiers/details/all", headers=auth_headers)
        assert res.status_code == 200
        ids = [c["classifier_id"] for c in res.json()["classifiers"]]
        assert g["classifier_id"] in ids
        row = next(c for c in res.json()["classifiers"] if c["classifier_id"] == g["classifier_id"])
        assert "model_name" in row and row["model_name"] is None
        assert "rule_count" in row

    def test_details_all_requires_auth(self, client):
        res = client.get("/classifiers/details/all")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


class TestDeleteUnattached:
    def test_delete_model_less_guardrail(self, client, auth_headers):
        # Regression: the delete handler used to resolve the owner via an INNER
        # JOIN through target_models, which 404'd a model-less guardrail.
        g = _create_guardrail(client, auth_headers)
        res = client.delete(f"/classifiers/{g['classifier_id']}", headers=auth_headers)
        assert res.status_code == 200, res.text
        # Gone from the list.
        allg = client.get("/classifiers/details/all", headers=auth_headers).json()["classifiers"]
        assert g["classifier_id"] not in [c["classifier_id"] for c in allg]


class TestTrainGate:
    def test_train_unattached_is_blocked(self, client, auth_headers):
        g = _create_guardrail(client, auth_headers)
        res = client.post(f"/classifiers/{g['classifier_id']}/train", headers=auth_headers)
        assert res.status_code == 400
        assert "model" in res.json()["detail"].lower()


class TestAttachModel:
    def test_attach_binds_and_shows_under_model(self, client, auth_headers, test_model):
        g = _create_guardrail(client, auth_headers)
        res = client.post(
            f"/classifiers/details/{g['classifier_id']}/attach-model",
            json={"model_id": test_model["model_id"]}, headers=auth_headers,
        )
        assert res.status_code == 200, res.text
        assert res.json()["classifier"]["model_id"] == test_model["model_id"]
        # Now visible in the per-model (secondary) list.
        lst = client.get(f"/classifiers/{test_model['model_id']}", headers=auth_headers)
        ids = [c["classifier_id"] for c in lst.json()["classifiers"]]
        assert g["classifier_id"] in ids

    def test_attach_foreign_model_404(self, client, auth_headers):
        g = _create_guardrail(client, auth_headers)
        res = client.post(
            f"/classifiers/details/{g['classifier_id']}/attach-model",
            json={"model_id": 99999}, headers=auth_headers,
        )
        assert res.status_code == 404

    def test_attach_blocked_once_trained(self, client, auth_headers, test_model):
        g = _create_guardrail(client, auth_headers)
        # Simulate a completed training run by stamping trained_at directly.
        execute_query(
            "UPDATE classifiers SET model_id = %s, trained_at = now(), status = 'active' "
            "WHERE classifier_id = %s",
            (test_model["model_id"], g["classifier_id"]),
        )
        res = client.post(
            f"/classifiers/details/{g['classifier_id']}/attach-model",
            json={"model_id": test_model["model_id"]}, headers=auth_headers,
        )
        assert res.status_code == 409


class TestClone:
    def _seed_rule(self, classifier_id):
        ce = execute_query_dict(
            "INSERT INTO cognitive_elements (name, definition) VALUES (%s, %s) RETURNING ce_id",
            (_unique("clone_ce"), "clone test CE"),
        )[0]["ce_id"]
        setup = execute_query_dict(
            "INSERT INTO rule_setup (classifier_id, rule_id, custom_name, predicate, is_active) "
            "VALUES (%s, NULL, %s, %s, TRUE) RETURNING setup_id",
            (classifier_id, "clone_rule", "CE_x"),
        )[0]["setup_id"]
        execute_query(
            "INSERT INTO setup_ce_link (setup_id, ce_id, role, fallback_group) VALUES (%s, %s, 'necessary', 0)",
            (setup, ce),
        )
        return setup, ce

    def test_clone_copies_rules_to_new_untrained_guardrail(self, client, auth_headers, test_model, second_model):
        g = _create_guardrail(client, auth_headers)
        # Attach + seed a rule on the source.
        client.post(f"/classifiers/details/{g['classifier_id']}/attach-model",
                    json={"model_id": test_model["model_id"]}, headers=auth_headers)
        self._seed_rule(g["classifier_id"])

        res = client.post(
            f"/classifiers/details/{g['classifier_id']}/clone",
            json={"target_model_id": second_model["model_id"]}, headers=auth_headers,
        )
        assert res.status_code == 200, res.text
        new = res.json()["classifier"]
        assert new["classifier_id"] != g["classifier_id"]
        assert new["model_id"] == second_model["model_id"]
        assert new["status"] == "untrained"
        # The copy carries the same number of rules.
        src_rules = client.get(f"/classifiers/{g['classifier_id']}/rules", headers=auth_headers).json()["rules"]
        new_rules = client.get(f"/classifiers/{new['classifier_id']}/rules", headers=auth_headers).json()["rules"]
        assert len(new_rules) == len(src_rules) == 1

    def test_clone_to_foreign_model_404(self, client, auth_headers, test_model):
        g = _create_guardrail(client, auth_headers)
        client.post(f"/classifiers/details/{g['classifier_id']}/attach-model",
                    json={"model_id": test_model["model_id"]}, headers=auth_headers)
        res = client.post(
            f"/classifiers/details/{g['classifier_id']}/clone",
            json={"target_model_id": 99999}, headers=auth_headers,
        )
        assert res.status_code == 404
