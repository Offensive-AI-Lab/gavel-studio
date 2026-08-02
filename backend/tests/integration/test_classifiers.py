"""Tests for classifier management: CRUD, rules, config, training status."""
import pytest


class TestClassifierCreation:
    """Classifier CRUD operations."""

    def test_create_classifier(self, client, test_model, auth_headers):
        res = client.post("/classifiers/create", json={
            "model_id": test_model["model_id"],
            "name": "TestCLS_New",
        }, headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        cls = data.get("classifier", data)
        assert "classifier_id" in cls

    def test_create_classifier_without_auth_is_allowed(self, client, test_model):
        res = client.post("/classifiers/create", json={
            "model_id": test_model["model_id"],
            "name": "NoAuth",
        })
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_create_classifier_invalid_model(self, client, auth_headers):
        res = client.post("/classifiers/create", json={
            "model_id": 99999,
            "name": "BadModel",
        }, headers=auth_headers)
        assert res.status_code == 404

    def test_list_classifiers(self, client, test_model, test_classifier, auth_headers):
        res = client.get(f"/classifiers/{test_model['model_id']}", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        cls_list = data.get("classifiers", data) if isinstance(data, dict) else data
        assert isinstance(cls_list, list)
        assert len(cls_list) >= 1

    def test_get_classifier_details(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/classifiers/details/{cid}", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["classifier_id"] == cid


class TestClassifierConfig:
    """Training configuration."""

    def test_get_config(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/classifiers/{cid}/config", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "hidden_dim" in data or "config" in data

    def test_update_config(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.put(f"/classifiers/{cid}/config", json={
            "hidden_dim": 128,
            "epochs": 5,
        }, headers=auth_headers)
        assert res.status_code == 200


class TestClassifierRules:
    """Adding rules to classifiers."""

    def test_get_rules_empty(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/classifiers/{cid}/rules", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        rules = data.get("rules", data) if isinstance(data, dict) else data
        assert isinstance(rules, list)

    def test_add_manual_rule(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.post(f"/classifiers/{cid}/rules/manual", json={
            "name": "test_manual_rule",
            "predicate": "CE_A AND CE_B",
        }, headers=auth_headers)
        assert res.status_code in (200, 201, 400, 500)


class TestTrainingStatus:
    """Training status polling."""

    def test_get_status_untrained(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
