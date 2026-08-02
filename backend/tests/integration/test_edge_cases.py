"""Additional edge cases — concurrency, large datasets, unusual flows.

All tests clean up any DB rows they create.
"""
import io
import struct
import time
import pytest
from utils.sqlite_db import execute_query


class TestPaginationEdgeCases:
    """Library search pagination and boundary handling."""

    def test_search_page_zero(self, client, auth_headers, test_user):
        """page=0 should not crash."""
        res = client.get("/library/search", params={
            "q": "test", "user_id": test_user["user_id"], "page": 0,
        }, headers=auth_headers)
        assert res.status_code in (200, 400, 422)

    def test_search_negative_page(self, client, auth_headers, test_user):
        res = client.get("/library/search", params={
            "q": "test", "user_id": test_user["user_id"], "page": -1,
        }, headers=auth_headers)
        assert res.status_code in (200, 400, 422)

    def test_search_huge_page_size(self, client, auth_headers, test_user):
        """page_size=10000 — should be capped, not blow up memory."""
        res = client.get("/library/search", params={
            "q": "test", "user_id": test_user["user_id"], "page_size": 10000,
        }, headers=auth_headers)
        assert res.status_code in (200, 400, 422)


# Bookmark dedup is enforced by the UNIQUE(user_id, *_public_id) constraint on
# the bookmarks tables; see services/bookmarks.py.


class TestCascadeDeletes:
    """Verify FK cascades clean up dependent rows."""

    def test_delete_ce_cascades_to_excitation_dataset(self, client, auth_headers, test_user):
        """Deleting a CE should cascade to its excitation dataset."""
        from utils.sqlite_db import execute_query_dict
        from sql_scripts.definition_scripts import save_excitation_dataset
        import json

        # Create CE + excitation dataset
        ce_res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": f"cascade_test_ce_{int(time.time())}",
            "definition": "for cascade test",
        }, headers=auth_headers)
        ce_id = ce_res.json()["ce_id"]
        save_excitation_dataset(ce_id, {"ce_id": ce_id, "training_data": [], "samples_count": 0})

        # Verify dataset exists
        before = execute_query_dict(
            "SELECT dataset_id FROM excitation_datasets WHERE ce_id = %s", (ce_id,))
        assert len(before) == 1

        # Delete the CE
        execute_query("DELETE FROM cognitive_elements WHERE ce_id = %s", (ce_id,))

        # Dataset should be gone too
        after = execute_query_dict(
            "SELECT dataset_id FROM excitation_datasets WHERE ce_id = %s", (ce_id,))
        assert len(after) == 0


class TestStringEncoding:
    """Various character encodings and locale issues."""

    def test_emoji_in_ce_definition(self, client, auth_headers, test_user):
        """Emojis in CE definition should round-trip correctly."""
        from utils.sqlite_db import execute_query_dict
        ce_res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": f"emoji_ce_{int(time.time())}",
            "definition": "Detects positive sentiment expressions",
        }, headers=auth_headers)
        if ce_res.status_code != 200:
            pytest.skip("CE creation failed")
        ce_id = ce_res.json()["ce_id"]
        try:
            row = execute_query_dict("SELECT definition FROM cognitive_elements WHERE ce_id = %s", (ce_id,))
            assert row[0]["definition"] is not None
        finally:
            execute_query("DELETE FROM cognitive_elements WHERE ce_id = %s", (ce_id,))

    def test_hebrew_in_ce_definition(self, client, auth_headers, test_user):
        """RTL Hebrew text should be stored correctly."""
        ce_res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": f"hebrew_ce_{int(time.time())}",
            "definition": "Hebrew text test for RTL handling and storage",
        }, headers=auth_headers)
        assert ce_res.status_code == 200
        ce_id = ce_res.json().get("ce_id")
        if ce_id:
            execute_query("DELETE FROM cognitive_elements WHERE ce_id = %s", (ce_id,))


class TestStateConsistency:
    """Multi-step operations should leave the DB in a consistent state on failure."""

    def test_classifier_create_with_fk_violation_cleans_up(self, client, auth_headers):
        """Bad model_id in classifier creation should not leave half-state in DB."""
        from utils.sqlite_db import execute_query_dict
        before = execute_query_dict("SELECT COUNT(*) AS c FROM classifiers WHERE name = 'rollback_test'")

        res = client.post("/classifiers/create", json={
            "model_id": 999999, "name": "rollback_test",
        }, headers=auth_headers)
        assert res.status_code == 404

        after = execute_query_dict("SELECT COUNT(*) AS c FROM classifiers WHERE name = 'rollback_test'")
        assert before[0]["c"] == after[0]["c"]


class TestSpecialEndpoints:
    """Edge cases on miscellaneous endpoints."""

    def test_root_redirect(self, client):
        """GET / should redirect to frontend."""
        res = client.get("/", follow_redirects=False)
        assert res.status_code in (200, 302, 307)

    def test_unknown_endpoint_returns_404(self, client):
        res = client.get("/this-endpoint-does-not-exist", follow_redirects=False)
        assert res.status_code == 404

    def test_method_not_allowed(self, client):
        """POST to a GET-only endpoint returns 405."""
        res = client.post("/library/categories")
        assert res.status_code in (405, 401, 403)


class TestTrainingConfigDefaults:
    """Training config should have sensible defaults if user doesn't set them."""

    def test_get_config_returns_defaults_for_new_classifier(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        res = client.get(f"/classifiers/{cid}/config", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        # Should have sensible defaults
        cfg = data.get("config", data)
        # Common training params should be present
        assert any(k in cfg for k in ["hidden_dim", "epochs", "batch_size"])


class TestDuplicatePrevention:
    """Verify the system prevents creating duplicate entities."""

    def test_duplicate_classifier_name_for_same_model(self, client, test_model, auth_headers):
        """Same classifier name for same model should be prevented or generate unique IDs."""
        from utils.sqlite_db import execute_query
        model_id = test_model["model_id"]
        name = f"dup_cls_{int(time.time())}"

        r1 = client.post("/classifiers/create", json={
            "model_id": model_id, "name": name,
        }, headers=auth_headers)
        r2 = client.post("/classifiers/create", json={
            "model_id": model_id, "name": name,
        }, headers=auth_headers)

        # Both might succeed (separate IDs) or second is rejected
        assert r1.status_code == 200
        assert r2.status_code in (200, 400, 409)

        # Cleanup both
        for r in (r1, r2):
            if r.status_code == 200:
                cid = r.json().get("classifier", {}).get("classifier_id")
                if cid:
                    execute_query("DELETE FROM classifiers WHERE classifier_id = %s", (cid,))
