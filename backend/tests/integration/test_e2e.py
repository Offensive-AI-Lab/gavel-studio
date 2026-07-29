"""End-to-end tests: full user workflows from first contact through evaluation."""
import pytest
import time


class TestFullUserWorkflow:
    """E2E: Identify -> Create Model -> Create Classifier -> Add Rules -> Check Status."""

    def test_complete_setup_workflow(self, client):
        """Test the entire setup flow a new operator would follow.

        There is no register/login anymore — the app serves one fixed local
        user, so the flow starts at /user/me and every request is implicitly
        that user (headers stay empty)."""
        suffix = int(time.time() * 1000) % 1000000
        headers = {}

        # 1. The local identity is available immediately — no registration.
        me_res = client.get("/user/me", headers=headers)
        assert me_res.status_code == 200
        user_id = me_res.json()["user_id"]

        # 3. Create model (a lightweight public model). The session fixtures
        # register the same HF repo, and a model source can only be registered
        # once — on 409 reuse the already-registered model, which exercises the
        # same "user has a usable model" outcome.
        model_res = client.post("/models/create", json={
            "user_id": user_id,
            "name": f"E2EModel_{suffix}",
            "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        }, headers=headers)
        assert model_res.status_code in (200, 409), f"Model creation failed: {model_res.text}"
        if model_res.status_code == 200:
            model_data = model_res.json()
            model_id = model_data.get("model", model_data).get("model_id") or model_data.get("model_id")
        else:
            listing = client.get(f"/models/{user_id}", headers=headers)
            assert listing.status_code == 200
            listing_data = listing.json()
            models_list = listing_data.get("models", listing_data) if isinstance(listing_data, dict) else listing_data
            matches = [m for m in models_list if m.get("storage_path") == "HuggingFaceTB/SmolLM2-360M-Instruct"]
            assert matches, "409 said the model exists but it is not in the listing"
            model_id = matches[0]["model_id"]

        # 4. List models
        models_res = client.get(f"/models/{user_id}", headers=headers)
        assert models_res.status_code == 200
        models_data = models_res.json()
        models_list = models_data.get("models", models_data) if isinstance(models_data, dict) else models_data
        assert any(m["model_id"] == model_id for m in models_list)

        # 5. Create classifier
        cls_res = client.post("/classifiers/create", json={
            "model_id": model_id,
            "name": f"E2ECLS_{suffix}",
        }, headers=headers)
        assert cls_res.status_code == 200, f"Classifier creation failed: {cls_res.text}"
        cls_data = cls_res.json()
        classifier_id = cls_data.get("classifier", cls_data).get("classifier_id") or cls_data.get("classifier_id")

        # 6. Verify classifier appears in list
        cls_list = client.get(f"/classifiers/{model_id}", headers=headers)
        assert cls_list.status_code == 200
        cls_list_data = cls_list.json()
        cls_items = cls_list_data.get("classifiers", cls_list_data) if isinstance(cls_list_data, dict) else cls_list_data
        assert any(c["classifier_id"] == classifier_id for c in cls_items)

        # 7. Get classifier details
        details = client.get(f"/classifiers/details/{classifier_id}", headers=headers)
        assert details.status_code == 200
        assert details.json()["status"] in ("untrained", "active")

        # 8. Check training status
        status_res = client.get(f"/classifiers/{classifier_id}/training-status", headers=headers)
        assert status_res.status_code == 200

        # 9. Check dashboard
        dash_res = client.get(f"/dashboard/{user_id}", headers=headers)
        assert dash_res.status_code == 200
        dash_data = dash_res.json()
        assert "stats" in dash_data
        assert dash_data["stats"]["total_models"] >= 1


class TestCEWorkflow:
    """E2E: Create CEs -> Bookmark -> Search -> Create rule from CEs."""

    def test_ce_to_rule_workflow(self, client, test_user, auth_headers):
        uid = test_user["user_id"]
        suffix = int(time.time()) % 100000

        # 1. Create two CEs
        ce_ids = []
        for i in range(2):
            res = client.post("/cognitive/create", json={
                "user_id": uid,
                "name": f"e2e_ce_{i}_{suffix}",
                "definition": f"E2E test CE number {i}",
            }, headers=auth_headers)
            assert res.status_code == 200
            ce_ids.append(res.json()["ce_id"])

        # 2. Bookmarks are local rows again (the central server is gone), so
        #    bookmarking a DRAFT works — a draft has a perfectly good local id.
        for ce_id in ce_ids:
            bm_res = client.post("/cognitive/bookmark", json={
                "user_id": uid,
                "ce_id": ce_id,
            }, headers=auth_headers)
            assert bm_res.status_code == 200

        # 3. The bookmark list must now contain both drafts.
        bm_list = client.get(f"/cognitive/bookmarks/{uid}", headers=auth_headers)
        assert bm_list.status_code == 200
        bm_data = bm_list.json()
        bm_items = bm_data.get("bookmarks", bm_data) if isinstance(bm_data, dict) else bm_data
        assert isinstance(bm_items, list)
        bookmarked_ids = {b.get("ce_id") for b in bm_items if isinstance(b, dict)}
        assert set(ce_ids).issubset(bookmarked_ids)

        # 4. Search library
        search_res = client.get("/library/search", params={
            "q": f"e2e_ce_0_{suffix}",
            "user_id": uid,
        }, headers=auth_headers)
        assert search_res.status_code == 200


class TestDashboard:
    """Dashboard data integrity."""

    def test_dashboard_stats_structure(self, client, test_user, auth_headers):
        res = client.get(f"/dashboard/{test_user['user_id']}", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "stats" in data
        stats = data["stats"]
        required_keys = ["total_models", "total_classifiers", "total_rules", "total_ces"]
        for key in required_keys:
            assert key in stats, f"Missing stat: {key}"
            assert isinstance(stats[key], int)

    def test_dashboard_nonexistent_user(self, client, auth_headers):
        # Dashboard is strictly single-operator: any user_id other than the
        # local user is rejected with 403.
        res = client.get("/dashboard/99999", headers=auth_headers)
        assert res.status_code == 403


class TestLibrarySearch:
    """Library search functionality."""

    def test_search_empty_query(self, client, auth_headers, test_user):
        res = client.get("/library/search", params={
            "q": "",
            "user_id": test_user["user_id"],
        }, headers=auth_headers)
        assert res.status_code in (200, 400, 422)

    def test_search_with_category(self, client, auth_headers, test_user):
        res = client.get("/library/search", params={
            "q": "security",
            "user_id": test_user["user_id"],
            "categories": "Security & Defense",
        }, headers=auth_headers)
        assert res.status_code == 200

    def test_get_categories(self, client, auth_headers):
        res = client.get("/library/categories", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) > 0  # Default categories should exist
