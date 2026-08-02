"""Tests for Cognitive Element management: CRUD, bookmarks."""
import pytest
import time


class TestCECreation:
    """CE creation and retrieval."""

    def test_create_ce(self, client, auth_headers, test_user):
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": f"test_ce_{int(time.time()) % 100000}",
            "definition": "A test cognitive element for unit testing.",
            "categories": [],
        }, headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert "ce_id" in data

    def test_create_ce_duplicate_name(self, client, auth_headers, test_user):
        name = f"dup_ce_{int(time.time()) % 100000}"
        # Create first
        client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": name,
            "definition": "First",
        }, headers=auth_headers)
        # Create again - should upsert, not error
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": name,
            "definition": "Updated",
        }, headers=auth_headers)
        assert res.status_code == 200

    def test_create_ce_empty_name(self, client, auth_headers, test_user):
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": "",
            "definition": "empty name",
        }, headers=auth_headers)
        assert res.status_code in (400, 422)

    def test_get_ces(self, client, auth_headers, test_user):
        res = client.get(f"/cognitive/{test_user['user_id']}", headers=auth_headers)
        assert res.status_code == 200
        assert isinstance(res.json(), list)


class TestCEBookmarks:
    """CE bookmark operations."""

    def test_bookmark_draft_ce_is_allowed(self, client, auth_headers, test_user):
        """A freshly-created LOCAL draft CE CAN be bookmarked.

        Bookmarks are keyed by the local SERIAL id (not public_id), so a
        draft is bookmarkable like anything else — which is what a
        single-user workspace wants."""
        ce_res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": f"bm_ce_{int(time.time()) % 100000}",
            "definition": "bookmarkable",
        }, headers=auth_headers)
        if ce_res.status_code != 200:
            pytest.skip("Could not create CE")
        ce_id = ce_res.json()["ce_id"]

        res = client.post("/cognitive/bookmark", json={
            "user_id": test_user["user_id"],
            "ce_id": ce_id,
        }, headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["status"] == "bookmarked"

        # ...and it comes back in the listing.
        listed = client.get(f"/cognitive/bookmarks/{test_user['user_id']}", headers=auth_headers)
        assert listed.status_code == 200
        assert any(b["ce_id"] == ce_id for b in listed.json()["bookmarks"])

    def test_bookmark_nonexistent_ce_rejected(self, client, auth_headers, test_user):
        """Bookmarking a CE id that doesn't exist locally is a 400, not a 500."""
        res = client.post("/cognitive/bookmark", json={
            "user_id": test_user["user_id"],
            "ce_id": 999999999,
        }, headers=auth_headers)
        assert res.status_code == 400

    def test_get_bookmarks(self, client, auth_headers, test_user):
        res = client.get(f"/cognitive/bookmarks/{test_user['user_id']}", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        bms = data.get("bookmarks", data) if isinstance(data, dict) else data
        assert isinstance(bms, list)

    def test_remove_bookmark_nonexistent(self, client, auth_headers, test_user):
        res = client.delete(f"/cognitive/bookmark/{test_user['user_id']}/99999", headers=auth_headers)
        assert res.status_code in (200, 404)
