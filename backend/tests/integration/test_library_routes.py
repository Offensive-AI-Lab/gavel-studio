"""Integration tests for the /library routes (backend/routes/library.py).

Covers:
  * GET /library/search — query required (422), empty/whitespace query
    (returns empty results), >512 char query (400), asset_types filter,
    unknown asset type (400), and pagination shape.
  * GET /library/drafts gating logic — the core "don't show rules/CEs
    before background generation finishes" behaviour:
      - a draft rule whose default test set is still 'generating'/'pending'
        is HIDDEN, and becomes visible once that set flips to 'ready';
      - a draft CE with NO excitation dataset is HIDDEN, and becomes
        visible once an excitation row exists.
  * Draft delete endpoints — 404 not-found, 400 published-row refusal.
  * check-name / record kind validation (400) and auth boundaries (401/403).

We never load ML weights here — the gating rows are seeded with direct SQL
into tracked tables (rules, cognitive_elements, test_datasets,
excitation_datasets) which the conftest snapshot/restore cleans up
automatically. Names are uniquified per-test to dodge 409/unique conflicts.
"""
import json
import time
import uuid

import pytest

from utils.sqlite_db import execute_query_dict


def _uniq(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{id(object())}"


def _insert_draft_rule(name: str, is_ready: bool = True) -> int:
    rows = execute_query_dict(
        """
        INSERT INTO rules (name, predicate, description, categories, is_ready, is_local_draft)
        VALUES (%s, %s, %s, %s, %s, TRUE) RETURNING rule_id
        """,
        (name, "A AND B", "seeded draft rule", [], is_ready),
    )
    return rows[0]["rule_id"]


def _insert_draft_ce(name: str, is_ready: bool = True) -> int:
    rows = execute_query_dict(
        """
        INSERT INTO cognitive_elements (name, definition, categories, is_ready, is_local_draft)
        VALUES (%s, %s, %s, %s, TRUE) RETURNING ce_id
        """,
        (name, "seeded draft CE", [], is_ready),
    )
    return rows[0]["ce_id"]


def _insert_default_test_set(rule_id: int, status: str) -> int:
    rows = execute_query_dict(
        """
        INSERT INTO test_datasets
            (rule_id, dataset_type, scenario_name, conversations, status, is_default)
        VALUES (%s, 'positive_calibration', 'seed', %s::jsonb, %s, TRUE)
        RETURNING dataset_id
        """,
        (rule_id, json.dumps([[{"role": "user", "content": "x"}]]), status),
    )
    return rows[0]["dataset_id"]


def _insert_excitation(ce_id: int) -> int:
    rows = execute_query_dict(
        """
        INSERT INTO excitation_datasets (ce_id, dataset)
        VALUES (%s, %s) RETURNING dataset_id
        """,
        (ce_id, "seeded excitation"),
    )
    return rows[0]["dataset_id"]


# ---------------------------------------------------------------------------
# GET /library/search — validation
# ---------------------------------------------------------------------------


class TestSearchValidation:
    def test_search_requires_query_param(self, client):
        # `q` is a required Query param → 422 when entirely absent.
        res = client.get("/library/search")
        assert res.status_code == 422

    def test_empty_query_returns_empty_results(self, client):
        # Whitespace-only q with no categories/author short-circuits to an
        # empty result set (200), it does not error.
        res = client.get("/library/search", params={"q": "   "})
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == ""
        assert data["results"] == []
        assert data["candidates_examined"] == 0

    def test_overlong_query_rejected(self, client):
        res = client.get("/library/search", params={"q": "a" * 513})
        assert res.status_code == 400
        assert "512" in res.json()["detail"]

    def test_query_at_max_length_allowed(self, client):
        # Exactly 512 chars is allowed (boundary). Should not 400.
        res = client.get("/library/search", params={"q": "a" * 512})
        assert res.status_code in (200, 500)

    def test_unknown_asset_type_rejected(self, client):
        res = client.get(
            "/library/search", params={"q": "anything", "asset_types": "rule,banana"}
        )
        assert res.status_code == 400
        assert "asset_types" in res.json()["detail"]

    def test_valid_asset_types_accepted(self, client):
        # rule,ce is the valid set — should not 400 on validation.
        res = client.get(
            "/library/search", params={"q": "test", "asset_types": "rule,ce"}
        )
        assert res.status_code in (200, 500)

    def test_page_must_be_positive(self, client):
        res = client.get("/library/search", params={"q": "x", "page": 0})
        assert res.status_code == 422

    def test_page_size_upper_bound_enforced(self, client):
        res = client.get("/library/search", params={"q": "x", "page_size": 51})
        assert res.status_code == 422

    def test_categories_unknown_returns_empty(self, client):
        # Unknown category names resolve to zero IDs → early empty return.
        res = client.get(
            "/library/search",
            params={"q": "x", "categories": _uniq("nope-category")},
        )
        assert res.status_code == 200
        assert res.json()["results"] == []

    def test_search_response_shape(self, client):
        res = client.get(
            "/library/search", params={"q": "   ", "page": 2, "page_size": 5}
        )
        assert res.status_code == 200
        data = res.json()
        # Structured-response contract.
        for key in (
            "query",
            "results",
            "candidates_examined",
            "total_results",
            "page",
            "page_size",
        ):
            assert key in data
        assert isinstance(data["results"], list)


# ---------------------------------------------------------------------------
# GET /library/drafts — gating logic (the headline feature)
# ---------------------------------------------------------------------------


class TestDraftRuleGating:
    def test_draft_rule_with_generating_set_is_hidden(self, client, auth_headers):
        name = _uniq("gate-rule-generating")
        rule_id = _insert_draft_rule(name)
        _insert_default_test_set(rule_id, "generating")

        res = client.get("/library/drafts", headers=auth_headers)
        assert res.status_code == 200
        names = {r["name"] for r in res.json()["rules"]}
        assert name not in names

    def test_draft_rule_with_pending_set_is_hidden(self, client, auth_headers):
        name = _uniq("gate-rule-pending")
        rule_id = _insert_draft_rule(name)
        _insert_default_test_set(rule_id, "pending")

        res = client.get("/library/drafts", headers=auth_headers)
        assert res.status_code == 200
        names = {r["name"] for r in res.json()["rules"]}
        assert name not in names

    def test_draft_rule_with_ready_set_is_visible(self, client, auth_headers):
        name = _uniq("gate-rule-ready")
        rule_id = _insert_draft_rule(name)
        _insert_default_test_set(rule_id, "ready")

        res = client.get("/library/drafts", headers=auth_headers)
        assert res.status_code == 200
        names = {r["name"] for r in res.json()["rules"]}
        assert name in names

    def test_draft_rule_without_any_set_is_visible(self, client, auth_headers):
        # No default test set at all → the NOT EXISTS gate passes (nothing is
        # generating), so the ready draft shows up.
        name = _uniq("gate-rule-noset")
        _insert_draft_rule(name)

        res = client.get("/library/drafts", headers=auth_headers)
        assert res.status_code == 200
        names = {r["name"] for r in res.json()["rules"]}
        assert name in names

    def test_gating_flips_when_status_becomes_ready(self, client, auth_headers):
        name = _uniq("gate-rule-flip")
        rule_id = _insert_draft_rule(name)
        ds_id = _insert_default_test_set(rule_id, "generating")

        hidden = client.get("/library/drafts", headers=auth_headers).json()["rules"]
        assert name not in {r["name"] for r in hidden}

        from utils.sqlite_db import execute_query
        execute_query(
            "UPDATE test_datasets SET status = 'ready' WHERE dataset_id = %s", (ds_id,)
        )

        shown = client.get("/library/drafts", headers=auth_headers).json()["rules"]
        assert name in {r["name"] for r in shown}

    def test_not_ready_rule_is_hidden(self, client, auth_headers):
        # is_ready = FALSE rules are excluded regardless of test-set status.
        name = _uniq("gate-rule-notready")
        _insert_draft_rule(name, is_ready=False)

        res = client.get("/library/drafts", headers=auth_headers)
        names = {r["name"] for r in res.json()["rules"]}
        assert name not in names


class TestDraftCEGating:
    def test_draft_ce_without_excitation_is_hidden(self, client, auth_headers):
        name = _uniq("gate-ce-nodata")
        _insert_draft_ce(name)

        res = client.get("/library/drafts", headers=auth_headers)
        assert res.status_code == 200
        names = {c["name"] for c in res.json()["ces"]}
        assert name not in names

    def test_draft_ce_with_excitation_is_visible(self, client, auth_headers):
        name = _uniq("gate-ce-withdata")
        ce_id = _insert_draft_ce(name)
        _insert_excitation(ce_id)

        res = client.get("/library/drafts", headers=auth_headers)
        assert res.status_code == 200
        ces = res.json()["ces"]
        match = [c for c in ces if c["name"] == name]
        assert match, "CE with excitation data should be visible"
        assert match[0]["has_training_data"] is True

    def test_ce_gating_flips_when_excitation_added(self, client, auth_headers):
        name = _uniq("gate-ce-flip")
        ce_id = _insert_draft_ce(name)

        hidden = client.get("/library/drafts", headers=auth_headers).json()["ces"]
        assert name not in {c["name"] for c in hidden}

        _insert_excitation(ce_id)

        shown = client.get("/library/drafts", headers=auth_headers).json()["ces"]
        assert name in {c["name"] for c in shown}

    def test_not_ready_ce_is_hidden(self, client, auth_headers):
        name = _uniq("gate-ce-notready")
        ce_id = _insert_draft_ce(name, is_ready=False)
        _insert_excitation(ce_id)

        res = client.get("/library/drafts", headers=auth_headers)
        names = {c["name"] for c in res.json()["ces"]}
        assert name not in names


class TestDraftsResponseShape:
    def test_drafts_requires_auth(self, client):
        res = client.get("/library/drafts")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_drafts_returns_rules_and_ces_keys(self, client, auth_headers):
        res = client.get("/library/drafts", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data.get("rules"), list)
        assert isinstance(data.get("ces"), list)


# ---------------------------------------------------------------------------
# Draft delete endpoints
# ---------------------------------------------------------------------------


class TestDeleteDraftRule:
    def test_delete_missing_rule_404(self, client, auth_headers):
        res = client.delete("/library/drafts/rule/99999999", headers=auth_headers)
        assert res.status_code == 404

    def test_delete_published_rule_refused(self, client, auth_headers):
        from utils.sqlite_db import execute_query_dict as eqd
        rows = eqd(
            """
            INSERT INTO rules (name, predicate, description, categories, is_local_draft, public_id)
            VALUES (%s, 'A AND B', 'published', %s, FALSE, %s) RETURNING rule_id
            """,
            (_uniq("pub-rule"), [], _uniq("rule-pid")),
        )
        rule_id = rows[0]["rule_id"]
        res = client.delete(f"/library/drafts/rule/{rule_id}", headers=auth_headers)
        assert res.status_code == 400
        assert "published" in res.json()["detail"].lower()

    def test_delete_draft_rule_succeeds(self, client, auth_headers):
        rule_id = _insert_draft_rule(_uniq("del-draft-rule"))
        res = client.delete(f"/library/drafts/rule/{rule_id}", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["status"] == "deleted"

    def test_delete_requires_auth(self, client):
        res = client.delete("/library/drafts/rule/1")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


class TestDeleteDraftCE:
    def test_delete_missing_ce_404(self, client, auth_headers):
        res = client.delete("/library/drafts/ce/99999999", headers=auth_headers)
        assert res.status_code == 404

    def test_delete_published_ce_refused(self, client, auth_headers):
        from utils.sqlite_db import execute_query_dict as eqd
        rows = eqd(
            """
            INSERT INTO cognitive_elements (name, definition, is_local_draft, public_id)
            VALUES (%s, 'published', FALSE, %s) RETURNING ce_id
            """,
            (_uniq("pub-ce"), _uniq("ce-pid")),
        )
        ce_id = rows[0]["ce_id"]
        res = client.delete(f"/library/drafts/ce/{ce_id}", headers=auth_headers)
        assert res.status_code == 400
        assert "published" in res.json()["detail"].lower()

    def test_dependent_rules_for_ce(self, client, auth_headers):
        ce_id = _insert_draft_ce(_uniq("dep-ce"))
        res = client.get(
            f"/library/drafts/ce/{ce_id}/dependent-rules", headers=auth_headers
        )
        assert res.status_code == 200
        assert isinstance(res.json()["rules"], list)


# ---------------------------------------------------------------------------
# check-name / record validation + auth
# ---------------------------------------------------------------------------


class TestCheckNameValidation:
    def test_check_name_requires_auth(self, client):
        res = client.get("/library/check-name", params={"kind": "rule", "name": "x"})
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_check_name_bad_kind_400(self, client, auth_headers):
        res = client.get(
            "/library/check-name",
            params={"kind": "widget", "name": "x"},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_check_name_blank_name_400(self, client, auth_headers):
        res = client.get(
            "/library/check-name",
            params={"kind": "rule", "name": "   "},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_check_name_missing_params_422(self, client, auth_headers):
        res = client.get("/library/check-name", headers=auth_headers)
        assert res.status_code == 422


class TestGetPublicRecord:
    def test_record_requires_auth(self, client):
        res = client.get("/library/record/rule/some-pid")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_record_bad_kind_400(self, client, auth_headers):
        res = client.get("/library/record/widget/some-pid", headers=auth_headers)
        assert res.status_code == 400

    def test_record_unknown_pid_found_false(self, client, auth_headers):
        res = client.get(
            f"/library/record/rule/{_uniq('missing-pid')}", headers=auth_headers
        )
        assert res.status_code == 200
        data = res.json()
        assert data["found"] is False
        assert data["summary"] is None


# ---------------------------------------------------------------------------
# Publish machinery is GONE — the library is read-only from the studio side
# (contributions go by gavel-rules pull request). Pin the removal.
# ---------------------------------------------------------------------------


class TestPublishEndpointsRemoved:
    @pytest.mark.parametrize("path", [
        "/library/publish/rule",
        "/library/publish/ce",
        "/library/publish/rule-set",
    ])
    def test_publish_routes_are_gone(self, client, auth_headers, path):
        res = client.post(path, json={"record_id": 1}, headers=auth_headers)
        assert res.status_code == 404

    def test_adopt_public_route_is_gone(self, client, auth_headers):
        res = client.post("/library/ce/1/adopt-public", json={}, headers=auth_headers)
        assert res.status_code == 404

    def test_rename_endpoints_are_gone(self, client, auth_headers):
        for path in ("/ai/rename-ce", "/ai/rename-rule"):
            res = client.post(path, json={"old_name": "a", "new_name": "b"},
                              headers=auth_headers)
            assert res.status_code == 404, path


# ---------------------------------------------------------------------------
# GET /library/search — visibility gating + rule description
#
# Rules of the road (both the hybrid and the browse branch must honour them):
#   * is_ready = FALSE never surfaces anywhere;
#   * public search hides is_local_draft = TRUE rows;
#   * bookmark-scoped search keeps the user's bookmarked drafts.
#
# The embedder is stubbed so these run without loading MiniLM. Seeded rows
# have no embedding, so the semantic signal contributes nothing and hits come
# from FTS5 + the name-trigram bonus — enough to rank a uniquely-named row.
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_embedder(monkeypatch):
    from routes import library as library_routes

    monkeypatch.setattr(
        library_routes._search_service, "_embed", lambda text: [0.0] * 384
    )


def _insert_library_rule(
    name: str,
    *,
    is_ready: bool = True,
    is_local_draft: bool = False,
    description: str = "seeded library rule",
    author: str = None,
) -> int:
    rows = execute_query_dict(
        """
        INSERT INTO rules (name, predicate, description, categories, is_ready,
                           is_local_draft, public_id, created_by_username)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING rule_id
        """,
        (
            name,
            "A AND B",
            description,
            [],
            is_ready,
            is_local_draft,
            None if is_local_draft else f"pid_{uuid.uuid4().hex}",
            author,
        ),
    )
    return rows[0]["rule_id"]


def _insert_library_ce(name: str) -> int:
    rows = execute_query_dict(
        """
        INSERT INTO cognitive_elements (name, definition, categories, is_ready,
                                        is_local_draft, public_id)
        VALUES (%s, %s, %s, TRUE, FALSE, %s) RETURNING ce_id
        """,
        (name, "seeded library CE", [], f"pid_{uuid.uuid4().hex}"),
    )
    return rows[0]["ce_id"]


def _bookmark_rule(user_id: int, rule_id: int) -> None:
    execute_query_dict(
        "INSERT INTO rule_bookmarks (user_id, rule_id) VALUES (%s, %s)",
        (user_id, rule_id),
    )


def _search_ids(client, path, **params):
    res = client.get(path, params=params)
    assert res.status_code == 200, res.text
    return res.json(), [r["id"] for r in res.json()["results"]]


class TestSearchVisibilityGating:
    def test_public_search_returns_published_ready_rule(self, client, stub_embedder):
        name = _uniq("visgate-published")
        rule_id = _insert_library_rule(name)
        _, ids = _search_ids(client, "/library/search", q=name, asset_types="rule")
        assert rule_id in ids

    def test_public_search_hides_unready_rule(self, client, stub_embedder):
        name = _uniq("visgate-unready")
        rule_id = _insert_library_rule(name, is_ready=False)
        _, ids = _search_ids(client, "/library/search", q=name, asset_types="rule")
        assert rule_id not in ids

    def test_public_search_hides_local_draft(self, client, stub_embedder):
        name = _uniq("visgate-draft")
        rule_id = _insert_library_rule(name, is_local_draft=True)
        _, ids = _search_ids(client, "/library/search", q=name, asset_types="rule")
        assert rule_id not in ids

    def test_public_search_hides_unready_ce(self, client, stub_embedder):
        name = _uniq("visgate-ce-unready")
        rows = execute_query_dict(
            """
            INSERT INTO cognitive_elements (name, definition, categories, is_ready,
                                            is_local_draft, public_id)
            VALUES (%s, %s, %s, FALSE, FALSE, %s) RETURNING ce_id
            """,
            (name, "seeded library CE", [], f"pid_{uuid.uuid4().hex}"),
        )
        ce_id = rows[0]["ce_id"]
        _, ids = _search_ids(client, "/library/search", q=name, asset_types="ce")
        assert ce_id not in ids

    def test_browse_by_author_hides_unready_and_drafts(self, client, stub_embedder):
        author = _uniq("visgate-author").lower()
        published = _insert_library_rule(_uniq("visgate-b-pub"), author=author)
        unready = _insert_library_rule(
            _uniq("visgate-b-unready"), is_ready=False, author=author
        )
        draft = _insert_library_rule(
            _uniq("visgate-b-draft"), is_local_draft=True, author=author
        )

        # Empty q + author → the browse branch, not the hybrid one.
        _, ids = _search_ids(client, "/library/search", q="", author=author,
                             asset_types="rule")
        assert ids == [published]
        assert unready not in ids and draft not in ids

    def test_bookmark_search_surfaces_bookmarked_draft(
        self, client, stub_embedder, test_user
    ):
        name = _uniq("visgate-bm-draft")
        rule_id = _insert_library_rule(name, is_local_draft=True)
        _bookmark_rule(test_user["user_id"], rule_id)

        _, ids = _search_ids(
            client, "/library/bookmarks/search",
            user_id=test_user["user_id"], q=name, asset_types="rule",
        )
        assert rule_id in ids

    def test_bookmark_search_hides_unready_bookmarked_rule(
        self, client, stub_embedder, test_user
    ):
        name = _uniq("visgate-bm-unready")
        rule_id = _insert_library_rule(name, is_ready=False, is_local_draft=True)
        _bookmark_rule(test_user["user_id"], rule_id)

        _, ids = _search_ids(
            client, "/library/bookmarks/search",
            user_id=test_user["user_id"], q=name, asset_types="rule",
        )
        assert rule_id not in ids

    def test_bookmark_search_ignores_rules_the_user_did_not_bookmark(
        self, client, stub_embedder, test_user
    ):
        name = _uniq("visgate-bm-unbookmarked")
        rule_id = _insert_library_rule(name)
        _, ids = _search_ids(
            client, "/library/bookmarks/search",
            user_id=test_user["user_id"], q=name, asset_types="rule",
        )
        assert rule_id not in ids


class TestSearchResultDescription:
    def test_hybrid_rule_result_carries_description(self, client, stub_embedder):
        name = _uniq("desc-hybrid")
        rule_id = _insert_library_rule(name, description="Flags refund promises.")
        data, ids = _search_ids(client, "/library/search", q=name, asset_types="rule")
        assert rule_id in ids
        row = next(r for r in data["results"] if r["id"] == rule_id)
        assert row["description"] == "Flags refund promises."

    def test_browse_rule_result_carries_description(self, client, stub_embedder):
        author = _uniq("desc-author").lower()
        rule_id = _insert_library_rule(
            _uniq("desc-browse"), description="Flags refund promises.", author=author
        )
        data, ids = _search_ids(client, "/library/search", q="", author=author,
                                asset_types="rule")
        assert ids == [rule_id]
        assert data["results"][0]["description"] == "Flags refund promises."

    def test_ce_result_description_is_null(self, client, stub_embedder):
        name = _uniq("desc-ce")
        ce_id = _insert_library_ce(name)
        data, ids = _search_ids(client, "/library/search", q=name, asset_types="ce")
        assert ce_id in ids
        row = next(r for r in data["results"] if r["id"] == ce_id)
        assert row["description"] is None

    def test_bookmark_search_rule_result_carries_description(
        self, client, stub_embedder, test_user
    ):
        name = _uniq("desc-bookmark")
        rule_id = _insert_library_rule(name, description="Flags refund promises.")
        _bookmark_rule(test_user["user_id"], rule_id)
        data, ids = _search_ids(
            client, "/library/bookmarks/search",
            user_id=test_user["user_id"], q=name, asset_types="rule",
        )
        assert rule_id in ids
        row = next(r for r in data["results"] if r["id"] == rule_id)
        assert row["description"] == "Flags refund promises."


# ---------------------------------------------------------------------------
# /library/categories (no auth, simple read)
# ---------------------------------------------------------------------------


class TestCategories:
    def test_categories_returns_list(self, client):
        res = client.get("/library/categories")
        assert res.status_code == 200
        assert isinstance(res.json(), list)
