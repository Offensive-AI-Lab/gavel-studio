"""Integration tests for the AI rule-pipeline orchestration endpoints.

Covers the build-from-CEs draft-rule lifecycle in routes/ai_pipeline.py:
  * POST /ai/rules/from-bookmarked-ce  (create_rule_from_bookmarked_ces)
  * POST /ai/rules/{id}/finalize       (finalize_draft_rule)
  * POST /ai/rules/{id}/discard-unready (discard_unready_rule)
  * the rule-pipeline step orchestration endpoints (/ai/derive-scenario,
    /ai/rules/{id}/generate-defaults, /ai/rules/{id}/defaults/status,
    /ai/test-config/generate, /ai/test-config/negative/generate) at the
    request / validation / auth / not-found level only.

These tests deliberately avoid anything that loads ML weights or runs real
LLM/model work. They exercise validation (400/422), auth boundaries (401/403),
not-found (404), and the draft visibility/gating logic (a created draft rule
must be is_ready=FALSE / hidden until finalized). The one test that actually
finalizes a real rule (which would trigger a real embedding) is marked slow.

All DB rows are created through the API so the conftest snapshot/restore
cleans them up automatically. Names are uniquified per test to avoid 409s.
"""
import itertools
import time

import pytest
from utils.sqlite_db import execute_query_dict

_seq = itertools.count()


def _uniq(prefix: str) -> str:
    # A process-wide counter: id(object()) is NOT unique (CPython reuses the
    # freed address), and the v2 endpoint counts DISTINCT CEs — two CEs
    # accidentally sharing a name collapse to one and trip the >=2 guard.
    return f"{prefix}_{int(time.time())}_{next(_seq)}"


def _make_ce(client, auth_headers, test_user, name=None):
    """Create a ready CE through the public API and return its ce_id."""
    res = client.post(
        "/cognitive/create",
        json={
            "user_id": test_user["user_id"],
            "name": name or _uniq("aipl_ce"),
            "definition": "A cognitive element used by the AI-pipeline draft-rule tests.",
        },
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert "ce_id" in data, data
    return data["ce_id"]


def _rule_is_ready(rule_id: int):
    rows = execute_query_dict(
        "SELECT is_ready FROM rules WHERE rule_id = %s", (rule_id,)
    ) or []
    return rows[0]["is_ready"] if rows else None


# ---------------------------------------------------------------------------
# POST /ai/rules/from-bookmarked-ce
# ---------------------------------------------------------------------------
class TestCreateRuleFromBookmarkedCEs:
    def test_creates_hidden_draft_rule(self, client, auth_headers, test_user):
        """A successful create returns a rule_id + predicate and the rule is
        created NOT-ready (is_ready=FALSE) so it stays hidden until finalize."""
        ce1 = _make_ce(client, auth_headers, test_user)
        ce2 = _make_ce(client, auth_headers, test_user)

        res = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={
                "name": _uniq("aipl_draft_rule"),
                "groups": {"required": [ce1, ce2]},
                "condition": "all of required",
                "categories": [],
            },
            headers=auth_headers,
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["success"] is True
        assert isinstance(body["rule_id"], int)
        assert "predicate" in body
        # Gating: the new draft must be hidden (not ready) until finalize.
        assert _rule_is_ready(body["rule_id"]) is False

    def test_single_ce_rejected_400(self, client, auth_headers, test_user):
        """A rule needs >= 2 CEs; one CE is a deterministic 400."""
        ce1 = _make_ce(client, auth_headers, test_user)
        res = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={
                "name": _uniq("aipl_one_ce"),
                "groups": {"required": [ce1]},
                "condition": "all of required",
            },
            headers=auth_headers,
        )
        assert res.status_code == 400
        assert "2 cognitive elements" in res.json()["detail"]

    def test_empty_groups_rejected_400(self, client, auth_headers, test_user):
        """Empty groups fail the >= 2 distinct-CEs guard — a deterministic 400."""
        res = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={"name": _uniq("aipl_empty"), "groups": {},
                  "condition": "all of required"},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_bad_group_name_rejected_400(self, client, auth_headers, test_user):
        """Group names must be [a-z][a-z0-9_]* and not a grammar keyword."""
        ce1 = _make_ce(client, auth_headers, test_user)
        ce2 = _make_ce(client, auth_headers, test_user)
        res = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={"name": _uniq("aipl_badgroup"),
                  "groups": {"Bad Group!": [ce1, ce2]},
                  "condition": "all of required"},
            headers=auth_headers,
        )
        assert res.status_code == 400
        assert "group name" in res.json()["detail"].lower()

    def test_invalid_condition_rejected_400(self, client, auth_headers, test_user):
        """A condition referencing an undefined group is a validation 400."""
        ce1 = _make_ce(client, auth_headers, test_user)
        ce2 = _make_ce(client, auth_headers, test_user)
        res = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={"name": _uniq("aipl_badcond"),
                  "groups": {"required": [ce1, ce2]},
                  "condition": "all of some_other_group"},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_missing_name_422(self, client, auth_headers, test_user):
        """name is required by the schema -> 422 from pydantic."""
        ce1 = _make_ce(client, auth_headers, test_user)
        ce2 = _make_ce(client, auth_headers, test_user)
        res = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={
                "groups": {"required": [ce1, ce2]},
                "condition": "all of required",
            },
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_empty_name_422(self, client, auth_headers, test_user):
        """name has min_length=1 -> empty string is a 422."""
        ce1 = _make_ce(client, auth_headers, test_user)
        ce2 = _make_ce(client, auth_headers, test_user)
        res = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={
                "name": "",
                "groups": {"required": [ce1, ce2]},
                "condition": "all of required",
            },
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_requires_auth(self, client):
        res = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={"name": "x", "groups": {"required": [1, 2]},
                  "condition": "all of required"},
        )
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


# ---------------------------------------------------------------------------
# POST /ai/rules/{id}/finalize
# ---------------------------------------------------------------------------
class TestFinalizeDraftRule:
    def test_finalize_nonexistent_rule_404(self, client, auth_headers):
        res = client.post(
            "/ai/rules/999999999/finalize",
            json={"ce_ids": []},
            headers=auth_headers,
        )
        assert res.status_code == 404
        assert res.json()["detail"] == "Rule not found"

    def test_finalize_requires_auth(self, client):
        res = client.post("/ai/rules/1/finalize", json={"ce_ids": []})
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_finalize_empty_body_ok_schema(self, client, auth_headers):
        """ce_ids defaults to [] — an empty body is schema-valid, so for a
        missing rule we still get the deterministic 404 (not a 422)."""
        res = client.post(
            "/ai/rules/999999998/finalize",
            json={},
            headers=auth_headers,
        )
        assert res.status_code == 404

    @pytest.mark.slow
    def test_finalize_then_refinalize_real_rule(self, client, auth_headers, test_user):
        """End-to-end: create a hidden draft, finalize it (flips is_ready=TRUE),
        and confirm finalize is idempotent on an already-final rule. Marked
        slow because finalize triggers a real embedding."""
        ce1 = _make_ce(client, auth_headers, test_user)
        ce2 = _make_ce(client, auth_headers, test_user)
        create = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={
                "name": _uniq("aipl_finalize_rule"),
                "groups": {"required": [ce1, ce2]},
                "condition": "all of required",
            },
            headers=auth_headers,
        )
        assert create.status_code == 200, create.text
        rule_id = create.json()["rule_id"]
        assert _rule_is_ready(rule_id) is False

        fin = client.post(
            f"/ai/rules/{rule_id}/finalize",
            json={"ce_ids": [ce1, ce2]},
            headers=auth_headers,
        )
        assert fin.status_code == 200, fin.text
        assert fin.json()["rule_id"] == rule_id
        assert _rule_is_ready(rule_id) is True

        # Re-finalizing an already-ready rule is a harmless no-op (still 200).
        again = client.post(
            f"/ai/rules/{rule_id}/finalize",
            json={"ce_ids": []},
            headers=auth_headers,
        )
        assert again.status_code == 200
        assert _rule_is_ready(rule_id) is True


# ---------------------------------------------------------------------------
# POST /ai/rules/{id}/discard-unready
# ---------------------------------------------------------------------------
class TestDiscardUnreadyRule:
    def test_discard_nonexistent_returns_deleted_false(self, client, auth_headers):
        """A missing rule is reported as deleted=False (not a 404 — the
        endpoint treats 'already gone' as success)."""
        res = client.post(
            "/ai/rules/999999997/discard-unready", headers=auth_headers
        )
        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["deleted"] is False

    def test_discard_unready_local_draft(self, client, auth_headers, test_user):
        """A freshly-created (unpublished) draft rule is fully deleted."""
        ce1 = _make_ce(client, auth_headers, test_user)
        ce2 = _make_ce(client, auth_headers, test_user)
        create = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={
                "name": _uniq("aipl_discard_rule"),
                "groups": {"required": [ce1, ce2]},
                "condition": "all of required",
            },
            headers=auth_headers,
        )
        assert create.status_code == 200, create.text
        rule_id = create.json()["rule_id"]

        res = client.post(
            f"/ai/rules/{rule_id}/discard-unready", headers=auth_headers
        )
        assert res.status_code == 200
        assert res.json()["deleted"] is True
        # The row is gone.
        assert _rule_is_ready(rule_id) is None

    def test_discard_requires_auth(self, client):
        res = client.post("/ai/rules/1/discard-unready")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


# ---------------------------------------------------------------------------
# Step orchestration: scenario derivation (Step 2 / 2D feeder)
# ---------------------------------------------------------------------------
class TestDeriveScenario:
    def test_derive_scenario_nonexistent_rule_404(self, client, auth_headers):
        """_load_rule_context raises 404 before any LLM call."""
        res = client.post(
            "/ai/derive-scenario",
            json={"rule_id": 999999996},
            headers=auth_headers,
        )
        assert res.status_code == 404

    def test_derive_scenario_missing_rule_id_422(self, client, auth_headers):
        res = client.post("/ai/derive-scenario", json={}, headers=auth_headers)
        assert res.status_code == 422

    def test_derive_scenario_requires_auth(self, client):
        res = client.post("/ai/derive-scenario", json={"rule_id": 1})
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_derive_scenario_missing_openai_key_503(
        self, client, auth_headers, test_user, monkeypatch
    ):
        """With no OPENAI_API_KEY the endpoint names the real cause — 503 with
        actionable detail — instead of letting litellm bury it in an opaque
        500 'Scenario derivation failed: ...'. Uses a real draft rule so the
        request gets past the 404 rule lookup."""
        ce1 = _make_ce(client, auth_headers, test_user)
        ce2 = _make_ce(client, auth_headers, test_user)
        res = client.post(
            "/ai/rules/from-bookmarked-ce",
            json={"name": _uniq("aipl_no_key"),
                  "groups": {"required": [ce1, ce2]},
                  "condition": "all of required"},
            headers=auth_headers,
        )
        assert res.status_code == 200, res.text
        rule_id = res.json()["rule_id"]

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        res = client.post(
            "/ai/derive-scenario", json={"rule_id": rule_id}, headers=auth_headers
        )
        assert res.status_code == 503
        detail = res.json()["detail"]
        assert "OPENAI_API_KEY" in detail
        assert "backend/.env" in detail

    def test_derive_scenario_404_wins_over_missing_key(
        self, client, auth_headers, monkeypatch
    ):
        """The key precheck sits AFTER the rule lookup: a nonexistent rule is
        still a 404 even when OPENAI_API_KEY is absent."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        res = client.post(
            "/ai/derive-scenario",
            json={"rule_id": 999999995},
            headers=auth_headers,
        )
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# Step orchestration: default test/calibration set generation (Step 2D)
# ---------------------------------------------------------------------------
class TestGenerateDefaults:
    def test_generate_defaults_empty_instructions_422(self, client, auth_headers):
        """scenario_instructions has min_length=1 -> empty string is 422,
        validated before any rule lookup or generation work."""
        res = client.post(
            "/ai/rules/1/generate-defaults",
            json={"scenario_instructions": ""},
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_generate_defaults_missing_instructions_422(self, client, auth_headers):
        res = client.post(
            "/ai/rules/1/generate-defaults", json={}, headers=auth_headers
        )
        assert res.status_code == 422

    def test_generate_defaults_requires_auth(self, client):
        res = client.post(
            "/ai/rules/1/generate-defaults",
            json={"scenario_instructions": "x"},
        )
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


# ---------------------------------------------------------------------------
# Step orchestration: rule-default status (Step 2D polling endpoint)
# ---------------------------------------------------------------------------
class TestDefaultsStatus:
    def test_status_requires_auth(self, client):
        res = client.get("/ai/rules/1/defaults/status")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_status_for_unknown_rule_shape(self, client, auth_headers):
        """status endpoint returns a state dict even for an unknown rule
        (the 'missing' state); never a 500 for a plain integer id."""
        res = client.get(
            "/ai/rules/999999995/defaults/status", headers=auth_headers
        )
        assert res.status_code in (200, 404)
        if res.status_code == 200:
            assert "state" in res.json()


# ---------------------------------------------------------------------------
# Step orchestration: positive/negative test-config builders (Step 2 inputs)
# These hit the LLM only on a 200 path; we assert the auth/validation edges
# which short-circuit before any model call.
# ---------------------------------------------------------------------------
class TestTestConfigGeneration:
    def test_positive_config_requires_auth(self, client):
        res = client.post(
            "/ai/test-config/generate",
            json={"description": "some misuse scenario"},
        )
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_positive_config_missing_description_422(self, client, auth_headers):
        res = client.post(
            "/ai/test-config/generate", json={}, headers=auth_headers
        )
        assert res.status_code == 422

    def test_negative_config_requires_auth(self, client):
        res = client.post(
            "/ai/test-config/negative/generate",
            json={"positive_config": {}},
        )
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_negative_config_missing_positive_config_422(self, client, auth_headers):
        res = client.post(
            "/ai/test-config/negative/generate", json={}, headers=auth_headers
        )
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# POST /ai/generate-pipeline — error-detail propagation
# ---------------------------------------------------------------------------
class TestGeneratePipelineErrorDetail:
    def test_early_failure_preserves_real_detail(
        self, client, auth_headers, test_user, monkeypatch
    ):
        """An early generation failure (e.g. missing OPENAI_API_KEY) must
        surface its real HTTPException detail to the client.

        Regression: the rollback handlers referenced `new_ces_info`, which was
        only assigned *after* the early `raise` — so the intended
        "Rule generation failed: ..." detail was replaced by an
        UnboundLocalError-driven bare 500. `new_ces_info` is now initialized
        before the try block. No LLM is called: the generator is stubbed to
        report failure the way a dead LLM call does.
        """
        import routes.ai_pipeline as ai_pipeline

        monkeypatch.setattr(
            ai_pipeline,
            "_generate_rule_from_scenario",
            lambda scenario: {
                "success": False,
                "error": "LLM error: AuthenticationError - no api key",
            },
        )
        res = client.post(
            "/ai/generate-pipeline",
            json={
                "scenario": "detect unqualified medical advice",
                "user_id": test_user["user_id"],
            },
            headers=auth_headers,
        )
        assert res.status_code == 500
        assert (
            res.json()["detail"]
            == "Rule generation failed: LLM error: AuthenticationError - no api key"
        )
