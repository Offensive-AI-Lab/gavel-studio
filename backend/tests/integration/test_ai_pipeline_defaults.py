"""Integration tests for the AI-pipeline rule-defaults / test-set endpoints.

Covers the rule default-set lifecycle endpoints under the `/ai` router
(mounted with prefix `/ai` in main.py) WITHOUT triggering real LLM
generation. The handlers we exercise here return BEFORE any background
thread is started, except where explicitly noted:

  * POST /ai/rules/{rule_id}/generate-defaults   (GenerateDefaultsRequest)
  * GET  /ai/rules/{rule_id}/defaults/status      (states: missing/generating/ready/error)
  * POST /ai/test-set/generate                    (validation + conflict only)
  * GET  /ai/test-set/{dataset_id}/status
  * POST /ai/embed-resources                      (no-op guard path)

Validation focus: missing/blank fields (422), reserved/duplicate names
(400/409), not-found (404), auth boundaries (401/403), and the rolled-up
status state machine. We never assert finished artifacts.

A draft rule is created by direct SQL insert into `rules` (a tracked table —
the conftest snapshot/restore cleans it up). We do NOT start real generation:
the generate-defaults handler hands off to a daemon thread and returns
immediately, and we only assert the immediate response shape / row creation,
not the (LLM-driven) result.
"""
import json
import time

import pytest


def _insert_draft_rule(name: str) -> int:
    """Insert a minimal draft rule row and return its rule_id.

    Mirrors the pattern used by test_aux_datasets.py. `rules` is tracked by
    the integration conftest, so the row is removed on test teardown.
    """
    from utils.sqlite_db import execute_query_dict
    rows = execute_query_dict(
        """
        INSERT INTO rules (name, predicate, description, categories, is_ready, public_id, is_local_draft)
        VALUES (%s, %s, %s, %s, FALSE, %s, %s) RETURNING rule_id
        """,
        (name, "A AND B", "test rule", [], None, True),
    )
    return rows[0]["rule_id"]


def _insert_default_dataset(rule_id: int, dataset_type: str, status: str,
                            generation_log: str = None) -> int:
    """Insert a default (is_default=TRUE, user_id NULL) test_datasets row in a
    given status, so we can drive the rolled-up status state machine without
    running generation. Tracked table -> auto-cleaned."""
    from utils.sqlite_db import execute_query_dict
    rows = execute_query_dict(
        """
        INSERT INTO test_datasets
            (rule_id, user_id, is_default, dataset_type, scenario_name, config, status,
             generation_log)
        VALUES (%s, NULL, TRUE, %s, %s, %s::jsonb, %s, %s)
        RETURNING dataset_id
        """,
        (rule_id, dataset_type, "Test Set", json.dumps({"scenario_instructions": "x"}),
         status, generation_log),
    )
    return rows[0]["dataset_id"]


def _insert_custom_dataset(rule_id: int, user_id: int, dataset_type: str,
                           scenario_name: str, status: str = "ready") -> int:
    """Insert a private custom test_datasets row owned by user_id."""
    from utils.sqlite_db import execute_query_dict
    rows = execute_query_dict(
        """
        INSERT INTO test_datasets
            (rule_id, user_id, is_default, dataset_type, scenario_name, config, status)
        VALUES (%s, %s, FALSE, %s, %s, %s::jsonb, %s)
        RETURNING dataset_id
        """,
        (rule_id, user_id, dataset_type, scenario_name, json.dumps({}), status),
    )
    return rows[0]["dataset_id"]


# ---------------------------------------------------------------------------
# POST /ai/rules/{rule_id}/generate-defaults  — validation + auth
# ---------------------------------------------------------------------------


class TestGenerateDefaultsValidation:
    def test_blank_scenario_instructions_422(self, client, auth_headers):
        # scenario_instructions has Field(..., min_length=1); empty -> 422.
        rule_id = _insert_draft_rule(f"defrule_blank_{int(time.time())}_{id(self)}")
        res = client.post(
            f"/ai/rules/{rule_id}/generate-defaults",
            json={"scenario_instructions": ""},
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_missing_scenario_instructions_422(self, client, auth_headers):
        rule_id = _insert_draft_rule(f"defrule_missing_{int(time.time())}_{id(self)}")
        res = client.post(
            f"/ai/rules/{rule_id}/generate-defaults",
            json={},
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_no_auth_is_allowed(self, client):
        res = client.post(
            "/ai/rules/123456/generate-defaults",
            json={"scenario_instructions": "some scenario"},
        )
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_non_integer_rule_id_422(self, client, auth_headers):
        # Path param is typed int; a non-numeric segment -> 422.
        res = client.post(
            "/ai/rules/not-a-number/generate-defaults",
            json={"scenario_instructions": "scenario"},
            headers=auth_headers,
        )
        assert res.status_code == 422

    def test_missing_openai_key_503(self, client, auth_headers, monkeypatch):
        # Generation is LLM work: without a key the route refuses up front
        # rather than starting a thread whose output never arrives.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        rule_id = _insert_draft_rule(f"defrule_nokey_{int(time.time())}_{id(self)}")
        res = client.post(
            f"/ai/rules/{rule_id}/generate-defaults",
            json={"scenario_instructions": "users coaxing the model into X"},
            headers=auth_headers,
        )
        assert res.status_code == 503
        assert res.json()["detail"]["code"] == "OPENAI_KEY_MISSING"

    def test_accept_kicks_off_and_reports_generating(self, client, auth_headers, monkeypatch):
        # The handler hands off to a daemon thread and returns immediately with
        # a {"success": True, "state": "generating"} envelope. The background
        # generation will fail without usable LLM creds, but that does not
        # affect the synchronous response we assert here.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        rule_id = _insert_draft_rule(f"defrule_accept_{int(time.time())}_{id(self)}")
        res = client.post(
            f"/ai/rules/{rule_id}/generate-defaults",
            json={"scenario_instructions": "users coaxing the model into X"},
            headers=auth_headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert data.get("success") is True
        assert data.get("rule_id") == rule_id
        assert data.get("state") == "generating"

    def test_custom_counts_accepted(self, client, auth_headers, monkeypatch):
        # target_count / calibration_count are optional ints with defaults
        # (100 / 50). Supplying explicit values must be accepted by the schema.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        rule_id = _insert_draft_rule(f"defrule_counts_{int(time.time())}_{id(self)}")
        res = client.post(
            f"/ai/rules/{rule_id}/generate-defaults",
            json={"scenario_instructions": "scenario", "target_count": 10, "calibration_count": 5},
            headers=auth_headers,
        )
        assert res.status_code == 200
        assert res.json().get("state") == "generating"


# ---------------------------------------------------------------------------
# GET /ai/rules/{rule_id}/defaults/status  — state machine
# ---------------------------------------------------------------------------


class TestDefaultsStatus:
    def test_status_missing_for_rule_without_defaults(self, client, auth_headers):
        rule_id = _insert_draft_rule(f"statrule_missing_{int(time.time())}_{id(self)}")
        res = client.get(f"/ai/rules/{rule_id}/defaults/status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["rule_id"] == rule_id
        assert data["state"] == "missing"
        assert data["datasets"] == []

    def test_status_missing_for_nonexistent_rule(self, client, auth_headers):
        # rule_defaults_status doesn't probe rule existence — no default rows
        # means state 'missing' regardless of whether the rule exists.
        res = client.get("/ai/rules/99999999/defaults/status", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["state"] == "missing"

    def test_status_error_when_any_bucket_errored(self, client, auth_headers):
        rule_id = _insert_draft_rule(f"statrule_err_{int(time.time())}_{id(self)}")
        _insert_default_dataset(rule_id, "positive", "ready")
        _insert_default_dataset(rule_id, "negative", "error")
        res = client.get(f"/ai/rules/{rule_id}/defaults/status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["state"] == "error"
        assert len(data["datasets"]) == 2

    def test_status_generating_when_partial(self, client, auth_headers):
        # Some buckets present but not all three ready, none errored -> generating.
        rule_id = _insert_draft_rule(f"statrule_gen_{int(time.time())}_{id(self)}")
        _insert_default_dataset(rule_id, "positive", "ready")
        _insert_default_dataset(rule_id, "negative", "generating")
        res = client.get(f"/ai/rules/{rule_id}/defaults/status", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["state"] == "generating"

    def test_status_no_auth_is_allowed(self, client):
        res = client.get("/ai/rules/1/defaults/status")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


# ---------------------------------------------------------------------------
# Failure handling: the status payload carries WHY a bucket failed, and a
# failed generation reveals the rule (is_ready=TRUE) instead of stranding it
# hidden (where boot-time crash recovery would silently delete it).
# ---------------------------------------------------------------------------


class TestDefaultsFailureRevealAndReason:
    def test_status_carries_per_bucket_error_reason(self, client, auth_headers):
        rule_id = _insert_draft_rule(f"failreason_{int(time.time())}_{id(self)}")
        _insert_default_dataset(rule_id, "positive", "ready",
                                generation_log="Completed: 100 positive conversations")
        _insert_default_dataset(rule_id, "negative", "error",
                                generation_log="negative config generation failed: LLM quota exceeded")
        res = client.get(f"/ai/rules/{rule_id}/defaults/status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["state"] == "error"
        # Rolled-up reason names the bucket + its failure message.
        assert data["error"] == "negative: negative config generation failed: LLM quota exceeded"
        by_type = {d["dataset_type"]: d for d in data["datasets"]}
        assert by_type["negative"]["error"] == "negative config generation failed: LLM quota exceeded"
        # Non-errored buckets don't grow an error field.
        assert "error" not in by_type["positive"]

    def test_failed_generation_reveals_rule_with_reason(self, client, auth_headers, monkeypatch):
        # Simulate the real failure mode: the generation thread's first LLM
        # call (build_positive_config) blows up. Run the thread body
        # synchronously so we can assert its terminal effects.
        import routes.ai_pipeline as ai_pipeline
        from services.default_datasets import _run_rule_defaults
        from utils.sqlite_db import execute_query_dict

        def boom(scenario):
            raise RuntimeError("LLM exploded mid-config")

        monkeypatch.setattr(ai_pipeline, "build_positive_config", boom)

        rule_id = _insert_draft_rule(f"failreveal_{int(time.time())}_{id(self)}")
        rows = execute_query_dict("SELECT is_ready FROM rules WHERE rule_id = %s", (rule_id,))
        assert not rows[0]["is_ready"]  # starts hidden

        _run_rule_defaults(rule_id, "some scenario", 5, 3, finalize_ce_ids=None)

        # The rule is REVEALED despite the failure — it lands in Drafts
        # instead of being wiped by crash recovery on next boot.
        rows = execute_query_dict("SELECT is_ready FROM rules WHERE rule_id = %s", (rule_id,))
        assert bool(rows[0]["is_ready"]) is True

        # And the status endpoint reports the reason for every bucket.
        res = client.get(f"/ai/rules/{rule_id}/defaults/status", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        assert data["state"] == "error"
        assert "LLM exploded mid-config" in data["error"]
        assert len(data["datasets"]) == 3
        assert all("LLM exploded mid-config" in d["error"] for d in data["datasets"])


# ---------------------------------------------------------------------------
# POST /ai/test-set/generate  — validation + conflict (pre-thread paths only)
# ---------------------------------------------------------------------------


class TestTestSetGenerateValidation:
    def test_missing_config_422(self, client, auth_headers):
        # `config: dict` is a required field on TestGenerateRequest.
        res = client.post("/ai/test-set/generate", json={"target_count": 5}, headers=auth_headers)
        assert res.status_code == 422

    def test_no_auth_is_allowed(self, client):
        res = client.post("/ai/test-set/generate", json={"config": {}})
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_reserved_default_name_rejected_400(self, client, auth_headers):
        # "Test Set" is reserved for the rule's public default; a custom set may
        # not use it. Rejected before any thread starts.
        rule_id = _insert_draft_rule(f"tsg_reserved_{int(time.time())}_{id(self)}")
        res = client.post(
            "/ai/test-set/generate",
            json={
                "rule_id": rule_id,
                "config": {"scenario_instructions": "x"},
                "scenario_name": "Test Set",
                "dataset_type": "positive",
            },
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_duplicate_custom_name_conflict_409(self, client, test_user, auth_headers):
        # Pre-seed a custom set with a given name/type for this rule+user, then
        # a second generate with the same (name, type) must 409 before any
        # thread starts.
        rule_id = _insert_draft_rule(f"tsg_dup_{int(time.time())}_{id(self)}")
        name = f"My Custom Set {int(time.time())}_{id(self)}"
        _insert_custom_dataset(rule_id, test_user["user_id"], "positive", name)
        res = client.post(
            "/ai/test-set/generate",
            json={
                "rule_id": rule_id,
                "config": {"scenario_instructions": "x"},
                "scenario_name": name,
                "dataset_type": "positive",
            },
            headers=auth_headers,
        )
        assert res.status_code == 409


# ---------------------------------------------------------------------------
# GET /ai/test-set/{dataset_id}/status
# ---------------------------------------------------------------------------


class TestTestSetStatus:
    def test_status_not_found_404(self, client):
        res = client.get("/ai/test-set/99999999/status")
        assert res.status_code == 404

    def test_status_returns_row_shape(self, client, test_user):
        # Seed a custom row directly and confirm the status endpoint echoes its
        # core fields plus the computed conversation_count.
        rule_id = _insert_draft_rule(f"tss_shape_{int(time.time())}_{id(self)}")
        ds_id = _insert_custom_dataset(
            rule_id, test_user["user_id"], "positive",
            f"set_{int(time.time())}_{id(self)}", status="generating",
        )
        res = client.get(f"/ai/test-set/{ds_id}/status")
        assert res.status_code == 200
        data = res.json()
        assert data["dataset_id"] == ds_id
        assert data["rule_id"] == rule_id
        assert data["dataset_type"] == "positive"
        assert data["status"] == "generating"
        assert "conversation_count" in data
        # status != 'ready' -> count stays 0 without touching conversations col.
        assert data["conversation_count"] == 0


# ---------------------------------------------------------------------------
# POST /ai/embed-resources  — no-op guard path
# ---------------------------------------------------------------------------


class TestEmbedResourcesGuard:
    def test_missing_openai_key_503_when_it_would_generate(self, client, monkeypatch):
        # A rule with no default rows yet ('missing') means this call would
        # start LLM generation, so it needs a key like the rest of the pipeline.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        rule_id = _insert_draft_rule(f"embed_nokey_{int(time.time())}_{id(self)}")
        res = client.post("/ai/embed-resources", json={"ce_ids": [], "rule_id": rule_id})
        assert res.status_code == 503
        assert res.json()["detail"]["code"] == "OPENAI_KEY_MISSING"

    def test_no_key_needed_when_nothing_would_generate(self, client, monkeypatch):
        # CE-only embedding is local work (MiniLM), so a missing key must not
        # turn the request away.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        res = client.post("/ai/embed-resources", json={"ce_ids": [], "rule_id": None})
        assert res.status_code == 200

    def test_empty_payload_is_noop_success(self, client, monkeypatch):
        # No ce_ids and no rule_id -> nothing embedded, nothing generated.
        # This exercises the success envelope without touching embeddings,
        # is_ready flips, or default-set generation.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        res = client.post("/ai/embed-resources", json={"ce_ids": [], "rule_id": None})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["embedded_ces"] == 0
        assert data["embedded_rule"] == 0

    def test_default_payload_is_noop_success(self, client, monkeypatch):
        # All fields optional with sensible defaults — empty body is accepted.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        res = client.post("/ai/embed-resources", json={})
        assert res.status_code == 200
        assert res.json()["status"] == "success"
