"""Idempotency, double-submit, and out-of-order guards.

Covers the "user behaves unexpectedly" cases: double-clicking an action,
repeating a create, or doing things in the wrong order. We assert the system
guards the operation rather than corrupting state or silently duplicating.
"""
import time

from utils.sqlite_db import execute_query, execute_query_dict


def _uniq(p: str) -> str:
    return f"{p}_{int(time.time() * 1000) % 100_000_000}"


class TestTrainingDoubleSubmit:
    """Double-clicking "Train" must not launch two training runs."""

    def test_train_while_already_training_returns_409(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        prev = execute_query_dict(
            "SELECT status FROM classifiers WHERE classifier_id=%s", (cid,)
        )[0]["status"]
        # Simulate a run already in flight.
        execute_query("UPDATE classifiers SET status='training' WHERE classifier_id=%s", (cid,))
        try:
            res = client.post(f"/classifiers/{cid}/train", headers=auth_headers)
            assert res.status_code == 409
            assert "in progress" in res.json().get("detail", "").lower()
        finally:
            execute_query("UPDATE classifiers SET status=%s WHERE classifier_id=%s", (prev, cid))

    def test_train_nonexistent_classifier_is_404_not_crash(self, client, auth_headers):
        res = client.post("/classifiers/999999999/train", headers=auth_headers)
        assert res.status_code == 404


class TestCreateCEIdempotent:
    """CEs dedupe by name — re-submitting the same CE returns the existing row
    instead of creating a duplicate (the create endpoint is idempotent)."""

    def test_create_same_ce_twice_returns_same_row(self, client, auth_headers, test_user):
        name = _uniq("idem_ce")
        body = {"user_id": test_user["user_id"], "name": name, "definition": "dedup me"}
        r1 = client.post("/cognitive/create", json=body, headers=auth_headers)
        r2 = client.post("/cognitive/create", json=body, headers=auth_headers)
        assert r1.status_code == 200 and r2.status_code == 200, (r1.text, r2.text)
        assert r1.json()["ce_id"] == r2.json()["ce_id"]
        count = execute_query_dict(
            "SELECT count(*) c FROM cognitive_elements WHERE name=%s", (name,)
        )[0]["c"]
        assert count == 1, f"expected exactly one CE named {name}, found {count}"

    def test_rapid_repeated_create_stays_single(self, client, auth_headers, test_user):
        """Five quick repeats (a stuck button / retry storm) → still one row."""
        name = _uniq("storm_ce")
        body = {"user_id": test_user["user_id"], "name": name, "definition": "storm"}
        ids = set()
        for _ in range(5):
            res = client.post("/cognitive/create", json=body, headers=auth_headers)
            if res.status_code == 200:
                ids.add(res.json()["ce_id"])
        assert len(ids) == 1, f"repeated creates should collapse to one id, got {ids}"
        count = execute_query_dict(
            "SELECT count(*) c FROM cognitive_elements WHERE name=%s", (name,)
        )[0]["c"]
        assert count == 1


class TestOutOfOrderOperations:
    """Actions taken before their prerequisites exist must be rejected cleanly,
    not 500 or silently 'succeed'."""

    def test_realtime_analyze_before_training_rejected(self, client, test_classifier, auth_headers):
        cid = test_classifier["classifier_id"]
        prev = execute_query_dict(
            "SELECT status FROM classifiers WHERE classifier_id=%s", (cid,)
        )[0]["status"]
        execute_query(
            "UPDATE classifiers SET status='untrained' WHERE classifier_id=%s "
            "AND status NOT IN ('active','needs_retraining')",
            (cid,),
        )
        try:
            res = client.post(
                f"/realtime/{cid}/analyze-stored",
                json={"messages": [{"role": "assistant", "content": "hi"}]},
                headers=auth_headers,
            )
            # Untrained classifier has no model on disk → must be a clean 400/404.
            assert res.status_code in (400, 404)
        finally:
            execute_query("UPDATE classifiers SET status=%s WHERE classifier_id=%s", (prev, cid))

    def test_finalize_nonexistent_rule_is_404(self, client, auth_headers):
        res = client.post("/ai/rules/999999999/finalize", headers=auth_headers)
        assert res.status_code in (404, 400, 422)
