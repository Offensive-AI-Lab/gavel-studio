"""Resilience tests — failures that happen *during* a running task.

test_crash_recovery.py exercises the boot-time recovery sweep in isolation.
These tests instead inject the failure MID-FLIGHT (the LLM/network drops while
a generation is in progress, or the central server is unreachable during auth)
and assert two things:

  1. Immediate handling — the system lands in a clean, explicit state (rows
     flip to 'error', never stuck in 'generating'; auth returns an error
     instead of a half-written local user).
  2. Reclaimability — an orphaned draft left behind by the failure is wiped by
     the incomplete-pipeline recovery sweep.

Everything created lives in conftest-tracked tables, so cleanup is automatic.
"""
import time

import pytest

from utils.sqlite_db import execute_query, execute_query_dict


def _uniq(p: str) -> str:
    return f"{p}_{int(time.time() * 1000) % 100_000_000}"


def _raise_runtime(*_a, **_k):
    raise RuntimeError("LLM connection reset")


def _make_draft_rule() -> int:
    return execute_query_dict(
        "INSERT INTO rules (name, predicate, is_ready) VALUES (%s, %s, FALSE) RETURNING rule_id",
        (_uniq("res_rule"), "CE"),
    )[0]["rule_id"]


class TestLLMFailureDuringDefaultsGeneration:
    """The OpenAI/LLM call dies while building the default test/calibration
    sets. The 3 placeholder rows must flip to 'error', not stay 'generating'."""

    def test_config_llm_failure_marks_all_rows_error(self, monkeypatch):
        rule_id = _make_draft_rule()

        # Simulate the LLM connection dropping mid-generation. The generator
        # imports build_positive_config from routes.ai_pipeline at call time,
        # so patching it there is what the running task sees.
        import routes.ai_pipeline as aip
        monkeypatch.setattr(aip, "build_positive_config", _raise_runtime)

        from services.default_datasets import _run_rule_defaults
        # Positive-config failure aborts the run right after the 3 placeholder
        # rows are created — no real dialogue generation happens.
        _run_rule_defaults(rule_id, "some scenario", target_count=4, calibration_count=2)

        rows = execute_query_dict(
            "SELECT dataset_type, status, generation_log FROM test_datasets WHERE rule_id=%s",
            (rule_id,),
        )
        assert rows, "the 3 placeholder rows should have been created up-front"
        statuses = [r["status"] for r in rows]
        # The whole point: NOTHING is left stuck in 'generating'.
        assert all(s == "error" for s in statuses), statuses
        assert any("LLM connection reset" in (r["generation_log"] or "") for r in rows)

    def test_failed_rule_is_revealed_and_survives_recovery(self, monkeypatch):
        """A generation FAILURE reveals the rule (is_ready=TRUE) so it lands in
        Drafts with the failure reason instead of being stranded hidden — where
        the boot-time incomplete-pipeline sweep would silently delete it. The
        revealed rule must therefore SURVIVE a recovery run (the user retries
        the set generation from it). Only crash-interrupted rows (still
        is_ready=FALSE) are reclaimed by the sweep."""
        rule_id = _make_draft_rule()
        import routes.ai_pipeline as aip
        monkeypatch.setattr(aip, "build_positive_config", _raise_runtime)
        from services.default_datasets import _run_rule_defaults
        _run_rule_defaults(rule_id, "scenario", 4, 2)

        rows = execute_query_dict("SELECT is_ready FROM rules WHERE rule_id=%s", (rule_id,))
        assert rows and bool(rows[0]["is_ready"]), "failed rule must be revealed, not stranded hidden"

        from utils.crash_recovery import IncompletePipelineRecovery
        IncompletePipelineRecovery().run()

        still_there = execute_query_dict("SELECT 1 FROM rules WHERE rule_id=%s", (rule_id,))
        assert still_there, "revealed failed rule must survive the recovery sweep (user can retry)"

    def test_failed_generation_keeps_rule_out_of_public_library(self, client, monkeypatch, auth_headers):
        """A rule whose default generation failed is revealed as a LOCAL DRAFT
        (it surfaces via /library/drafts) but must not leak into the public
        library, which lists is_local_draft=FALSE rules only."""
        rule_id = _make_draft_rule()
        import routes.ai_pipeline as aip
        monkeypatch.setattr(aip, "build_positive_config", _raise_runtime)
        from services.default_datasets import _run_rule_defaults
        _run_rule_defaults(rule_id, "scenario", 4, 2)

        res = client.get("/rules/public/library", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        rules = data.get("rules", data) if isinstance(data, dict) else data
        ids = {r.get("rule_id") for r in rules} if isinstance(rules, list) else set()
        assert rule_id not in ids, "failed local draft must not leak into the public library"


# NOTE — TestCentralServerDownDuringAuth was removed with login. It asserted that
# a central-server outage during register/login failed cleanly and wrote no
# partial local user. There is no register/login, and the backend no longer calls
# central for identity at all — only to publish.
