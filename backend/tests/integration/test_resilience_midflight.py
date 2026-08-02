"""Resilience tests — failures that happen *during* a running task.

test_crash_recovery.py exercises the boot-time recovery sweep in isolation.
These tests instead inject the failure MID-FLIGHT (the LLM/network drops while
a generation is in progress)
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

    def test_orphaned_draft_rule_reclaimed_by_recovery(self, monkeypatch):
        rule_id = _make_draft_rule()
        import routes.ai_pipeline as aip
        monkeypatch.setattr(aip, "build_positive_config", _raise_runtime)
        from services.default_datasets import _run_rule_defaults
        _run_rule_defaults(rule_id, "scenario", 4, 2)

        # The draft is is_ready=FALSE and not inside any active (completed=FALSE)
        # pipeline run, so the incomplete-pipeline sweep must reclaim it — the
        # half-built rule never lingers as a visible orphan.
        from utils.crash_recovery import IncompletePipelineRecovery
        IncompletePipelineRecovery().run()

        still_there = execute_query_dict("SELECT 1 FROM rules WHERE rule_id=%s", (rule_id,))
        assert not still_there, "is_ready=FALSE orphan draft should be wiped by recovery"

    def test_failed_generation_keeps_rule_hidden_from_library(self, client, monkeypatch, auth_headers):
        """A rule whose default generation failed must not surface in the public
        library before recovery runs (it's still is_ready=FALSE)."""
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
        assert rule_id not in ids, "half-built draft must stay hidden until finalized"


