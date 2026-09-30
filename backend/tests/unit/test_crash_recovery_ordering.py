"""Regression: the post-warmup recovery pass must not re-run strategies that
already ran early. HTTP serving starts immediately but run_all_recovery waits
~30s for model warmup, so re-running "wipe unfinished pipeline runs" then
deleted wizard runs the operator opened during that window ("Pipeline run not
found" on the next PATCH).
"""
import pytest

from utils import crash_recovery


class _Strategy:
    def __init__(self, name, safe_before_warmup, fail=False):
        self.name = name
        self.safe_before_warmup = safe_before_warmup
        self.fail = fail
        self.runs = 0

    def run(self):
        self.runs += 1
        if self.fail:
            raise RuntimeError("boom")


@pytest.fixture
def strategies(monkeypatch):
    early = _Strategy("early", True)
    late = _Strategy("late", False)
    monkeypatch.setattr(crash_recovery, "RECOVERY_STRATEGIES", [early, late])
    monkeypatch.setattr(crash_recovery, "_ran_early", set())
    return early, late


def test_an_early_strategy_is_not_rerun_by_the_post_warmup_pass(strategies):
    early, late = strategies
    crash_recovery.run_early_recovery()
    crash_recovery.run_all_recovery()
    assert early.runs == 1
    assert late.runs == 1


def test_run_all_alone_still_runs_everything(strategies):
    early, late = strategies
    crash_recovery.run_all_recovery()
    assert early.runs == 1
    assert late.runs == 1


def test_an_early_strategy_that_failed_is_retried_by_the_post_warmup_pass(monkeypatch):
    flaky = _Strategy("flaky", True, fail=True)
    monkeypatch.setattr(crash_recovery, "RECOVERY_STRATEGIES", [flaky])
    monkeypatch.setattr(crash_recovery, "_ran_early", set())
    crash_recovery.run_early_recovery()
    crash_recovery.run_all_recovery()
    assert flaky.runs == 2
