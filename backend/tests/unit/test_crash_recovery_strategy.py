"""Unit tests for the RecoveryStrategy abstraction.

The integration tests (tests/integration/test_crash_recovery.py) verify that
each concrete strategy works against a real DB. These tests target the
orchestrator-level guarantees — the parts of the design that hold regardless
of what any individual strategy does:

  1. RECOVERY_STRATEGIES is iterable and every entry is a RecoveryStrategy.
  2. Every strategy carries a non-empty `name` for log readability.
  3. `run_all_recovery` continues to the next strategy even when one raises.
  4. A new strategy can be added to the list without changing the orchestrator.

Item 3 is the core promise of the Strategy pattern here and the most likely
thing to silently regress.
"""
import logging
import pytest

from utils.crash_recovery import (
    RECOVERY_STRATEGIES,
    RecoveryStrategy,
    StuckTrainingRecovery,
    StuckTestGenerationRecovery,
    IncompletePipelineRecovery,
    OrphanedClassifierDirRecovery,
    run_all_recovery,
)


class TestRegistry:
    def test_registry_is_non_empty(self):
        assert len(RECOVERY_STRATEGIES) > 0

    def test_every_entry_is_a_recovery_strategy(self):
        for strategy in RECOVERY_STRATEGIES:
            assert isinstance(strategy, RecoveryStrategy), (
                f"{strategy!r} does not inherit RecoveryStrategy"
            )

    def test_every_strategy_has_a_name(self):
        # Names are surfaced in error messages and logs. Empty names break
        # log search after a real incident.
        for strategy in RECOVERY_STRATEGIES:
            assert strategy.name, f"strategy {type(strategy).__name__} has empty name"
            assert isinstance(strategy.name, str)

    def test_strategy_names_are_unique(self):
        # Two strategies sharing a name would make log triage ambiguous.
        names = [s.name for s in RECOVERY_STRATEGIES]
        assert len(names) == len(set(names)), f"duplicate names in: {names}"

    def test_expected_strategies_present(self):
        # Sanity check: every documented strategy is wired in. If a future
        # change drops one from the registry the boot path silently stops
        # cleaning up that class of stuck state.
        types = {type(s) for s in RECOVERY_STRATEGIES}
        assert StuckTrainingRecovery in types
        assert StuckTestGenerationRecovery in types
        assert IncompletePipelineRecovery in types
        assert OrphanedClassifierDirRecovery in types


class TestOrchestratorIsolation:
    """run_all_recovery must keep going if one strategy raises.

    This is the central promise of the Strategy pattern wiring here. We patch
    the registry temporarily with a strategy that always raises, sandwiched
    between two that record they ran, and verify both no-op strategies still
    executed and the run_all_recovery call returned normally.
    """

    def test_failing_strategy_does_not_block_subsequent_strategies(self, monkeypatch, caplog):
        ran = []

        class _RecordingStrategy(RecoveryStrategy):
            name = "recording-A"

            def run(self):
                ran.append("A")

        class _FailingStrategy(RecoveryStrategy):
            name = "boom"

            def run(self):
                raise RuntimeError("intentional test failure")

        class _RecordingStrategyB(RecoveryStrategy):
            name = "recording-B"

            def run(self):
                ran.append("B")

        # Replace the live registry contents in place, so run_all_recovery
        # iterates only our test strategies. monkeypatch restores it after.
        from utils import crash_recovery as cr

        monkeypatch.setattr(
            cr,
            "RECOVERY_STRATEGIES",
            [_RecordingStrategy(), _FailingStrategy(), _RecordingStrategyB()],
        )

        # Call: should not raise, even though the middle strategy throws.
        with caplog.at_level(logging.ERROR, logger="utils.crash_recovery"):
            run_all_recovery()

        # Both surrounding strategies executed.
        assert ran == ["A", "B"], (
            f"failing strategy short-circuited the loop. ran={ran}"
        )

        # The error was logged with the failing strategy's name (so a real
        # operator can find it).
        log_text = caplog.text
        assert "boom" in log_text
        assert "intentional test failure" in log_text

    def test_run_all_recovery_with_empty_registry(self, monkeypatch):
        # An empty registry should be a no-op, not crash.
        from utils import crash_recovery as cr

        monkeypatch.setattr(cr, "RECOVERY_STRATEGIES", [])
        # Should return cleanly.
        run_all_recovery()


class TestRecoveryStrategyAbstract:
    def test_cannot_instantiate_abstract_base(self):
        # ABC enforcement: instantiating RecoveryStrategy directly should fail.
        with pytest.raises(TypeError):
            RecoveryStrategy()  # type: ignore[abstract]

    def test_subclass_without_run_cannot_be_instantiated(self):
        class _Incomplete(RecoveryStrategy):
            name = "incomplete"
            # missing run()

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]
