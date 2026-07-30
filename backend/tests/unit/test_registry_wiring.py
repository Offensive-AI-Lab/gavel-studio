"""Tests for the backend wiring: build_poller() gating and that the
reconcile action PROBES freshness (without mutating the DB) and pushes the
badge state to the frontend."""
from services.registry_sync import wiring
from services.registry_sync.poller import RegistryUpdatePoller


def test_build_poller_none_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_REGISTRY_POLLER", "0")
    assert wiring.build_poller() is None


def test_build_poller_returns_wired_poller(monkeypatch):
    monkeypatch.delenv("ENABLE_REGISTRY_POLLER", raising=False)
    monkeypatch.delenv("REGISTRY_POLL_S", raising=False)
    p = wiring.build_poller()
    assert isinstance(p, RegistryUpdatePoller)
    assert p.interval_s == 300.0
    assert isinstance(p.client, wiring._LibraryUpdateNotifier)


def test_build_poller_respects_interval_env(monkeypatch):
    monkeypatch.delenv("ENABLE_REGISTRY_POLLER", raising=False)
    monkeypatch.setenv("REGISTRY_POLL_S", "42")
    assert wiring.build_poller().interval_s == 42.0


def test_build_poller_bad_interval_falls_back(monkeypatch):
    monkeypatch.delenv("ENABLE_REGISTRY_POLLER", raising=False)
    monkeypatch.setenv("REGISTRY_POLL_S", "not-a-number")
    assert wiring.build_poller().interval_s == 300.0


def test_notifier_checks_and_pushes_badge_without_mutating(monkeypatch):
    seen = []
    monkeypatch.setattr("services.library_sync.check_for_updates",
                        lambda: {"available": True, "checked": True, "reason": None})
    monkeypatch.setattr("services.library_events.set_available",
                        lambda available: seen.append(available))
    # The notifier must NEVER pull records — that's the user's click.
    def _no_sync(*a, **k):
        raise AssertionError("reconcile must not sync_library")
    monkeypatch.setattr("services.library_sync.sync_library", _no_sync)

    out = wiring._LibraryUpdateNotifier().reconcile()

    assert out["available"] is True
    assert seen == [True]


def test_notifier_pushes_synced_when_up_to_date(monkeypatch):
    seen = []
    monkeypatch.setattr("services.library_sync.check_for_updates",
                        lambda: {"available": False, "checked": True, "reason": None})
    monkeypatch.setattr("services.library_events.set_available",
                        lambda available: seen.append(available))
    wiring._LibraryUpdateNotifier().reconcile()
    assert seen == [False]


def test_notifier_skips_badge_when_probe_unchecked(monkeypatch):
    """A transient registry outage (checked=False) must not touch the badge —
    a stale 'synced' beats a phantom 'updates available'."""
    seen = []
    monkeypatch.setattr("services.library_sync.check_for_updates",
                        lambda: {"available": False, "checked": False, "reason": "down"})
    monkeypatch.setattr("services.library_events.set_available",
                        lambda available: seen.append(available))
    wiring._LibraryUpdateNotifier().reconcile()
    assert seen == []


def test_built_poller_runs_freshness_checks(monkeypatch):
    """Integration: the REAL wired poller ticks the real notifier ->
    check_for_updates (no DB mutation)."""
    import asyncio

    monkeypatch.delenv("ENABLE_REGISTRY_POLLER", raising=False)
    checks = []
    monkeypatch.setattr("services.library_sync.check_for_updates",
                        lambda: checks.append(1) or {"available": False, "checked": True})
    monkeypatch.setattr("services.library_events.set_available", lambda available: None)

    p = wiring.build_poller()
    assert p is not None
    p.interval_s = 0.01                          # speed the loop up for the test
    p.initial_delay_s = 0.0

    async def run():
        await p.start()
        await asyncio.sleep(0.1)
        await p.stop()

    asyncio.run(run())
    assert sum(checks) >= 2
