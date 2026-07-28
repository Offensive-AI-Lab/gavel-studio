"""Tests for services.registry_sync.poller — the periodic HF freshness probe
that replaced the central-server WebSocket subscriber.

Pure asyncio, no network: the poller's `client` is a stub whose `reconcile()`
records calls (or raises). Intervals are tiny real delays — the loop sleeps on
an asyncio Event wait, so sub-second intervals keep the tests fast.
"""
import asyncio
import time

from services.registry_sync.poller import RegistryUpdatePoller


class _Client:
    def __init__(self, raise_on_call=False):
        self.calls = 0
        self.raise_on_call = raise_on_call

    def reconcile(self):
        self.calls += 1
        if self.raise_on_call:
            raise RuntimeError("HF is down")
        return {"available": False, "checked": True}


def test_polls_repeatedly_until_stopped():
    client = _Client()

    async def run():
        p = RegistryUpdatePoller(client, interval_s=0.01, initial_delay_s=0)
        await p.start()
        await asyncio.sleep(0.15)
        await p.stop()

    asyncio.run(run())
    assert client.calls >= 2


def test_initial_delay_defers_first_tick():
    client = _Client()

    async def run():
        p = RegistryUpdatePoller(client, interval_s=0.01, initial_delay_s=30)
        await p.start()
        await asyncio.sleep(0.1)
        assert client.calls == 0          # still inside the initial delay
        await p.stop()

    asyncio.run(run())
    assert client.calls == 0


def test_reconcile_failure_does_not_kill_loop():
    client = _Client(raise_on_call=True)

    async def run():
        p = RegistryUpdatePoller(client, interval_s=0.01, initial_delay_s=0)
        await p.start()
        await asyncio.sleep(0.15)
        await p.stop()

    asyncio.run(run())
    # Every tick raised, and the loop kept ticking anyway.
    assert client.calls >= 2


def test_stop_during_long_sleep_exits_promptly():
    client = _Client()

    async def run():
        p = RegistryUpdatePoller(client, interval_s=600, initial_delay_s=0)
        await p.start()
        await asyncio.sleep(0.05)         # let the first tick land
        t0 = time.monotonic()
        await p.stop()
        return time.monotonic() - t0

    elapsed = asyncio.run(run())
    assert client.calls == 1
    assert elapsed < 1.0                  # did not wait out the 600s interval


def test_stop_before_initial_delay_never_ticks():
    client = _Client()

    async def run():
        p = RegistryUpdatePoller(client, interval_s=0.01, initial_delay_s=600)
        await p.start()
        await p.stop()

    asyncio.run(run())
    assert client.calls == 0
