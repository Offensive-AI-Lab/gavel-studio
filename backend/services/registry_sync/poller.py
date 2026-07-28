"""Registry update poller — periodic freshness probe against HuggingFace.

Replaces the old WebSocket subscriber. That subscriber listened for the
central server's `version_update` push; with the central server gone, the
backend checks HF itself on a timer. Each tick runs the same reconcile action
the subscriber used (see wiring._LibraryUpdateNotifier): a cheap manifest-hash
compare against `sync_state.last_manifest_hash` — no records are pulled, so a
tick costs one small HF read. The result drives `library_events.set_available`
and from there the sidebar's "Updates available" badge over SSE.

It deliberately does NOT apply the update: pulling records mid-session would
silently change the user's library underneath them. The user applies updates
on their own click (sidebar indicator / manual sync).

The first probe is delayed a little so the boot-time library-sync thread
(main.py `library-sync-bootstrap`) lands first — otherwise the badge could
flash "updates available" for content the boot sync is about to pull anyway.

`client.reconcile()` does blocking HF I/O, so it's dispatched to a thread.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RegistryUpdatePoller:
    def __init__(self, client, *, interval_s: float = 300.0,
                 initial_delay_s: float = 20.0):
        self.client = client
        self.interval_s = max(0.01, float(interval_s))
        self.initial_delay_s = max(0.0, float(initial_delay_s))
        self._stop = asyncio.Event()
        self._tasks: list = []

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._stop.clear()
        self._tasks = [asyncio.create_task(self._run_loop(), name="registry-poll")]

    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []

    async def wait(self) -> None:
        """Await the loop tasks (used by tests / a blocking host)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------------ #
    # the poll loop
    # ------------------------------------------------------------------ #
    async def _run_loop(self) -> None:
        if await self._sleep_or_stop(self.initial_delay_s):
            return
        while not self._stop.is_set():
            await self._reconcile()
            if await self._sleep_or_stop(self.interval_s):
                break

    async def _sleep_or_stop(self, delay: float) -> bool:
        """Sleep up to `delay` seconds; return True if stop() fired meanwhile."""
        if delay <= 0:
            return self._stop.is_set()
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
            return True
        except asyncio.TimeoutError:
            return False

    async def _reconcile(self) -> Optional[dict]:
        # reconcile() is blocking (HF I/O + DB) → run off the event loop. A
        # failed probe must never kill the loop — the next tick retries.
        try:
            return await asyncio.to_thread(self.client.reconcile)
        except Exception as e:
            logger.error("[registry] reconcile crashed: %s", e)
            return None
