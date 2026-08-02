"""In-process event bus: the backend -> frontend live push.

The registry poller PROBES whether this backend is behind (without touching the
DB) and calls `set_available()` HERE. The `/library/events` SSE stream fans the resulting
`update_available` / `synced` event out to every connected browser tab, so the
sidebar surfaces a "click to sync" badge the instant an update lands — no
frontend polling, and no silent mid-session DB mutation (the user applies the
update on their click).

Tiny by design: a set of asyncio.Queues (one per connected SSE client) plus a
remembered event loop so `publish()` can be called from the subscriber's worker
THREAD (the probe runs off the loop via asyncio.to_thread) and still hand the
event to the loop safely. A one-field `_state` records the latest availability
so the SSE greet can replay it to a tab that connects after the push.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Each connected SSE client owns one bounded queue. Bounded so a stuck client
# can't grow memory without limit — when full we drop the oldest (latest state
# wins; these events are "something changed, re-fetch" signals, not a log).
_QUEUE_MAXSIZE = 64

_subscribers: Set["asyncio.Queue[dict]"] = set()
_loop: Optional[asyncio.AbstractEventLoop] = None


def _remember_loop() -> None:
    """Capture the running event loop the first time a client connects. publish()
    (called from a worker thread) needs it to schedule the put thread-safely."""
    global _loop
    try:
        _loop = asyncio.get_running_loop()
    except RuntimeError:
        pass  # no running loop (shouldn't happen from a request handler)


def subscribe() -> "asyncio.Queue[dict]":
    """Register a new SSE client. Call from the request handler (on the loop)."""
    _remember_loop()
    q: "asyncio.Queue[dict]" = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    _subscribers.add(q)
    return q


def unsubscribe(q: "asyncio.Queue[dict]") -> None:
    _subscribers.discard(q)


def subscriber_count() -> int:
    return len(_subscribers)


def publish(event: dict) -> None:
    """Fan `event` out to every connected client. Thread-safe: callable from the
    registry subscriber's worker thread. A no-op when nobody is listening."""
    if not _subscribers or _loop is None:
        return
    for q in list(_subscribers):
        _loop.call_soon_threadsafe(_offer, q, event)


def _offer(q: "asyncio.Queue[dict]", event: dict) -> None:
    """Enqueue on the loop thread; on a full (slow) client, drop the oldest."""
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
            q.put_nowait(event)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Availability state — "is this backend behind the registry?"
# --------------------------------------------------------------------------- #
# Last-known value, so the SSE greet can replay it to a tab that connected after
# the push fired. Updated by set_available(); never mutates the DB.
_state = {"available": False}


def current_state() -> dict:
    return dict(_state)


def set_available(available: bool) -> None:
    """Record whether the local library is behind the registry and notify the
    frontend (`update_available` / `synced`). Does NOT pull anything — applying
    the update is the user's click."""
    available = bool(available)
    _state["available"] = available
    publish({"event": "update_available" if available else "synced"})
