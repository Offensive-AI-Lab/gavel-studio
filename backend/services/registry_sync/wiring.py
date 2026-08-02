"""Factory + wiring for the registry update poller (local backend).

`build_poller()` assembles the periodic freshness probe and returns it (or
None when disabled). Each tick PROBES the registry — a cheap manifest-hash
compare, anonymous, no records pulled — and tells the frontend whether this
backend is behind, so the sidebar surfaces a "click to sync" badge.

It deliberately does NOT apply the update: pulling records mid-session would
silently change the user's library underneath them. The user applies updates
on their own click (sidebar indicator / manual sync).

The backend polls the library repo itself (REGISTRY_POLL_S, default 300s).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .poller import RegistryUpdatePoller

logger = logging.getLogger(__name__)


class _LibraryUpdateNotifier:
    """Adapts the poller's tick to a NON-mutating freshness probe:
    `check_for_updates()` (a cheap manifest-hash compare, anonymous, no records
    pulled), then `library_events.set_available()` to push the badge state to the
    frontend. Safe to call on every tick."""

    def reconcile(self):
        from services.library_sync import check_for_updates
        from services import library_events
        try:
            status = check_for_updates()
            if status.get("checked"):
                library_events.set_available(bool(status.get("available")))
            return status
        except Exception as e:
            logger.warning("[registry] update check failed: %s", e)
            return None


def build_poller() -> Optional[RegistryUpdatePoller]:
    """Build the registry update poller from env, or None if disabled."""
    if os.getenv("ENABLE_REGISTRY_POLLER", "1") == "0":
        logger.info("[registry] poller disabled (ENABLE_REGISTRY_POLLER=0)")
        return None
    try:
        interval = float(os.getenv("REGISTRY_POLL_S", "300"))
    except ValueError:
        interval = 300.0
    return RegistryUpdatePoller(_LibraryUpdateNotifier(), interval_s=interval)
