"""Provider resolution from the environment.

The active provider is whatever the env says (or auto-detected from which vars
are set), with a fallback to the always-present LocalProvider. Selection is
per-workload + capability-checked, so a provider that can't do (say) realtime
transparently falls back to local for that one workload.

Env:
    GPU_PROVIDER      = auto | local | remote_worker             (default: auto)
    GPU_FALLBACK_LOCAL = true | false                            (default: true)
    GPU_WORKER_URL    = https://...      (presence enables remote_worker in auto)
    GPU_WORKER_TOKEN  = <bearer>
"""
import os
import threading
import time
from typing import List, Optional

from .base import ComputeError, ComputeProvider, Workload

_PROVIDER = os.getenv("GPU_PROVIDER", "auto").strip().lower() or "auto"
_FALLBACK_LOCAL = os.getenv("GPU_FALLBACK_LOCAL", "true").strip().lower() in ("1", "true", "yes", "on")
_WORKER_URL = os.getenv("GPU_WORKER_URL", "").strip()

# Availability probes can be network round-trips (worker /health), so
# cache the result briefly — get_provider/status can be called per workload-start
# and per UI poll, and we don't want to ping on every one.
_AVAIL_TTL = float(os.getenv("GPU_AVAIL_TTL", "30"))
_avail_cache: dict = {}
_probing: set = set()
_lock = threading.Lock()
_instances: dict = {}


# --- lazy provider singletons (heavy imports happen only when first needed) ---

def _local() -> ComputeProvider:
    with _lock:
        if "local" not in _instances:
            from .providers.local import LocalProvider
            _instances["local"] = LocalProvider()
        return _instances["local"]


def _remote() -> Optional[ComputeProvider]:
    with _lock:
        if "remote_worker" not in _instances:
            try:
                from .providers.remote_worker import RemoteWorkerProvider
                _instances["remote_worker"] = RemoteWorkerProvider()
            except Exception:
                # Not built yet / not configured → simply unavailable.
                _instances["remote_worker"] = None
        return _instances["remote_worker"]


def all_providers() -> list:
    """Every loadable provider instance. Used for lifecycle ops (boot orphan
    recovery, shutdown cleanup, inference cancel) that should run across ALL
    backends — each provider no-ops if the work isn't its own."""
    return [p for p in (_local(), _remote()) if p is not None]


def failover_providers(workload: "Workload", include_cpu_tier: bool = False) -> List[str]:
    """Ordered failover ladder of provider NAMES for `workload`, most-capable
    first: remote_worker -> local, each included only when configured
    (mirrors get_provider's priority). Honors GPU_PROVIDER (a forced non-local
    provider yields just that, plus 'local' when GPU_FALLBACK_LOCAL is on).

    `include_cpu_tier` appends a trailing 'local_cpu' attempt after 'local' — used
    by inference (which can ride out a GPU-tier crash on CPU). Training omits it
    (CPU training of the readout model is impractical; it stops at local GPU).

    The dispatcher walks this list, advancing to the next name only when the
    current tier fails. 'local_cpu' resolves to the local provider with CPU forced.
    """
    chain: List[str] = []
    if _PROVIDER in ("auto", "remote_worker") and _remote() is not None \
            and (_WORKER_URL or _PROVIDER == "remote_worker"):
        chain.append("remote_worker")
    if _PROVIDER == "local" or _FALLBACK_LOCAL or not chain:
        chain.append("local")
        if include_cpu_tier:
            chain.append("local_cpu")
    return chain


def provider_by_name(name: str) -> Optional[ComputeProvider]:
    """Resolve a failover-chain name to a provider instance. 'local' and
    'local_cpu' both map to the local provider — the CPU-ness of 'local_cpu' is
    applied by the caller (utils.device.force_cpu), not by a separate provider."""
    if name in ("local", "local_cpu"):
        return _local()
    if name == "remote_worker":
        return _remote()
    return None


def _cached_available(p: ComputeProvider, allow_probe: bool = True):
    """Cached availability. Returns True/False, or None when the answer isn't
    cached AND `allow_probe` is False (so a cheap status call never blocks on an
    SSH/HTTP probe). `is_available()` can be a multi-second round trip, so the
    probe result is cached for `_AVAIL_TTL`."""
    now = time.monotonic()
    key = p.name
    with _lock:
        ent = _avail_cache.get(key)
        if ent and ent[1] > now:
            return ent[0]
    if not allow_probe:
        return None  # unknown; caller (status) treats optimistically
    try:
        ok = bool(p.is_available())
    except Exception:
        ok = False
    # Stamp expiry from NOW (after the probe) so a slow probe still caches.
    with _lock:
        _avail_cache[key] = (ok, time.monotonic() + _AVAIL_TTL)
    return ok


def _kick_background_probe(p: ComputeProvider) -> None:
    """Refresh a provider's availability OFF the request path, so a cache-only
    status() call self-corrects within seconds without ever blocking. Deduped:
    at most one in-flight probe per provider, and skipped while the cache is
    still fresh."""
    now = time.monotonic()
    with _lock:
        if p.name in _probing:
            return
        ent = _avail_cache.get(p.name)
        if ent and ent[1] > now:
            return
        _probing.add(p.name)

    def _run():
        try:
            _cached_available(p, allow_probe=True)
        finally:
            with _lock:
                _probing.discard(p.name)

    threading.Thread(target=_run, name=f"compute-probe-{p.name}", daemon=True).start()


def _candidates() -> List[ComputeProvider]:
    """Configured providers in priority order (most-specific first); LocalProvider
    is always the final floor."""
    out: List[ComputeProvider] = []
    if _PROVIDER in ("auto", "remote_worker"):
        r = _remote()
        if r is not None and (_WORKER_URL or _PROVIDER == "remote_worker"):
            out.append(r)
    out.append(_local())
    return out


def get_provider(workload: Workload, probe: bool = True) -> ComputeProvider:
    """Resolve the provider to run `workload`. Honors GPU_PROVIDER, checks
    availability + capability, and falls back to LocalProvider when allowed.

    `probe=False` (used by status()) never blocks on a reachability check: an
    uncached provider is assumed available, so the status chip resolves instantly
    and the real probe happens at dispatch time."""
    local = _local()

    # Explicit, non-local choice: try it, else fall back to local if permitted.
    if _PROVIDER == "remote_worker":
        chosen = _remote()
        if chosen is not None and chosen.capabilities().supports(workload):
            if _cached_available(chosen, allow_probe=probe) is not False:  # True or unknown
                return chosen
        if _FALLBACK_LOCAL:
            return local
        raise ComputeError(
            f"GPU_PROVIDER={_PROVIDER} is unavailable or can't run {workload.value}, "
            "and GPU_FALLBACK_LOCAL is off.",
        )

    if _PROVIDER == "local":
        return local

    # auto: first configured + (available or unknown) + capable, else local.
    for p in _candidates():
        if p is local:
            return local
        if not p.capabilities().supports(workload):
            continue
        if _cached_available(p, allow_probe=probe) is not False:
            return p
    return local


def status() -> dict:
    """Snapshot for the UI 'which GPU am I on' indicator. Reports the provider
    that WOULD run each workload right now."""
    # Refresh configured non-local providers in the background so a cache-only
    # status reflects reality on the next poll, without blocking this one.
    for p in _candidates():
        if not getattr(p.capabilities(), "is_local", True):
            _kick_background_probe(p)

    per_workload = {}
    for w in Workload:
        try:
            p = get_provider(w, probe=False)  # never block the status chip on SSH/HTTP
            cap = p.capabilities()
            per_workload[w.value] = {
                "provider": p.name,
                "accelerator": cap.accelerator.value,
                "detail": cap.detail,
            }
        except Exception as e:
            per_workload[w.value] = {"provider": None, "error": str(e)}

    # A single headline = whatever runs inference (the most common heavy path).
    head = per_workload.get(Workload.INFERENCE.value, {})
    return {
        "configured_provider": _PROVIDER,
        "fallback_local": _FALLBACK_LOCAL,
        "headline": head,
        "workloads": per_workload,
    }


def local_provider() -> ComputeProvider:
    """The always-present LocalProvider — used by dispatch sites for an explicit
    fall-back after a remote ComputeError(retryable_local=True)."""
    return _local()


def invalidate_cache() -> None:
    """Drop the availability cache (e.g. after the user edits env / a worker comes
    back online). Cheap; next call re-probes."""
    with _lock:
        _avail_cache.clear()
