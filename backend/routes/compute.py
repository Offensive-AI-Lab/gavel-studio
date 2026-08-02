"""Compute-backend status — powers the UI 'which GPU am I on' indicator.

Intentionally UNAUTHENTICATED (like /health): it returns only non-sensitive infra
info (which provider would run each workload + the accelerator), never secrets or
the worker URL/token. Keeping it auth-free means the indicator always renders,
even before login or when the central auth server is unreachable.
"""
from fastapi import APIRouter

from services import compute

router = APIRouter()


@router.get("/status")
def compute_status():
    """Active provider per workload + accelerator, for the status chip."""
    try:
        return compute.status()
    except Exception as e:
        # Never let the indicator take down the page.
        return {"configured_provider": None, "error": str(e), "workloads": {}}


# Friendly labels for the machine picker.
_TARGET_LABELS = {
    "local": "This machine",
    "remote_worker": "Remote GPU",
}


@router.get("/targets")
def compute_targets(workload: str = "training"):
    """Selectable compute targets for a workload — the configured failover chain
    (e.g. remote_worker / local). Powers the 'choose a machine' picker at
    train time: only when more than one is configured does the UI prompt."""
    from services.compute.base import Workload
    try:
        w = Workload(workload)
    except ValueError:
        w = Workload.TRAINING

    out, seen = [], set()
    for name in compute.failover_providers(w):
        if name in seen or name == "local_cpu":
            continue
        seen.add(name)
        accel = None
        try:
            p = compute.provider_by_name(name)
            accel = p.capabilities().accelerator.value if p else None
        except Exception:
            accel = None
        out.append({"name": name, "label": _TARGET_LABELS.get(name, name), "accelerator": accel})
    return {"workload": w.value, "targets": out}
