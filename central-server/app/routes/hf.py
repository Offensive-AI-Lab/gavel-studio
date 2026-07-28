"""HuggingFace write proxy.

The HF token lives only here. Local backends send a list of file
operations + a commit message; this server commits them to the public
library repo on HuggingFace.

Read operations (manifest fetch, registry sync) do NOT need this proxy
— they use the public anonymous endpoints. Only writes route through
here, which is the security boundary we care about.
"""
import base64
import hashlib
import json
import os
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..utils.rate_limit import rate_limit

router = APIRouter(prefix="/hf", tags=["huggingface"])

# A publish does blocking HF network I/O on a threadpool thread, so cap how fast
# any one client can fire them to keep the threadpool free for auth/ratings.
_rl_commit = rate_limit("hf-commit", limit=20, window_seconds=60)

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("HF_REPO_ID", "GavelPublicData/public-library")
REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")

# Retry the WHOLE atomic commit on transient failures (network blips, HF 5xx),
# so a flaky connection lands the full commit instead of failing to nothing.
_COMMIT_ATTEMPTS = max(1, int(os.getenv("HF_COMMIT_ATTEMPTS", "3")))
_COMMIT_BACKOFF = float(os.getenv("HF_COMMIT_BACKOFF", "0.6"))


class FileOp(BaseModel):
    """One file to upload. `content_b64` is base64-encoded bytes."""
    path: str = Field(..., min_length=1, max_length=512)
    content_b64: str


class CommitRequest(BaseModel):
    operations: List[FileOp] = Field(..., min_length=1, max_length=100)
    commit_message: str = Field(..., min_length=1, max_length=512)
    parent_commit: Optional[str] = None  # SHA for race detection


class CommitResponse(BaseModel):
    status: str  # "success" or "race" or "error"
    commit_sha: Optional[str] = None
    # sha256 of the manifest.json bytes ACTUALLY committed (after we version-stamp
    # it). The publisher caches this so its own next reconcile short-circuits and
    # it never flags itself "behind" over the stamp it didn't compute locally.
    manifest_sha256: Optional[str] = None
    error: Optional[str] = None


def _is_race_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "412" in msg
        or "precondition" in msg.lower()
        or "stale" in msg.lower()
        or "fetch first" in msg.lower()
        or "out-of-date" in msg.lower()
    )


def _is_transient(exc: Exception) -> bool:
    """Worth retrying: network/timeout/HF-5xx hiccups. A permanent error (bad
    repo, auth, 4xx other than race) is not retried — retrying just wastes time."""
    m = str(exc).lower()
    return any(s in m for s in (
        "timeout", "timed out", "connection", "temporarily", "max retries",
        "500", "502", "503", "504", "remotedisconnected", "reset by peer",
    ))


def _manifest_required_record_paths(manifest: dict) -> set:
    """The PRIMARY record files the manifest references — one per rule and CE.
    These are the files whose absence would leave a client pulling a manifest
    that points at a rule/CE that doesn't exist, breaking its local DB. (We
    don't enforce datasets/calibration here — a missing test set doesn't make a
    rule/CE itself dangling.)"""
    req = set()
    for pid in (manifest.get("rules") or {}):
        req.add(f"public_rules/{pid}.json")
    for pid in (manifest.get("ces") or {}):
        req.add(f"public_ces/{pid}.json")
    # A rule set's own record is a primary file; its member RULES are checked
    # via the public_rules entries above (they must be published first), so we
    # don't re-derive them here.
    for pid in (manifest.get("rule_sets") or {}):
        req.add(f"public_rule_sets/{pid}.json")
    return req


@router.post("/commit", response_model=CommitResponse)
def hf_commit(req: CommitRequest, _rl=Depends(_rl_commit)):
    """Commit a batch of files to the HF registry — ALL or NOTHING.

    Unauthenticated: there are no accounts. Note this was ALWAYS the case
    from HuggingFace's point of view — the commit is attributed to the account
    that owns HF_TOKEN, never to a caller. Authorship travels inside the
    payload's `created_by_username` field, which this server does not verify.
    The rate limit below is the only gate.

    Atomicity: every file goes into ONE HfApi.create_commit, which HF applies
    as a single atomic commit. We also decode + validate EVERY file before
    touching HF, so a bad file aborts before anything is uploaded. If the commit
    fails for any reason (crash, network, HF bug), nothing lands in the repo —
    there is no partial state. Transient failures retry the whole atomic commit
    (each attempt is still all-or-nothing); a stale `parent_commit` on a retry
    surfaces as a race, which prevents a duplicate commit if a prior attempt
    secretly succeeded before the connection dropped.
    """
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="Central server has no HF_TOKEN configured")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=HF_TOKEN)

    # Decode + validate ALL files first. If any is bad, we abort here, before a
    # single byte is sent to HF — so a partial set can never be committed.
    decoded: dict = {}
    for f in req.operations:
        try:
            decoded[f.path] = base64.b64decode(f.content_b64)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid base64 in '{f.path}'")

    # Referential-integrity guard. If this batch updates the manifest, every
    # rule/CE it references must be present — either uploaded in THIS batch or
    # already in the registry. Otherwise we'd publish a manifest that points at
    # missing records, and clients pulling it would corrupt their local DB. If
    # the set is incomplete, we refuse and commit NOTHING.
    manifest_sha256: Optional[str] = None
    if "manifest.json" in decoded:
        try:
            manifest = json.loads(decoded["manifest.json"])
        except Exception:
            raise HTTPException(status_code=400, detail="manifest.json is not valid JSON")
        unmet = _manifest_required_record_paths(manifest) - set(decoded.keys())
        if unmet:
            try:
                existing = set(api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE))
            except Exception as e:
                # Can't confirm what's already on HF → refuse rather than risk a
                # broken publish. Nothing is committed.
                return CommitResponse(status="error", error=f"Could not verify registry state before commit: {e}")
            still_missing = sorted(unmet - existing)
            if still_missing:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Refusing to publish an incomplete set: the manifest references "
                        f"{len(still_missing)} record file(s) that are neither in this upload "
                        f"nor already in the registry (e.g. '{still_missing[0]}'). Nothing was uploaded."
                    ),
                )

        # Stamp the content version map so the control-plane watcher detects this
        # publish and broadcasts a version_update (single write chokepoint = here).
        from ..services.manifest_versions import augment_manifest
        decoded["manifest.json"] = json.dumps(
            augment_manifest(manifest), ensure_ascii=False).encode("utf-8")
        # Authoritative hash of exactly what lands on HF — handed back so the
        # publisher caches it as last_manifest_hash (see CommitResponse).
        manifest_sha256 = hashlib.sha256(decoded["manifest.json"]).hexdigest()

    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=c) for p, c in decoded.items()]

    last_err: Optional[Exception] = None
    for attempt in range(_COMMIT_ATTEMPTS):
        try:
            info = api.create_commit(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                operations=ops,
                commit_message=req.commit_message,
                parent_commit=req.parent_commit,
            )
            # Ring the watcher's doorbell so it reconciles and broadcasts
            # `version_update` to every connected backend NOW, instead of waiting
            # up to the safety-poll window. Only matters when the manifest changed
            # (a real publish). The publisher dedups itself — it cached
            # manifest_sha256, so its own freshness probe comes back "synced" — so
            # only OTHER users' sidebars light up. Best-effort: the safety poll
            # still converges everyone if this trigger is a no-op (e.g. control
            # plane disabled in tests).
            if manifest_sha256 is not None:
                try:
                    from ..services import control_plane
                    control_plane.WATCHER.trigger()
                except Exception as trig_err:
                    import logging
                    logging.getLogger(__name__).warning(
                        "[hf] watcher trigger after commit failed: %s", trig_err)
            return CommitResponse(
                status="success",
                commit_sha=getattr(info, "oid", None) or getattr(info, "commit_oid", None),
                manifest_sha256=manifest_sha256,
            )
        except Exception as e:
            if _is_race_error(e):
                return CommitResponse(status="race", error=str(e))
            last_err = e
            if _is_transient(e) and attempt < _COMMIT_ATTEMPTS - 1:
                time.sleep(_COMMIT_BACKOFF * (2 ** attempt))
                continue
            break

    return CommitResponse(status="error", error=str(last_err))


@router.get("/head-sha")
def get_head_sha():
    """Returns the current HEAD commit SHA of the HF repo. Used by the
    local backend to set `parent_commit` for race detection, and by the
    startup library-sync thread to check whether the cached manifest is
    stale.

    Non-sensitive: anyone with curl can read the same value from the public
    HuggingFace REST API without a token.
    """
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="Central server has no HF_TOKEN configured")
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    try:
        info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
        return {"sha": info.sha}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HF read failed: {e}")
