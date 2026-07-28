"""Direct HuggingFace WRITE client — commits publish batches to the registry.

This replaces the central-server proxy (the old `services/central_server.py`
HTTP client plus the separate `central-server/` deployment). The central server
existed so that MANY users' studios could publish to one shared registry
without any of them holding the write token. In the local-only, single-operator
deployment there is exactly one publisher, so the write token lives in
`backend/.env` (HF_TOKEN) and the backend commits to HF itself.

What was kept from the proxy, verbatim in behavior:
  * the response contract `{"status": "success"|"race"|"error", ...}` that
    `hf_publish._push_atomic` consumes,
  * race detection via `parent_commit` (compare-and-swap on the repo HEAD),
  * whole-commit retry on transient network/5xx failures — each attempt is
    still all-or-nothing; a stale `parent_commit` on a retry surfaces as a
    race, which prevents a duplicate commit if a prior attempt secretly
    succeeded before the connection dropped,
  * the referential-integrity guard: a manifest must never be committed while
    it references rule/CE/rule-set record files that are neither in the batch
    nor already in the repo — a broken index would poison every later sync.

What was deliberately dropped: manifest version stamping (`global_signature` /
`namespaces`). Those signatures existed so OTHER studios could cheaply detect
changes; the local backend compares the raw manifest hash instead (see
`hf_sync.check_for_updates`). Because nothing rewrites the manifest anymore,
the bytes committed are exactly the bytes the publisher built — and
`manifest_sha256` in the response is simply their hash.

Reads never come through here. Pulling the public library from HuggingFace is
anonymous (see `services/hf_sync.py`), so Browse / Sync / Fork all work with no
token configured at all; only publishing needs HF_TOKEN.
"""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger("hf-write")

# Retry the WHOLE atomic commit on transient failures (network blips, HF 5xx),
# so a flaky connection lands the full commit instead of failing to nothing.
_COMMIT_ATTEMPTS = max(1, int(os.getenv("HF_COMMIT_ATTEMPTS", "3")))
_COMMIT_BACKOFF = float(os.getenv("HF_COMMIT_BACKOFF", "0.6"))


class HFWriteError(Exception):
    """Raised when a write cannot be attempted (no token / bad input /
    integrity refusal). Mirrors the old CentralServerError shape so callers
    can surface `status_code` in HTTP responses."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def _resolve_token() -> Optional[str]:
    """HF WRITE token from backend/.env. Empty string is coerced to None — a
    blank token would produce an illegal ``Bearer `` header (see
    hf_sync._resolve_token for the original rationale)."""
    return (os.environ.get("HF_TOKEN") or "").strip() or None


def is_enabled() -> bool:
    """True when publishing is possible (a write token is configured). When
    False, publishing is unavailable — but every local feature, and library
    READ-sync, still work."""
    return bool(_resolve_token())


def _repo_ref():
    """(repo_id, repo_type) shared with the read side — one source of truth."""
    from services.hf_sync import REPO_ID, REPO_TYPE
    return REPO_ID, REPO_TYPE


def _get_api():
    from huggingface_hub import HfApi

    token = _resolve_token()
    if not token:
        raise HFWriteError(
            "HF_TOKEN is not configured — publishing is unavailable. "
            "Everything else (including library sync) works without it.",
            status_code=503,
        )
    return HfApi(token=token)


def _is_race_error(exc: Exception) -> bool:
    """HF returns 412 (Precondition Failed) when parent_commit is stale.
    String match because the SDK has no typed exception for this case."""
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
    repo, auth, 4xx other than race) is not retried — retrying just wastes
    time."""
    m = str(exc).lower()
    return any(s in m for s in (
        "timeout", "timed out", "connection", "temporarily", "max retries",
        "500", "502", "503", "504", "remotedisconnected", "reset by peer",
    ))


def _manifest_required_record_paths(manifest: dict) -> set:
    """The PRIMARY record files the manifest references — one per rule, CE and
    rule set. These are the files whose absence would leave a client pulling a
    manifest that points at a record that doesn't exist, breaking its local
    DB. (Datasets/calibration are not enforced — a missing test set doesn't
    make a rule/CE itself dangling. A rule set's member RULES are covered via
    the public_rules entries — they must be published first.)"""
    req = set()
    for pid in (manifest.get("rules") or {}):
        req.add(f"public_rules/{pid}.json")
    for pid in (manifest.get("ces") or {}):
        req.add(f"public_ces/{pid}.json")
    for pid in (manifest.get("rule_sets") or {}):
        req.add(f"public_rule_sets/{pid}.json")
    return req


def hf_head_sha() -> Optional[str]:
    """Current HEAD commit SHA of the registry repo, used as `parent_commit`
    for race detection on publish."""
    repo_id, repo_type = _repo_ref()
    api = _get_api()
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
        return info.sha
    except Exception as e:
        raise HFWriteError(f"HF read failed: {e}", status_code=502)


def hf_commit(*, operations: List[Dict[str, bytes]], commit_message: str,
              parent_commit: Optional[str] = None) -> dict:
    """Commit a batch of files to the HF registry — ALL or NOTHING.

    `operations` is a list of {"path": str, "content": bytes}. Every file goes
    into ONE HfApi.create_commit, which HF applies as a single atomic commit;
    if it fails for any reason, nothing lands in the repo — there is no
    partial state.

    Returns {"status": "success"|"race"|"error", "commit_sha": ...,
    "manifest_sha256": ..., "error": ...} — the same contract the central
    server's /hf/commit used to return, so `hf_publish` is unchanged.
    """
    from huggingface_hub import CommitOperationAdd

    repo_id, repo_type = _repo_ref()
    api = _get_api()

    files = {op["path"]: op["content"] for op in operations}

    # Referential-integrity guard. If this batch updates the manifest, every
    # record it references must be present — either uploaded in THIS batch or
    # already in the registry. Otherwise we'd publish a manifest that points at
    # missing records, and clients pulling it would corrupt their local DB. If
    # the set is incomplete, we refuse and commit NOTHING. (The only publisher
    # is now our own hf_publish, which builds records and manifest together —
    # this guard is kept as a cheap self-check against future publish bugs.)
    manifest_sha256: Optional[str] = None
    if "manifest.json" in files:
        try:
            manifest = json.loads(files["manifest.json"])
        except Exception:
            raise HFWriteError("manifest.json is not valid JSON", status_code=400)
        unmet = _manifest_required_record_paths(manifest) - set(files.keys())
        if unmet:
            try:
                existing = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
            except Exception as e:
                # Can't confirm what's already on HF → refuse rather than risk
                # a broken publish. Nothing is committed.
                return {"status": "error",
                        "error": f"Could not verify registry state before commit: {e}"}
            still_missing = sorted(unmet - existing)
            if still_missing:
                raise HFWriteError(
                    "Refusing to publish an incomplete set: the manifest references "
                    f"{len(still_missing)} record file(s) that are neither in this upload "
                    f"nor already in the registry (e.g. '{still_missing[0]}'). "
                    "Nothing was uploaded.",
                    status_code=409,
                )
        # No version stamping anymore — the committed bytes are exactly what
        # the publisher built, so this hash is what the next freshness probe
        # will see on HF. hf_publish caches it as last_manifest_hash so its own
        # publish never flashes a phantom "update available".
        manifest_sha256 = hashlib.sha256(files["manifest.json"]).hexdigest()

    ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=c) for p, c in files.items()]

    last_err: Optional[Exception] = None
    for attempt in range(_COMMIT_ATTEMPTS):
        try:
            info = api.create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                operations=ops,
                commit_message=commit_message,
                parent_commit=parent_commit,
            )
            return {
                "status": "success",
                "commit_sha": getattr(info, "oid", None) or getattr(info, "commit_oid", None),
                "manifest_sha256": manifest_sha256,
            }
        except Exception as e:
            if _is_race_error(e):
                return {"status": "race", "error": str(e)}
            last_err = e
            if _is_transient(e) and attempt < _COMMIT_ATTEMPTS - 1:
                logger.warning("[hf-write] transient commit failure (attempt %d/%d): %s",
                               attempt + 1, _COMMIT_ATTEMPTS, e)
                time.sleep(_COMMIT_BACKOFF * (2 ** attempt))
                continue
            break

    return {"status": "error", "error": str(last_err)}
