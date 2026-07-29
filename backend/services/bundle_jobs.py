"""Server-side background jobs for guardrail bundle export / import.

Export and import can take a while (export may publish draft rules to HF first;
import syncs the whole library, then rebuilds the guardrail). Running them as
detached background tasks means the work survives the user closing the modal —
only a backend crash ends it. The `bundle_jobs` table is the durable record the
frontend polls, and the breadcrumb crash recovery uses to clean up a partially
imported guardrail.

Job lifecycle: a row is created with status='running'; the runner updates
`phase` as it goes and finishes by setting status='done' (with a `result`, and
for export an `artifact_path` to the finished zip) or status='error' (+`error`).
The runners NEVER raise — every path ends in a terminal status — so a job can't
get wedged in 'running' except by an actual process death (handled on next boot
by `recover_interrupted_jobs`).
"""
import json
import logging
import os
import shutil
import uuid

from utils.sqlite_db import execute_query, execute_query_dict

logger = logging.getLogger(__name__)

_BASE = os.path.dirname(os.path.dirname(__file__))  # backend/
TRAINED_MODELS_DIR = os.path.join(_BASE, "trained_classifiers")
_ARTIFACT_DIR = os.path.join(_BASE, "bundle_exports")   # finished export zips
_IMPORT_DIR = os.path.join(_BASE, "bundle_imports")     # staged upload payloads

# A finished export artifact older than this is pruned on boot (the user has had
# ample time to download it; it's re-creatable anyway).
_ARTIFACT_TTL_SECONDS = 24 * 3600


# ---------------------------------------------------------------------------
# Job record CRUD
# ---------------------------------------------------------------------------

def create_job(user_id: int, job_type: str, *, classifier_id=None, tier=None, phase=None) -> int:
    rows = execute_query_dict(
        "INSERT INTO bundle_jobs (user_id, job_type, classifier_id, tier, phase) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING job_id",
        (user_id, job_type, classifier_id, tier, phase),
    )
    return rows[0]["job_id"]


def _set_phase(job_id: int, phase: str) -> None:
    execute_query("UPDATE bundle_jobs SET phase = %s, updated_at = now() WHERE job_id = %s",
                  (phase, job_id))


def _set_classifier(job_id: int, classifier_id: int) -> None:
    execute_query("UPDATE bundle_jobs SET classifier_id = %s, updated_at = now() WHERE job_id = %s",
                  (classifier_id, job_id))


def _set_done(job_id: int, *, result=None, artifact_path=None, filename=None, classifier_id=None) -> None:
    execute_query(
        "UPDATE bundle_jobs SET status = 'done', phase = NULL, "
        "result = %s::jsonb, "
        "artifact_path = COALESCE(%s, artifact_path), "
        "filename = COALESCE(%s, filename), "
        "classifier_id = COALESCE(%s, classifier_id), "
        "updated_at = now() WHERE job_id = %s",
        (json.dumps(result) if result is not None else None,
         artifact_path, filename, classifier_id, job_id),
    )


def _set_error(job_id: int, error: str) -> None:
    execute_query(
        "UPDATE bundle_jobs SET status = 'error', error = %s, phase = NULL, updated_at = now() "
        "WHERE job_id = %s",
        (str(error)[:2000], job_id),
    )


def get_job(job_id: int, user_id: int):
    """Client-facing job view (no on-disk path leaked)."""
    rows = execute_query_dict(
        "SELECT job_id, job_type, status, phase, error, classifier_id, tier, filename, "
        "result, created_at, updated_at "
        "FROM bundle_jobs WHERE job_id = %s AND user_id = %s",
        (job_id, user_id),
    )
    return rows[0] if rows else None


def get_artifact(job_id: int, user_id: int):
    rows = execute_query_dict(
        "SELECT artifact_path, filename, status, job_type FROM bundle_jobs "
        "WHERE job_id = %s AND user_id = %s",
        (job_id, user_id),
    )
    return rows[0] if rows else None


def latest_export_job(user_id: int, classifier_id: int):
    """The most recent running-or-ready export job for a guardrail, so the UI
    can resume showing progress / offer a download after a modal reopen. A 'done'
    job whose artifact has since been pruned is treated as gone."""
    rows = execute_query_dict(
        "SELECT job_id, status, phase, error, filename, artifact_path, created_at "
        "FROM bundle_jobs "
        "WHERE user_id = %s AND classifier_id = %s AND job_type = 'export' "
        "AND status IN ('running', 'done') "
        "ORDER BY created_at DESC LIMIT 1",
        (user_id, classifier_id),
    )
    if not rows:
        return None
    job = rows[0]
    if job["status"] == "done" and not (job.get("artifact_path") and os.path.isfile(job["artifact_path"])):
        return None
    job.pop("artifact_path", None)  # don't leak the path
    return job


# ---------------------------------------------------------------------------
# Upload / artifact files
# ---------------------------------------------------------------------------

def save_upload(data: bytes) -> str:
    os.makedirs(_IMPORT_DIR, exist_ok=True)
    path = os.path.join(_IMPORT_DIR, f"{uuid.uuid4().hex}.zip")
    with open(path, "wb") as f:
        f.write(data)
    return path


def _artifact_path(job_id: int, filename: str) -> str:
    os.makedirs(_ARTIFACT_DIR, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (filename or "bundle.zip"))
    return os.path.join(_ARTIFACT_DIR, f"{job_id}_{safe}")


def cleanup_prior_export_artifacts(user_id: int, classifier_id: int, keep_job_id: int) -> None:
    """When a new export for a guardrail starts, drop the old finished zips for
    that same guardrail so they don't pile up on disk."""
    rows = execute_query_dict(
        "SELECT job_id, artifact_path FROM bundle_jobs "
        "WHERE user_id = %s AND classifier_id = %s AND job_type = 'export' "
        "AND job_id <> %s AND artifact_path IS NOT NULL",
        (user_id, classifier_id, keep_job_id),
    ) or []
    for r in rows:
        _safe_unlink(r.get("artifact_path"))
        execute_query("UPDATE bundle_jobs SET artifact_path = NULL WHERE job_id = %s", (r["job_id"],))


def _safe_unlink(path) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Runners (executed on a background-task thread; must never raise)
# ---------------------------------------------------------------------------

def run_export_job(job_id: int, user_id: int, classifier_id: int, tier: str) -> None:
    from services import classifier_bundle as cb
    try:
        a = cb.assess_export(classifier_id)
        if a["blockers"]:
            return _set_error(job_id, a["blockers"][0])
        if a["drift"]:
            return _set_error(job_id, a["reason"] or "The policy changed since training; retrain first.")

        # Publish any draft rules in the policy (the user approved this by
        # starting the export). publish_rule cascades each rule's draft CEs.
        if not a["can_export"] and a["unpublished"]:
            _set_phase(job_id, "Publishing rules to the library…")
            from services.hf_publish import publish_rule
            for item in a["unpublished"]:
                res = publish_rule(item["rule_id"], publisher_user_id=user_id)
                ok = getattr(res, "status", None) is not None and res.status.name == "SUCCESS"
                if not ok:
                    return _set_error(
                        job_id,
                        f"Couldn't publish rule “{item['name']}”: {getattr(res, 'error', 'publish failed')}",
                    )
            a = cb.assess_export(classifier_id)

        if not a["can_export"]:
            return _set_error(job_id, a["reason"] or "This rule set can't be exported.")
        if tier not in a["tiers_available"]:
            return _set_error(job_id, f"Tier '{tier}' isn't available for this rule set.")

        _set_phase(job_id, "Building the bundle…")
        data, filename = cb.build_bundle_zip(classifier_id, tier)
        path = _artifact_path(job_id, filename)
        with open(path, "wb") as f:
            f.write(data)
        _set_done(job_id, result={"filename": filename, "tier": tier}, artifact_path=path, filename=filename)
    except cb.BundleError as e:
        _set_error(job_id, e.message)
    except Exception as e:
        logger.exception("[bundle-jobs] export job %s failed", job_id)
        _set_error(job_id, f"Export failed: {e}")


def run_import_job(job_id: int, user_id: int, upload_path: str) -> None:
    from services import classifier_bundle as cb
    try:
        with open(upload_path, "rb") as f:
            data = f.read()
        result = cb.import_bundle(
            data, user_id,
            on_phase=lambda t: _set_phase(job_id, t),
            on_classifier_created=lambda cid: _set_classifier(job_id, cid),
        )
        _set_done(job_id, result=result, classifier_id=result.get("classifier_id"))
    except cb.BundleError as e:
        _set_error(job_id, e.message)
    except Exception as e:
        logger.exception("[bundle-jobs] import job %s failed", job_id)
        _set_error(job_id, f"Import failed: {e}")
    finally:
        _safe_unlink(upload_path)


# ---------------------------------------------------------------------------
# Crash recovery (called on boot — DB + filesystem only, no heavy imports)
# ---------------------------------------------------------------------------

def recover_interrupted_jobs() -> dict:
    """Reconcile jobs left 'running' by a dead process.

    Export: just mark errored (its half-written zip, if any, is pruned).
    Import: if the guardrail it created actually reached 'active', the import
    finished and only the job record was lost → mark the job done. Otherwise the
    guardrail is a half-built partial → delete it (DB row + workdir) and error
    the job. This is the "only a backend crash cancels the import" guarantee.
    """
    summary = {"errored": 0, "rolled_back": 0, "completed": 0}
    rows = execute_query_dict(
        "SELECT job_id, job_type, classifier_id FROM bundle_jobs WHERE status = 'running'"
    ) or []
    for r in rows:
        jid, jtype, cid = r["job_id"], r["job_type"], r.get("classifier_id")
        try:
            if jtype == "import" and cid:
                st = execute_query_dict("SELECT status FROM classifiers WHERE classifier_id = %s", (cid,))
                if st and st[0]["status"] == "active":
                    _set_done(jid, result={"classifier_id": cid, "recovered": True}, classifier_id=cid)
                    summary["completed"] += 1
                    continue
                if _delete_partial_classifier(cid):
                    summary["rolled_back"] += 1
            _set_error(jid, "Interrupted by a server restart.")
            summary["errored"] += 1
        except Exception as e:
            logger.warning("[bundle-jobs] recovery of job %s failed: %s", jid, e)

    _cleanup_stale_artifacts()
    if any(summary.values()):
        logger.info("[bundle-jobs] recovery: %s", summary)
    return summary


def _delete_partial_classifier(classifier_id: int) -> bool:
    """Delete a half-imported guardrail's DB row (cascades to its rule setups)
    and its on-disk workdir. Resolves the workdir from user_id directly so this
    stays import-light (no transformers) for early boot recovery."""
    try:
        rows = execute_query_dict(
            "SELECT m.user_id FROM classifiers c JOIN target_models m ON c.model_id = m.model_id "
            "WHERE c.classifier_id = %s",
            (classifier_id,),
        )
        execute_query("DELETE FROM classifiers WHERE classifier_id = %s", (classifier_id,))
        if rows:
            wd = os.path.join(TRAINED_MODELS_DIR, str(rows[0]["user_id"]), f"classifier_{classifier_id}")
            if os.path.isdir(wd):
                shutil.rmtree(wd, ignore_errors=True)
        return True
    except Exception as e:
        logger.warning("[bundle-jobs] could not roll back partial classifier %s: %s", classifier_id, e)
        return False


def _cleanup_stale_artifacts() -> None:
    """Prune export zips older than the TTL (best-effort)."""
    try:
        if not os.path.isdir(_ARTIFACT_DIR):
            return
        import time
        cutoff = time.time() - _ARTIFACT_TTL_SECONDS
        for name in os.listdir(_ARTIFACT_DIR):
            p = os.path.join(_ARTIFACT_DIR, name)
            try:
                if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                    os.remove(p)
            except Exception:
                pass
    except Exception:
        pass
    # Also clear any orphaned import staging files.
    try:
        if os.path.isdir(_IMPORT_DIR):
            import time
            cutoff = time.time() - _ARTIFACT_TTL_SECONDS
            for name in os.listdir(_IMPORT_DIR):
                p = os.path.join(_IMPORT_DIR, name)
                try:
                    if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                        os.remove(p)
                except Exception:
                    pass
    except Exception:
        pass
