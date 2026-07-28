# backend/routes/classifiers.py
import io
import json
import logging
import os
import time
import zipfile
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from utils.auth import get_current_user
from utils.ownership import (
    assert_owns_classifier, assert_owns_model,
    require_classifier_owner, require_model_owner,
)
from sql_scripts.model_scripts import (
    create_classifier,
    get_model_classifiers,
    get_user_classifiers,
    attach_model_to_classifier,
    clone_classifier_policy,
    fork_public_rule_set_to_classifier,
    get_classifier_rules,
    add_rule_to_classifier,
    create_custom_rule_setup,
    create_and_link_global_rule,
    delete_classifier,
    reconcile_classifier_status,
)
from sql_scripts.definition_scripts import create_ce
from utils.PostgreSQL import execute_query_dict

logger = logging.getLogger(__name__)
router = APIRouter()

# Per-guardrail lock for cluster downloads. Status polling fires every 5s
# from the frontend but a download takes ~50s — without this, two or three
# polls would race on the same trained_rnn.pth / classifier_meta.json files,
# clobbering temp renames and sometimes marking the guardrail active with
# only a partial set of files.
import threading
_download_locks: dict = {}
_download_locks_lock = threading.Lock()


def _get_download_lock(classifier_id: int) -> threading.Lock:
    with _download_locks_lock:
        if classifier_id not in _download_locks:
            _download_locks[classifier_id] = threading.Lock()
        return _download_locks[classifier_id]


# ---------------------------------------------------------------------------
# Cluster GPU downgrade race (training)
#
# Training submit must stay fast (it's in the HTTP request), so the race runs in
# a background thread: it watches the primary (powerful) GPU job and, if it stays
# queued past the wait window, submits the SAME training on a weaker GPU and keeps
# whichever STARTS first. The winner is swapped into the guardrail's training_log
# under the per-guardrail download lock — the SAME lock the status poller takes —
# so the poller never observes the deliberately-cancelled loser as a failed job.
# A `racing` flag in training_log tells the poller to hold off failing the job
# over a primary that dies mid-race (the race itself is recovering it).
# ---------------------------------------------------------------------------

def _update_training_log(classifier_id: int, mutate) -> None:
    """Read-modify-write the guardrail's training_log JSON under the download lock
    (serialized vs the status poller). `mutate(dict)` edits the parsed log in
    place; a no-op if the guardrail is no longer training."""
    from utils.PostgreSQL import execute_query, execute_query_dict
    lock = _get_download_lock(classifier_id)
    with lock:
        row = execute_query_dict(
            "SELECT training_log, status FROM classifiers WHERE classifier_id = %s",
            (classifier_id,))
        if not row or row[0]["status"] != "training":
            return
        try:
            tl = json.loads(row[0]["training_log"]) if row[0]["training_log"] else {}
        except Exception:
            tl = {}
        if not isinstance(tl, dict):
            return
        mutate(tl)
        execute_query(
            "UPDATE classifiers SET training_log = %s "
            "WHERE classifier_id = %s AND status = 'training'",
            (json.dumps(tl), classifier_id))


def _set_training_racing(classifier_id: int, racing: bool) -> None:
    _update_training_log(classifier_id, lambda tl: tl.__setitem__("racing", bool(racing)))


def _spawn_training_gpu_race(classifier_id: int, user_id: int, job) -> None:
    """If a secondary GPU is configured, race the just-submitted primary training
    job against a weaker GPU in the background. SLURM only."""
    from services.compute.providers.slurm import cluster_direct as cd
    secondary = cd.SLURM_GPU_SECONDARY
    primary_gpu = cd.SLURM_GPU_PRIMARY
    if not secondary or secondary == primary_gpu:
        return
    if not job or not job.raw or not job.raw.get("slurm_job_id"):
        return

    primary = {"slurm_job_id": job.raw["slurm_job_id"],
               "remote_job_dir": job.raw.get("remote_job_dir"),
               "job_id": job.raw.get("job_id"), "gpu": primary_gpu}

    def _resub(gpu):
        mh, labels, tcfg, dsets, calib = _build_training_inputs(classifier_id)
        tcfg = {**(tcfg or {}), "gpu_type": gpu}
        return cd.submit_training_job(
            classifier_id=classifier_id, user_id=user_id, model_hf_path=mh,
            labels=labels, training_config=tcfg, dataset_files=dsets,
            calibration_entries=calib)

    def _switch(winner):
        # Re-point the tracked job to the winner BEFORE run_gpu_race cancels the
        # loser, so the poller starts following the winner immediately.
        def _mut(tl):
            jobd = tl.get("job") if isinstance(tl.get("job"), dict) else {}
            raw = jobd.get("raw") if isinstance(jobd.get("raw"), dict) else {}
            raw.update({"slurm_job_id": winner["slurm_job_id"],
                        "remote_job_dir": winner.get("remote_job_dir"),
                        "job_id": winner.get("job_id"), "mode": "cluster"})
            jobd["id"] = str(winner["slurm_job_id"])
            jobd["raw"] = raw
            tl["job"] = jobd
            tl["last_contact"] = time.time()
        _update_training_log(classifier_id, _mut)

    def _run():
        _set_training_racing(classifier_id, True)
        try:
            cd.run_gpu_race(primary, _resub, on_switch=_switch)
        except Exception as e:
            print(f"[train] Classifier {classifier_id} | GPU race failed: {e}")
        finally:
            _set_training_racing(classifier_id, False)

    threading.Thread(target=_run, daemon=True, name=f"gpu-race-train-{classifier_id}").start()


def _cancel_cluster_job_for_classifier(classifier_id: int) -> None:
    """If the guardrail has an in-flight cluster training job, scancel it
    and remove its remote job directory. Called from delete paths so a
    user removing a guardrail mid-training doesn't leave the GPU running
    and the job dir orphaned on the cluster. Best-effort — every failure
    is swallowed so the DB delete still proceeds."""
    try:
        rows = execute_query_dict(
            "SELECT status, training_log FROM classifiers WHERE classifier_id = %s",
            (classifier_id,),
        )
        if not rows:
            return
        row = rows[0]
        # Only mid-flight cluster jobs need scancel. A successfully
        # downloaded guardrail (status='active') already had cleanup_job
        # called on its remote dir.
        if row["status"] != "training":
            return
        tl = row.get("training_log")
        if isinstance(tl, str):
            try:
                tl = json.loads(tl)
            except (json.JSONDecodeError, TypeError):
                return
        if not isinstance(tl, dict):
            return
        # Reconstruct (provider, job) from new or legacy training_log, then cancel
        # through the compute interface — no transport-specific code here.
        from services.compute.base import TrainingJob
        prov = tl.get("provider")
        if prov in ("slurm", "remote_worker") and isinstance(tl.get("job"), dict):
            job = TrainingJob(provider=prov, classifier_id=classifier_id,
                              id=str(tl["job"].get("id")), raw=tl["job"].get("raw") or {})
        elif tl.get("mode") == "cluster" and tl.get("slurm_job_id"):
            prov = "slurm"
            job = TrainingJob(provider="slurm", classifier_id=classifier_id, id=str(tl["slurm_job_id"]),
                              raw={"slurm_job_id": tl["slurm_job_id"],
                                   "remote_job_dir": tl.get("remote_job_dir"), "mode": "cluster"})
        elif tl.get("mode") == "remote_worker" and tl.get("worker_job_id"):
            prov = "remote_worker"
            job = TrainingJob(provider="remote_worker", classifier_id=classifier_id,
                              id=str(tl["worker_job_id"]), raw={})
        else:
            return
        from services import compute
        p = compute.get_provider(compute.Workload.TRAINING, probe=False)
        if p.name == prov:
            p.cancel_training(job)
    except Exception as e:
        print(f"[train] Cancel-on-delete for classifier {classifier_id} failed: {e}")


def _cancel_cluster_inference_for_classifier(classifier_id: int) -> None:
    """If the guardrail has an in-flight cluster calibration/evaluation job,
    scancel it and clean its remote job dir BEFORE the DB delete. The job's
    {slurm_job_id, remote_job_dir} is stashed on the '*_running' evaluation_results
    row's `plots` column (see routes/evaluation.py _run_inference._stash); the
    cascade delete will wipe that row, so we must read it first. Best-effort —
    every failure is swallowed so the DB delete still proceeds."""
    try:
        rows = execute_query_dict(
            """SELECT plots FROM evaluation_results
               WHERE classifier_id = %s
                 AND eval_type IN ('calibration_running', 'evaluation_running')""",
            (classifier_id,),
        ) or []
        if not rows:
            return
        from services import compute
        providers = compute.all_providers()
        for row in rows:
            plots = row.get("plots")
            if isinstance(plots, str):
                try:
                    plots = json.loads(plots)
                except (json.JSONDecodeError, TypeError):
                    continue
            pointer = (plots or {}).get("cluster") if isinstance(plots, dict) else None
            if not isinstance(pointer, dict):
                continue
            # Cancel through the interface — the provider that owns the job acts.
            for p in providers:
                p.cancel_inference(pointer)
    except Exception as e:
        print(f"[train] Inference cancel-on-delete for classifier {classifier_id} failed: {e}")


def _end_realtime_session_for_classifier(classifier_id: int) -> None:
    """Tear down any warm realtime cluster session for this guardrail (stop
    sentinel + scancel + cleanup) so deleting a guardrail mid-monitoring doesn't
    leave a GPU job running. Best-effort."""
    try:
        from services import compute
        from services.compute.base import RealtimeSession
        provider = compute.get_provider(compute.Workload.REALTIME, probe=False)
        provider.end_realtime(RealtimeSession(
            provider=provider.name, classifier_id=classifier_id, id=str(classifier_id)))
    except Exception as e:
        print(f"[realtime] Session end-on-delete for classifier {classifier_id} failed: {e}")

# --- Request Schemas ---
class ClassifierCreate(BaseModel):
    name: str
    # Optional: a guardrail can be created model-less and have a model attached
    # later (at train time). When provided, the user must own the model.
    model_id: Optional[int] = None

class AttachModelRequest(BaseModel):
    model_id: int

class CloneClassifierRequest(BaseModel):
    target_model_id: int
    name: Optional[str] = None

class ForkRuleSetRequest(BaseModel):
    rule_set_public_id: str
    name: Optional[str] = None

class AddExistingRuleRequest(BaseModel):
    rule_id: int

class CreateManualRuleRequest(BaseModel):
    name: str


class CreateAIRuleRequest(BaseModel):
    name: str
    predicate: str
    active_ces: List[str] # List of CE names (e.g. ["Tax_Evasion"])
    user_id: int # Needed to create CEs if they don't exist

class TrainingConfigUpdate(BaseModel):
    hidden_dim: Optional[int] = None
    num_rnn_layers: Optional[int] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None
    rnn_sequence_length: Optional[int] = None
    num_layers_to_use: Optional[int] = None
    max_length: Optional[int] = None
    batch_size_text: Optional[int] = None

# --- Endpoints ---

@router.get("/{model_id}", dependencies=[Depends(require_model_owner)])
def get_classifiers_endpoint(model_id: int, _: int = Depends(get_current_user)):
    classifiers = get_model_classifiers(model_id)
    # Recompute 'needs_retraining' from real policy drift (and self-heal the
    # stored status) so the card badge reflects whether the current policy
    # actually differs from what the model was trained on.
    for c in classifiers:
        c["status"] = reconcile_classifier_status(c["classifier_id"])
    return {"classifiers": classifiers}

@router.post("/create")
def create_new_classifier(classifier: ClassifierCreate, uid: int = Depends(get_current_user)):
    name = (classifier.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Rule Set name is required")

    model_id = classifier.model_id
    if model_id is not None:
        # Attached at creation (the secondary per-model flow). Must own the model.
        assert_owns_model(uid, model_id)
        # A name must be unique within its parent model — two guardrails under
        # one model with the same name are indistinguishable everywhere the user
        # sees them (sidebar, download zips), so reject the duplicate up front.
        existing = execute_query_dict(
            "SELECT classifier_id FROM classifiers WHERE model_id = %s AND LOWER(name) = LOWER(%s)",
            (model_id, name),
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"A rule set named '{name}' already exists under this model.",
            )
    else:
        # Model-less guardrail (the primary flow): unique among this user's
        # other unattached guardrails so the Guardrails list stays unambiguous.
        existing = execute_query_dict(
            "SELECT classifier_id FROM classifiers "
            "WHERE user_id = %s AND model_id IS NULL AND LOWER(name) = LOWER(%s)",
            (uid, name),
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"You already have a rule set named '{name}'.",
            )

    result = create_classifier(uid, name, model_id)
    if result:
        return {"success": True, "classifier": result}
    raise HTTPException(status_code=500, detail="Failed to create rule set")

@router.get("/details/all")
def get_all_user_classifiers(uid: int = Depends(get_current_user)):
    """Every guardrail the user owns, across all models and the unattached ones
    (model_name NULL until a model is picked). Backs the primary Guardrails page."""
    classifiers = get_user_classifiers(uid)
    # Self-heal 'needs_retraining' from real policy drift, like the per-model list.
    for c in classifiers:
        c["status"] = reconcile_classifier_status(c["classifier_id"])
    return {"classifiers": classifiers}

@router.get("/details/{classifier_id}", dependencies=[Depends(require_classifier_owner)])
def get_classifier_details(classifier_id: int, _: int = Depends(get_current_user)):
    # LEFT JOIN so an unattached guardrail (model_id NULL) still returns, with
    # model_name NULL — an INNER JOIN would 404 it.
    query = """
        SELECT c.*, tm.name as model_name
        FROM classifiers c
        LEFT JOIN target_models tm ON c.model_id = tm.model_id
        WHERE c.classifier_id = %s
    """
    res = execute_query_dict(query, (classifier_id,))
    if res:
        return res[0]
    raise HTTPException(status_code=404, detail="Not found")

@router.post("/details/{classifier_id}/attach-model", dependencies=[Depends(require_classifier_owner)])
def attach_model_endpoint(classifier_id: int, req: AttachModelRequest,
                          uid: int = Depends(get_current_user)):
    """Bind a guardrail to a model (the model-last step before training). Only
    allowed while the guardrail is still untrained — once trained, the weights
    are locked to their model, so moving to another model is the clone action."""
    assert_owns_model(uid, req.model_id)
    row = execute_query_dict(
        "SELECT name, trained_at FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Rule Set not found")
    if row[0].get("trained_at") is not None:
        raise HTTPException(
            status_code=409,
            detail="This rule set is already trained on a model. Use 'Apply to another model' to copy it.",
        )
    # Name must stay unique within the target model.
    name = row[0]["name"]
    clash = execute_query_dict(
        "SELECT classifier_id FROM classifiers "
        "WHERE model_id = %s AND LOWER(name) = LOWER(%s) AND classifier_id <> %s",
        (req.model_id, name, classifier_id),
    )
    if clash:
        raise HTTPException(
            status_code=409,
            detail=f"A rule set named '{name}' already exists under this model. Rename this one first.",
        )
    updated = attach_model_to_classifier(classifier_id, req.model_id)
    return {"success": True, "classifier": updated}

@router.post("/details/{classifier_id}/clone", dependencies=[Depends(require_classifier_owner)])
def clone_classifier_endpoint(classifier_id: int, req: CloneClassifierRequest,
                              uid: int = Depends(get_current_user)):
    """Apply this guardrail's rule set to another model: deep-copy it into a new,
    untrained guardrail attached to target_model_id (independent copy thereafter,
    retrained for its model)."""
    assert_owns_model(uid, req.target_model_id)
    new = clone_classifier_policy(classifier_id, req.target_model_id, uid, req.name)
    return {"success": True, "classifier": new}


@router.post("/from-rule-set")
def fork_rule_set_endpoint(req: ForkRuleSetRequest, uid: int = Depends(get_current_user)):
    """Fork a PUBLIC rule set into a new private, model-less rule set the caller
    owns. Add-by-reference: members are the existing public rules, no model is
    attached, the copy starts untrained. Returns the new guardrail so the
    frontend can navigate straight into its rule editor."""
    try:
        new = fork_public_rule_set_to_classifier(req.rule_set_public_id, uid, req.name)
        return {"success": True, "classifier": new}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- RULES LOGIC ---

@router.get("/{classifier_id}/rules", dependencies=[Depends(require_classifier_owner)])
def get_rules(classifier_id: int, _: int = Depends(get_current_user)):
    return {"rules": get_classifier_rules(classifier_id)}


@router.get("/{classifier_id}/policy-comparison", dependencies=[Depends(require_classifier_owner)])
def get_policy_comparison(classifier_id: int, mode: str = "same_policy",
                          _: int = Depends(get_current_user)):
    """Compare trained guardrails side by side, in one of two modes:

      mode='same_policy' (default) — guardrails sharing this one's
        `trained_policy_fingerprint` (a model-INDEPENDENT hash of the rule/CE
        composition). Surfaces how the SAME detection policy performs across
        DIFFERENT base models. If a guardrail's policy is later changed and it
        is retrained, its fingerprint changes, so it naturally drops out of this
        group (it's no longer the same policy).

      mode='same_model' — every trained guardrail on this one's base model,
        regardless of policy. Lets you compare different policies (or repeat
        training runs) on the SAME model.

    Each entry carries its latest post-training evaluation metrics.
    """
    mode = "same_model" if mode == "same_model" else "same_policy"

    src = execute_query_dict(
        "SELECT c.classifier_id, c.model_id, c.name, c.trained_policy_fingerprint, "
        "c.trained_rule_names, tm.name AS model_name "
        "FROM classifiers c JOIN target_models tm ON c.model_id = tm.model_id "
        "WHERE c.classifier_id = %s",
        (classifier_id,),
    )
    if not src:
        raise HTTPException(status_code=404, detail="Rule Set not found")
    src = src[0]
    fingerprint = (src.get("trained_policy_fingerprint") or "").strip()
    model_id = src.get("model_id")

    _SELECT = (
        "SELECT c.classifier_id, c.name, c.model_id, c.status, c.trained_at, "
        "c.trained_rule_names, c.trained_policy_fingerprint, tm.name AS model_name "
        "FROM classifiers c JOIN target_models tm ON c.model_id = tm.model_id "
    )
    if mode == "same_model":
        peers = execute_query_dict(
            _SELECT + "WHERE c.model_id = %s AND c.trained_at IS NOT NULL ORDER BY c.trained_at ASC",
            (model_id,),
        ) or [] if model_id is not None else []
    else:
        peers = execute_query_dict(
            _SELECT + "WHERE c.trained_policy_fingerprint = %s AND c.trained_at IS NOT NULL ORDER BY c.trained_at ASC",
            (fingerprint,),
        ) or [] if fingerprint else []

    out = []
    for p in peers:
        cid = p["classifier_id"]
        ev = execute_query_dict(
            """
            SELECT metrics, created_at FROM evaluation_results
            WHERE classifier_id = %s AND eval_type = 'evaluation' AND metrics IS NOT NULL
              AND created_at >= COALESCE(
                  (SELECT trained_at FROM classifiers WHERE classifier_id = %s),
                  '-infinity'::timestamptz)
            ORDER BY created_at DESC LIMIT 1
            """,
            (cid, cid),
        )
        metrics = (ev[0]["metrics"] if ev else None) or {}
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}
        usecase_rows = [
            r for r in (metrics.get("metrics") or [])
            if (r.get("Support_Pos") or 0) > 0 or (r.get("Support_Neg") or 0) > 0
        ]
        out.append({
            "classifier_id": cid,
            "name": p.get("name"),
            "model_id": p.get("model_id"),
            "model_name": p.get("model_name"),
            "status": p.get("status"),
            "trained_at": p.get("trained_at"),
            "is_source": cid == classifier_id,
            "has_eval": bool(ev),
            "evaluated_at": ev[0]["created_at"] if ev else None,
            "weighted_averages": metrics.get("weighted_averages") or {},
            "usecase_rows": usecase_rows,
            "rule_names": p.get("trained_rule_names") or [],
            # Whether this peer's policy matches the source's (only meaningful in
            # same_model mode, where policies can differ).
            "same_policy_as_source": bool(
                fingerprint and (p.get("trained_policy_fingerprint") or "").strip() == fingerprint
            ),
        })

    return {
        "mode": mode,
        "policy_fingerprint": fingerprint or None,
        "rule_names": src.get("trained_rule_names") or [],
        "source_classifier_id": classifier_id,
        "source_model_name": src.get("model_name"),
        "classifiers": out,
    }

def _mark_needs_retraining(classifier_id: int):
    """If the guardrail is 'active', mark it as needing retraining due to rule changes."""
    from utils.PostgreSQL import execute_query as _eq
    _eq(
        "UPDATE classifiers SET status = 'needs_retraining' WHERE classifier_id = %s AND status = 'active'",
        (classifier_id,),
    )

# Option 1: Add from List (Public Space)
@router.post("/{classifier_id}/rules/add")
def add_existing_rule(classifier_id: int, req: AddExistingRuleRequest, auth_uid: int = Depends(get_current_user)):
    assert_owns_classifier(auth_uid, classifier_id)
    setup_id = add_rule_to_classifier(classifier_id, req.rule_id)
    _mark_needs_retraining(classifier_id)
    return {"success": True, "setup_id": setup_id}

# Option 2: Create by Hand (Local Only)
@router.post("/{classifier_id}/rules/manual")
def create_manual_rule(classifier_id: int, req: CreateManualRuleRequest, auth_uid: int = Depends(get_current_user)):
    assert_owns_classifier(auth_uid, classifier_id)
    setup_id = create_custom_rule_setup(classifier_id, req.name)
    if setup_id:
        _mark_needs_retraining(classifier_id)
        return {"success": True, "setup_id": setup_id}
    raise HTTPException(status_code=500, detail="Failed to create rule")

# Option 3: Generate with AI (Global -> Local)
@router.post("/{classifier_id}/rules/ai")
def create_ai_rule(classifier_id: int, req: CreateAIRuleRequest, auth_uid: int = Depends(get_current_user)):
    """
    Receives the AI-generated logic, saves it to the global library,
    and then links it to this guardrail.
    """
    assert_owns_classifier(auth_uid, classifier_id)
    try:
        # 1. Resolve CE names to IDs
        ce_ids = []
        for name in req.active_ces:
            # Create/Find the CE globally
            ce_rec = create_ce(req.user_id, name)
            ce_ids.append(ce_rec['ce_id'])

        # 2. Create Global Rule & Link
        setup_id = create_and_link_global_rule(
            classifier_id,
            req.name,
            req.predicate,
            ce_ids
        )
        _mark_needs_retraining(classifier_id)
        return {"success": True, "setup_id": setup_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.delete("/{classifier_id}", dependencies=[Depends(require_classifier_owner)])
def remove_classifier_endpoint(classifier_id: int, _: int = Depends(get_current_user)):
    """
    Removes a guardrail.
    Cascade logic: The DB will automatically remove all associated
    entries in rule_setup and setup_ce_link. The on-disk workdir
    (trained_classifiers/<user_id>/classifier_<id>/) is also swept
    so deleted guardrails don't leak weights / datasets to disk
    until the next boot-time orphan recovery.
    """
    try:
        # Resolve user_id BEFORE the DB delete (needed for the workdir path).
        # Read it straight off the guardrail row — joining through
        # target_models would miss a model-less guardrail (model_id NULL),
        # which has no model to join to.
        owner_rows = execute_query_dict(
            "SELECT user_id FROM classifiers WHERE classifier_id = %s",
            (classifier_id,),
        )
        if not owner_rows:
            raise HTTPException(status_code=404, detail="Rule Set not found")
        user_id = owner_rows[0]["user_id"]

        # Stop any in-flight work BEFORE the DB delete. Each pointer we need
        # (slurm_job_id / remote_job_dir / running-row) lives on a row the cascade
        # delete is about to wipe, so they MUST run first:
        #   * cluster TRAINING job,
        #   * cluster CALIBRATION / EVALUATION inference job,
        #   * warm REALTIME monitoring session.
        # Local in-process tasks (training + calibration/evaluation inference)
        # self-abort cooperatively: their loops check whether the guardrail row
        # still exists and stop at the next checkpoint once delete_classifier runs
        # (see trainer.TrainingCancelled / inference_core.InferenceCancelled).
        _cancel_cluster_job_for_classifier(classifier_id)
        _cancel_cluster_inference_for_classifier(classifier_id)
        _end_realtime_session_for_classifier(classifier_id)

        delete_classifier(classifier_id)

        # Best-effort disk cleanup. DB is the source of truth; if this
        # fails the boot-time OrphanedClassifierDirRecovery will sweep
        # it on next process start.
        try:
            from classifier_engine.trainer import delete_classifier_workdir
            delete_classifier_workdir(classifier_id, user_id)
        except Exception as cleanup_err:
            logging.getLogger(__name__).warning(
                f"classifier {classifier_id} workdir cleanup failed: {cleanup_err}"
            )

        return {"success": True, "message": f"Rule Set {classifier_id} removed successfully."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- TRAINING CONFIG ENDPOINTS ---

TRAINING_CONFIG_DEFAULTS = {
    "hidden_dim": 256,
    "num_rnn_layers": 3,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 3e-4,
    "rnn_sequence_length": 5,
    "num_layers_to_use": 8,
    "max_length": 256,
    "batch_size_text": 4,
}

@router.get("/{classifier_id}/config")
def get_training_config(classifier_id: int, auth_uid: int = Depends(get_current_user)):
    """Return stored training config merged with defaults."""
    assert_owns_classifier(auth_uid, classifier_id)
    result = execute_query_dict(
        "SELECT training_config FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    if not result:
        raise HTTPException(status_code=404, detail="Rule Set not found")

    stored = result[0].get("training_config") or {}
    merged = {**TRAINING_CONFIG_DEFAULTS, **stored}
    return {"config": merged, "defaults": TRAINING_CONFIG_DEFAULTS}


@router.put("/{classifier_id}/config")
def update_training_config(
    classifier_id: int,
    req: TrainingConfigUpdate,
    _: int = Depends(get_current_user),
):
    """Save training hyperparameters for this guardrail."""
    assert_owns_classifier(_, classifier_id)
    # Verify guardrail exists
    check = execute_query_dict(
        "SELECT classifier_id FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    if not check:
        raise HTTPException(status_code=404, detail="Rule Set not found")

    # Build config dict from non-None fields only
    config = {k: v for k, v in req.model_dump().items() if v is not None}

    from utils.PostgreSQL import execute_query
    execute_query(
        "UPDATE classifiers SET training_config = %s::jsonb WHERE classifier_id = %s",
        (json.dumps(config), classifier_id),
    )
    merged = {**TRAINING_CONFIG_DEFAULTS, **config}
    return {"success": True, "config": merged}


# --- TRAINING ENDPOINTS ---

# Maps raw trainer stages to short user-facing labels. The trainer emits
# stages from a fixed vocabulary (see _progress() calls in
# classifier_engine/trainer.py); anything we don't recognize here falls
# back to a Title-cased version of the raw key.
_TRAINING_PHASE_LABELS = {
    "init":      "Preparing",
    "data":      "Loading datasets",
    "load_llm":  "Loading language model",
    "split":     "Splitting train/validation",
    "extract":   "Extracting embeddings",
    "train_rnn": "Training RNN",
    "save":      "Saving model",
}


def _run_training_task(classifier_id: int):
    """Background task wrapper for training.

    Wires a progress callback into run_training so each stage boundary
    persists a `training_phase` + `training_phase_detail` row update.
    The /training-status route reads those columns; the UI polls and
    renders them so the user sees something more informative than a
    plain "Training..." spinner during the multi-minute pipeline.
    Phase columns are cleared back to NULL on completion or error so
    stale text doesn't linger past the run.
    """
    import traceback
    from utils.PostgreSQL import execute_query as _eq

    def _on_progress(stage: str, detail: str = ""):
        label = _TRAINING_PHASE_LABELS.get(stage, stage.replace("_", " ").title())
        try:
            _eq(
                "UPDATE classifiers SET training_phase = %s, training_phase_detail = %s WHERE classifier_id = %s",
                (label, detail or None, classifier_id),
            )
        except Exception:
            # Progress writes are best-effort; never let a transient DB
            # blip kill the actual training run.
            pass

    try:
        from classifier_engine.trainer import run_training
        run_training(classifier_id, progress_callback=_on_progress)
        # Clear the phase signal on the success path. run_training's own
        # success block flips status to 'active', so by the time we get
        # here the run is fully done.
        try:
            _eq(
                "UPDATE classifiers SET training_phase = NULL, training_phase_detail = NULL WHERE classifier_id = %s",
                (classifier_id,),
            )
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Background training failed for classifier {classifier_id}: {e}")
        logger.error(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        # Ensure status is set to 'error' even if trainer didn't handle it,
        # and clear the phase columns so the UI doesn't keep showing the
        # last in-flight stage after a failure.
        try:
            _eq(
                "UPDATE classifiers SET status = 'error', training_log = %s, training_phase = NULL, training_phase_detail = NULL "
                "WHERE classifier_id = %s AND status = 'training'",
                (f"Training failed: {str(e)[:500]}", classifier_id),
            )
        except Exception:
            pass


def _build_training_inputs(classifier_id: int):
    """Assemble model path + labels + per-CE dataset files + selection
    calibration dialogues for a training run. Shared by the cluster and
    remote-worker submit paths so they can never drift."""
    from classifier_engine.trainer import (
        get_classifier_info, get_classifier_ces_with_datasets, get_training_config,
        _extract_training_data, _sanitize_label, fetch_calibration_entries,
    )
    info = get_classifier_info(classifier_id)
    if not info:
        raise ValueError("Guardrail not found")
    model_hf_path = info["storage_path"]
    training_config = get_training_config(classifier_id)
    ces = get_classifier_ces_with_datasets(classifier_id)
    labels, dataset_files, idx = {}, {}, 0
    for ce in ces:
        conversations = _extract_training_data(ce.get("dataset"))
        if not conversations:
            continue
        safe = _sanitize_label(ce["name"])
        labels[safe] = idx
        dataset_files[f"{safe}.json"] = conversations
        idx += 1
    if not labels:
        raise ValueError("No valid training data found")
    calibration_entries = fetch_calibration_entries(
        classifier_id, ces, per_ce=int(training_config.get("selection_calib_per_ce", 25)),
    )
    return model_hf_path, labels, training_config, dataset_files, calibration_entries


@router.post("/{classifier_id}/train", dependencies=[Depends(require_classifier_owner)])
def start_training(
    classifier_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    target: str = None,   # optional: force a compute target (local | slurm | remote_worker)
    user_id: int = Depends(get_current_user),
):
    """
    Starts training the RNN guardrail.

    Compute is resolved via services.compute (remote_worker / slurm / local —
    see the provider registry). Off-box providers get the job payload built
    locally (CE datasets, model path, labels, config) and submitted; LOCAL
    mode runs training on this machine's GPU in a background task.

    Returns immediately; poll /training-status for progress.
    """
    # Verify guardrail exists
    check = execute_query_dict(
        "SELECT classifier_id, status, model_id FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    if not check:
        raise HTTPException(status_code=404, detail="Rule Set not found")

    # A guardrail must be attached to a model before it can train. Training is
    # irreducibly model-specific (the RNN's geometry comes from the LLM), so an
    # unattached guardrail (model_id NULL) has nothing to extract features from.
    if check[0].get("model_id") is None:
        raise HTTPException(
            status_code=400,
            detail="Pick a model for this rule set before training.",
        )

    current_status = check[0]["status"]
    if current_status == "training":
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Lazy-load any HF-synced CE excitations that aren't cached locally yet.
    try:
        from services.hf_sync import ensure_excitations_for_classifier
        ensure_excitations_for_classifier(classifier_id)
    except Exception as lazy_err:
        print(f"[train] lazy excitation prefetch failed: {lazy_err}")

    # Verify there are CEs with datasets
    ces_check = execute_query_dict(
        """
        SELECT COUNT(DISTINCT ce.ce_id) as ce_count
        FROM rule_setup rs
        JOIN setup_ce_link scl ON rs.setup_id = scl.setup_id
        JOIN cognitive_elements ce ON scl.ce_id = ce.ce_id
        JOIN excitation_datasets ed ON ce.ce_id = ed.ce_id
        WHERE rs.classifier_id = %s
        """,
        (classifier_id,),
    )
    ce_count = (ces_check[0]["ce_count"] if ces_check else 0) or 0
    if ce_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No CEs with excitation datasets found. Generate training data for your CEs first.",
        )

    # Set status to training immediately so UI can reflect it.
    from utils.PostgreSQL import execute_query
    execute_query(
        "UPDATE classifiers SET status = 'training', training_log = NULL, "
        "training_phase = NULL, training_phase_detail = NULL "
        "WHERE classifier_id = %s",
        (classifier_id,),
    )

    # ---- Off-box training (remote GPU worker / SLURM cluster) via the compute interface ----
    # One provider-driven path for both. The provider resolution accounts for
    # reachability (an unreachable cluster/worker resolves to 'local'), so on any
    # failure we fall through to local training below. training_log uses the
    # generic {provider, mode, job:{id,raw}, last_contact} shape the status route
    # reconstructs from.
    from services import compute
    # If the user explicitly picked a machine, honor it; otherwise auto-resolve.
    # An unknown/unconfigured target falls back to auto-resolution.
    provider = (compute.provider_by_name(target) if target else None) \
        or compute.get_provider(compute.Workload.TRAINING)
    if provider.name in ("remote_worker", "slurm"):
        try:
            model_hf_path, labels, training_config, dataset_files, calibration_entries = \
                _build_training_inputs(classifier_id)
            spec = compute.TrainingSpec(
                classifier_id=classifier_id, user_id=user_id, model_hf_path=model_hf_path,
                labels=labels, training_config=training_config, dataset_files=dataset_files,
                calibration_entries=calibration_entries,
            )
            job = provider.submit_training(spec)
            mode = "remote_worker" if provider.name == "remote_worker" else "cluster"
            from utils.PostgreSQL import execute_update
            # Failover ladder for this run (remote_worker -> slurm -> local GPU; no
            # CPU tier for training). chain_pos marks the tier we're on; if it dies
            # mid-run the status poller advances to the next tier (see _failover).
            chain = compute.failover_providers(compute.Workload.TRAINING)
            training_log = {"provider": provider.name, "mode": mode,
                            "job": {"id": job.id, "raw": job.raw},
                            "chain": chain,
                            "chain_pos": chain.index(provider.name) if provider.name in chain else 0,
                            "user_id": user_id,
                            # epoch of last confirmed contact; the status poller fails
                            # (or fails over) the job if this goes stale.
                            "last_contact": time.time()}
            recorded = execute_update(
                "UPDATE classifiers SET training_log = %s "
                "WHERE classifier_id = %s AND status = 'training'",
                (json.dumps(training_log), classifier_id))
            if not recorded:
                # Guardrail deleted mid-submit — cancel the job we just made so it
                # doesn't run orphaned (closes the submit-window race atomically).
                provider.cancel_training(job)
                return {"success": False, "mode": mode, "classifier_id": classifier_id,
                        "message": "Rule Set was removed before training started; the job was cancelled."}
            where = "the GPU worker" if provider.name == "remote_worker" else "the cluster"
            # Cluster only: race the powerful GPU against a weaker fallback so a
            # long queue doesn't block the user (background; no-op if unconfigured).
            if provider.name == "slurm":
                try:
                    _spawn_training_gpu_race(classifier_id, user_id, job)
                except Exception as race_err:
                    print(f"[train] Classifier {classifier_id} | could not start GPU race: {race_err}")
            return {"success": True, "mode": mode, "classifier_id": classifier_id,
                    "ce_count": int(ce_count), "message": f"Training submitted to {where}"}
        except Exception as e:
            print(f"[train] Classifier {classifier_id} | {provider.name} submit failed, falling back to local: {e}")
            import traceback
            traceback.print_exc()
            execute_query(
                "UPDATE classifiers SET training_phase = %s, training_phase_detail = %s "
                "WHERE classifier_id = %s",
                ("fallback", f"{provider.name} unavailable ({e}). Training locally.", classifier_id))
            # fall through to local


    # LOCAL MODE: best available accelerator (CUDA > MPS > CPU).
    from utils.device import get_torch_device
    device_name = get_torch_device().type.upper()
    print(f"[train] Running locally on {device_name}")
    background_tasks.add_task(_run_training_task, classifier_id)

    return {
        "success": True,
        "message": f"Training started locally on {device_name}",
        "classifier_id": classifier_id,
        "ce_count": int(ce_count),
        "mode": f"local_{device_name.lower()}",
    }


@router.get("/{classifier_id}/download", dependencies=[Depends(require_classifier_owner)])
def download_classifier(classifier_id: int, _: int = Depends(get_current_user)):
    """
    Zips the trained guardrail folder and returns it as a downloadable file.
    Only available when the guardrail status is 'active'.
    """
    result = execute_query_dict(
        "SELECT classifier_id, name, status FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    if not result:
        raise HTTPException(status_code=404, detail="Rule Set not found")

    row = result[0]
    if row["status"] != "active":
        raise HTTPException(status_code=400, detail="Rule Set is not trained yet")

    from classifier_engine.trainer import classifier_workdir
    classifier_dir = classifier_workdir(classifier_id)
    if not os.path.isdir(classifier_dir):
        raise HTTPException(status_code=404, detail="Trained model files not found")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in row["name"])
    zip_filename = f"classifier_{classifier_id}_{safe_name}.zip"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(classifier_dir):
            fpath = os.path.join(classifier_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, fname)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


# ---------------------------------------------------------------------------
# Guardrail bundle: export (preflight → publish policy → download) + import
# ---------------------------------------------------------------------------

@router.get("/{classifier_id}/export/preflight", dependencies=[Depends(require_classifier_owner)])
def export_preflight(classifier_id: int, _: int = Depends(get_current_user)):
    """Report whether this guardrail can be exported and what's outstanding.

    The UI uses this to decide whether to show the Export button (only when the
    guardrail is trained AND its policy hasn't drifted), which tiers are
    offerable, and which draft rules still need publishing first.
    """
    from services import classifier_bundle
    try:
        a = classifier_bundle.assess_export(classifier_id)
    except classifier_bundle.BundleError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    # Strip internal payloads before returning to the client.
    return {
        "classifier_id": a["classifier_id"],
        "name": a["name"],
        "can_export": a["can_export"],
        "reason": a["reason"],
        "drift": a["drift"],
        "tiers_available": a["tiers_available"],
        "unpublished": a["unpublished"],
        "blockers": a["blockers"],
    }


@router.post("/{classifier_id}/export/start", dependencies=[Depends(require_classifier_owner)])
def export_start(
    classifier_id: int,
    background_tasks: BackgroundTasks,
    tier: str = "model+calibration",
    user_id: int = Depends(get_current_user),
):
    """Kick off an export as a background job and return its job_id immediately.

    The job publishes any draft rules in the policy (the user approved this by
    starting the export), then builds the bundle. It survives the user closing
    the modal — only a backend crash ends it. Poll GET /classifiers/bundle-jobs/
    {job_id}; download from .../download when status is 'done'.
    """
    from services import classifier_bundle, bundle_jobs
    try:
        a = classifier_bundle.assess_export(classifier_id)
    except classifier_bundle.BundleError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    # Fail obvious cases synchronously so the UI gets an immediate, precise error.
    if a["drift"] or a["blockers"]:
        raise HTTPException(status_code=409, detail=a["reason"] or (a["blockers"][0] if a["blockers"] else "Cannot export."))
    if tier not in a["tiers_available"]:
        raise HTTPException(status_code=409, detail=f"Tier '{tier}' isn't available for this rule set.")

    job_id = bundle_jobs.create_job(user_id, "export", classifier_id=classifier_id, tier=tier, phase="Queued…")
    bundle_jobs.cleanup_prior_export_artifacts(user_id, classifier_id, keep_job_id=job_id)
    background_tasks.add_task(bundle_jobs.run_export_job, job_id, user_id, classifier_id, tier)
    return {"job_id": job_id}


@router.get("/{classifier_id}/export/active-job", dependencies=[Depends(require_classifier_owner)])
def export_active_job(classifier_id: int, user_id: int = Depends(get_current_user)):
    """The latest running-or-ready export job for this guardrail, so the UI can
    resume showing progress / offer a download after the modal is reopened."""
    from services import bundle_jobs
    return {"job": bundle_jobs.latest_export_job(user_id, classifier_id)}


@router.post("/import/start")
async def import_start(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user),
):
    """Stage an uploaded bundle and kick off the import as a background job.
    Returns a job_id to poll; the import survives the user navigating away."""
    from services import bundle_jobs
    try:
        data = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    upload_path = bundle_jobs.save_upload(data)
    job_id = bundle_jobs.create_job(user_id, "import", phase="Queued…")
    background_tasks.add_task(bundle_jobs.run_import_job, job_id, user_id, upload_path)
    return {"job_id": job_id}


@router.get("/bundle-jobs/{job_id}")
def bundle_job_status(job_id: int, user_id: int = Depends(get_current_user)):
    """Poll a bundle export/import job."""
    from services import bundle_jobs
    job = bundle_jobs.get_job(job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@router.get("/bundle-jobs/{job_id}/download")
def bundle_job_download(job_id: int, user_id: int = Depends(get_current_user)):
    """Stream a finished export bundle."""
    from fastapi.responses import FileResponse
    from services import bundle_jobs
    art = bundle_jobs.get_artifact(job_id, user_id)
    if not art:
        raise HTTPException(status_code=404, detail="Job not found.")
    if art["job_type"] != "export" or art["status"] != "done" or not art.get("artifact_path"):
        raise HTTPException(status_code=409, detail="This export isn't ready to download.")
    path = art["artifact_path"]
    if not os.path.isfile(path):
        raise HTTPException(status_code=410, detail="This export is no longer available — re-export it.")
    return FileResponse(path, media_type="application/zip", filename=art.get("filename") or "bundle.gavel.zip")


@router.get("/{classifier_id}/training-status")
def get_training_status(classifier_id: int, auth_uid: int = Depends(get_current_user)):
    """
    Returns the current training status of a guardrail.
    Status values: untrained | training | active | error

    For cluster-submitted jobs: polls the central server for real-time
    status + training log. When central reports "completed", updates the
    local guardrail to 'active'. When central reports "failed"/"oom"/
    "timeout", updates to 'error' with the message.
    """
    assert_owns_classifier(auth_uid, classifier_id)
    result = execute_query_dict(
        "SELECT classifier_id, name, status, model_path, training_log, "
        "training_phase, training_phase_detail, created_at "
        "FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    if not result:
        raise HTTPException(status_code=404, detail="Rule Set not found")

    row = result[0]
    # Recompute 'needs_retraining' from real policy drift before anything reads
    # the status, so the banner/badge reflect whether the current policy differs
    # from the trained one (self-healing — a no-op for training/error states and
    # guardrails without a fingerprint snapshot).
    if row["status"] in ("active", "needs_retraining"):
        row["status"] = reconcile_classifier_status(classifier_id)
    is_training = row["status"] == "training"

    # ---- Off-box training (SLURM cluster OR remote GPU worker) ----
    # Driven entirely through the compute provider interface — this route is
    # transport-agnostic. training_log carries either the new
    # {provider, mode, job:{id,raw}, last_contact} shape or a legacy
    # {mode, slurm_job_id/worker_job_id, ...} shape; we reconstruct a TrainingJob
    # from either, so in-flight jobs submitted before this change keep working.
    training_log_data = row.get("training_log")
    _lp = training_log_data
    if isinstance(_lp, str):
        try:
            _lp = json.loads(_lp)
        except (json.JSONDecodeError, TypeError):
            _lp = None

    def _reconstruct_job():
        from services.compute.base import TrainingJob
        if not isinstance(_lp, dict):
            return None
        prov = _lp.get("provider")
        if prov in ("slurm", "remote_worker") and isinstance(_lp.get("job"), dict):
            return prov, TrainingJob(provider=prov, classifier_id=classifier_id,
                                     id=str(_lp["job"].get("id")), raw=_lp["job"].get("raw") or {})
        mode = _lp.get("mode")
        if mode == "cluster" and _lp.get("slurm_job_id"):
            return "slurm", TrainingJob(
                provider="slurm", classifier_id=classifier_id, id=str(_lp["slurm_job_id"]),
                raw={"slurm_job_id": _lp["slurm_job_id"], "remote_job_dir": _lp.get("remote_job_dir"),
                     "job_id": _lp.get("job_id"), "mode": "cluster"})
        if mode == "remote_worker" and _lp.get("worker_job_id"):
            return "remote_worker", TrainingJob(
                provider="remote_worker", classifier_id=classifier_id,
                id=str(_lp["worker_job_id"]), raw={})
        return None

    _job_info = _reconstruct_job() if is_training else None
    if _job_info:
        from services import compute
        from services.compute.base import JobState
        from utils.PostgreSQL import execute_query
        expected_provider, job = _job_info
        provider = compute.get_provider(compute.Workload.TRAINING)
        where = "the GPU worker" if expected_provider == "remote_worker" else "the cluster"
        phase_label = "Training on GPU worker" if expected_provider == "remote_worker" else "Training on cluster"

        def _ret(status, *, model_path=None, phase=None, detail=None, training_log=None,
                 is_tr=False, is_trn=True, err=False):
            return {"classifier_id": row["classifier_id"], "name": row["name"], "status": status,
                    "model_path": model_path,
                    "training_log": training_log if training_log is not None else training_log_data,
                    "training_phase": phase, "training_phase_detail": detail,
                    "is_trained": is_tr, "is_training": is_trn, "has_error": err,
                    "mode": expected_provider}

        def _failover(reason: str):
            """The current tier died mid-run. Advance the failover ladder stored in
            training_log (chain/chain_pos) to the next tier and re-dispatch, so a
            crashed/vanished cluster or worker re-runs the training elsewhere
            instead of just failing. Returns a status payload (still 'training') on
            success, or None when there's no chain / the ladder is exhausted — in
            which case the caller fails the job as before. Additive: legacy in-flight
            jobs (no 'chain') get None and keep the old behavior."""
            chain = _lp.get("chain") if isinstance(_lp, dict) else None
            pos = _lp.get("chain_pos") if isinstance(_lp, dict) else None
            uid = (_lp.get("user_id") if isinstance(_lp, dict) else None) or 0
            if not chain or pos is None or pos + 1 >= len(chain):
                return None
            # Best-effort: stop the dead job so it can't keep running orphaned.
            try:
                provider.cancel_training(job)
            except Exception:
                pass
            for nxt in range(pos + 1, len(chain)):
                name = chain[nxt]
                try:
                    if name == "local":
                        import threading
                        threading.Thread(target=_run_training_task,
                                         args=(classifier_id,), daemon=True).start()
                        detail = f"{reason} Retraining on your local GPU."
                        nl = {"mode": "local", "chain": chain, "chain_pos": nxt, "user_id": uid}
                        execute_query(
                            "UPDATE classifiers SET training_phase = %s, "
                            "training_phase_detail = %s, training_log = %s WHERE classifier_id = %s",
                            ("Retrying locally", detail, json.dumps(nl), classifier_id))
                        return _ret("training", phase="Retrying locally", detail=detail,
                                    training_log=json.dumps(nl))
                    prov2 = compute.provider_by_name(name)
                    if prov2 is None:
                        continue
                    mhp, labels, tcfg, dfiles, calib = _build_training_inputs(classifier_id)
                    spec2 = compute.TrainingSpec(
                        classifier_id=classifier_id, user_id=uid, model_hf_path=mhp,
                        labels=labels, training_config=tcfg, dataset_files=dfiles,
                        calibration_entries=calib)
                    job2 = prov2.submit_training(spec2)
                    mode2 = "remote_worker" if name == "remote_worker" else "cluster"
                    where2 = "the GPU worker" if name == "remote_worker" else "the cluster"
                    detail = f"{reason} Retrying on {where2}."
                    nl = {"provider": name, "mode": mode2,
                          "job": {"id": job2.id, "raw": job2.raw},
                          "chain": chain, "chain_pos": nxt, "user_id": uid,
                          "last_contact": time.time()}
                    execute_query(
                        "UPDATE classifiers SET training_phase = %s, "
                        "training_phase_detail = %s, training_log = %s WHERE classifier_id = %s",
                        ("Retrying", detail, json.dumps(nl), classifier_id))
                    return _ret("training", phase="Retrying", detail=detail,
                                training_log=json.dumps(nl))
                except Exception as fe:
                    print(f"[train] Classifier {classifier_id} | failover to {name} failed: {fe}")
                    continue
            return None

        # The configured provider must match the one that submitted; if the GPU
        # config changed under a running job we can't poll it — surface as still
        # training so the user can recover manually.
        if provider.name != expected_provider:
            return _ret("training", phase=phase_label, detail="Waiting - GPU provider changed since submit.")

        # Serialize per-guardrail so two polls don't race the download/finalize.
        cls_lock = _get_download_lock(classifier_id)
        if not cls_lock.acquire(blocking=False):
            return _ret("training", phase="Processing", detail="Downloading trained model...")
        try:
            fresh = execute_query_dict(
                "SELECT status, model_path, training_phase, training_phase_detail "
                "FROM classifiers WHERE classifier_id = %s", (classifier_id,))
            if fresh and fresh[0]["status"] != "training":
                f = fresh[0]
                return _ret(f["status"], model_path=f["model_path"], phase=f["training_phase"],
                            detail=f["training_phase_detail"], is_tr=f["status"] == "active",
                            is_trn=False, err=f["status"] == "error")

            st = provider.poll_training(job)
            now = time.time()

            if st.state == JobState.DONE:
                from classifier_engine.trainer import classifier_workdir
                work_dir = classifier_workdir(classifier_id)
                provider.fetch_trained_model(job, work_dir)
                model_path = os.path.join(work_dir, "trained_rnn.pth")
                detail = f"Trained on {where}"
                execute_query(
                    "UPDATE classifiers SET status = 'active', model_path = %s, "
                    "training_phase = 'complete', training_phase_detail = %s WHERE classifier_id = %s",
                    (model_path, detail, classifier_id))
                try:
                    from sql_scripts.model_scripts import commit_trained_policy_snapshot
                    commit_trained_policy_snapshot(classifier_id)
                except Exception as snap_err:
                    print(f"[train] Classifier {classifier_id} | snapshot commit failed: {snap_err}")
                return _ret("active", model_path=model_path, phase="complete", detail=detail,
                            is_tr=True, is_trn=False)

            if st.state in (JobState.ERROR, JobState.CANCELLED):
                # GPU race in flight: the tracked job may be the queued primary
                # that the race is about to cancel + replace with the faster-
                # available GPU. Don't fail/failover yet — let the race re-point
                # us to the winner (it clears `racing` when it's done).
                if isinstance(_lp, dict) and _lp.get("racing"):
                    return _ret("training", phase="Selecting GPU",
                                detail="Waiting for the first available GPU…")
                fo = _failover(f"{where} reported a failure.")
                if fo is not None:
                    return fo
                emsg = st.error or f"Training failed on {where}"
                execute_query(
                    "UPDATE classifiers SET status = 'error', training_phase = 'failed', "
                    "training_phase_detail = %s WHERE classifier_id = %s", (emsg, classifier_id))
                return _ret("error", phase="failed", detail=emsg, is_trn=False, err=True)

            # Running - liveness deadline fails a job whose cluster/worker vanished.
            dead_timeout = float(os.getenv("GPU_WORKER_DEAD_TIMEOUT", "600"))
            last_contact = (_lp.get("last_contact") if isinstance(_lp, dict) else None) or now
            if not st.reachable:
                gone = now - last_contact
                if gone > dead_timeout:
                    fo = _failover(f"{where} was unreachable for {gone:.0f}s.")
                    if fo is not None:
                        return fo
                    emsg = f"{where} unreachable for {gone:.0f}s (> {dead_timeout:.0f}s) - failing the training job."
                    execute_query(
                        "UPDATE classifiers SET status = 'error', training_phase = 'failed', "
                        "training_phase_detail = %s WHERE classifier_id = %s", (emsg, classifier_id))
                    return _ret("error", phase="failed", detail=emsg, is_trn=False, err=True)
                detail = f"Cannot reach {where} (retrying, {gone:.0f}s/{dead_timeout:.0f}s)..."
                execute_query(
                    "UPDATE classifiers SET training_phase = %s, training_phase_detail = %s WHERE classifier_id = %s",
                    (phase_label, detail, classifier_id))
                return _ret("training", phase=phase_label, detail=detail)

            # Reachable + running - advance the liveness clock (new-format logs) + show progress.
            detail = st.detail or st.phase or f"Optimizing rule set on {where}..."
            tl_out = training_log_data
            if isinstance(_lp, dict) and _lp.get("provider"):
                new_log = dict(_lp); new_log["last_contact"] = now
                tl_out = json.dumps(new_log)
                execute_query(
                    "UPDATE classifiers SET training_phase = %s, training_phase_detail = %s, "
                    "training_log = %s WHERE classifier_id = %s",
                    (st.phase or phase_label, detail, tl_out, classifier_id))
            else:
                execute_query(
                    "UPDATE classifiers SET training_phase = %s, training_phase_detail = %s WHERE classifier_id = %s",
                    (st.phase or phase_label, detail, classifier_id))
            return _ret("training", phase=st.phase or phase_label, detail=detail, training_log=tl_out)
        except Exception as e:
            print(f"[train] Classifier {classifier_id} | poll failed: {e}")
            return _ret("training", phase=phase_label, detail=f"Cannot reach {where}: {e}")
        finally:
            cls_lock.release()

    return {
        "classifier_id": row["classifier_id"],
        "name": row["name"],
        "status": row["status"],
        "model_path": row["model_path"],
        "training_log": row["training_log"],
        # Surface phase detail while training AND on error, so the frontend can
        # show WHY a run failed (e.g. a model with no chat template) — including
        # after a page refresh once the 'error' status is persisted.
        "training_phase": row["training_phase"] if (is_training or row["status"] == "error") else None,
        "training_phase_detail": row["training_phase_detail"] if (is_training or row["status"] == "error") else None,
        "is_trained": row["status"] == "active",
        "is_training": is_training,
        "has_error": row["status"] == "error",
    }