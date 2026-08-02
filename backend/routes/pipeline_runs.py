"""REST routes for the `pipeline_runs` table — wizard state persistence
for both Pipeline A (rule generation) and Pipeline B (test + eval).

The wizard itself is a frontend concern (see RuleGenerationWizard.jsx
and TestEvalWizard.jsx); these endpoints just store the user's progress
so they can resume after a browser close. Every step transition fires
a PATCH; reading the row back gives the wizard everything it needs to
rehydrate.

All routes are auth-gated and scoped to the requester's own rows.
"""
from typing import Optional, Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from sql_scripts.pipeline_run_scripts import (
    create_pipeline_run,
    get_pipeline_run,
    get_active_runs,
    update_step,
    set_run_links,
    complete_run,
    delete_pipeline_run,
)
from utils.auth import get_current_user

router = APIRouter()


def _serialize(row: dict) -> dict:
    """Shape a DB row into the JSON contract the wizard frontend expects."""
    return {
        "run_id": row["run_id"],
        "pipeline_type": row.get("pipeline_type", "rule"),
        "classifier_id": row.get("classifier_id"),
        "rule_id": row.get("rule_id"),
        "current_step": row["current_step"],
        "steps": row.get("steps") or {},
        "completed": row["completed"],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
    }


def _owned(run_id: int, user_id: int) -> dict:
    """Fetch a run + verify the requester owns it. 404 on miss or
    mismatch (the latter to avoid leaking which ids exist)."""
    row = get_pipeline_run(run_id)
    if not row or row["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return row


class CreateRunRequest(BaseModel):
    pipeline_type: str = "rule"
    classifier_id: Optional[int] = None
    rule_id: Optional[int] = None


class StepUpdateRequest(BaseModel):
    step_id: str       # "1" | "2A" | "2B" | "2C" | "3A" | "3B" | "3C" | "3D" | "cal" | "eval"
    status: str        # "pending" | "in_progress" | "completed" | "skipped" | "error"
    data: Optional[Dict[str, Any]] = None
    # Optional explicit advancement. Wizard sets this when the user
    # clicks Next; leaving it unset means "just record this step's
    # state, don't change which step the wizard is on".
    advance_to: Optional[str] = None


class LinksUpdateRequest(BaseModel):
    rule_id: Optional[int] = None


@router.post("")
def start_run(
    req: CreateRunRequest,
    user_id: int = Depends(get_current_user),
):
    """Open a new wizard run for the requester.

    Pipeline A runs (pipeline_type='rule') are guardrail-agnostic —
    pass no classifier_id. Pipeline B runs (pipeline_type='test_eval')
    require a classifier_id + (ideally) a rule_id.
    """
    try:
        row = create_pipeline_run(
            user_id=user_id,
            pipeline_type=req.pipeline_type,
            classifier_id=req.classifier_id,
            rule_id=req.rule_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create pipeline run")
    return _serialize(row)


@router.get("/active")
def list_active(
    classifier_id: Optional[int] = Query(default=None),
    pipeline_type: Optional[str] = Query(default=None),
    rule_id: Optional[int] = Query(default=None),
    user_id: int = Depends(get_current_user),
):
    """List the requester's active (completed=FALSE) runs.

    Optional filters: `pipeline_type` ('rule' or 'test_eval'),
    `classifier_id`, `rule_id`. Useful for surfacing "resume your
    previous run" banners scoped to whatever context the user just
    arrived in.
    """
    try:
        rows = get_active_runs(
            user_id=user_id,
            classifier_id=classifier_id,
            pipeline_type=pipeline_type,
            rule_id=rule_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"runs": [_serialize(r) for r in rows]}


@router.get("/{run_id}")
def get_run(run_id: int, user_id: int = Depends(get_current_user)):
    """Fetch one run + its step state."""
    row = _owned(run_id, user_id)
    return _serialize(row)


@router.patch("/{run_id}/step")
def patch_step(
    run_id: int,
    req: StepUpdateRequest,
    user_id: int = Depends(get_current_user),
):
    """Update one step's state. Optionally bump `current_step`."""
    _owned(run_id, user_id)
    try:
        row = update_step(
            run_id=run_id,
            step_id=req.step_id,
            status=req.status,
            data=req.data,
            advance_to=req.advance_to,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not row:
        raise HTTPException(status_code=500, detail="Step update failed")
    return _serialize(row)


@router.patch("/{run_id}/links")
def patch_links(
    run_id: int,
    req: LinksUpdateRequest,
    user_id: int = Depends(get_current_user),
):
    """Attach rule_id to the run. Called by Pipeline A's step 2A once
    the rule has been generated and persisted."""
    _owned(run_id, user_id)
    row = set_run_links(run_id, rule_id=req.rule_id)
    if not row:
        raise HTTPException(status_code=500, detail="Link update failed")
    return _serialize(row)


@router.post("/{run_id}/complete")
def finish_run(run_id: int, user_id: int = Depends(get_current_user)):
    """Mark the run completed. Pipeline A's Finish also calls
    /ai/embed-resources externally (separate concern — the wizard
    decides what to do at the end of its flow)."""
    _owned(run_id, user_id)
    row = complete_run(run_id)
    if not row:
        raise HTTPException(status_code=500, detail="Complete failed")
    return _serialize(row)


@router.delete("/{run_id}")
def abandon_run(run_id: int, user_id: int = Depends(get_current_user)):
    """Hard-delete a run. The rule / CEs / test sets it produced stay
    around — those have their own lifecycle and the user typically
    wants to keep them."""
    ok = delete_pipeline_run(run_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return {"success": True}
