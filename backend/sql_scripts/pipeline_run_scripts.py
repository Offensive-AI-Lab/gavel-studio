"""SQL helpers for the `pipeline_runs` table.

A `pipeline_run` row is one in-progress (or completed) walk through one
of two reference-parity flows:

* **Pipeline A — Rule Generation** (`pipeline_type='rule'`): steps 1,
  2A, 2B, 2C. Guardrail-agnostic — produces public-library artifacts
  (a rule + its CEs with training and calibration data).
* **Pipeline B — Test + Evaluation** (`pipeline_type='test_eval'`):
  steps 3A, 3B, 3C, 3D, plus a Calibrate and an Evaluate step. Per-
  guardrail, per-rule — produces test sets, calibrated thresholds, and
  an evaluation result for one rule in a specific guardrail.

The frontend posts a new row when the user enters either wizard, then
PATCHes per-step state as they advance. A run is "active" while
`completed = FALSE`; the wizard offers to resume the most recent active
run for a given (user, pipeline_type, guardrail) tuple on next visit.

The `steps` JSONB column holds a step-id-keyed map. Each entry's
shape is `{status: pending|in_progress|completed|skipped|error,
data: {...step-specific fields...}}`. The backend doesn't validate the
inner shape — that's the wizard's job. Treating it as opaque keeps
step evolution cheap (no migration for a new step-data field).
"""
from typing import Optional, List, Dict, Any
import json

from utils.sqlite_db import execute_query, execute_query_dict


# Step ids per pipeline. The wizard frontend imports its own copy of
# these lists for the sidebar; the backend just uses them to (a) seed
# the initial `steps` map and (b) validate the `step_id` parameter
# coming in on PATCH calls.
_STEP_IDS_RULE = ("1", "2A", "2B", "2C", "2D")
# Test/Eval (Pipeline B): the user-authored Define step (positive + negative
# instructions + counts) replaced the old 3A/3B/3C/3D config ceremony.
_STEP_IDS_TEST_EVAL = ("define", "cal", "eval")
# CE generation (Pipeline C): mirrors the rule wizard for a single Cognitive
# Element — ideation, generation, training (excitation), calibration.
_STEP_IDS_CE = ("1", "2.1", "2.2", "2.3")

_VALID_PIPELINE_TYPES = ("rule", "test_eval", "ce")

# Union of every legal step id — used by update_step() to validate a
# `step_id` from the route layer (we cross-check against the actual
# pipeline_type below, but this catches typos quickly).
_ALL_STEP_IDS = _STEP_IDS_RULE + _STEP_IDS_TEST_EVAL + _STEP_IDS_CE

# What step a fresh run lands on, by pipeline_type.
_FIRST_STEP = {
    "rule": "1",
    "test_eval": "define",
    "ce": "1",
}

# All columns selected on read — kept as a constant so every helper
# returns the same shape (matters for the wizard's setRun() merges).
_COLS = (
    "run_id, user_id, classifier_id, rule_id, "
    "pipeline_type, current_step, steps, completed, created_at, updated_at"
)


def _step_ids_for(pipeline_type: str) -> tuple:
    if pipeline_type == "rule":
        return _STEP_IDS_RULE
    if pipeline_type == "test_eval":
        return _STEP_IDS_TEST_EVAL
    if pipeline_type == "ce":
        return _STEP_IDS_CE
    raise ValueError(f"Unknown pipeline_type: {pipeline_type}")


def _default_steps_state(pipeline_type: str) -> Dict[str, Dict[str, Any]]:
    """Initial steps map for the given flavor. Every step starts
    pending with empty data."""
    return {sid: {"status": "pending", "data": {}} for sid in _step_ids_for(pipeline_type)}


def create_pipeline_run(
    user_id: int,
    pipeline_type: str = "rule",
    classifier_id: Optional[int] = None,
    rule_id: Optional[int] = None,
) -> Dict:
    """Open a new wizard run. Returns the new row.

    * `pipeline_type='rule'` runs MUST omit classifier_id — the rule is
      a library artifact and isn't bound to any guardrail yet.
    * `pipeline_type='test_eval'` runs MUST set classifier_id (the test
      sets are per-guardrail) and SHOULD set rule_id (the wizard is
      always scoped to a single rule).
    """
    if pipeline_type not in _VALID_PIPELINE_TYPES:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}")
    if pipeline_type == "test_eval" and classifier_id is None:
        raise ValueError("test_eval runs require a classifier_id")

    first_step = _FIRST_STEP[pipeline_type]
    rows = execute_query_dict(
        f"""
        INSERT INTO pipeline_runs
            (user_id, classifier_id, rule_id, pipeline_type, current_step, steps)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        RETURNING {_COLS}
        """,
        (
            user_id, classifier_id, rule_id,
            pipeline_type, first_step, json.dumps(_default_steps_state(pipeline_type)),
        ),
    )
    return rows[0] if rows else None


def get_pipeline_run(run_id: int) -> Optional[Dict]:
    """Fetch a run by id. Ownership is checked at the route layer."""
    rows = execute_query_dict(
        f"SELECT {_COLS} FROM pipeline_runs WHERE run_id = %s",
        (run_id,),
    )
    return rows[0] if rows else None


def get_active_runs(
    user_id: int,
    classifier_id: Optional[int] = None,
    pipeline_type: Optional[str] = None,
    rule_id: Optional[int] = None,
) -> List[Dict]:
    """List the user's in-progress runs (completed=FALSE).

    Filters compose: pass `pipeline_type='test_eval', classifier_id=N,
    rule_id=M` to find an active Pipeline-B run for one specific rule.
    Pass none of them to get every active run across both flavors (used
    by the global nav badge).
    """
    where = ["user_id = %s", "completed = FALSE"]
    params: List[Any] = [user_id]
    if classifier_id is not None:
        where.append("classifier_id = %s")
        params.append(classifier_id)
    if pipeline_type is not None:
        if pipeline_type not in _VALID_PIPELINE_TYPES:
            raise ValueError(f"Unknown pipeline_type: {pipeline_type}")
        where.append("pipeline_type = %s")
        params.append(pipeline_type)
    if rule_id is not None:
        where.append("rule_id = %s")
        params.append(rule_id)

    return execute_query_dict(
        f"SELECT {_COLS} FROM pipeline_runs WHERE {' AND '.join(where)} ORDER BY updated_at DESC",
        tuple(params),
    ) or []


def update_step(
    run_id: int,
    step_id: str,
    status: str,
    data: Optional[Dict[str, Any]] = None,
    advance_to: Optional[str] = None,
) -> Optional[Dict]:
    """Merge a step's status+data into the run's steps JSONB.

    `advance_to` optionally bumps `current_step`. The wizard hand-rolls
    "user clicked Next" instead of inferring advancement from status
    changes — a user can revisit a completed step without leaving it.

    SQLite's `json_set` mutates a single key without rewriting the whole
    blob (it creates the key if missing, like jsonb_set's create_missing),
    so concurrent step updates for different keys don't clobber each other.
    """
    if step_id not in _ALL_STEP_IDS:
        raise ValueError(f"Unknown step id: {step_id}")

    new_value = json.dumps({"status": status, "data": data or {}})
    # step_id comes from the _ALL_STEP_IDS whitelist, so quoting it into a
    # JSON path is safe. Quotes keep ids like "2A" working as object keys.
    step_path = f'$."{step_id}"'

    if advance_to is not None and advance_to not in _ALL_STEP_IDS:
        raise ValueError(f"Unknown step id for advance_to: {advance_to}")

    if advance_to is not None:
        rows = execute_query_dict(
            f"""
            UPDATE pipeline_runs
            SET steps = json_set(steps, %s, json(%s)),
                current_step = %s,
                updated_at = now()
            WHERE run_id = %s
            RETURNING {_COLS}
            """,
            (step_path, new_value, advance_to, run_id),
        )
    else:
        rows = execute_query_dict(
            f"""
            UPDATE pipeline_runs
            SET steps = json_set(steps, %s, json(%s)),
                updated_at = now()
            WHERE run_id = %s
            RETURNING {_COLS}
            """,
            (step_path, new_value, run_id),
        )
    return rows[0] if rows else None


def set_run_links(
    run_id: int,
    rule_id: Optional[int] = None,
) -> Optional[Dict]:
    """Attach the rule_id FK once it exists. Pipeline A sets it after
    step 2A completes. Pipeline B has it populated at create time (the
    wizard enters with a rule in mind).

    Pass None to leave the field unchanged.
    """
    parts: List[str] = []
    params: List[Any] = []
    if rule_id is not None:
        parts.append("rule_id = %s")
        params.append(rule_id)
    if not parts:
        return get_pipeline_run(run_id)
    params.append(run_id)
    rows = execute_query_dict(
        f"""
        UPDATE pipeline_runs
        SET {', '.join(parts)}, updated_at = now()
        WHERE run_id = %s
        RETURNING {_COLS}
        """,
        tuple(params),
    )
    return rows[0] if rows else None


def complete_run(run_id: int) -> Optional[Dict]:
    """Mark a run completed. The row stays around — useful for
    auditing and for "show me my last 5 finished runs"."""
    rows = execute_query_dict(
        f"""
        UPDATE pipeline_runs
        SET completed = TRUE, updated_at = now()
        WHERE run_id = %s
        RETURNING {_COLS}
        """,
        (run_id,),
    )
    return rows[0] if rows else None


def delete_pipeline_run(run_id: int, user_id: int) -> bool:
    """Hard-delete a run the user owns. Returns True on success.

    We don't cascade to the run's rule / CEs / test sets — those have
    their own lifecycle and are typically what the user wants to
    *keep* even when they abandon the wizard partway through.
    """
    rows = execute_query(
        "DELETE FROM pipeline_runs WHERE run_id = %s AND user_id = %s RETURNING run_id",
        (run_id, user_id),
    )
    return bool(rows)
