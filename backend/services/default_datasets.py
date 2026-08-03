"""Rule-level *default* test/calibration dataset generation.

Every rule carries a default test set generated at rule-creation time:
three `test_datasets` rows tagged `is_default = TRUE` and keyed by
`rule_id` (user_id NULL; test sets are rule-scoped, not guardrail-scoped):

    * positive             — positive test bucket
    * negative             — hard-negative test bucket
    * positive_calibration — usecase-level calibration bucket

These are the canonical, shared sets; library rules ship theirs in the
gavel-rules registry (rules/<name>/tests/, pulled lazily by
services/library_sync.py). A user who wants different
coverage generates their own *private* set instead (is_default = FALSE,
user_id set) via the existing /ai/test-set/generate path.

Generation is fire-and-forget: `generate_rule_defaults` returns
immediately after spawning a daemon thread. The thread builds the
positive config (LLM), derives the negative config (LLM), upserts the
three rows, then fills each with judged dialogues. Idempotent — re-running
for the same rule UPSERTs onto the existing default rows (partial unique
index `uq_default_per_rule_type`).

The heavy lifting reuses the existing reference-parity helpers in
routes/ai_pipeline.py (`build_positive_config`, `build_negative_config`,
`_run_test_generation` → `_generate_judged_dialogues`). Imports are lazy
to avoid a circular import (ai_pipeline triggers this module, and this
module calls back into ai_pipeline's generators).
"""
import json
import threading
from typing import Optional

from utils.sqlite_db import execute_query, execute_query_dict

# The three buckets that make up a rule's default test set.
DEFAULT_DATASET_TYPES = ("positive", "negative", "positive_calibration")

# Public name for a rule's DEFAULT (HF-published) test set. Reserved — users
# cannot name a private custom set this (enforced in /ai/test-set/generate).
DEFAULT_TEST_SET_NAME = "Test Set"


def _upsert_default_row(rule_id: int, dataset_type: str, config: dict) -> int:
    """Insert (or reset) the default row for (rule_id, dataset_type).

    Relies on the partial unique index `uq_default_per_rule_type` so a
    regeneration overwrites the existing default rather than duplicating
    it. Registry columns (public_id/published_at) are intentionally NOT
    cleared on conflict — the row identity (rule + type) is stable, and a
    republish refreshes the HF content under the same id.
    """
    rows = execute_query_dict(
        """
        INSERT INTO test_datasets
            (rule_id, user_id, is_default, dataset_type, scenario_name,
             config, status, generation_log)
        VALUES (%s, NULL, TRUE, %s, %s, %s::jsonb, 'generating',
                'Starting generation...')
        ON CONFLICT (rule_id, dataset_type) WHERE is_default = TRUE
        DO UPDATE SET config = EXCLUDED.config,
                      scenario_name = EXCLUDED.scenario_name,
                      status = 'generating',
                      conversations = NULL,
                      generation_log = 'Regenerating...'
        RETURNING dataset_id
        """,
        (rule_id, dataset_type, DEFAULT_TEST_SET_NAME, json.dumps(config).replace("\\u0000", "")),
    )
    return rows[0]["dataset_id"]


def _mark_row_error(dataset_id: int, message: str) -> None:
    try:
        execute_query(
            "UPDATE test_datasets SET status = 'error', generation_log = %s WHERE dataset_id = %s",
            (message[:500], dataset_id),
        )
    except Exception:
        pass


def rule_defaults_ready(rule_id: int) -> bool:
    """True only when all three default buckets exist and are 'ready'.

    Consumed by the publish gate (a rule can't publish until its default
    set is fully generated) and by the frontend status poll.
    """
    rows = execute_query_dict(
        "SELECT dataset_type, status FROM test_datasets WHERE rule_id = %s AND is_default = TRUE",
        (rule_id,),
    ) or []
    by_type = {r["dataset_type"]: r["status"] for r in rows}
    return all(by_type.get(t) == "ready" for t in DEFAULT_DATASET_TYPES)


def rule_defaults_status(rule_id: int) -> dict:
    """Per-bucket status map for the frontend, plus a rolled-up state.

    Rolled-up `state` is one of: 'missing' (no rows yet), 'generating'
    (some not ready, none errored), 'error' (any errored), 'ready' (all
    three ready).

    Failure reasons: an errored bucket's entry carries an `error` field (its
    generation_log, which holds the failure message written by
    _mark_row_error / _run_test_generation), and the top level gets a
    rolled-up `error` string. The frontend tray chip surfaces it so the user
    sees WHY generation failed. Both fields are absent when nothing errored.
    """
    rows = execute_query_dict(
        "SELECT dataset_id, dataset_type, status, generation_log FROM test_datasets "
        "WHERE rule_id = %s AND is_default = TRUE",
        (rule_id,),
    ) or []
    by_type = {r["dataset_type"]: r["status"] for r in rows}
    if not rows:
        state = "missing"
    elif any(s == "error" for s in by_type.values()):
        state = "error"
    elif all(by_type.get(t) == "ready" for t in DEFAULT_DATASET_TYPES):
        state = "ready"
    else:
        state = "generating"
    datasets = []
    errors = []
    for r in rows:
        entry = {"dataset_id": r["dataset_id"], "dataset_type": r["dataset_type"], "status": r["status"]}
        if r["status"] == "error":
            reason = r.get("generation_log") or "generation failed"
            entry["error"] = reason
            errors.append(f"{r['dataset_type']}: {reason}")
        datasets.append(entry)
    result = {"rule_id": rule_id, "state": state, "datasets": datasets}
    if errors:
        result["error"] = "; ".join(errors)
    return result


def _flip_visibility(rule_id: int, finalize_ce_ids: Optional[list] = None) -> None:
    """Flip the rule (and any deferred new CEs) to is_ready=TRUE — the moment
    they become visible in Drafts/Browse/bookmarks/the CE picker."""
    if finalize_ce_ids:
        execute_query(
            "UPDATE cognitive_elements SET is_ready = TRUE WHERE ce_id = ANY(%s)",
            (finalize_ce_ids,),
        )
    execute_query(
        "UPDATE rules SET is_ready = TRUE WHERE rule_id = %s",
        (rule_id,),
    )


def _run_rule_defaults(
    rule_id: int,
    scenario_instructions: str,
    target_count: int,
    calibration_count: int,
    finalize_ce_ids: Optional[list] = None,
) -> None:
    """Daemon-thread body: build configs, then generate the 3 buckets.

    `finalize_ce_ids` (when provided) flips the rule + those CEs to
    is_ready=TRUE once the buckets are generated — keeping them hidden
    everywhere until the rule is fully built.
    """
    from routes.ai_pipeline import (
        build_positive_config,
        build_negative_config,
        _run_test_generation,
    )

    # Create the three rows up front (status='generating') so the UI and
    # the publish gate see a deterministic "3 expected, generating" state
    # immediately — before the slow config LLM calls return.
    placeholder = {"scenario_instructions": scenario_instructions}
    pos_id = _upsert_default_row(rule_id, "positive", placeholder)
    cal_id = _upsert_default_row(rule_id, "positive_calibration", placeholder)
    neg_id = _upsert_default_row(rule_id, "negative", placeholder)

    # --- positive config (drives positive + positive_calibration) ---
    try:
        pos_config = build_positive_config(scenario_instructions)
        # Guarantee the scenario is embedded in the persisted config — this
        # is the provenance + regeneration source now that `scenarios` is gone.
        pos_config["scenario_instructions"] = (
            pos_config.get("scenario_instructions") or scenario_instructions
        )
    except Exception as e:
        for did in (pos_id, cal_id, neg_id):
            _mark_row_error(did, f"positive config generation failed: {e}")
        # Terminal failure — reveal the rule rather than stranding it hidden
        # (see the failure-reveal comment at the end of this function).
        try:
            _flip_visibility(rule_id, finalize_ce_ids)
        except Exception as flip_err:
            print(f"[default_datasets] failure-reveal failed for rule {rule_id}: {flip_err}")
        return
    pos_json = json.dumps(pos_config).replace("\\u0000", "")
    execute_query(
        "UPDATE test_datasets SET config = %s::jsonb WHERE dataset_id = ANY(%s)",
        (pos_json, [pos_id, cal_id]),
    )

    # --- negative config (polar-context derivation) ---
    neg_config = None
    try:
        neg_config, _reasoning = build_negative_config(pos_config)
        neg_config["scenario_instructions"] = (
            neg_config.get("scenario_instructions") or pos_config["scenario_instructions"]
        )
        execute_query(
            "UPDATE test_datasets SET config = %s::jsonb WHERE dataset_id = %s",
            (json.dumps(neg_config).replace("\\u0000", ""), neg_id),
        )
    except Exception as e:
        _mark_row_error(neg_id, f"negative config generation failed: {e}")

    # --- dialogue generation (sequential; each call flips status to ready) ---
    _run_test_generation(pos_id, pos_config, target_count, "positive")
    _run_test_generation(cal_id, pos_config, calibration_count, "positive_calibration")
    if neg_config is not None:
        _run_test_generation(neg_id, neg_config, target_count, "negative")

    # Visibility finalize: the rule + its new CEs are kept is_ready=FALSE
    # (hidden in Browse / Drafts / bookmarks / CE picker everywhere) until the
    # whole default test/calibration set is built — so a rule never appears
    # half-finished. Flip them ready now that generation is done.
    #
    # FAILURE reveal: if any bucket errored, the rule is revealed REGARDLESS of
    # which flow owns the finalize (the AI pipeline passes finalize_ce_ids; the
    # build-from-CEs wizard finalizes from the frontend, which never runs on
    # failure). A rule left is_ready=FALSE would be invisible everywhere and
    # silently DELETED by boot-time crash recovery. Same philosophy as the
    # embed path: don't strand the rule hidden forever — it's still usable,
    # and the user can regenerate its sets (retry from the tray chip, or Edit
    # the revealed rule). The status endpoint carries the per-bucket reason.
    failed = rule_defaults_status(rule_id).get("state") == "error"
    if failed or finalize_ce_ids is not None:
        try:
            _flip_visibility(rule_id, finalize_ce_ids)
        except Exception as e:
            print(f"[default_datasets] is_ready finalize failed for rule {rule_id}: {e}")


def generate_rule_defaults(
    rule_id: int,
    scenario_instructions: str,
    target_count: int = 100,
    calibration_count: int = 50,
    finalize_ce_ids: Optional[list] = None,
) -> dict:
    """Kick off default-set generation for `rule_id` in the background.

    Returns immediately; poll `rule_defaults_status(rule_id)` for progress.
    Safe to call again to regenerate (idempotent upsert per bucket).

    `finalize_ce_ids`: when the AI pipeline passes the rule's CE ids here, the
    background thread flips the rule + those CEs to is_ready=TRUE only after the
    set is built — so the rule stays hidden everywhere until it's complete.
    """
    if not scenario_instructions or not scenario_instructions.strip():
        raise ValueError("scenario_instructions is required to generate rule defaults")

    threading.Thread(
        target=_run_rule_defaults,
        args=(rule_id, scenario_instructions, target_count, calibration_count, finalize_ce_ids),
        daemon=True,
    ).start()
    return {"success": True, "rule_id": rule_id, "state": "generating"}
