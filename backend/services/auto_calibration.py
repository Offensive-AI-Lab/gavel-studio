"""Auto-calibration: run the post-training chain the moment training ends.

    training → calibration → evaluation

Stage 2 (evaluation) runs inline on the calibration thread, so the stages are
serialized and the Evaluation page's existing per-stage phase lines read as one
pipeline: "Calibrating… <phase>", then "Evaluating… <phase>".


Calibration takes zero user decisions — it's a threshold × patience sweep over
data the studio already has — yet monitoring is statistically meaningless
without it (an uncalibrated session falls back to 0.5 for every CE). Leaving it
as a manual click meant it got forgotten, so training now chains straight into
it.

Two callers, one per training-completion path:
  * `classifier_engine.trainer.run_training` — the in-process/local provider.
  * `routes.classifiers.get_training_status` — the poll that finalizes a
    cluster / remote-worker job.
Both call `schedule_post_training_calibration`, which is idempotent: a second
call while a run is in flight (or after one already succeeded) is a no-op, so
the two paths can't double-submit.

Kept out of `routes/` on purpose: the trainer imports this, and importing a
route module from the trainer would drag FastAPI into the compute path.
"""

import json
import logging
import os
import threading

from utils.sqlite_db import execute_query_dict

logger = logging.getLogger(__name__)


def _auto_calibration_enabled() -> bool:
    """AUTO_CALIBRATE_AFTER_TRAINING=0 turns the chain off (default: on).

    Exists for callers that drive `run_training` directly and don't want a
    second GPU workload starting behind them — the integration suite sets it,
    and an operator debugging a training run can too.
    """
    return _env_on("AUTO_CALIBRATE_AFTER_TRAINING")


def _auto_evaluation_enabled() -> bool:
    """AUTO_EVALUATE_AFTER_CALIBRATION=0 stops the chain after calibration."""
    return _env_on("AUTO_EVALUATE_AFTER_CALIBRATION")


def _env_on(name: str) -> bool:
    return os.getenv(name, "1").strip().lower() not in ("0", "false", "no", "off")


def calibration_state(classifier_id: int) -> str:
    """One of 'calibrated' | 'running' | 'missing', for the CURRENT training.

    Rows from before the guardrail's last training don't count — retraining
    moves the model, so old thresholds are stale by definition. That cutoff is
    `_POST_TRAIN_CLAUSE`, shared with the evaluation routes so the two can
    never disagree about what "calibrated" means.
    """
    from routes.evaluation import _POST_TRAIN_CLAUSE

    rows = execute_query_dict(
        f"""SELECT eval_type FROM evaluation_results
           WHERE classifier_id = %s
             AND eval_type IN ('calibration', 'calibration_running')
             AND (eval_type = 'calibration_running' OR thresholds IS NOT NULL)
             {_POST_TRAIN_CLAUSE}
           ORDER BY created_at DESC, eval_id DESC""",
        (classifier_id, classifier_id),
    ) or []

    types = {r["eval_type"] for r in rows}
    if "calibration" in types:
        return "calibrated"
    if "calibration_running" in types:
        return "running"
    return "missing"


def is_calibrated(classifier_id: int) -> bool:
    return calibration_state(classifier_id) == "calibrated"


def chain_progress(classifier_id: int) -> dict | None:
    """The post-training stage in flight, or None. Shape: {phase, detail}.

    Training's own progress banner goes quiet the moment status flips to
    'active', which left the auto-chain invisible: the user watched
    "Extracting embeddings…", then nothing, while calibration ran for minutes.
    This lets the rule-set page keep that banner alive through "Calibrating"
    and "Evaluating".

    Reads the same `*_running` marker rows the Evaluation page already polls —
    no new bookkeeping, and `metrics.phase` supplies the live sub-step
    ("Loading calibration datasets…") for free. Taking the LATEST row per kind
    is what makes a finished or failed run stop reporting as running.
    """
    from routes.evaluation import _POST_TRAIN_CLAUSE

    def _latest(kinds):
        rows = execute_query_dict(
            f"""SELECT eval_type, metrics FROM evaluation_results
               WHERE classifier_id = %s AND eval_type = ANY(%s)
                 {_POST_TRAIN_CLAUSE}
               ORDER BY created_at DESC, eval_id DESC LIMIT 1""",
            (classifier_id, list(kinds), classifier_id),
        ) or []
        return rows[0] if rows else None

    def _detail(row, fallback):
        metrics = row.get("metrics")
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except (ValueError, TypeError):
                metrics = None
        return (metrics or {}).get("phase") or fallback

    try:
        cal = _latest(("calibration", "calibration_error", "calibration_running"))
        if cal and cal["eval_type"] == "calibration_running":
            return {"phase": "Calibrating",
                    "detail": _detail(cal, "Tuning per-CE thresholds…")}

        ev = _latest(("evaluation", "evaluation_error", "evaluation_running"))
        if ev and ev["eval_type"] == "evaluation_running":
            return {"phase": "Evaluating",
                    "detail": _detail(ev, "Scoring the rules against their test sets…")}
    except Exception as err:
        logger.error(f"[auto-calibration] chain progress failed for {classifier_id}: {err}")
    return None


def policy_drifted(classifier_id: int) -> bool:
    """True when the live policy differs from the one the model was trained on.

    `calibration_state` can't see this: a calibration row stays 'post-train'
    forever, so editing a rule set's rules/CEs WITHOUT retraining leaves stale
    thresholds looking perfectly valid. The model doesn't know the new CEs and
    the thresholds were tuned for the old ones — monitoring that is the same
    class of silently-meaningless output as monitoring uncalibrated.

    Delegates to `reconcile_classifier_status`, which compares the live policy
    fingerprint against the trained snapshot (and self-heals the stored status),
    so drift means exactly what the rest of the app means by it.
    """
    try:
        from sql_scripts.model_scripts import reconcile_classifier_status
        return reconcile_classifier_status(classifier_id) == "needs_retraining"
    except Exception as err:
        # Never let a drift-check failure block monitoring outright — fall back
        # to the calibration check alone.
        logger.error(f"[auto-calibration] drift check failed for {classifier_id}: {err}")
        return False


def run_post_calibration_evaluation(classifier_id: int) -> bool:
    """Stage 2 of the train → calibrate → evaluate chain. Returns True if it ran.

    Runs INLINE on the calibration worker thread rather than spawning another:
    both stages are heavy GPU work, and serializing them is the point — the
    user watches one stage finish and the next begin.

    Skipped when: disabled by env, already evaluated for this training, or the
    rule set has no default test sets to evaluate against (a guardrail whose
    rules ship no positive/negative sets — nothing to measure, and a noisy
    error row would be worse than silence).
    """
    try:
        if not _auto_evaluation_enabled():
            logger.info(f"[auto-evaluation] classifier {classifier_id}: disabled by env")
            return False

        from routes.evaluation import (
            _has_post_train_success, _load_default_eval_pairs, _run_evaluation,
        )

        if _has_post_train_success(classifier_id, "evaluation"):
            logger.info(f"[auto-evaluation] classifier {classifier_id}: skipped (already evaluated)")
            return False

        dataset_pairs = _load_default_eval_pairs(classifier_id)
        if not dataset_pairs:
            logger.info(f"[auto-evaluation] classifier {classifier_id}: skipped (no test sets)")
            return False

        logger.info(f"[auto-evaluation] classifier {classifier_id}: started")
        _run_evaluation(classifier_id, dataset_pairs)
        return True
    except Exception as err:
        # Never propagate: this runs at the tail of the calibration worker, and
        # a failed evaluation must not invalidate a good calibration.
        logger.error(f"[auto-evaluation] classifier {classifier_id} failed: {err}")
        return False


def schedule_post_training_calibration(classifier_id: int) -> bool:
    """Start calibration in a daemon thread. Returns True if it was started.

    No-ops when the guardrail is already calibrated or a run is in flight.
    Best-effort by design: a failure here must never turn a successful
    training run into a failed one, so everything is caught and logged. The
    user can always fall back to the manual Recalibrate button.
    """
    try:
        if not _auto_calibration_enabled():
            logger.info(
                f"[auto-calibration] classifier {classifier_id}: disabled by env"
            )
            return False
        state = calibration_state(classifier_id)
        if state != "missing":
            logger.info(
                f"[auto-calibration] classifier {classifier_id}: skipped ({state})"
            )
            return False

        from routes.evaluation import _run_calibration

        def _worker():
            try:
                _run_calibration(classifier_id)
            except Exception as err:
                # _run_calibration writes its own calibration_error row; this
                # only catches a failure to even get that far.
                logger.error(
                    f"[auto-calibration] classifier {classifier_id} failed: {err}"
                )
                return
            # Stage 2 of the chain. Only on a real success — evaluation reads
            # the calibrated thresholds, so running it after a failed sweep
            # would just fail again with a worse message.
            if calibration_state(classifier_id) == "calibrated":
                run_post_calibration_evaluation(classifier_id)

        threading.Thread(
            target=_worker,
            name=f"auto-calibrate-{classifier_id}",
            daemon=True,
        ).start()
        logger.info(f"[auto-calibration] classifier {classifier_id}: started")
        return True
    except Exception as err:
        logger.error(
            f"[auto-calibration] could not schedule for classifier {classifier_id}: {err}"
        )
        return False
