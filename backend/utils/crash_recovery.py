"""Crash recovery — runs on server startup to clean up stale/orphaned state.

Implemented as a Strategy pattern (GoF): each `RecoveryStrategy` subclass owns
one isolated recovery workflow. The orchestrator iterates a list of strategies
without caring what any of them does, and one strategy failing never prevents
the next one from running.

Adding a new recovery type:
  1. Subclass `RecoveryStrategy` with a `name` and a `run()` method.
  2. Append an instance to `RECOVERY_STRATEGIES`.
No changes to `run_all_recovery` or to existing strategies are required.

Currently registered:
  1. Guardrails stuck in 'training' status (server crashed during training)
     → Reset to previous status, delete partial training files
  2. Test datasets stuck in 'generating' status (server crashed during generation)
     → Mark as 'error'
  3. Orphaned guardrail directories with no matching DB record
     → Delete files
"""
from abc import ABC, abstractmethod
from typing import List
import logging
import os
import shutil

logger = logging.getLogger(__name__)

TRAINED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_classifiers")


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class RecoveryStrategy(ABC):
    """One crash-recovery action.

    Subclasses implement `run()`. They may raise on internal errors — the
    orchestrator (`run_all_recovery`) catches and logs them so a single
    strategy's failure cannot cascade and prevent the others from running.
    """

    #: Human-readable label used in log messages.
    name: str = ""

    #: True for strategies that touch ONLY DB state (no `transformers`/`torch`
    #: imports) and therefore can run EARLY on boot — before the heavy model
    #: warmup — without racing its `from transformers import ...`. These are the
    #: strategies that clear phantom "Running…"/"Generating…" UI markers, so we
    #: want them to fire promptly after a crash-restart instead of waiting out
    #: the ~30s warmup. Heavy strategies (training reset imports the trainer,
    #: which pulls transformers) leave this False and run only post-warmup.
    safe_before_warmup: bool = False

    @abstractmethod
    def run(self) -> None: ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class StuckTrainingRecovery(RecoveryStrategy):
    """Reset guardrails stuck in 'training' status and clean up partial files.

    If a guardrail is in 'training' but the server restarted, the background
    task is dead. Reset to 'error' (or 'needs_retraining' if a previously
    trained model exists on disk) and delete any partial training artifacts.
    """

    name = "training"
    # DB + filesystem only — runs in EARLY recovery (before the model warmup) so
    # a crash-restart clears a stuck 'training' status (and the frontend's
    # "Training…" badge) promptly, instead of only after the ~30s warmup — or
    # never, if the warmup hangs. We resolve the workdir from user_id directly
    # (below) rather than importing classifier_engine.trainer, which would pull
    # transformers and reintroduce the import-race that kept this strategy late.
    safe_before_warmup = True

    def run(self) -> None:
        from utils.sqlite_db import execute_query, execute_query_dict

        # Join target_models for user_id so the on-disk workdir
        # (trained_classifiers/<user_id>/classifier_<id>/) is locatable WITHOUT
        # importing the trainer (= no transformers import → safe pre-warmup).
        stuck = execute_query_dict(
            """SELECT c.classifier_id, c.model_path, tm.user_id
               FROM classifiers c
               LEFT JOIN target_models tm ON c.model_id = tm.model_id
               WHERE c.status = 'training'"""
        ) or []

        for row in stuck:
            cid = row["classifier_id"]
            logger.warning(f"[Recovery] Classifier {cid} stuck in 'training' — resetting to 'error'")

            user_id = row.get("user_id")
            work_dir = (
                os.path.join(TRAINED_MODELS_DIR, str(user_id), f"classifier_{cid}")
                if user_id is not None else None
            )

            if work_dir and os.path.isdir(work_dir):
                # If there's a valid trained model on disk, preserve it and let
                # the user decide whether to retrain.
                rnn_path = os.path.join(work_dir, "trained_rnn.pth")
                meta_path = os.path.join(work_dir, "classifier_meta.json")
                if os.path.exists(rnn_path) and os.path.exists(meta_path):
                    execute_query(
                        "UPDATE classifiers SET status = 'needs_retraining', training_log = %s, "
                        "training_phase = NULL, training_phase_detail = NULL WHERE classifier_id = %s",
                        ("Server restarted during training. Previous model preserved.", cid),
                    )
                    logger.info(f"[Recovery] Classifier {cid}: previous model found, set to 'needs_retraining'")
                else:
                    try:
                        shutil.rmtree(work_dir)
                        logger.info(f"[Recovery] Deleted partial training dir: {work_dir}")
                    except Exception as e:
                        logger.error(f"[Recovery] Failed to delete {work_dir}: {e}")

                    execute_query(
                        "UPDATE classifiers SET status = 'error', model_path = NULL, training_log = %s, "
                        "training_phase = 'failed', training_phase_detail = %s WHERE classifier_id = %s",
                        ("Training interrupted by server restart. Please retrain.",
                         "Training interrupted by server restart. Please retrain.", cid),
                    )
            else:
                execute_query(
                    "UPDATE classifiers SET status = 'error', model_path = NULL, training_log = %s, "
                    "training_phase = 'failed', training_phase_detail = %s WHERE classifier_id = %s",
                    ("Training interrupted by server restart. Please retrain.",
                     "Training interrupted by server restart. Please retrain.", cid),
                )

        if stuck:
            logger.info(f"[Recovery] Recovered {len(stuck)} stuck training classifier(s)")


class StuckTestGenerationRecovery(RecoveryStrategy):
    """Mark test datasets stuck in 'generating' as 'error'.

    These were daemon threads that died with the server.
    """

    name = "test-generation"
    safe_before_warmup = True  # DB-only; clears a phantom "Generating…" marker

    def run(self) -> None:
        from utils.sqlite_db import execute_query, execute_query_dict

        stuck = execute_query_dict(
            "SELECT dataset_id FROM test_datasets WHERE status = 'generating'"
        ) or []

        for row in stuck:
            did = row["dataset_id"]
            logger.warning(f"[Recovery] Test dataset {did} stuck in 'generating' — marking as error")
            execute_query(
                "UPDATE test_datasets SET status = 'error', generation_log = %s WHERE dataset_id = %s",
                ("Generation interrupted by server restart.", did),
            )

        if stuck:
            logger.info(f"[Recovery] Recovered {len(stuck)} stuck test dataset(s)")


class StuckEvaluationRecovery(RecoveryStrategy):
    """Wipe any calibration/evaluation left 'running' by a crash.

    Calibration and evaluation run as background tasks (they may block-poll a
    remote inference job for minutes). If the server dies before the FULL
    result lands, the only trace is a 'calibration_running'/'evaluation_running'
    marker row — there is NO partial result, because the final row is written
    only on complete success. We DELETE those markers so the run looks like it
    never happened (re-runnable, button re-enabled), and best-effort cancel +
    clean the orphaned remote job whose pointer was stashed on the marker.
    """

    name = "evaluation"
    safe_before_warmup = True  # DB-only; clears phantom "Running…"

    def run(self) -> None:
        from utils.sqlite_db import execute_query, execute_query_dict

        stuck = execute_query_dict(
            "SELECT eval_id, classifier_id, eval_type, plots FROM evaluation_results "
            "WHERE eval_type IN ('calibration_running', 'evaluation_running')"
        ) or []

        for row in stuck:
            plots = row.get("plots") or {}
            pointer = plots.get("remote_job") if isinstance(plots, dict) else None
            if isinstance(pointer, dict) and pointer:
                try:
                    from services import compute
                    for p in compute.all_providers():
                        p.cancel_inference(pointer)
                    logger.info(
                        f"[Recovery] Cancelled orphaned inference job {pointer} "
                        f"for classifier {row.get('classifier_id')}"
                    )
                except Exception as e:
                    logger.warning(f"[Recovery] Could not cancel orphaned inference job: {e}")

        if stuck:
            execute_query(
                "DELETE FROM evaluation_results "
                "WHERE eval_type IN ('calibration_running', 'evaluation_running')"
            )
            logger.info(
                f"[Recovery] Cleared {len(stuck)} interrupted calibration/evaluation run(s) "
                f"(treated as never-run)"
            )


class IncompletePipelineRecovery(RecoveryStrategy):
    """Wipe rules / CEs that were generated but never fully finished.

    Both `cognitive_elements` and `rules` carry an `is_ready` boolean. Every
    generation flow — the AI rule pipeline, the CE pipeline, and the manual
    build-from-CEs wizard — inserts with FALSE and flips TRUE only after
    EVERYTHING has landed (training data, embeddings, AND the default
    test/calibration set). So a row still FALSE on boot means generation died
    mid-flight: the server crashed, the user closed the tab, the network or
    power dropped, or the OS killed the process. Those background threads do
    not survive a restart, so the row can never finish on its own.

    Product contract (per ofek): any such failure must look EXACTLY as if the
    rule/CE was never generated. So we delete every is_ready=FALSE row on
    startup — no exceptions. We also drop the matching unfinished rule/CE
    wizard runs (pipeline_runs, completed=FALSE) so nothing offers to "resume"
    into a draft that no longer exists. (This intentionally replaces the old
    resume-after-crash behavior, which kept active-run drafts alive.)

    Cascades take care of rule_ce_link / setup_ce_link / excitation_datasets /
    test_datasets; pipeline_runs.rule_id and rule_setup.rule_id are ON DELETE
    SET NULL, so deleting a referenced rule never throws.
    """

    name = "incomplete-pipeline-cleanup"
    # Pure DB deletes (no torch/transformers import) → run EARLY, before the
    # ~30s model warmup, so a half-generated rule/CE disappears from
    # Drafts/Browse the instant the backend is back, not 30s later.
    safe_before_warmup = True

    def run(self) -> None:
        from utils.sqlite_db import execute_query, execute_query_dict

        ce_rows = execute_query_dict(
            "SELECT ce_id, name FROM cognitive_elements WHERE is_ready = FALSE"
        ) or []
        for row in ce_rows:
            logger.warning(
                f"[Recovery] Wiping incomplete CE {row['ce_id']} ('{row['name']}') — generation never finished"
            )
            try:
                execute_query("DELETE FROM cognitive_elements WHERE ce_id = %s", (row["ce_id"],))
            except Exception as e:
                logger.error(f"[Recovery] Failed to delete CE {row['ce_id']}: {e}")

        rule_rows = execute_query_dict(
            "SELECT rule_id, name FROM rules WHERE is_ready = FALSE"
        ) or []
        for row in rule_rows:
            logger.warning(
                f"[Recovery] Wiping incomplete rule {row['rule_id']} ('{row['name']}') — generation never finished"
            )
            try:
                execute_query("DELETE FROM rules WHERE rule_id = %s", (row["rule_id"],))
            except Exception as e:
                logger.error(f"[Recovery] Failed to delete rule {row['rule_id']}: {e}")

        # Drop the unfinished rule/CE wizard sessions too, so the UI never
        # offers to resume into a now-deleted draft. Test/eval runs
        # (pipeline_type='test_eval') don't generate rules/CEs — leave them.
        try:
            execute_query(
                "DELETE FROM pipeline_runs "
                "WHERE completed = FALSE AND pipeline_type IN ('rule', 'ce')"
            )
        except Exception as e:
            logger.error(f"[Recovery] Failed to clear unfinished pipeline runs: {e}")

        if ce_rows or rule_rows:
            logger.info(
                f"[Recovery] Wiped {len(ce_rows)} incomplete CE(s) and {len(rule_rows)} incomplete rule(s)"
            )


class OrphanedClassifierDirRecovery(RecoveryStrategy):
    """Delete guardrail directories that have no matching DB record.

    Layout walked:  trained_classifiers/<user_id>/classifier_<classifier_id>/
    Anything that doesn't fit that pattern (or whose classifier_id is gone
    from the DB) is swept. Empty user-id folders left over after the sweep
    are also removed.
    """

    name = "orphan-cleanup"

    def run(self) -> None:
        from utils.sqlite_db import execute_query_dict

        if not os.path.isdir(TRAINED_MODELS_DIR):
            return

        rows = execute_query_dict("SELECT classifier_id FROM classifiers") or []
        valid_ids = {row["classifier_id"] for row in rows}

        for user_dirname in os.listdir(TRAINED_MODELS_DIR):
            user_path = os.path.join(TRAINED_MODELS_DIR, user_dirname)
            if not os.path.isdir(user_path):
                continue
            # User-id subfolders are integer-named; anything else (e.g. a
            # legacy flat classifier_<id> that survived a migration) gets
            # the same orphan check inline.
            if not user_dirname.isdigit():
                # Legacy flat classifier_<id> at the top level.
                if user_dirname.startswith("classifier_"):
                    try:
                        cid = int(user_dirname.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    if cid not in valid_ids:
                        logger.warning(f"[Recovery] Legacy orphan {user_path} — deleting")
                        try:
                            shutil.rmtree(user_path)
                        except Exception as e:
                            logger.error(f"[Recovery] Failed to delete {user_path}: {e}")
                continue

            for dirname in os.listdir(user_path):
                if not dirname.startswith("classifier_"):
                    continue
                try:
                    cid = int(dirname.split("_")[1])
                except (IndexError, ValueError):
                    continue

                if cid not in valid_ids:
                    orphan_path = os.path.join(user_path, dirname)
                    logger.warning(f"[Recovery] Orphaned directory {orphan_path} — deleting")
                    try:
                        shutil.rmtree(orphan_path)
                    except Exception as e:
                        logger.error(f"[Recovery] Failed to delete orphan {orphan_path}: {e}")

            # Tidy: remove a now-empty user folder so the disk doesn't
            # accumulate dead namespaces.
            try:
                if os.path.isdir(user_path) and not os.listdir(user_path):
                    os.rmdir(user_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Registry — open/closed extension point.
# ---------------------------------------------------------------------------


class BundleJobRecovery(RecoveryStrategy):
    """Reconcile export/import background jobs left 'running' by a dead process.

    A crashed import may have created a partial guardrail; this rolls it back
    (unless it actually reached 'active', in which case the import finished and
    only the job record was lost). DB + filesystem only, so it runs early.
    """

    name = "bundle-jobs"
    safe_before_warmup = True

    def run(self) -> None:
        from services.bundle_jobs import recover_interrupted_jobs
        recover_interrupted_jobs()


RECOVERY_STRATEGIES: List[RecoveryStrategy] = [
    StuckTrainingRecovery(),
    StuckTestGenerationRecovery(),
    StuckEvaluationRecovery(),
    IncompletePipelineRecovery(),
    OrphanedClassifierDirRecovery(),
    BundleJobRecovery(),
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_early_recovery() -> None:
    """Run ONLY the DB-state strategies that are safe before model warmup.

    Called synchronously-early on boot (off a tiny daemon thread that does NOT
    wait for the heavy warmup) so a crash-restart promptly clears phantom
    "Running…" / "Generating…" markers — otherwise the UI keeps showing a dead
    calibration/evaluation as if it were still in progress for the whole ~30s
    warmup window. Idempotent with run_all_recovery (which re-runs these as a
    no-op once the markers are already gone).
    """
    early = [s for s in RECOVERY_STRATEGIES if s.safe_before_warmup]
    if not early:
        return
    logger.info("[Recovery] Running early (pre-warmup) crash recovery...")
    for strategy in early:
        try:
            strategy.run()
        except Exception as e:
            logger.error(f"[Recovery] early {strategy.name} recovery failed: {e}")
    logger.info("[Recovery] Early crash recovery complete")


def run_all_recovery() -> None:
    """Run every registered recovery strategy. Called once on server startup.

    A strategy raising is logged and skipped — the next strategy still runs.
    The `safe_before_warmup` strategies may already have run via
    run_early_recovery(); re-running them here is a harmless no-op.
    """
    logger.info("[Recovery] Running crash recovery checks...")
    for strategy in RECOVERY_STRATEGIES:
        try:
            strategy.run()
        except Exception as e:
            logger.error(f"[Recovery] {strategy.name} recovery failed: {e}")
    logger.info("[Recovery] Crash recovery complete")


# ---------------------------------------------------------------------------
# Backward-compatible function-level entry points.
# Tests and any external callers can still import these by name; they are
# now thin wrappers over the corresponding strategy.
# ---------------------------------------------------------------------------


def recover_stuck_training() -> None:
    StuckTrainingRecovery().run()


def recover_stuck_test_generation() -> None:
    StuckTestGenerationRecovery().run()


def cleanup_orphaned_classifier_dirs() -> None:
    OrphanedClassifierDirRecovery().run()
