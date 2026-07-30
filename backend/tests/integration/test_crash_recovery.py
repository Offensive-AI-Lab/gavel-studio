"""Tests for crash recovery: stuck statuses, orphaned files, pipeline rollback."""
import os
import json
import pytest
import shutil


# Resolve backend/ from this file's location, regardless of how deep the test
# nests under tests/. tests/integration/test_crash_recovery.py → backend/.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStuckEvaluationRecovery:
    """Interrupted calibration/evaluation is wiped on boot — 'as if never ran',
    so a crash mid-run never leaves a partial/stale result and is re-runnable."""

    def test_running_rows_deleted(self, client, test_classifier, auth_headers):
        from utils.sqlite_db import execute_query, execute_query_dict
        cid = test_classifier["classifier_id"]
        # Two interrupted runs: a plain one + one that had a cluster job stashed.
        execute_query(
            "INSERT INTO evaluation_results (classifier_id, eval_type) "
            "VALUES (%s, 'calibration_running')",
            (cid,),
        )
        execute_query(
            "INSERT INTO evaluation_results (classifier_id, eval_type, plots) "
            "VALUES (%s, 'evaluation_running', %s::jsonb)",
            (cid, json.dumps({"cluster": {"slurm_job_id": "999999", "remote_job_dir": "/tmp/nope"}})),
        )
        before = execute_query_dict(
            "SELECT COUNT(*) AS n FROM evaluation_results WHERE classifier_id = %s "
            "AND eval_type IN ('calibration_running','evaluation_running')", (cid,),
        )
        assert before[0]["n"] == 2

        from utils.crash_recovery import StuckEvaluationRecovery
        StuckEvaluationRecovery().run()

        after = execute_query_dict(
            "SELECT COUNT(*) AS n FROM evaluation_results WHERE classifier_id = %s "
            "AND eval_type IN ('calibration_running','evaluation_running')", (cid,),
        )
        assert after[0]["n"] == 0

    def test_completed_results_untouched(self, client, test_classifier, auth_headers):
        from utils.sqlite_db import execute_query, execute_query_dict
        cid = test_classifier["classifier_id"]
        execute_query(
            "INSERT INTO evaluation_results (classifier_id, eval_type, thresholds) "
            "VALUES (%s, 'calibration', '{}'::jsonb)", (cid,),
        )
        from utils.crash_recovery import StuckEvaluationRecovery
        StuckEvaluationRecovery().run()
        rows = execute_query_dict(
            "SELECT COUNT(*) AS n FROM evaluation_results "
            "WHERE classifier_id = %s AND eval_type = 'calibration'", (cid,),
        )
        assert rows[0]["n"] >= 1


class TestStuckTrainingRecovery:
    """Verify classifiers stuck in 'training' get recovered on startup."""

    def test_stuck_training_reset_to_error(self, client, test_classifier, auth_headers):
        """Simulate a server crash during training — status should be recoverable."""
        from utils.sqlite_db import execute_query, execute_query_dict
        cid = test_classifier["classifier_id"]

        # Simulate stuck training
        execute_query(
            "UPDATE classifiers SET status = 'training', training_log = NULL WHERE classifier_id = %s",
            (cid,),
        )

        # Verify it's stuck
        row = execute_query_dict("SELECT status FROM classifiers WHERE classifier_id = %s", (cid,))
        assert row[0]["status"] == "training"

        # Run recovery
        from utils.crash_recovery import recover_stuck_training
        recover_stuck_training()

        # Should no longer be 'training'
        row = execute_query_dict("SELECT status FROM classifiers WHERE classifier_id = %s", (cid,))
        assert row[0]["status"] != "training"
        assert row[0]["status"] in ("error", "needs_retraining")

    def test_stuck_training_with_existing_model_preserves(self, client, test_classifier, auth_headers):
        """If a previous trained model exists, recovery should set 'needs_retraining' not 'error'."""
        from utils.sqlite_db import execute_query, execute_query_dict
        cid = test_classifier["classifier_id"]

        # Create fake model files to simulate previous training
        from classifier_engine.trainer import classifier_workdir
        work_dir = classifier_workdir(cid)
        os.makedirs(work_dir, exist_ok=True)
        # Create fake model + metadata
        with open(os.path.join(work_dir, "trained_rnn.pth"), "w") as f:
            f.write("fake model")
        with open(os.path.join(work_dir, "classifier_meta.json"), "w") as f:
            json.dump({"labels": {}}, f)

        # Simulate stuck training
        execute_query(
            "UPDATE classifiers SET status = 'training' WHERE classifier_id = %s",
            (cid,),
        )

        # Run recovery
        from utils.crash_recovery import recover_stuck_training
        recover_stuck_training()

        # Should preserve previous model
        row = execute_query_dict("SELECT status FROM classifiers WHERE classifier_id = %s", (cid,))
        assert row[0]["status"] == "needs_retraining"

        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)

    def test_stuck_training_with_only_rnn_treated_as_partial(self, client, test_classifier, auth_headers):
        """Only `trained_rnn.pth` exists, no `classifier_meta.json`. The model
        is incomplete — recovery must NOT preserve it. Status should become
        'error', work dir should be deleted.

        If the existence check is ever weakened from `and` to `or`, this case
        would be wrongly preserved as 'needs_retraining'.
        """
        from utils.sqlite_db import execute_query, execute_query_dict
        cid = test_classifier["classifier_id"]

        # Use the helper so the test path stays in lockstep with whatever
        # layout the trainer actually writes to (currently
        # trained_classifiers/<user_id>/classifier_<id>/).
        from classifier_engine.trainer import classifier_workdir
        work_dir = classifier_workdir(cid)
        os.makedirs(work_dir, exist_ok=True)
        # ONLY the rnn weights — no meta.json. This is what a half-finished
        # save looks like in practice.
        with open(os.path.join(work_dir, "trained_rnn.pth"), "w") as f:
            f.write("fake model bytes")

        execute_query(
            "UPDATE classifiers SET status = 'training' WHERE classifier_id = %s",
            (cid,),
        )

        from utils.crash_recovery import recover_stuck_training
        recover_stuck_training()

        row = execute_query_dict(
            "SELECT status FROM classifiers WHERE classifier_id = %s", (cid,)
        )
        assert row[0]["status"] == "error", (
            "incomplete training (rnn without meta) was wrongly preserved as "
            f"'{row[0]['status']}' instead of being marked 'error'"
        )
        # Partial files should be cleaned up.
        assert not os.path.isdir(work_dir)

    def test_stuck_training_with_only_meta_treated_as_partial(self, client, test_classifier, auth_headers):
        """Symmetric to the rnn-only case: only `classifier_meta.json` exists,
        no weights. Also incomplete; must not be preserved.
        """
        from utils.sqlite_db import execute_query, execute_query_dict
        cid = test_classifier["classifier_id"]

        # Use the helper so the test path stays in lockstep with whatever
        # layout the trainer actually writes to (currently
        # trained_classifiers/<user_id>/classifier_<id>/).
        from classifier_engine.trainer import classifier_workdir
        work_dir = classifier_workdir(cid)
        os.makedirs(work_dir, exist_ok=True)
        with open(os.path.join(work_dir, "classifier_meta.json"), "w") as f:
            json.dump({"labels": {}}, f)

        execute_query(
            "UPDATE classifiers SET status = 'training' WHERE classifier_id = %s",
            (cid,),
        )

        from utils.crash_recovery import recover_stuck_training
        recover_stuck_training()

        row = execute_query_dict(
            "SELECT status FROM classifiers WHERE classifier_id = %s", (cid,)
        )
        assert row[0]["status"] == "error", (
            "incomplete training (meta without rnn) was wrongly preserved as "
            f"'{row[0]['status']}' instead of being marked 'error'"
        )
        assert not os.path.isdir(work_dir)

    def test_stuck_training_no_model_deletes_files(self, client, test_classifier, auth_headers):
        """If no valid model, recovery should delete partial files and set 'error'."""
        from utils.sqlite_db import execute_query, execute_query_dict
        cid = test_classifier["classifier_id"]

        # Create partial training directory (no valid model)
        from classifier_engine.trainer import classifier_workdir
        work_dir = classifier_workdir(cid)
        os.makedirs(os.path.join(work_dir, "sequences", "train"), exist_ok=True)
        with open(os.path.join(work_dir, "partial_data.pt"), "w") as f:
            f.write("partial")

        # Simulate stuck
        execute_query(
            "UPDATE classifiers SET status = 'training' WHERE classifier_id = %s",
            (cid,),
        )

        from utils.crash_recovery import recover_stuck_training
        recover_stuck_training()

        # Directory should be deleted
        assert not os.path.isdir(work_dir)

        # Status should be error
        row = execute_query_dict("SELECT status FROM classifiers WHERE classifier_id = %s", (cid,))
        assert row[0]["status"] == "error"


class TestStuckTestGenerationRecovery:
    """Verify test datasets stuck in 'generating' get recovered."""

    def test_stuck_generating_marked_error(self, client, test_classifier, auth_headers):
        """Simulate stuck test generation — should be marked as error."""
        from utils.sqlite_db import execute_query, execute_query_dict
        cid = test_classifier["classifier_id"]

        # Insert a stuck generating record (test sets are rule-scoped now;
        # this recovery test doesn't need a rule link).
        result = execute_query_dict(
            """INSERT INTO test_datasets (dataset_type, status, generation_log)
               VALUES ('positive', 'generating', 'Stuck...')
               RETURNING dataset_id""",
        )
        dataset_id = result[0]["dataset_id"]

        # Run recovery
        from utils.crash_recovery import recover_stuck_test_generation
        recover_stuck_test_generation()

        # Should be marked as error
        row = execute_query_dict(
            "SELECT status, generation_log FROM test_datasets WHERE dataset_id = %s",
            (dataset_id,),
        )
        assert row[0]["status"] == "error"
        assert "server restart" in row[0]["generation_log"].lower()

        # Cleanup
        execute_query("DELETE FROM test_datasets WHERE dataset_id = %s", (dataset_id,))

    def test_ready_datasets_not_affected(self, client, test_classifier, auth_headers):
        """Recovery should NOT touch datasets with status='ready'."""
        from utils.sqlite_db import execute_query, execute_query_dict
        cid = test_classifier["classifier_id"]

        result = execute_query_dict(
            """INSERT INTO test_datasets (dataset_type, status, generation_log)
               VALUES ('positive', 'ready', 'Done')
               RETURNING dataset_id""",
        )
        dataset_id = result[0]["dataset_id"]

        from utils.crash_recovery import recover_stuck_test_generation
        recover_stuck_test_generation()

        row = execute_query_dict("SELECT status FROM test_datasets WHERE dataset_id = %s", (dataset_id,))
        assert row[0]["status"] == "ready"

        execute_query("DELETE FROM test_datasets WHERE dataset_id = %s", (dataset_id,))


class TestOrphanedDirectoryCleanup:
    """Verify orphaned classifier directories are cleaned up."""

    def test_orphaned_dir_deleted(self, client):
        """Directory with no matching DB record should be deleted."""
        base = os.path.join(_BACKEND_DIR, "trained_classifiers")
        orphan_dir = os.path.join(base, "classifier_999888")
        os.makedirs(orphan_dir, exist_ok=True)
        with open(os.path.join(orphan_dir, "dummy.txt"), "w") as f:
            f.write("orphan")

        from utils.crash_recovery import cleanup_orphaned_classifier_dirs
        cleanup_orphaned_classifier_dirs()

        assert not os.path.isdir(orphan_dir)

    def test_valid_dir_not_deleted(self, client, test_classifier):
        """Directory matching a real classifier should NOT be deleted."""
        cid = test_classifier["classifier_id"]
        base = os.path.join(_BACKEND_DIR, "trained_classifiers")
        valid_dir = os.path.join(base, f"classifier_{cid}")
        os.makedirs(valid_dir, exist_ok=True)
        marker = os.path.join(valid_dir, "keep_me.txt")
        with open(marker, "w") as f:
            f.write("valid")

        try:
            from utils.crash_recovery import cleanup_orphaned_classifier_dirs
            cleanup_orphaned_classifier_dirs()

            assert os.path.isdir(valid_dir)
            assert os.path.exists(marker)
        finally:
            shutil.rmtree(valid_dir, ignore_errors=True)

    def test_orphan_cleanup_continues_past_non_classifier_dir(self, client):
        """A stray directory that doesn't match the `classifier_*` prefix must
        be skipped, not used as a stop signal. With `continue` the loop keeps
        going; if it ever became `break`, an unrelated folder landing earlier
        in os.listdir() order would shield every later orphan from cleanup.
        """
        base = os.path.join(_BACKEND_DIR, "trained_classifiers")
        os.makedirs(base, exist_ok=True)

        # 'a_' sorts before 'classifier_' so it lands first in any
        # alphabetical enumeration of os.listdir().
        stray_dir = os.path.join(base, "a_unrelated_folder")
        orphan_dir = os.path.join(base, "classifier_888777")
        os.makedirs(stray_dir, exist_ok=True)
        os.makedirs(orphan_dir, exist_ok=True)

        try:
            from utils.crash_recovery import cleanup_orphaned_classifier_dirs
            cleanup_orphaned_classifier_dirs()

            # The stray dir is left alone (not our concern).
            assert os.path.isdir(stray_dir), (
                "non-classifier_* directory should be skipped, not deleted"
            )
            # The orphan AFTER the stray dir is still cleaned up. If the loop
            # were `break` instead of `continue`, this orphan would survive.
            assert not os.path.isdir(orphan_dir), (
                "orphan directory was not deleted — recovery loop bailed out "
                "after hitting a non-classifier_* directory"
            )
        finally:
            # Cleanup whatever is left so we don't pollute later test runs.
            shutil.rmtree(stray_dir, ignore_errors=True)
            shutil.rmtree(orphan_dir, ignore_errors=True)


class TestIncompletePipelineRecovery:
    """IncompletePipelineRecovery wipes EVERY is_ready=FALSE rule and CE on
    boot — no exceptions. A row still FALSE after a restart means its
    generation died mid-flight (crash, closed tab, power/network loss), and
    the background thread that would finish it is gone. Per the product
    contract, that must look exactly as if the rule/CE was never generated.
    Unfinished rule/CE wizard runs (pipeline_runs, completed=FALSE) are
    deleted too, so nothing offers to resume into a deleted draft.

    These are the highest-value tests: getting the predicate wrong here
    either leaves half-baked rules in the library or silently destroys
    finalized work.
    """

    @staticmethod
    def _insert_pending_rule(name: str) -> int:
        from utils.sqlite_db import execute_query_dict
        rows = execute_query_dict(
            """
            INSERT INTO rules (name, predicate, description, categories, is_ready)
            VALUES (%s, %s, %s, %s, FALSE) RETURNING rule_id
            """,
            (name, "A AND B", "x", []),
        )
        return rows[0]["rule_id"]

    @staticmethod
    def _insert_pending_ce(name: str) -> int:
        from utils.sqlite_db import execute_query_dict
        rows = execute_query_dict(
            """
            INSERT INTO cognitive_elements (name, definition, categories, is_ready)
            VALUES (%s, %s, %s, FALSE) RETURNING ce_id
            """,
            (name, "x", []),
        )
        return rows[0]["ce_id"]

    @staticmethod
    def _link_ce_to_rule(rule_id: int, ce_id: int):
        from utils.sqlite_db import execute_query
        execute_query(
            "INSERT INTO rule_ce_link (rule_id, ce_id) VALUES (%s, %s)",
            (rule_id, ce_id),
        )

    @staticmethod
    def _start_active_run(rule_id: int, user_id: int, classifier_id: int):
        # Pin the rule to an active wizard run. The recovery query uses
        # `completed = FALSE` as its "user will be back" signal.
        from utils.sqlite_db import execute_query
        execute_query(
            """
            INSERT INTO pipeline_runs (user_id, classifier_id, rule_id, current_step, steps, completed)
            VALUES (%s, %s, %s, '2A', '{}'::jsonb, FALSE)
            """,
            (user_id, classifier_id, rule_id),
        )

    def test_inactive_pending_rule_wiped(self, client):
        from utils.sqlite_db import execute_query_dict
        rule_id = self._insert_pending_rule("ipr_inactive_rule")

        from utils.crash_recovery import IncompletePipelineRecovery
        IncompletePipelineRecovery().run()

        rows = execute_query_dict("SELECT rule_id FROM rules WHERE rule_id = %s", (rule_id,))
        assert not rows, "is_ready=FALSE rule with no active wizard run should be wiped"

    def test_unattached_pending_ce_wiped(self, client):
        from utils.sqlite_db import execute_query_dict
        ce_id = self._insert_pending_ce("ipr_unattached_ce")

        from utils.crash_recovery import IncompletePipelineRecovery
        IncompletePipelineRecovery().run()

        rows = execute_query_dict("SELECT ce_id FROM cognitive_elements WHERE ce_id = %s", (ce_id,))
        assert not rows, "is_ready=FALSE CE with no active-run parent should be wiped"

    def test_rule_with_active_run_is_wiped(self, client, test_classifier, test_user):
        # New contract: a crash wipes the draft even if a wizard run pointed
        # at it — "like we didn't generate it". The unfinished run is dropped
        # too so nothing tries to resume into the deleted rule.
        from utils.sqlite_db import execute_query_dict
        rule_id = self._insert_pending_rule("ipr_active_rule")
        self._start_active_run(rule_id, test_user["user_id"], test_classifier["classifier_id"])

        from utils.crash_recovery import IncompletePipelineRecovery
        IncompletePipelineRecovery().run()

        assert not execute_query_dict(
            "SELECT rule_id FROM rules WHERE rule_id = %s", (rule_id,)
        ), "is_ready=FALSE rule must be wiped even with an active wizard run"
        assert not execute_query_dict(
            "SELECT run_id FROM pipeline_runs WHERE rule_id = %s", (rule_id,)
        ), "the unfinished rule wizard run should be cleared too"

    def test_ces_linked_to_active_rule_are_wiped(self, client, test_classifier, test_user):
        # The dependent CE is wiped along with its incomplete rule.
        from utils.sqlite_db import execute_query_dict
        rule_id = self._insert_pending_rule("ipr_active_rule_w_ce")
        ce_id = self._insert_pending_ce("ipr_dep_ce")
        self._link_ce_to_rule(rule_id, ce_id)
        self._start_active_run(rule_id, test_user["user_id"], test_classifier["classifier_id"])

        from utils.crash_recovery import IncompletePipelineRecovery
        IncompletePipelineRecovery().run()

        assert not execute_query_dict("SELECT rule_id FROM rules WHERE rule_id = %s", (rule_id,))
        assert not execute_query_dict(
            "SELECT ce_id FROM cognitive_elements WHERE ce_id = %s", (ce_id,)
        ), "CE of an incomplete rule must be wiped, active run or not"

    def test_test_eval_run_not_deleted(self, client, test_classifier, test_user):
        # Only rule/CE generation runs are cleared. A test_eval wizard run is
        # a different flow and must survive recovery.
        from utils.sqlite_db import execute_query, execute_query_dict
        rows = execute_query_dict(
            """
            INSERT INTO pipeline_runs (user_id, classifier_id, rule_id, current_step, steps, completed, pipeline_type)
            VALUES (%s, %s, NULL, '1', '{}'::jsonb, FALSE, 'test_eval')
            RETURNING run_id
            """,
            (test_user["user_id"], test_classifier["classifier_id"]),
        )
        run_id = rows[0]["run_id"]

        from utils.crash_recovery import IncompletePipelineRecovery
        IncompletePipelineRecovery().run()

        assert execute_query_dict(
            "SELECT run_id FROM pipeline_runs WHERE run_id = %s", (run_id,)
        ), "test_eval runs must not be swept by rule/CE incomplete-pipeline recovery"

    def test_mixed_run_wipes_both_active_and_inactive(self, client, test_classifier, test_user):
        # Hybrid: one rule with an active run and one with none — both are
        # wiped now, since neither finished generating.
        from utils.sqlite_db import execute_query_dict
        active_id = self._insert_pending_rule("ipr_mixed_active")
        inactive_id = self._insert_pending_rule("ipr_mixed_inactive")
        self._start_active_run(active_id, test_user["user_id"], test_classifier["classifier_id"])

        from utils.crash_recovery import IncompletePipelineRecovery
        IncompletePipelineRecovery().run()

        assert not execute_query_dict("SELECT rule_id FROM rules WHERE rule_id = %s", (active_id,))
        assert not execute_query_dict("SELECT rule_id FROM rules WHERE rule_id = %s", (inactive_id,))

    def test_finalized_rule_untouched(self, client):
        # Sanity: the recovery query is gated on is_ready=FALSE. A live,
        # finalized rule (is_ready=TRUE) must never be touched, parked or not.
        from utils.sqlite_db import execute_query_dict
        rows = execute_query_dict(
            """
            INSERT INTO rules (name, predicate, description, categories, is_ready)
            VALUES (%s, %s, %s, %s, TRUE) RETURNING rule_id
            """,
            ("ipr_finalized", "A AND B", "x", []),
        )
        rule_id = rows[0]["rule_id"]

        from utils.crash_recovery import IncompletePipelineRecovery
        IncompletePipelineRecovery().run()

        assert execute_query_dict("SELECT rule_id FROM rules WHERE rule_id = %s", (rule_id,))


class TestFullRecovery:
    """Test the full run_all_recovery function."""

    def test_run_all_recovery_no_crash(self, client):
        """Running recovery with nothing stuck should not error."""
        from utils.crash_recovery import run_all_recovery
        # Should complete without exceptions
        run_all_recovery()


class TestTrainingTaskErrorHandling:
    """Verify the training background task sets error status on failure."""

    def test_training_task_sets_error_on_exception(self, client, test_classifier, auth_headers):
        """If training fails, status should be 'error', not stuck in 'training'."""
        from utils.sqlite_db import execute_query, execute_query_dict
        cid = test_classifier["classifier_id"]

        # Set to training
        execute_query("UPDATE classifiers SET status = 'training' WHERE classifier_id = %s", (cid,))

        # Simulate training failure via the wrapper
        from routes.classifiers import _run_training_task
        try:
            _run_training_task(cid)
        except Exception:
            pass

        # Status should be 'error', not stuck in 'training'
        row = execute_query_dict("SELECT status FROM classifiers WHERE classifier_id = %s", (cid,))
        assert row[0]["status"] == "error"


class TestClassifierModelValidation:
    """Verify classifier creation validates model_id."""

    def test_invalid_model_id_returns_404(self, client, auth_headers):
        res = client.post("/classifiers/create", json={
            "model_id": 99999,
            "name": "BadModel",
        }, headers=auth_headers)
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_zero_model_id_returns_404(self, client, auth_headers):
        res = client.post("/classifiers/create", json={
            "model_id": 0,
            "name": "ZeroModel",
        }, headers=auth_headers)
        assert res.status_code == 404

    def test_negative_model_id_returns_404(self, client, auth_headers):
        res = client.post("/classifiers/create", json={
            "model_id": -1,
            "name": "NegModel",
        }, headers=auth_headers)
        assert res.status_code == 404
