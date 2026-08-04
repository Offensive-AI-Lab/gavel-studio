"""TRUE end-to-end local training through the SUBPROCESS path (needs a GPU).

The rest of the local-training tests fabricate job directories so they can run
in milliseconds. This one runs the real thing: POST /train spawns
compute_jobs/train_job.py against a real base model, the status route polls it
to completion, and the run finalizes into an 'active' guardrail with artifacts
on disk. It is the only test that proves the parent and the child actually agree
about the job directory, the payload format, the phase file and the artifacts.

Marked `slow` and skipped without CUDA — on CPU the real extraction takes far
too long to be part of any routine run. Run it with:

    python -m pytest tests/integration/test_local_training_subprocess_e2e.py

CLEANUP: DB rows land in conftest-tracked tables; the on-disk workdir is removed
in a finally block.
"""
import json
import os
import time

import pytest

from utils.sqlite_db import execute_query, execute_query_dict

pytestmark = pytest.mark.slow

# Import the seeding machinery from the existing ML e2e test so the two use the
# same conversation shapes and the same speed-tuned config.
from tests.integration.test_ml_pipeline_e2e import (  # noqa: E402
    FAST_TRAINING_CONFIG, _cleanup_workdir, _seed_trainable_classifier,
)

# The whole run (model download excluded) on a modern GPU is a couple of
# minutes; give it room without hanging a suite forever.
_DEADLINE_S = float(os.getenv("GAVEL_E2E_TRAIN_DEADLINE", "1800"))
_POLL_S = 3.0


def _requires_cuda():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device — the real subprocess training run is GPU-only")
    except ImportError:
        pytest.skip("torch not installed")


class TestLocalSubprocessTrainingEndToEnd:
    def test_train_polls_and_finalizes_through_the_child_process(
            self, client, test_user, test_model, auth_headers, monkeypatch):
        _requires_cuda()

        # Auto-calibration would run a second multi-minute GPU job after the
        # finish line; this test is about training, so record instead of run.
        import services.auto_calibration as ac
        scheduled = []
        monkeypatch.setattr(ac, "schedule_post_training_calibration", lambda cid: scheduled.append(cid))

        cid, user_id, _ce_ids, _ce_names = _seed_trainable_classifier(
            client, auth_headers, test_user, test_model)
        try:
            res = client.post(f"/classifiers/{cid}/train", headers=auth_headers)
            assert res.status_code == 200, res.text
            assert res.json()["success"] is True
            assert res.json()["mode"].startswith("local_")

            # The handle is durable and points at a live child process.
            row = execute_query_dict(
                "SELECT status, training_log FROM classifiers WHERE classifier_id = %s", (cid,))[0]
            assert row["status"] == "training"
            tl = row["training_log"]
            tl = json.loads(tl) if isinstance(tl, str) else tl
            assert tl["provider"] == "local"
            pid = tl["job"]["raw"]["pid"]
            job_dir = tl["job"]["raw"]["job_dir"]
            assert os.path.isdir(job_dir)
            # It is a SEPARATE process — the whole point of the change.
            assert pid != os.getpid()

            # Poll the way the UI does, collecting the phases we're shown.
            phases, body = [], None
            deadline = time.time() + _DEADLINE_S
            while time.time() < deadline:
                body = client.get(f"/classifiers/{cid}/training-status",
                                  headers=auth_headers).json()
                if body.get("training_phase") and body["training_phase"] not in phases:
                    phases.append(body["training_phase"])
                if not body["is_training"]:
                    break
                time.sleep(_POLL_S)

            assert body is not None and not body["is_training"], (
                f"training did not finish within {_DEADLINE_S:.0f}s (phases seen: {phases})")
            assert body["status"] == "active", (
                f"training failed: {body.get('training_phase_detail')} (phases: {phases})")

            # Progress was VISIBLE mid-run, not a blank spinner: at least one
            # real stage label from the shared vocabulary showed up before the
            # terminal 'complete'.
            from services.compute.phases import TRAINING_PHASE_LABELS
            live = [p for p in phases if p != "complete"]
            assert live, f"no phase was ever surfaced during the run: {phases}"
            assert set(live) & set(TRAINING_PHASE_LABELS.values()), (
                f"phases {live} don't come from the shared vocabulary")

            # Artifacts were fetched out of the job dir into the workdir, and
            # the job dir was cleaned up.
            from classifier_engine.trainer import classifier_workdir
            work_dir = classifier_workdir(cid)
            assert os.path.isfile(os.path.join(work_dir, "trained_rnn.pth"))
            meta_path = os.path.join(work_dir, "classifier_meta.json")
            assert os.path.isfile(meta_path)
            assert not os.path.exists(job_dir)

            meta = json.load(open(meta_path))
            assert meta["labels"] and meta["num_classes"] == len(meta["labels"])
            # selected_layers is the [first, last+1] pair the inference side
            # expects, clamped to the model's real depth.
            first, stop = meta["selected_layers"]
            assert 0 <= first < stop <= meta["total_layers"]

            # Finalize happened in the PARENT: status, snapshot and the
            # post-training chain, exactly once.
            row = execute_query_dict(
                "SELECT status, model_path, trained_at, trained_policy_fingerprint "
                "FROM classifiers WHERE classifier_id = %s", (cid,))[0]
            assert row["status"] == "active"
            assert row["model_path"].endswith("trained_rnn.pth")
            assert row["trained_at"] is not None
            assert row["trained_policy_fingerprint"]
            assert scheduled == [cid]

            # The child is gone.
            from services.compute.providers.local import _pid_alive
            assert not _pid_alive(pid)
        finally:
            _cleanup_workdir(cid, user_id)

    def test_cancelling_via_delete_kills_the_child(
            self, client, test_user, test_model, auth_headers):
        """Deleting a guardrail mid-training must stop the GPU work: the child
        is DB-free and cannot notice its guardrail is gone."""
        _requires_cuda()
        from services.compute.providers.local import _pid_alive

        cid, user_id, _ce_ids, _ce_names = _seed_trainable_classifier(
            client, auth_headers, test_user, test_model)
        try:
            assert client.post(f"/classifiers/{cid}/train",
                               headers=auth_headers).status_code == 200
            tl = execute_query_dict(
                "SELECT training_log FROM classifiers WHERE classifier_id = %s", (cid,))[0]["training_log"]
            tl = json.loads(tl) if isinstance(tl, str) else tl
            pid = tl["job"]["raw"]["pid"]
            assert _pid_alive(pid)

            assert client.delete(f"/classifiers/{cid}", headers=auth_headers).status_code == 200

            deadline = time.time() + 30
            while time.time() < deadline and _pid_alive(pid):
                time.sleep(0.5)
            assert not _pid_alive(pid), "the training child survived the delete"
        finally:
            _cleanup_workdir(cid, user_id)
