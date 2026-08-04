"""Route wiring for local (subprocess) training.

Local training is submitted, polled and finalized through the compute provider
exactly like a remote-worker job. These tests drive the real HTTP routes against
a FABRICATED job handle and job directory — no model is loaded and no trainer
runs, so they stay fast — and pin the three things the route owns:

  * POST /train persists a durable handle instead of running training inline.
  * GET /training-status maps a job directory to a user-facing phase, and is
    where the run FINALIZES (artifacts, 'active', snapshot, calibration).
  * DELETE cancels the child, because a DB-free child cannot notice that its
    guardrail is gone.
"""
import json
import os
import shutil
import uuid

import pytest

from utils.sqlite_db import execute_query, execute_query_dict

pytestmark = pytest.mark.integration


def _classifier(client, auth_headers, test_model):
    res = client.post("/classifiers/create", json={
        "model_id": test_model["model_id"],
        "name": f"localtrain-{uuid.uuid4().hex[:8]}",
    }, headers=auth_headers)
    assert res.status_code == 200, res.text
    return res.json().get("classifier", res.json())["classifier_id"]


def _seed_ce(cid):
    """One CE with an excitation dataset, wired through a rule_setup."""
    name = f"localtrain_ce_{uuid.uuid4().hex[:8]}_probe"
    ce_id = execute_query_dict(
        "INSERT INTO cognitive_elements (name, definition) VALUES (%s, %s) RETURNING ce_id",
        (name, "local training route test CE"))[0]["ce_id"]
    execute_query(
        "INSERT INTO excitation_datasets (ce_id, dataset) VALUES (%s, %s)",
        (ce_id, json.dumps({"samples": [
            {"conversation": [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello"}]}]})))
    rule_id = execute_query_dict(
        "INSERT INTO rules (name, predicate, ce_groups, condition) "
        "VALUES (%s, %s, %s, %s) RETURNING rule_id",
        (f"localtrain_rule_{uuid.uuid4().hex[:8]}", name,
         json.dumps({"required": [name]}), "all of required"))[0]["rule_id"]
    setup_id = execute_query_dict(
        "INSERT INTO rule_setup (classifier_id, rule_id, custom_name, predicate, "
        "ce_groups, condition, is_active) VALUES (%s, %s, %s, %s, %s, %s, TRUE) "
        "RETURNING setup_id",
        (cid, rule_id, f"localtrain_setup_{uuid.uuid4().hex[:8]}", name,
         json.dumps({"required": [name]}), "all of required"))[0]["setup_id"]
    execute_query("INSERT INTO setup_ce_link (setup_id, ce_id) VALUES (%s, %s)",
                  (setup_id, ce_id))
    return ce_id


def _fabricate_job(cid, pid, files=None):
    """Put a guardrail into exactly the state a submitted local run leaves it in:
    status='training' plus a {job_dir, pid, started_at} handle in training_log,
    with a job directory on disk holding `files`."""
    from classifier_engine.trainer import classifier_workdir
    from services.compute.providers.local import JOB_DIRNAME
    job_dir = os.path.join(classifier_workdir(cid), JOB_DIRNAME)
    os.makedirs(job_dir, exist_ok=True)
    for name, content in (files or {}).items():
        with open(os.path.join(job_dir, name), "w") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))
    handle = {"provider": "local", "mode": "local",
              "job": {"id": str(pid), "raw": {"job_dir": job_dir, "pid": pid,
                                              "started_at": 0, "device": "cpu"}},
              "chain": ["local"], "chain_pos": 0, "user_id": 1, "last_contact": 0}
    execute_query(
        "UPDATE classifiers SET status = 'training', training_log = %s WHERE classifier_id = %s",
        (json.dumps(handle), cid))
    return job_dir


@pytest.fixture
def no_calibration(monkeypatch):
    """Auto-calibration would load a real model; record the call instead."""
    import services.auto_calibration as ac
    calls = []
    monkeypatch.setattr(ac, "schedule_post_training_calibration", lambda cid: calls.append(cid))
    return calls


# ---------------------------------------------------------------------------
# POST /train
# ---------------------------------------------------------------------------

class TestStartTrainingLocal:
    def test_submits_through_the_provider_and_persists_a_durable_handle(
            self, client, test_model, auth_headers, monkeypatch):
        from services.compute.base import TrainingJob
        from services.compute.providers.local import LocalProvider
        cid = _classifier(client, auth_headers, test_model)
        _seed_ce(cid)

        seen = {}

        def _fake_submit(self, spec):
            seen["spec"] = spec
            return TrainingJob(provider="local", classifier_id=spec.classifier_id, id="4242",
                               raw={"job_dir": "/tmp/fake/job", "pid": 4242, "started_at": 1.0})
        monkeypatch.setattr(LocalProvider, "submit_training", _fake_submit)

        res = client.post(f"/classifiers/{cid}/train", headers=auth_headers)
        assert res.status_code == 200, res.text
        assert res.json()["success"] is True
        assert res.json()["mode"].startswith("local_")

        # The route built the full spec for the provider — same helper the
        # remote path uses, so the two can't drift.
        spec = seen["spec"]
        assert spec.classifier_id == cid
        assert spec.labels and spec.dataset_files
        assert spec.model_hf_path

        row = execute_query_dict(
            "SELECT status, training_log FROM classifiers WHERE classifier_id = %s", (cid,))[0]
        assert row["status"] == "training"
        tl = row["training_log"]
        tl = json.loads(tl) if isinstance(tl, str) else tl
        # Durable: everything needed to re-find the run after a restart.
        assert tl["provider"] == "local"
        assert tl["job"]["raw"]["pid"] == 4242
        assert tl["job"]["raw"]["job_dir"] == "/tmp/fake/job"

    def test_guardrail_deleted_mid_submit_cancels_the_child(
            self, client, test_model, auth_headers, monkeypatch):
        """Closes the submit-window race: the handle write is conditional on the
        row still being 'training', and losing that race must kill the process
        we just spawned rather than leave it running for a guardrail that's gone."""
        from services.compute.base import TrainingJob
        from services.compute.providers.local import LocalProvider
        cid = _classifier(client, auth_headers, test_model)
        _seed_ce(cid)

        cancelled = []

        def _fake_submit(self, spec):
            # Simulate the delete landing between the status flip and the write.
            execute_query("DELETE FROM classifiers WHERE classifier_id = %s", (cid,))
            return TrainingJob(provider="local", classifier_id=cid, id="99",
                               raw={"job_dir": "/tmp/fake/job", "pid": 99})
        monkeypatch.setattr(LocalProvider, "submit_training", _fake_submit)
        monkeypatch.setattr(LocalProvider, "cancel_training",
                            lambda self, job: cancelled.append(job))

        res = client.post(f"/classifiers/{cid}/train", headers=auth_headers)
        assert res.status_code == 200
        assert res.json()["success"] is False
        assert len(cancelled) == 1


# ---------------------------------------------------------------------------
# GET /training-status — poll mapping + finalize
# ---------------------------------------------------------------------------

class TestStatusPollMapsLocalJob:
    def test_running_child_shows_the_mapped_phase(self, client, test_model, auth_headers):
        cid = _classifier(client, auth_headers, test_model)
        # os.getpid() is a live process, which is all poll needs to call it running.
        _fabricate_job(cid, os.getpid(), files={
            "progress.json": {"stage": "extract", "progress": 0.2,
                              "detail": "Extracting LLM representations for train set"}})
        try:
            res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
            assert res.status_code == 200
            body = res.json()
            assert body["is_training"] is True
            assert body["status"] == "training"
            assert body["training_phase"] == "Extracting embeddings"   # not "extract"
            assert "representations" in body["training_phase_detail"]
            assert body["mode"] == "local"
            # ...and it is persisted, so a page refresh shows the same thing.
            row = execute_query_dict(
                "SELECT training_phase FROM classifiers WHERE classifier_id = %s", (cid,))[0]
            assert row["training_phase"] == "Extracting embeddings"
        finally:
            _cleanup(cid)

    def test_overall_progress_is_shown_as_a_percentage(self, client, test_model, auth_headers):
        # The child reports a 0..1 completion fraction. There is no separate
        # percentage field in the status payload, so it has to ride the phase
        # detail line the UI already polls — otherwise it is produced and thrown
        # away, and the operator watches a phase name that doesn't move for
        # twenty minutes.
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_job(cid, os.getpid(), files={
            "progress.json": {"stage": "extract", "progress": 0.62,
                              "detail": "Extracting LLM representations for train set"}})
        try:
            body = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers).json()
            assert body["training_phase_detail"].endswith("62%")
            assert "representations" in body["training_phase_detail"]
            row = execute_query_dict(
                "SELECT training_phase_detail FROM classifiers WHERE classifier_id = %s", (cid,))[0]
            assert row["training_phase_detail"].endswith("62%")
        finally:
            _cleanup(cid)

    def test_a_detail_that_already_quotes_a_percentage_is_not_doubled_up(
            self, client, test_model, auth_headers):
        # The RNN fit reports its own figure; showing "… — 68% — 78%" would just
        # confuse the operator.
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_job(cid, os.getpid(), files={
            "progress.json": {"stage": "train_rnn", "progress": 0.78,
                              "detail": "Optimizing guardrail — 68%"}})
        try:
            body = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers).json()
            assert body["training_phase_detail"] == "Optimizing guardrail — 68%"
        finally:
            _cleanup(cid)

    def test_completed_run_is_finalized_in_the_parent(
            self, client, test_model, auth_headers, no_calibration):
        from classifier_engine.trainer import classifier_workdir
        cid = _classifier(client, auth_headers, test_model)
        job_dir = _fabricate_job(cid, os.getpid(), files={
            "status.json": {"status": "success", "elapsed_s": 12.0},
            "trained_rnn.pth": "weights",
            "classifier_meta.json": {"labels": {"a": 0}},
            "training_log.json": [{"progress": 100}],
        })
        try:
            res = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
            assert res.status_code == 200, res.text
            body = res.json()
            assert body["status"] == "active"
            assert body["is_trained"] is True and body["is_training"] is False

            work_dir = classifier_workdir(cid)
            # Artifacts fetched up out of the job dir, which is then dropped.
            assert os.path.isfile(os.path.join(work_dir, "trained_rnn.pth"))
            assert os.path.isfile(os.path.join(work_dir, "classifier_meta.json"))
            assert not os.path.exists(job_dir)

            row = execute_query_dict(
                "SELECT status, model_path, trained_at, training_log "
                "FROM classifiers WHERE classifier_id = %s", (cid,))[0]
            assert row["status"] == "active"
            assert row["model_path"].endswith("trained_rnn.pth")
            # The job handle is replaced by the run's metrics once it's spent.
            tl = row["training_log"]
            tl = json.loads(tl) if isinstance(tl, str) else tl
            assert tl == [{"progress": 100}]
            # The trained-policy snapshot is committed by the parent...
            assert row["trained_at"] is not None
            # ...and the post-training chain is scheduled here, exactly once.
            assert no_calibration == [cid]
        finally:
            _cleanup(cid)

    def test_finalize_runs_once_even_when_polled_again(
            self, client, test_model, auth_headers, no_calibration):
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_job(cid, os.getpid(), files={
            "status.json": {"status": "success"},
            "trained_rnn.pth": "weights",
            "classifier_meta.json": {"labels": {"a": 0}},
        })
        try:
            for _ in range(3):
                client.get(f"/classifiers/{cid}/training-status", headers=auth_headers)
            # The UI polls every 5s; re-running calibration on every poll would
            # queue a pile of duplicate GPU work.
            assert no_calibration == [cid]
        finally:
            _cleanup(cid)

    def test_failed_child_surfaces_its_error(self, client, test_model, auth_headers):
        cid = _classifier(client, auth_headers, test_model)
        job_dir = _fabricate_job(cid, os.getpid(), files={
            "status.json": {"status": "failed",
                            "error": "CUDA out of memory. Try a smaller model."}})
        try:
            body = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers).json()
            assert body["status"] == "error"
            assert body["has_error"] is True
            assert "out of memory" in body["training_phase_detail"]
            row = execute_query_dict(
                "SELECT status FROM classifiers WHERE classifier_id = %s", (cid,))[0]
            assert row["status"] == "error"
            # Nothing will ever collect from a failed run, so its staged inputs
            # and extracted representations don't get to sit in the guardrail's
            # workdir until the next retrain.
            assert not os.path.exists(job_dir)
        finally:
            _cleanup(cid)

    def test_success_without_artifacts_fails_instead_of_polling_forever(
            self, client, test_model, auth_headers, no_calibration):
        # status.json says success but the weights aren't there. Retrying the
        # fetch on every poll would pin the guardrail in 'training'.
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_job(cid, os.getpid(), files={"status.json": {"status": "success"}})
        try:
            body = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers).json()
            assert body["status"] == "error"
            assert "could not be collected" in body["training_phase_detail"]
            assert no_calibration == []
        finally:
            _cleanup(cid)

    def test_child_that_vanished_without_a_result_fails_the_run(
            self, client, test_model, auth_headers):
        # An OOM-killed child writes no status.json. Without a liveness check the
        # guardrail would poll 'training' forever.
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_job(cid, 2_147_480_000, files={"train_job.log": "Killed"})
        try:
            body = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers).json()
            assert body["status"] == "error"
            assert "without producing a result" in body["training_phase_detail"]
        finally:
            _cleanup(cid)


# ---------------------------------------------------------------------------
# delete cancels the child
# ---------------------------------------------------------------------------

class TestDeleteCancelsLocalTraining:
    def test_deleting_a_guardrail_mid_training_cancels_the_child(
            self, client, test_model, auth_headers, monkeypatch):
        from services.compute.providers.local import LocalProvider
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_job(cid, os.getpid())
        cancelled = []
        monkeypatch.setattr(LocalProvider, "cancel_training",
                            lambda self, job: cancelled.append(job))

        res = client.delete(f"/classifiers/{cid}", headers=auth_headers)
        assert res.status_code == 200, res.text
        assert len(cancelled) == 1
        assert cancelled[0].raw["pid"] == os.getpid()

    def test_deleting_the_parent_model_cancels_local_training_too(
            self, client, test_user, auth_headers, monkeypatch):
        from services.compute.providers.local import LocalProvider
        # Inserted directly: /models/create validates the HF repo over the
        # network, which the suite has no access to.
        model = execute_query_dict(
            "INSERT INTO target_models (user_id, name, storage_path) "
            "VALUES (%s, %s, %s) RETURNING model_id",
            (test_user["user_id"], f"localtrain-model-{uuid.uuid4().hex[:8]}",
             f"pytest-org/localtrain-{uuid.uuid4().hex[:8]}"))[0]
        cid = _classifier(client, auth_headers, model)
        _fabricate_job(cid, os.getpid())
        cancelled = []
        monkeypatch.setattr(LocalProvider, "cancel_training",
                            lambda self, job: cancelled.append(job))

        assert client.delete(f"/models/{model['model_id']}", headers=auth_headers).status_code == 200
        assert len(cancelled) == 1

    def test_a_finished_guardrail_is_not_cancelled_on_delete(
            self, client, test_model, auth_headers, monkeypatch):
        from services.compute.providers.local import LocalProvider
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_job(cid, os.getpid())
        execute_query("UPDATE classifiers SET status = 'active' WHERE classifier_id = %s", (cid,))
        cancelled = []
        monkeypatch.setattr(LocalProvider, "cancel_training",
                            lambda self, job: cancelled.append(job))
        client.delete(f"/classifiers/{cid}", headers=auth_headers)
        assert cancelled == []


def _cleanup(cid):
    try:
        from classifier_engine.trainer import classifier_workdir
        shutil.rmtree(classifier_workdir(cid), ignore_errors=True)
    except Exception:
        pass
