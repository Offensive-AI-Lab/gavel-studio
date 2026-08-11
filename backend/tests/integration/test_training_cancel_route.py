"""POST /classifiers/{id}/training/cancel — the user stops a training run.

The kill machinery already existed (it is what delete and failover use); this
route is what exposes it. What the tests pin is the contract around it:

  * the run is actually stopped through the provider named in the stored
    handle — a local child OR a remote worker job (monkeypatched: no network,
    no signals sent to the test process);
  * the row is left exactly where crash recovery would leave it —
    'needs_retraining' when a previous model survives on disk, otherwise
    'error' with the partial workdir gone — with the phase columns CLEARED,
    because the run was cancelled, not failed;
  * nothing that belongs to a successful run happens (no policy snapshot, no
    calibration);
  * cancelling something that isn't training is a 409 with a plain string, so
    a double-click or a poll that finished first can never 500;
  * cancel and the status poll's finalize can never both happen — they take the
    same per-guardrail lock, and the cancel claims the row (the conditional
    status write) before it kills anything.
"""
import json
import os
import shutil
import threading
import time
import uuid

import pytest

from utils.sqlite_db import execute_query, execute_query_dict

pytestmark = pytest.mark.integration


def _classifier(client, auth_headers, test_model) -> int:
    res = client.post("/classifiers/create", json={
        "model_id": test_model["model_id"],
        "name": f"cancel-{uuid.uuid4().hex[:8]}",
    }, headers=auth_headers)
    assert res.status_code == 200, res.text
    return res.json().get("classifier", res.json())["classifier_id"]


def _training_row(cid) -> dict:
    return execute_query_dict(
        "SELECT status, model_path, training_log, training_phase, training_phase_detail, "
        "trained_at FROM classifiers WHERE classifier_id = %s", (cid,))[0]


def _fabricate_local_job(cid, pid=None):
    """Put the guardrail into exactly the state a submitted LOCAL run leaves it
    in: status='training', a {job_dir, pid} handle, and a job directory on disk
    holding the run's intermediate files."""
    from classifier_engine.trainer import classifier_workdir
    from services.compute.providers.local import JOB_DIRNAME
    pid = os.getpid() if pid is None else pid
    job_dir = os.path.join(classifier_workdir(cid), JOB_DIRNAME)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, "progress.json"), "w") as f:
        json.dump({"stage": "extract", "progress": 0.2}, f)
    handle = {"provider": "local", "mode": "local",
              "job": {"id": str(pid), "raw": {"job_dir": job_dir, "pid": pid,
                                              "started_at": 0, "device": "cpu"}},
              "chain": ["local"], "chain_pos": 0, "user_id": 1, "last_contact": 0}
    _set_training(cid, json.dumps(handle))
    return job_dir


def _set_training(cid, training_log):
    execute_query(
        "UPDATE classifiers SET status = 'training', training_log = %s, "
        "training_phase = %s, training_phase_detail = %s WHERE classifier_id = %s",
        (training_log, "Extracting embeddings", "Extracting LLM representations — 20%", cid))


def _write_trained_model(cid):
    """A complete model from a PREVIOUS run sitting in the guardrail's workdir."""
    from classifier_engine.trainer import classifier_workdir
    work_dir = classifier_workdir(cid)
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "trained_rnn.pth"), "w") as f:
        f.write("weights")
    with open(os.path.join(work_dir, "classifier_meta.json"), "w") as f:
        json.dump({"labels": {"a": 0}}, f)
    return work_dir


def _cleanup(cid):
    try:
        from classifier_engine.trainer import classifier_workdir
        shutil.rmtree(classifier_workdir(cid), ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def local_cancels(monkeypatch):
    """Record LocalProvider.cancel_training instead of signalling anything — the
    fabricated handles carry this test process's own pid."""
    from services.compute.providers.local import LocalProvider
    calls = []
    monkeypatch.setattr(LocalProvider, "cancel_training", lambda self, job: calls.append(job))
    return calls


@pytest.fixture
def no_calibration(monkeypatch):
    """Auto-calibration would load a real model; record the call instead."""
    import services.auto_calibration as ac
    calls = []
    monkeypatch.setattr(ac, "schedule_post_training_calibration", lambda cid: calls.append(cid))
    return calls


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

class TestCancelWhileTraining:
    def test_cancels_the_child_and_writes_the_run_off(
            self, client, test_model, auth_headers, local_cancels, no_calibration):
        cid = _classifier(client, auth_headers, test_model)
        job_dir = _fabricate_local_job(cid)
        try:
            res = client.post(f"/classifiers/{cid}/training/cancel", headers=auth_headers)
            assert res.status_code == 200, res.text
            assert res.json() == {"success": True, "status": "error"}

            # The actual run was stopped through the provider in the handle.
            assert len(local_cancels) == 1
            assert local_cancels[0].raw["pid"] == os.getpid()

            row = _training_row(cid)
            assert row["status"] == "error"
            # Cancelled, not failed: the phase columns are cleared, and the one
            # line the UI has to explain the state says so.
            assert row["training_phase"] is None
            assert row["training_phase_detail"] is None
            assert "cancel" in str(row["training_log"]).lower()
            # A run that never finished leaves no model behind.
            assert row["model_path"] is None
            # Partial work is gone (the job dir went with the workdir).
            assert not os.path.exists(job_dir)
            # None of the success-path side effects fired.
            assert no_calibration == []
            assert row["trained_at"] is None
        finally:
            _cleanup(cid)

    def test_the_status_route_stops_reporting_a_run_in_progress(
            self, client, test_model, auth_headers, local_cancels):
        # What the UI actually polls: after the cancel the card must show the
        # new state, not a training banner that never moves again.
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_local_job(cid)
        try:
            client.post(f"/classifiers/{cid}/training/cancel", headers=auth_headers)
            body = client.get(f"/classifiers/{cid}/training-status", headers=auth_headers).json()
            assert body["is_training"] is False
            assert body["status"] == "error"
        finally:
            _cleanup(cid)

    def test_a_previous_model_on_disk_is_kept_and_the_row_needs_retraining(
            self, client, test_model, auth_headers, local_cancels):
        # Exactly what crash recovery does for the same situation: a retrain
        # that the user stops must not throw away the model that was working.
        cid = _classifier(client, auth_headers, test_model)
        work_dir = _write_trained_model(cid)
        _fabricate_local_job(cid)
        try:
            res = client.post(f"/classifiers/{cid}/training/cancel", headers=auth_headers)
            assert res.status_code == 200, res.text
            assert res.json()["status"] == "needs_retraining"

            row = _training_row(cid)
            assert row["status"] == "needs_retraining"
            assert row["training_phase"] is None and row["training_phase_detail"] is None
            assert os.path.isfile(os.path.join(work_dir, "trained_rnn.pth"))
            assert os.path.isfile(os.path.join(work_dir, "classifier_meta.json"))
        finally:
            _cleanup(cid)


# ---------------------------------------------------------------------------
# Handles the route must not choke on
# ---------------------------------------------------------------------------

class TestHandleHandling:
    @pytest.mark.parametrize("training_log", [
        None,                       # submit died before the handle was written
        "not json at all",          # legacy free-text log
        json.dumps({"provider": "local"}),   # truncated handle: no job
    ], ids=["missing", "corrupt", "no-job"])
    def test_a_row_with_no_usable_handle_still_flips(
            self, client, test_model, auth_headers, local_cancels, training_log):
        # There is nothing to kill, but the row is still stuck in 'training' —
        # the whole point of the button is getting out of that state.
        cid = _classifier(client, auth_headers, test_model)
        _set_training(cid, training_log)
        try:
            res = client.post(f"/classifiers/{cid}/training/cancel", headers=auth_headers)
            assert res.status_code == 200, res.text
            assert res.json()["status"] in ("error", "needs_retraining")
            assert _training_row(cid)["status"] != "training"
            assert local_cancels == []
        finally:
            _cleanup(cid)

    def test_a_remote_handle_is_cancelled_on_the_remote_provider(
            self, client, test_model, auth_headers, monkeypatch, local_cancels):
        """A job on the GPU worker must be cancelled there — the route reads the
        provider out of the handle instead of assuming the local one."""
        from services.compute.providers.remote_worker import RemoteWorkerProvider
        remote_cancels = []
        monkeypatch.setattr(RemoteWorkerProvider, "cancel_training",
                            lambda self, job: remote_cancels.append(job))

        cid = _classifier(client, auth_headers, test_model)
        _set_training(cid, json.dumps({
            "provider": "remote_worker", "mode": "remote_worker",
            "job": {"id": "worker-job-77", "raw": {}},
            "chain": ["remote_worker", "local"], "chain_pos": 0, "user_id": 1}))
        try:
            res = client.post(f"/classifiers/{cid}/training/cancel", headers=auth_headers)
            assert res.status_code == 200, res.text
            assert len(remote_cancels) == 1
            assert remote_cancels[0].id == "worker-job-77"
            # ...and the local provider was left out of it.
            assert local_cancels == []
            assert _training_row(cid)["status"] == "error"
        finally:
            _cleanup(cid)

    def test_a_provider_that_refuses_to_cancel_does_not_strand_the_row(
            self, client, test_model, auth_headers, monkeypatch):
        # The worker is unreachable / the child is unkillable. Failing the
        # request here would leave the user with a guardrail stuck in
        # 'training' and no way out.
        from services.compute.providers.local import LocalProvider

        def _boom(self, job):
            raise RuntimeError("worker unreachable")
        monkeypatch.setattr(LocalProvider, "cancel_training", _boom)

        cid = _classifier(client, auth_headers, test_model)
        _fabricate_local_job(cid)
        try:
            res = client.post(f"/classifiers/{cid}/training/cancel", headers=auth_headers)
            assert res.status_code == 200, res.text
            assert _training_row(cid)["status"] == "error"
        finally:
            _cleanup(cid)


# ---------------------------------------------------------------------------
# Everything that is not a run in progress
# ---------------------------------------------------------------------------

class TestNothingToCancel:
    @pytest.mark.parametrize("status", ["active", "untrained", "error", "needs_retraining"])
    def test_409_when_the_guardrail_is_not_training(
            self, client, test_model, auth_headers, local_cancels, status):
        cid = _classifier(client, auth_headers, test_model)
        execute_query("UPDATE classifiers SET status = %s WHERE classifier_id = %s", (status, cid))
        res = client.post(f"/classifiers/{cid}/training/cancel", headers=auth_headers)
        assert res.status_code == 409, res.text
        # Plain string, so the UI can show it as-is.
        assert isinstance(res.json()["detail"], str)
        # Nothing was touched.
        assert _training_row(cid)["status"] == status
        assert local_cancels == []

    def test_cancelling_twice_is_a_409_never_a_500(
            self, client, test_model, auth_headers, local_cancels):
        # Two clicks, or two browser tabs. The second one finds the run already
        # stopped — the frontend just refetches on this.
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_local_job(cid)
        try:
            assert client.post(f"/classifiers/{cid}/training/cancel",
                               headers=auth_headers).status_code == 200
            second = client.post(f"/classifiers/{cid}/training/cancel", headers=auth_headers)
            assert second.status_code == 409, second.text
            # The first cancel's outcome survived the second call untouched.
            assert _training_row(cid)["status"] == "error"
            assert len(local_cancels) == 1
        finally:
            _cleanup(cid)

    def test_unknown_guardrail_is_404(self, client, auth_headers):
        res = client.post("/classifiers/99999999/training/cancel", headers=auth_headers)
        assert res.status_code == 404, res.text


# ---------------------------------------------------------------------------
# The race with the status poll's finalize.
#
# A finished run finalizes inside the status route: artifacts, status='active',
# policy snapshot, auto-calibration — all under the per-guardrail lock
# (routes.classifiers._get_download_lock). Cancel takes the SAME lock and, once
# it has it, claims the row with a conditional write BEFORE killing anything.
# So exactly one of the two can happen, whichever order they arrive in.
# ---------------------------------------------------------------------------

class TestFinalizeRace:
    @staticmethod
    def _lock(cid):
        from routes.classifiers import _get_download_lock
        return _get_download_lock(cid)

    @staticmethod
    def _finalize_like_the_poll(cid):
        """What the status route writes when the provider reports DONE: the
        collected artifacts plus the flip to 'active'."""
        work_dir = _write_trained_model(cid)
        execute_query(
            "UPDATE classifiers SET status = 'active', model_path = %s, "
            "training_phase = 'complete', training_phase_detail = %s "
            "WHERE classifier_id = %s AND status = 'training'",
            (os.path.join(work_dir, "trained_rnn.pth"), "Trained on this computer", cid))
        return work_dir

    @staticmethod
    def _cancel_in_thread(client, auth_headers, cid, out):
        def _run():
            res = client.post(f"/classifiers/{cid}/training/cancel", headers=auth_headers)
            out["code"] = res.status_code
            out["body"] = res.json()
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def test_cancel_waits_for_an_in_flight_finalize_and_loses_to_it(
            self, client, test_model, auth_headers, local_cancels, no_calibration):
        """The user hits Cancel while a poll is already finalizing a run that
        finished. The cancel must not cut into the finalize: it waits, finds the
        run 'active', and leaves it completely alone (409, nothing killed, the
        model that was just collected still on disk)."""
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_local_job(cid)
        lock = self._lock(cid)
        out = {}
        lock.acquire()          # stand in for the poll that is mid-finalize
        try:
            t = self._cancel_in_thread(client, auth_headers, cid, out)
            time.sleep(0.4)
            # Still blocked: it has not answered, and — the whole point — it has
            # not written the row off underneath the finalize.
            assert out == {}
            assert _training_row(cid)["status"] == "training"
            work_dir = self._finalize_like_the_poll(cid)
        finally:
            lock.release()
        t.join(timeout=10)
        assert not t.is_alive()

        assert out["code"] == 409, out
        assert isinstance(out["body"]["detail"], str)
        # One outcome, and it is the finalize's: the run stands as trained.
        row = _training_row(cid)
        assert row["status"] == "active"
        assert row["model_path"] == os.path.join(work_dir, "trained_rnn.pth")
        assert row["training_phase"] == "complete"
        # Nothing was killed and nothing was deleted after the fact.
        assert local_cancels == []
        assert os.path.isfile(os.path.join(work_dir, "trained_rnn.pth"))
        assert os.path.isfile(os.path.join(work_dir, "classifier_meta.json"))
        _cleanup(cid)

    def test_cancel_waits_for_the_lock_and_wins_when_the_poll_does_not_finalize(
            self, client, test_model, auth_headers, local_cancels, no_calibration):
        """Same wait, other outcome: the poll that held the lock found the job
        still running, so the run is still 'training' when the lock comes free
        and the cancel takes it — once."""
        cid = _classifier(client, auth_headers, test_model)
        job_dir = _fabricate_local_job(cid)
        lock = self._lock(cid)
        out = {}
        lock.acquire()
        try:
            t = self._cancel_in_thread(client, auth_headers, cid, out)
            time.sleep(0.4)
            assert out == {}                 # waited for the lock
            assert local_cancels == []       # and killed nothing while waiting
        finally:
            lock.release()
        t.join(timeout=10)
        assert not t.is_alive()

        assert out["code"] == 200, out
        assert out["body"] == {"success": True, "status": "error"}
        row = _training_row(cid)
        assert row["status"] == "error"
        assert row["training_phase"] is None and row["training_phase_detail"] is None
        assert "cancel" in str(row["training_log"]).lower()
        assert len(local_cancels) == 1       # exactly one kill
        assert not os.path.exists(job_dir)
        assert no_calibration == []
        _cleanup(cid)

    def test_the_row_is_claimed_before_the_kill_starts(
            self, client, test_model, auth_headers, monkeypatch):
        """Killing a local child costs up to ~8s (SIGTERM grace). The row must
        already be off 'training' when that starts, so nothing can finalize the
        run being killed — and so the UI stops showing a live run."""
        entered, release = threading.Event(), threading.Event()

        from services.compute.providers.local import LocalProvider

        def _slow_cancel(self, job):
            """Stand in for the SIGTERM grace period: the kill is under way and
            the cancel request is still holding the lock."""
            entered.set()
            release.wait(timeout=10)
        monkeypatch.setattr(LocalProvider, "cancel_training", _slow_cancel)

        cid = _classifier(client, auth_headers, test_model)
        _fabricate_local_job(cid)
        out = {}
        t = self._cancel_in_thread(client, auth_headers, cid, out)
        try:
            assert entered.wait(timeout=10), "the provider was never asked to cancel"
            # Mid-kill: the write-off has already landed...
            assert _training_row(cid)["status"] == "error"
            # ...so the status route reports a finished run, not a live one,
            # even though the cancel is still holding the finalize lock.
            body = client.get(f"/classifiers/{cid}/training-status",
                              headers=auth_headers).json()
            assert body["is_training"] is False
            assert body["status"] == "error"
        finally:
            release.set()
        t.join(timeout=15)
        assert out["code"] == 200, out
        _cleanup(cid)

    def test_a_poll_blocked_by_the_finalize_lock_reports_the_row(
            self, client, test_model, auth_headers, local_cancels):
        """Polls never wait on the lock — they answer from the row. While a run
        really is in flight that answer is a progress line; once the cancel's
        write-off has landed (the lock is still held while the child is killed)
        it must be the new state, or the card would bounce back to 'Training'
        for another 5 seconds."""
        cid = _classifier(client, auth_headers, test_model)
        _fabricate_local_job(cid)
        lock = self._lock(cid)
        lock.acquire()
        try:
            body = client.get(f"/classifiers/{cid}/training-status",
                              headers=auth_headers).json()
            assert body["is_training"] is True
            assert body["status"] == "training"

            # The cancel's conditional write, still under the lock.
            from utils.crash_recovery import write_off_training_run
            assert write_off_training_run(
                cid, "Training was cancelled.", failed_phase=False) == "error"

            body = client.get(f"/classifiers/{cid}/training-status",
                              headers=auth_headers).json()
            assert body["is_training"] is False
            assert body["status"] == "error"
            assert "cancel" in str(body["training_log"]).lower()
        finally:
            lock.release()
            _cleanup(cid)
