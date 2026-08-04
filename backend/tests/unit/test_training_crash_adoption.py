"""Boot-time adopt-vs-flip for local training runs.

A local training child lives in its own process session, so it survives the API
restarting — and under `uvicorn --reload` the API restarts on every file save.
StuckTrainingRecovery therefore can't treat "status = training on boot" as proof
the run is dead any more. This pins the decision logic itself: which handles get
adopted, which get written off, and that nothing is left half-decided.

Pure logic — the provider and the DB write are both stubbed, so no process is
spawned and no database is touched.
"""
import json

import pytest

from services.compute.base import JobState, TrainingStatus
from utils.crash_recovery import StuckTrainingRecovery

pytestmark = pytest.mark.unit


class _FakeProvider:
    """Stands in for LocalProvider with scripted answers."""

    def __init__(self, alive=False, state=JobState.ERROR):
        self.name = "local"
        self._alive = alive
        self._state = state
        self.cancelled = []

    def training_is_alive(self, job):
        return self._alive

    def poll_training(self, job):
        return TrainingStatus(state=self._state)

    def cancel_training(self, job):
        self.cancelled.append(job)


@pytest.fixture
def stub(monkeypatch):
    """Install a fake provider + capture the phase write."""
    import services.compute as compute
    import utils.sqlite_db as db

    writes = []
    monkeypatch.setattr(db, "execute_query", lambda *a, **k: writes.append(a))

    def _install(provider, name="local"):
        monkeypatch.setattr(compute, "provider_by_name",
                            lambda n: provider if n == name else None)
        return writes

    return _install


def _handle(pid=4321, job_dir="/tmp/gavel/job"):
    return json.dumps({"provider": "local", "mode": "local",
                       "job": {"id": str(pid),
                               "raw": {"job_dir": job_dir, "pid": pid}}})


class TestAdoptVsFlip:
    def test_live_child_is_adopted(self, stub):
        p = _FakeProvider(alive=True)
        writes = stub(p)
        assert StuckTrainingRecovery._adopt_if_alive(7, _handle()) is True
        # Adoption must leave the row in 'training' (only the phase text is
        # touched) so the normal status poll goes on to finalize the run.
        sql = writes[0][0]
        assert "training_phase" in sql
        assert "SET status" not in sql

    def test_dead_child_is_not_adopted(self, stub):
        p = _FakeProvider(alive=False, state=JobState.ERROR)
        stub(p)
        assert StuckTrainingRecovery._adopt_if_alive(7, _handle()) is False

    def test_run_that_finished_while_the_server_was_down_is_adopted_to_be_finalized(self, stub):
        # The child completed, wrote its artifacts, and exited — then the API
        # restarted before anything polled it. Failing that would throw away a
        # finished model, so it stays 'training' for the next poll to finalize.
        p = _FakeProvider(alive=False, state=JobState.DONE)
        stub(p)
        assert StuckTrainingRecovery._adopt_if_alive(7, _handle()) is True

    def test_remote_handles_are_never_adopted(self, stub):
        # A remote worker job cannot be resumed by this process; it goes down
        # the normal reset path.
        stub(_FakeProvider(alive=True))
        remote = json.dumps({"provider": "remote_worker", "mode": "remote_worker",
                             "job": {"id": "abc", "raw": {}}})
        assert StuckTrainingRecovery._adopt_if_alive(7, remote) is False

    def test_row_with_no_handle_is_not_adopted(self, stub):
        stub(_FakeProvider(alive=True))
        assert StuckTrainingRecovery._adopt_if_alive(7, None) is False
        assert StuckTrainingRecovery._adopt_if_alive(7, "Training failed: boom") is False
        assert StuckTrainingRecovery._adopt_if_alive(7, "{}") is False

    def test_provider_blowing_up_falls_back_to_not_adopting(self, stub, monkeypatch):
        # Never leave a guardrail in 'training' on an inconclusive answer: the
        # user can retrain out of 'error', but nothing rescues a row that is
        # stuck 'training' with no process behind it.
        class _Exploding(_FakeProvider):
            def training_is_alive(self, job):
                raise RuntimeError("procfs unreadable")
        stub(_Exploding())
        assert StuckTrainingRecovery._adopt_if_alive(7, _handle()) is False

    def test_missing_provider_is_not_adopted(self, monkeypatch):
        import services.compute as compute
        monkeypatch.setattr(compute, "provider_by_name", lambda n: None)
        assert StuckTrainingRecovery._adopt_if_alive(7, _handle()) is False


class TestStaleCleanup:
    def test_a_written_off_local_run_gets_a_cancel_through_its_provider(self, stub):
        # A local child survives the restart, so a remnant we've decided to fail
        # can still be holding THIS machine's GPU — killing it is the only way to
        # get it back before the guardrail is marked failed.
        p = _FakeProvider()
        stub(p)
        StuckTrainingRecovery._cancel_stale(7, _handle())
        assert len(p.cancelled) == 1
        assert p.cancelled[0].raw["job_dir"] == "/tmp/gavel/job"

    def test_a_remote_handle_is_left_alone(self, stub):
        # Boot recovery only flips the row for a remote job, exactly as it did
        # before local training moved out of process. Reaching across to a shared
        # worker on the strength of a handle we already consider unreliable is
        # not this strategy's business.
        p = _FakeProvider()
        stub(p, name="remote_worker")
        remote = json.dumps({"provider": "remote_worker", "mode": "remote_worker",
                             "job": {"id": "abc", "raw": {"worker_job_id": "abc"}}})
        StuckTrainingRecovery._cancel_stale(7, remote)
        assert p.cancelled == []

    def test_a_legacy_remote_handle_is_left_alone(self, stub):
        p = _FakeProvider()
        stub(p, name="remote_worker")
        StuckTrainingRecovery._cancel_stale(
            7, json.dumps({"mode": "remote_worker", "worker_job_id": "77"}))
        assert p.cancelled == []

    def test_no_handle_means_nothing_to_cancel(self, stub):
        p = _FakeProvider()
        stub(p)
        StuckTrainingRecovery._cancel_stale(7, None)
        assert p.cancelled == []

    def test_cancel_failure_is_swallowed(self, stub):
        class _Exploding(_FakeProvider):
            def cancel_training(self, job):
                raise RuntimeError("kill failed")
        stub(_Exploding())
        StuckTrainingRecovery._cancel_stale(7, _handle())   # must not raise
