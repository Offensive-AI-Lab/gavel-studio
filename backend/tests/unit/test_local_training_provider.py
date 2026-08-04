"""LocalProvider's training methods — submit / poll / fetch / cancel.

No DB, no torch, no real trainer: the child is a tiny stub script written per
test, so we can drive it into exactly the states the real one reaches (running,
success, hard death with no status.json) in milliseconds. What is under test is
the PARENT's contract — what it stages on disk, how it maps a job directory to a
poll result, that a cancel takes out the whole process group, and that the
PID + job-dir handshake refuses to touch a process that isn't ours.
"""
import json
import os
import shutil
import signal
import sys
import textwrap
import time

import pytest

from services.compute.base import JobState, TrainingJob, TrainingSpec
from services.compute.providers import local as local_mod
from services.compute.providers.local import LocalProvider

pytestmark = pytest.mark.unit


def _write_stub(tmp_path, body: str) -> str:
    """A stand-in for compute_jobs/train_job.py, taking the same --job-dir /
    --device arguments."""
    path = tmp_path / "stub_train_job.py"
    path.write_text(textwrap.dedent(f"""\
        import argparse, json, os, sys, time
        p = argparse.ArgumentParser()
        p.add_argument("--job-dir", required=True)
        p.add_argument("--device", default="auto")
        args = p.parse_args()
        job_dir = args.job_dir

        def progress(stage, detail="", frac=0.0):
            with open(os.path.join(job_dir, "progress.json"), "w") as f:
                json.dump({{"stage": stage, "detail": detail, "progress": frac}}, f)

        def status(s, error=None):
            d = {{"status": s}}
            if error:
                d["error"] = error
            with open(os.path.join(job_dir, "status.json"), "w") as f:
                json.dump(d, f)

    """) + textwrap.dedent(body))
    return str(path)


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """LocalProvider whose child script is a stub and whose workdir is tmp."""
    p = LocalProvider()
    monkeypatch.setattr(p, "_device_type", lambda: "cpu")
    return p


def _spec(tmp_path, **kw):
    kw.setdefault("output_dir", str(tmp_path / "workdir"))
    return TrainingSpec(
        classifier_id=42, user_id=1, model_hf_path="stub/model",
        labels={"a": 0, "b": 1},
        training_config={"epochs": 1},
        dataset_files={"a.json": [{"conversation": "x"}], "b.json": [{"conversation": "y"}]},
        calibration_entries=[{"conversation": "c"}],
        **kw,
    )


def _submit(provider, tmp_path, monkeypatch, body) -> TrainingJob:
    monkeypatch.setattr(local_mod, "_train_script", lambda: _write_stub(tmp_path, body))
    monkeypatch.setattr(provider, "_hf_token_for", lambda ref: None)
    return provider.submit_training(_spec(tmp_path))


def _wait_for(pred, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _poll_until(provider, job, state, timeout=20.0):
    """Poll until the job reports `state`, returning THAT status.

    Terminal polls are one-shot for a failure — they reap the child and drop the
    job directory — so a test can't poll for the state and then poll again for
    the message.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = provider.poll_training(job)
        if st.state == state:
            return st
        time.sleep(0.05)
    raise AssertionError(f"job never reached {state}")


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return False


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

class TestSubmitTraining:
    def test_stages_payload_and_datasets_into_the_job_dir(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, "time.sleep(30)\n")
        try:
            job_dir = job.raw["job_dir"]
            payload = json.load(open(os.path.join(job_dir, "job_payload.json")))
            assert payload["classifier_id"] == 42
            assert payload["labels"] == {"a": 0, "b": 1}
            assert payload["config"] == {"epochs": 1}
            assert payload["model_hf_path"] == "stub/model"
            # Datasets land as one file per CE, exactly where train_job.py looks.
            assert sorted(os.listdir(os.path.join(job_dir, "dataset"))) == ["a.json", "b.json"]
            assert json.load(open(os.path.join(job_dir, "dataset", "a.json"))) == [{"conversation": "x"}]
            assert json.load(open(os.path.join(job_dir, "calibration_input.json"))) == [{"conversation": "c"}]
        finally:
            provider.cancel_training(job)

    def test_handle_is_durable_and_child_runs_in_its_own_process_group(
            self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, "time.sleep(30)\n")
        try:
            assert job.provider == "local"
            assert set(job.raw) >= {"job_dir", "pid", "started_at"}
            assert job.id == str(job.raw["pid"])
            # Everything the handle needs to be re-read after a restart is JSON.
            assert json.loads(json.dumps(job.raw))["pid"] == job.raw["pid"]
            # Own session => own process group, so cancel can signal the whole
            # tree (the real child forks DataLoader workers).
            assert os.getpgid(job.raw["pid"]) != os.getpgid(os.getpid())
        finally:
            provider.cancel_training(job)

    def test_retrain_wipes_the_previous_workdir(self, provider, tmp_path, monkeypatch):
        work_dir = tmp_path / "workdir"
        work_dir.mkdir()
        (work_dir / "trained_rnn.pth").write_text("stale weights")
        job = _submit(provider, tmp_path, monkeypatch, "time.sleep(30)\n")
        try:
            assert not (work_dir / "trained_rnn.pth").exists()
        finally:
            provider.cancel_training(job)

    def test_child_env_disables_tokenizer_parallelism_and_exports_backend_on_path(self, provider):
        env = provider._child_env()
        assert env["TOKENIZERS_PARALLELISM"] == "false"
        assert env["PYTHONPATH"].split(os.pathsep)[0].endswith("backend")

    def test_child_env_passes_through_force_cpu(self, provider, monkeypatch):
        monkeypatch.setenv("GAVEL_FORCE_CPU", "1")
        assert provider._child_env()["GAVEL_FORCE_CPU"] == "1"

    def test_spawn_failure_raises_compute_error(self, provider, tmp_path, monkeypatch):
        from services.compute.base import ComputeError
        monkeypatch.setattr(local_mod, "_train_script", lambda: str(tmp_path / "nope.py"))
        monkeypatch.setattr(provider, "_hf_token_for", lambda ref: None)

        def _boom(*a, **k):
            raise OSError("fork failed")
        monkeypatch.setattr(local_mod.subprocess, "Popen", _boom)
        with pytest.raises(ComputeError):
            provider.submit_training(_spec(tmp_path))


# ---------------------------------------------------------------------------
# poll
# ---------------------------------------------------------------------------

class TestPollTraining:
    def test_running_child_maps_stage_to_a_user_facing_phase(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, """\
            progress("extract", "Extracting LLM representations for train set", 0.2)
            time.sleep(30)
        """)
        try:
            assert _wait_for(lambda: os.path.exists(os.path.join(job.raw["job_dir"], "progress.json")))
            st = provider.poll_training(job)
            assert st.state == JobState.RUNNING
            assert st.phase == "Extracting embeddings"     # not the raw "extract"
            assert st.detail == "Extracting LLM representations for train set"
            assert st.progress == 0.2
            assert st.reachable is True
        finally:
            provider.cancel_training(job)

    def test_unknown_stage_degrades_to_a_readable_label(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, """\
            progress("some_new_stage", "hello")
            time.sleep(30)
        """)
        try:
            assert _wait_for(lambda: os.path.exists(os.path.join(job.raw["job_dir"], "progress.json")))
            assert provider.poll_training(job).phase == "Some New Stage"
        finally:
            provider.cancel_training(job)

    def test_success_becomes_done_and_carries_the_training_log(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, """\
            with open(os.path.join(job_dir, "training_log.json"), "w") as f:
                json.dump([{"progress": 100, "val_accuracy": 0.9}], f)
            status("success")
        """)
        assert _wait_for(lambda: provider.poll_training(job).state == JobState.DONE)
        st = provider.poll_training(job)
        assert st.log == [{"progress": 100, "val_accuracy": 0.9}]

    def test_failure_becomes_error_with_the_child_s_message(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, """\
            status("failed", "CUDA out of memory.")
            sys.exit(2)
        """)
        assert "out of memory" in _poll_until(provider, job, JobState.ERROR).error

    def test_child_that_dies_without_a_result_is_an_error_not_an_eternal_running(
            self, provider, tmp_path, monkeypatch):
        # The OOM-killer / a segfault: no status.json is ever written. Polling
        # must NOT keep reporting 'running' against a dead PID forever.
        job = _submit(provider, tmp_path, monkeypatch, """\
            print("about to die")
            os._exit(9)
        """)
        st = _poll_until(provider, job, JobState.ERROR)
        assert "stopped without producing a result" in st.error
        assert "about to die" in st.error   # the log tail is the only clue there is

    def test_missing_job_dir_is_an_error(self, provider):
        job = TrainingJob(provider="local", classifier_id=1, id="1",
                          raw={"job_dir": "/nonexistent/job", "pid": 1})
        assert provider.poll_training(job).state == JobState.ERROR


# ---------------------------------------------------------------------------
# reaping + job-dir hygiene on the way out
# ---------------------------------------------------------------------------

class TestTerminalOutcomeReapsTheChild:
    """Every terminal outcome — success, failure, cancel — must wait() the child.

    An unwaited child stays a zombie for the whole life of the API process, and
    a zombie still answers kill(pid, 0): the next handle that lands on that
    number reads as "still training" forever. The success path is the easy one
    to miss, because poll short-circuits on status.json without ever looking at
    the process.
    """

    def _assert_reaped(self, pid):
        assert pid not in local_mod._procs      # registry entry dropped
        assert not _is_zombie(pid)
        with pytest.raises(ChildProcessError):  # already wait()ed
            os.waitpid(pid, os.WNOHANG)
        assert local_mod._pid_alive(pid) is False

    def test_success_poll_reaps(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, 'status("success")\n')
        _poll_until(provider, job, JobState.DONE)
        self._assert_reaped(job.raw["pid"])

    def test_failure_poll_reaps(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, """\
            status("failed", "boom")
            sys.exit(2)
        """)
        _poll_until(provider, job, JobState.ERROR)
        self._assert_reaped(job.raw["pid"])

    def test_child_that_died_without_a_result_is_reaped(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, "os._exit(9)\n")
        _poll_until(provider, job, JobState.ERROR)
        self._assert_reaped(job.raw["pid"])

    def test_cancel_reaps(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, "time.sleep(60)\n")
        assert _wait_for(lambda: local_mod._pid_alive(job.raw["pid"]))
        provider.cancel_training(job)
        self._assert_reaped(job.raw["pid"])


class TestFailedRunDropsItsJobDir:
    """A failed run has nothing to collect, so nothing will ever read its job
    directory again — and it holds the staged datasets plus the multi-GB
    extracted representations. Only the SUCCESS path may leave it in place, for
    fetch_trained_model (and for boot recovery, which re-reads a job dir to
    adopt a run that finished while the server was down)."""

    def test_reported_failure_removes_the_job_dir(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, """\
            with open(os.path.join(job_dir, "representations.pt"), "w") as f:
                f.write("x" * 1024)
            status("failed", "CUDA out of memory.")
            sys.exit(2)
        """)
        st = _poll_until(provider, job, JobState.ERROR)
        assert "out of memory" in st.error   # message composed before the delete
        assert not os.path.exists(job.raw["job_dir"])

    def test_hard_death_removes_the_job_dir_after_reading_the_log_tail(
            self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, """\
            print("about to die")
            os._exit(9)
        """)
        st = _poll_until(provider, job, JobState.ERROR)
        assert "about to die" in st.error    # tail extracted before the delete
        assert not os.path.exists(job.raw["job_dir"])

    def test_success_keeps_the_job_dir_for_the_finalize_step(
            self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, """\
            with open(os.path.join(job_dir, "trained_rnn.pth"), "w") as f:
                f.write("weights")
            status("success")
        """)
        _poll_until(provider, job, JobState.DONE)
        assert os.path.isfile(os.path.join(job.raw["job_dir"], "trained_rnn.pth"))


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

class TestFetchTrainedModel:
    def test_moves_artifacts_up_into_the_workdir_and_drops_the_job_dir(
            self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, """\
            with open(os.path.join(job_dir, "trained_rnn.pth"), "w") as f:
                f.write("weights")
            with open(os.path.join(job_dir, "classifier_meta.json"), "w") as f:
                json.dump({"labels": {"a": 0}}, f)
            status("success")
        """)
        assert _wait_for(lambda: provider.poll_training(job).state == JobState.DONE)

        dest = tmp_path / "workdir"
        provider.fetch_trained_model(job, str(dest))
        assert (dest / "trained_rnn.pth").read_text() == "weights"
        assert json.load(open(dest / "classifier_meta.json")) == {"labels": {"a": 0}}
        # Staged inputs + extracted representations are dead weight afterwards.
        assert not os.path.exists(job.raw["job_dir"])

    def test_missing_artifacts_raise_rather_than_marking_a_guardrail_active(
            self, provider, tmp_path, monkeypatch):
        from services.compute.base import ComputeError
        job = _submit(provider, tmp_path, monkeypatch, 'status("success")\n')
        assert _wait_for(lambda: provider.poll_training(job).state == JobState.DONE)
        with pytest.raises(ComputeError):
            provider.fetch_trained_model(job, str(tmp_path / "workdir"))


# ---------------------------------------------------------------------------
# cancel + the PID/job-dir handshake
# ---------------------------------------------------------------------------

class TestCancelTraining:
    def test_kills_the_whole_process_group_not_just_the_leader(
            self, provider, tmp_path, monkeypatch):
        # The real child forks 4 DataLoader workers; killing only the leader
        # orphans them and they keep holding the GPU. Mirror that shape: the
        # stub forks a grandchild that ignores SIGTERM's usual inheritance path.
        job = _submit(provider, tmp_path, monkeypatch, """\
            pid = os.fork()
            if pid == 0:
                with open(os.path.join(job_dir, "worker.pid"), "w") as f:
                    f.write(str(os.getpid()))
                time.sleep(60)
                os._exit(0)
            time.sleep(60)
        """)
        worker_file = os.path.join(job.raw["job_dir"], "worker.pid")
        assert _wait_for(lambda: os.path.exists(worker_file))
        worker_pid = int(open(worker_file).read())

        provider.cancel_training(job)

        assert not local_mod._pid_alive(job.raw["pid"])
        assert _wait_for(lambda: not local_mod._pid_alive(worker_pid), timeout=5)

    def test_escalates_to_sigkill_when_the_child_ignores_sigterm(
            self, provider, tmp_path, monkeypatch):
        monkeypatch.setattr(local_mod, "_CANCEL_GRACE_S", 1.0)
        job = _submit(provider, tmp_path, monkeypatch, """\
            import signal as _s
            _s.signal(_s.SIGTERM, _s.SIG_IGN)
            with open(os.path.join(job_dir, "ready"), "w") as f:
                f.write("1")
            time.sleep(60)
        """)
        assert _wait_for(lambda: os.path.exists(os.path.join(job.raw["job_dir"], "ready")))
        provider.cancel_training(job)
        assert not local_mod._pid_alive(job.raw["pid"])

    def test_cancels_a_child_whose_job_dir_is_already_gone(
            self, provider, tmp_path, monkeypatch):
        """The job directory is not what proves ownership — the Popen we are
        holding is. A terminal poll (or a workdir sweep) can remove the dir while
        the child is still very much alive, and gating the cancel on the dir
        turns it into a no-op on exactly the process we most need to kill."""
        job = _submit(provider, tmp_path, monkeypatch, "time.sleep(60)\n")
        pid = job.raw["pid"]
        assert _wait_for(lambda: local_mod._pid_alive(pid))
        shutil.rmtree(job.raw["job_dir"])

        provider.cancel_training(job)

        assert not local_mod._pid_alive(pid)
        assert pid not in local_mod._procs

    def test_is_a_no_op_on_an_already_finished_job(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, 'status("success")\n')
        assert _wait_for(lambda: provider.poll_training(job).state == JobState.DONE)
        provider.cancel_training(job)   # must not raise
        provider.cancel_training(job)   # idempotent

    def test_refuses_to_signal_a_pid_that_is_not_our_training_child(
            self, provider, tmp_path, monkeypatch):
        # PID reuse after a reboot: the recorded number belongs to somebody else
        # entirely. Killing it would be a serious bug, so the handshake must
        # reject it — here, THIS pytest process.
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        job = TrainingJob(provider="local", classifier_id=1, id=str(os.getpid()),
                          raw={"job_dir": str(job_dir), "pid": os.getpid()})
        killed = []

        def _spy(target, sig):
            if sig:   # signal 0 is the liveness probe, not a kill
                killed.append((target, sig))
        monkeypatch.setattr(local_mod.os, "killpg", _spy)
        monkeypatch.setattr(local_mod.os, "kill", _spy)
        provider.cancel_training(job)
        assert killed == []


class TestAdoptVsFlipHandshake:
    """The boot-time decision: is this handle's run still ours and alive?"""

    def test_live_child_with_matching_job_dir_is_adopted(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, "time.sleep(30)\n")
        try:
            assert provider.training_is_alive(job) is True
        finally:
            provider.cancel_training(job)

    def test_dead_pid_is_not_adopted(self, provider, tmp_path, monkeypatch):
        job = _submit(provider, tmp_path, monkeypatch, 'status("success")\n')
        assert _wait_for(lambda: not local_mod._pid_alive(job.raw["pid"]))
        assert provider.training_is_alive(job) is False

    def test_live_pid_whose_job_dir_vanished_is_not_adopted(self, provider, tmp_path, monkeypatch):
        # Boot recovery always runs in a FRESH process, so the spawn registry is
        # empty and the handshake is the on-disk one: no job directory means
        # nothing left to finalize from, so the handle is not adopted. (Drop the
        # entry by hand to reproduce that, then put it back so the child dies.)
        job = _submit(provider, tmp_path, monkeypatch, "time.sleep(30)\n")
        pid = job.raw["pid"]
        shutil.rmtree(job.raw["job_dir"])
        entry = local_mod._procs.pop(pid)
        try:
            assert provider.training_is_alive(job) is False
        finally:
            local_mod._procs[pid] = entry
            provider.cancel_training(job)

    def test_live_but_unrelated_pid_is_not_adopted(self, provider, tmp_path):
        # A recycled PID belonging to some other program (this pytest process).
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        job = TrainingJob(provider="local", classifier_id=1, id=str(os.getpid()),
                          raw={"job_dir": str(job_dir), "pid": os.getpid()})
        assert provider.training_is_alive(job) is False

    def test_garbage_handle_is_not_adopted(self, provider):
        job = TrainingJob(provider="local", classifier_id=1, id="x", raw={})
        assert provider.training_is_alive(job) is False


class TestPidLiveness:
    def test_zombie_child_does_not_read_as_alive(self, tmp_path):
        """kill(pid, 0) succeeds against a zombie; a finished run must not look
        like it is still training just because nobody reaped it yet."""
        import subprocess
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()   # reaped by us
        assert local_mod._pid_alive(proc.pid) is False

    def test_pid_zero_is_never_alive(self):
        assert local_mod._pid_alive(0) is False
        assert local_mod._pid_alive(None) is False
