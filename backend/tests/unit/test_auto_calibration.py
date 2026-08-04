"""Pure unit tests for services.auto_calibration + the realtime guard it backs.

Covers the two halves of "never monitor silently uncalibrated":

  * calibration_state / is_calibrated — reading the post-train calibration rows
  * schedule_post_training_calibration — idempotence, thread hand-off, and the
    swallow-everything contract (training must never fail because calibration
    couldn't be scheduled)
  * _require_calibrated_classifier — the 409 that blocks monitoring

Every DB call goes through `execute_query_dict`, imported at module top into
services.auto_calibration, so we monkeypatch it there. `_run_calibration` is
imported lazily inside the scheduler, so we patch it on routes.evaluation.
No DB, no torch, no threads that outlive a test.
"""
import pytest

import services.auto_calibration as ac
import routes.realtime as rt
from fastapi import HTTPException


def _rows(*eval_types):
    return [{"eval_type": t} for t in eval_types]


# ---------------------------------------------------------------------------
# calibration_state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rows,expected", [
    ([], "missing"),
    (_rows("calibration"), "calibrated"),
    (_rows("calibration_running"), "running"),
    # A finished run wins over an older running marker — the marker row is left
    # behind by _run_calibration, so "both present" means done.
    (_rows("calibration", "calibration_running"), "calibrated"),
])
def test_calibration_state(monkeypatch, rows, expected):
    monkeypatch.setattr(ac, "execute_query_dict", lambda *a, **k: rows)
    assert ac.calibration_state(7) == expected
    assert ac.is_calibrated(7) is (expected == "calibrated")


def test_calibration_state_tolerates_null_result(monkeypatch):
    """execute_query_dict can return None; that must read as 'missing', not blow up."""
    monkeypatch.setattr(ac, "execute_query_dict", lambda *a, **k: None)
    assert ac.calibration_state(7) == "missing"


# ---------------------------------------------------------------------------
# schedule_post_training_calibration
# ---------------------------------------------------------------------------

def test_schedule_runs_calibration_when_missing(monkeypatch):
    monkeypatch.setattr(ac, "calibration_state", lambda cid: "missing")
    called = []
    import routes.evaluation as ev
    monkeypatch.setattr(ev, "_run_calibration", lambda cid: called.append(cid))

    started = []
    class _FakeThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target
        def start(self):
            started.append(True)
            self._target()          # run inline so the assertion is deterministic
    monkeypatch.setattr(ac.threading, "Thread", _FakeThread)

    assert ac.schedule_post_training_calibration(42) is True
    assert started == [True]
    assert called == [42]


@pytest.mark.parametrize("state", ["calibrated", "running"])
def test_schedule_is_idempotent(monkeypatch, state):
    """The local trainer and the remote-worker poll both call this — a second call
    while calibrated/in-flight must not submit a duplicate run."""
    monkeypatch.setattr(ac, "calibration_state", lambda cid: state)
    def _boom(*a, **k):
        raise AssertionError("should not spawn a calibration thread")
    monkeypatch.setattr(ac.threading, "Thread", _boom)

    assert ac.schedule_post_training_calibration(42) is False


@pytest.mark.parametrize("value,enabled", [
    ("0", False), ("false", False), ("no", False), ("off", False), ("OFF", False),
    ("1", True), ("true", True), ("", True),
])
def test_env_gate(monkeypatch, value, enabled):
    """AUTO_CALIBRATE_AFTER_TRAINING=0 turns the chain off; default is on."""
    monkeypatch.setenv("AUTO_CALIBRATE_AFTER_TRAINING", value)
    monkeypatch.setattr(ac, "calibration_state", lambda cid: "missing")

    spawned = []
    class _FakeThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target
        def start(self):
            spawned.append(True)
    monkeypatch.setattr(ac.threading, "Thread", _FakeThread)

    assert ac.schedule_post_training_calibration(42) is enabled
    assert bool(spawned) is enabled


def test_env_gate_defaults_on(monkeypatch):
    monkeypatch.delenv("AUTO_CALIBRATE_AFTER_TRAINING", raising=False)
    assert ac._auto_calibration_enabled() is True


def test_schedule_never_raises(monkeypatch):
    """A scheduling failure must not turn a successful training run into a
    failed one — it returns False and logs instead of propagating."""
    def _boom(cid):
        raise RuntimeError("db down")
    monkeypatch.setattr(ac, "calibration_state", _boom)

    assert ac.schedule_post_training_calibration(42) is False


def test_worker_swallows_calibration_errors(monkeypatch):
    """_run_calibration writes its own error row; an exception escaping the
    thread would only produce a noisy unhandled-exception traceback."""
    monkeypatch.setattr(ac, "calibration_state", lambda cid: "missing")
    import routes.evaluation as ev
    def _boom(cid):
        raise RuntimeError("calibration exploded")
    monkeypatch.setattr(ev, "_run_calibration", _boom)

    class _FakeThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target
        def start(self):
            self._target()          # must not raise
    monkeypatch.setattr(ac.threading, "Thread", _FakeThread)

    assert ac.schedule_post_training_calibration(42) is True


# ---------------------------------------------------------------------------
# _require_calibrated_classifier — the monitoring block
# ---------------------------------------------------------------------------

def test_require_calibrated_passes_when_calibrated(monkeypatch):
    monkeypatch.setattr(ac, "calibration_state", lambda cid: "calibrated")
    rt._require_calibrated_classifier(3)        # no raise


@pytest.mark.parametrize("state,marker", [
    ("missing", "uncalibrated"),
    ("running", "still running"),
])
def test_require_calibrated_blocks_with_409(monkeypatch, state, marker):
    monkeypatch.setattr(ac, "calibration_state", lambda cid: state)
    with pytest.raises(HTTPException) as exc:
        rt._require_calibrated_classifier(3)
    # 409, not 400 — the frontend keys off it to show the Calibration CTA
    # instead of the generic "session failed" error.
    assert exc.value.status_code == 409
    assert marker in exc.value.detail


# ---------------------------------------------------------------------------
# run_post_calibration_evaluation — stage 2 of the chain
# ---------------------------------------------------------------------------

def _patch_eval(monkeypatch, *, already=False, pairs=None, run=None):
    import routes.evaluation as ev
    monkeypatch.setattr(ev, "_has_post_train_success", lambda cid, kind: already)
    monkeypatch.setattr(ev, "_load_default_eval_pairs", lambda cid: pairs if pairs is not None else [([], "positive")])
    monkeypatch.setattr(ev, "_run_evaluation", run or (lambda cid, pairs, **kw: None))


def test_evaluation_runs_after_calibration(monkeypatch):
    calls = []
    _patch_eval(monkeypatch, run=lambda cid, pairs, **kw: calls.append((cid, pairs)))
    assert ac.run_post_calibration_evaluation(42) is True
    assert calls and calls[0][0] == 42


@pytest.mark.parametrize("kwargs,reason", [
    ({"already": True}, "already evaluated for this training"),
    ({"pairs": []}, "no default test sets to evaluate against"),
])
def test_evaluation_skipped(monkeypatch, kwargs, reason):
    def _boom(*a, **k):
        raise AssertionError(f"should not evaluate: {reason}")
    _patch_eval(monkeypatch, run=_boom, **kwargs)
    assert ac.run_post_calibration_evaluation(42) is False


def test_evaluation_respects_env_gate(monkeypatch):
    monkeypatch.setenv("AUTO_EVALUATE_AFTER_CALIBRATION", "0")
    def _boom(*a, **k):
        raise AssertionError("should not evaluate when disabled")
    _patch_eval(monkeypatch, run=_boom)
    assert ac.run_post_calibration_evaluation(42) is False


def test_evaluation_failure_never_propagates(monkeypatch):
    """It runs at the tail of the calibration worker — a failed evaluation must
    not take a good calibration down with it."""
    def _boom(cid, pairs, **kw):
        raise RuntimeError("evaluation exploded")
    _patch_eval(monkeypatch, run=_boom)
    assert ac.run_post_calibration_evaluation(42) is False


def test_chain_stops_when_calibration_did_not_succeed(monkeypatch):
    """Evaluation reads the calibrated thresholds, so a failed sweep must not
    be followed by an evaluation that can only fail too."""
    monkeypatch.setattr(ac, "calibration_state", lambda cid: "missing")
    import routes.evaluation as ev
    monkeypatch.setattr(ev, "_run_calibration", lambda cid: None)   # "ran", wrote nothing
    monkeypatch.setattr(ac, "run_post_calibration_evaluation",
                        lambda cid: (_ for _ in ()).throw(AssertionError("should not chain")))

    class _FakeThread:
        def __init__(self, target, name=None, daemon=None):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(ac.threading, "Thread", _FakeThread)

    assert ac.schedule_post_training_calibration(42) is True


# ---------------------------------------------------------------------------
# policy_drifted — the "edited but not retrained" hole
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,drifted", [
    ("needs_retraining", True),
    ("active", False),
    ("untrained", False),
])
def test_policy_drifted(monkeypatch, status, drifted):
    import sql_scripts.model_scripts as ms
    monkeypatch.setattr(ms, "reconcile_classifier_status", lambda cid: status)
    assert ac.policy_drifted(5) is drifted


def test_policy_drifted_fails_open(monkeypatch):
    """A broken drift check must not block monitoring on its own — the
    calibration check still applies."""
    import sql_scripts.model_scripts as ms
    def _boom(cid):
        raise RuntimeError("db down")
    monkeypatch.setattr(ms, "reconcile_classifier_status", _boom)
    assert ac.policy_drifted(5) is False


def test_require_calibrated_blocks_on_policy_drift(monkeypatch):
    """Calibrated by the row check, but the policy moved underneath it: the
    thresholds are post-train yet tuned for a policy the model no longer
    reflects, so monitoring is refused and the message points at retraining."""
    monkeypatch.setattr(ac, "policy_drifted", lambda cid: True)
    monkeypatch.setattr(ac, "calibration_state", lambda cid: "calibrated")
    with pytest.raises(HTTPException) as exc:
        rt._require_calibrated_classifier(3)
    assert exc.value.status_code == 409
    assert "retrain" in exc.value.detail.lower()


def test_require_calibrated_passes_without_drift(monkeypatch):
    monkeypatch.setattr(ac, "policy_drifted", lambda cid: False)
    monkeypatch.setattr(ac, "calibration_state", lambda cid: "calibrated")
    rt._require_calibrated_classifier(3)        # no raise


# ---------------------------------------------------------------------------
# chain_progress — what the rule-set page shows after training finishes
# ---------------------------------------------------------------------------

def _chain_rows(monkeypatch, cal=None, ev=None):
    """Serve the two 'latest row per kind' queries chain_progress issues."""
    calls = []
    def _fake(sql, params):
        calls.append(params)
        kinds = params[1]
        row = cal if "calibration" in kinds[0] else ev
        return [row] if row else []
    monkeypatch.setattr(ac, "execute_query_dict", _fake)
    return calls


def test_chain_progress_reports_calibrating(monkeypatch):
    _chain_rows(monkeypatch, cal={"eval_type": "calibration_running",
                                  "metrics": {"phase": "Loading calibration datasets…"}})
    assert ac.chain_progress(1) == {"phase": "Calibrating",
                                    "detail": "Loading calibration datasets…"}


def test_chain_progress_reports_evaluating(monkeypatch):
    _chain_rows(monkeypatch,
                cal={"eval_type": "calibration", "metrics": None},
                ev={"eval_type": "evaluation_running", "metrics": '{"phase": "Scoring…"}'})
    # metrics arrives as a JSON string on some paths — must still be read.
    assert ac.chain_progress(1) == {"phase": "Evaluating", "detail": "Scoring…"}


def test_chain_progress_falls_back_when_phase_missing(monkeypatch):
    _chain_rows(monkeypatch, cal={"eval_type": "calibration_running", "metrics": None})
    assert ac.chain_progress(1)["detail"] == "Tuning per-CE thresholds…"


@pytest.mark.parametrize("cal,ev", [
    (None, None),                                                     # nothing ran
    ({"eval_type": "calibration", "metrics": None}, None),            # calibration done, no eval
    ({"eval_type": "calibration_error", "metrics": None}, None),      # calibration failed
])
def test_chain_progress_none_when_idle(monkeypatch, cal, ev):
    _chain_rows(monkeypatch, cal=cal, ev=ev)
    assert ac.chain_progress(1) is None


def test_chain_progress_never_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(ac, "execute_query_dict", _boom)
    assert ac.chain_progress(1) is None
