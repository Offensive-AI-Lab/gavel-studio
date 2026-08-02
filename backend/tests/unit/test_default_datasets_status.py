"""Pure unit tests for services/default_datasets.py.

No database is available, so every DB call (execute_query / execute_query_dict
as imported into the target module) is monkeypatched to return canned rows.
We also stub `threading` for generate_rule_defaults so no real thread spawns.

Covered surface:
  * module constants: DEFAULT_TEST_SET_NAME, DEFAULT_DATASET_TYPES
  * rule_defaults_ready   — all-ready / partial / error / missing / None
  * rule_defaults_status  — every rolled-up `state` branch + payload shape
  * generate_rule_defaults — input guard + immediate return + thread spawn
"""
import pytest

import services.default_datasets as dd


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_test_set_name(self):
        assert dd.DEFAULT_TEST_SET_NAME == "Test Set"

    def test_default_dataset_types_exact(self):
        # Order + membership matter: the status/ready logic iterates this tuple.
        assert dd.DEFAULT_DATASET_TYPES == (
            "positive",
            "negative",
            "positive_calibration",
        )

    def test_default_dataset_types_is_tuple(self):
        assert isinstance(dd.DEFAULT_DATASET_TYPES, tuple)
        assert len(dd.DEFAULT_DATASET_TYPES) == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_query_dict(monkeypatch, rows):
    """Replace execute_query_dict (imported into dd) with a recorder returning `rows`."""
    calls = []

    def fake(sql, params=None):
        calls.append((sql, params))
        return rows

    monkeypatch.setattr(dd, "execute_query_dict", fake)
    return calls


def _all_ready_rows():
    return [
        {"dataset_id": 1, "dataset_type": "positive", "status": "ready"},
        {"dataset_id": 2, "dataset_type": "negative", "status": "ready"},
        {"dataset_id": 3, "dataset_type": "positive_calibration", "status": "ready"},
    ]


# ---------------------------------------------------------------------------
# rule_defaults_ready
# ---------------------------------------------------------------------------


class TestRuleDefaultsReady:
    def test_all_three_ready_returns_true(self, monkeypatch):
        _patch_query_dict(monkeypatch, _all_ready_rows())
        assert dd.rule_defaults_ready(42) is True

    def test_passes_rule_id_in_params(self, monkeypatch):
        calls = _patch_query_dict(monkeypatch, _all_ready_rows())
        dd.rule_defaults_ready(99)
        assert calls[0][1] == (99,)
        # Query filters on is_default = TRUE.
        assert "is_default = TRUE" in calls[0][0]

    def test_partial_generating_returns_false(self, monkeypatch):
        rows = [
            {"dataset_type": "positive", "status": "ready"},
            {"dataset_type": "negative", "status": "generating"},
            {"dataset_type": "positive_calibration", "status": "ready"},
        ]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_ready(1) is False

    def test_missing_bucket_returns_false(self, monkeypatch):
        # Only two of the three required buckets present.
        rows = [
            {"dataset_type": "positive", "status": "ready"},
            {"dataset_type": "negative", "status": "ready"},
        ]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_ready(1) is False

    def test_any_error_returns_false(self, monkeypatch):
        rows = [
            {"dataset_type": "positive", "status": "ready"},
            {"dataset_type": "negative", "status": "error"},
            {"dataset_type": "positive_calibration", "status": "ready"},
        ]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_ready(1) is False

    def test_empty_rows_returns_false(self, monkeypatch):
        _patch_query_dict(monkeypatch, [])
        assert dd.rule_defaults_ready(1) is False

    def test_none_rows_returns_false(self, monkeypatch):
        # `... or []` guard: a None from the DB layer must not blow up.
        _patch_query_dict(monkeypatch, None)
        assert dd.rule_defaults_ready(1) is False

    def test_extra_unknown_type_does_not_break_all_ready(self, monkeypatch):
        # An extra bucket type beyond the canonical three is ignored; the three
        # required ones are all ready, so result stays True.
        rows = _all_ready_rows() + [
            {"dataset_type": "bonus", "status": "generating"},
        ]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_ready(1) is True


# ---------------------------------------------------------------------------
# rule_defaults_status
# ---------------------------------------------------------------------------


class TestRuleDefaultsStatus:
    def test_missing_state_when_no_rows(self, monkeypatch):
        _patch_query_dict(monkeypatch, [])
        result = dd.rule_defaults_status(7)
        assert result["state"] == "missing"
        assert result["rule_id"] == 7
        assert result["datasets"] == []

    def test_missing_state_when_none(self, monkeypatch):
        _patch_query_dict(monkeypatch, None)
        result = dd.rule_defaults_status(7)
        assert result["state"] == "missing"
        assert result["datasets"] == []

    def test_ready_state_when_all_three_ready(self, monkeypatch):
        _patch_query_dict(monkeypatch, _all_ready_rows())
        result = dd.rule_defaults_status(7)
        assert result["state"] == "ready"
        assert len(result["datasets"]) == 3

    def test_error_state_takes_priority_over_ready(self, monkeypatch):
        # Even with rows present, any 'error' status rolls up to 'error'.
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "ready"},
            {"dataset_id": 2, "dataset_type": "negative", "status": "error"},
            {"dataset_id": 3, "dataset_type": "positive_calibration", "status": "ready"},
        ]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_status(7)["state"] == "error"

    def test_generating_state_when_partial_no_error(self, monkeypatch):
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "generating"},
            {"dataset_id": 2, "dataset_type": "negative", "status": "ready"},
            {"dataset_id": 3, "dataset_type": "positive_calibration", "status": "ready"},
        ]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_status(7)["state"] == "generating"

    def test_generating_state_when_buckets_missing(self, monkeypatch):
        # Rows exist but not all three required types are 'ready' and none errored.
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "ready"},
        ]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_status(7)["state"] == "generating"

    def test_datasets_payload_shape(self, monkeypatch):
        rows = [
            {"dataset_id": 11, "dataset_type": "positive", "status": "ready"},
            {"dataset_id": 22, "dataset_type": "negative", "status": "generating"},
        ]
        _patch_query_dict(monkeypatch, rows)
        result = dd.rule_defaults_status(7)
        assert result["datasets"] == [
            {"dataset_id": 11, "dataset_type": "positive", "status": "ready"},
            {"dataset_id": 22, "dataset_type": "negative", "status": "generating"},
        ]
        # Each entry exposes exactly the three public fields.
        for entry in result["datasets"]:
            assert set(entry.keys()) == {"dataset_id", "dataset_type", "status"}

    def test_passes_rule_id_in_params(self, monkeypatch):
        calls = _patch_query_dict(monkeypatch, [])
        dd.rule_defaults_status(123)
        assert calls[0][1] == (123,)
        assert "is_default = TRUE" in calls[0][0]

    def test_top_level_keys(self, monkeypatch):
        _patch_query_dict(monkeypatch, _all_ready_rows())
        result = dd.rule_defaults_status(7)
        assert set(result.keys()) == {"rule_id", "state", "datasets"}


# ---------------------------------------------------------------------------
# generate_rule_defaults — input guard, immediate return, thread spawn
# ---------------------------------------------------------------------------


class _FakeThread:
    """Captures Thread construction + .start() without spawning anything."""

    instances = []

    def __init__(self, target=None, args=(), daemon=None, **kwargs):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.started = False
        _FakeThread.instances.append(self)

    def start(self):
        self.started = True
        # IMPORTANT: never invoke self.target — that would call into the real
        # _run_rule_defaults (LLM + DB). Spawning is fire-and-forget.


@pytest.fixture
def fake_threading(monkeypatch):
    _FakeThread.instances = []

    class _FakeThreadingModule:
        Thread = _FakeThread

    monkeypatch.setattr(dd, "threading", _FakeThreadingModule)
    # Also guard the DB seam in case anything unexpectedly calls it.
    monkeypatch.setattr(
        dd, "execute_query_dict", lambda *a, **k: pytest.fail("DB hit unexpectedly")
    )
    return _FakeThread


class TestGenerateRuleDefaults:
    def test_empty_instructions_raises(self, monkeypatch):
        monkeypatch.setattr(
            dd, "threading", pytest.fail  # any thread attempt would error
        )
        with pytest.raises(ValueError, match="scenario_instructions is required"):
            dd.generate_rule_defaults(1, "")

    def test_none_instructions_raises(self):
        with pytest.raises(ValueError):
            dd.generate_rule_defaults(1, None)

    def test_whitespace_only_instructions_raises(self):
        with pytest.raises(ValueError):
            dd.generate_rule_defaults(1, "   \t\n")

    def test_returns_immediately_with_generating_state(self, fake_threading):
        result = dd.generate_rule_defaults(55, "judge politely")
        assert result == {"success": True, "rule_id": 55, "state": "generating"}

    def test_spawns_exactly_one_daemon_thread(self, fake_threading):
        dd.generate_rule_defaults(55, "judge politely")
        assert len(fake_threading.instances) == 1
        t = fake_threading.instances[0]
        assert t.daemon is True
        assert t.started is True

    def test_thread_targets_run_rule_defaults_with_args(self, fake_threading):
        dd.generate_rule_defaults(7, "instr", target_count=9, calibration_count=4)
        t = fake_threading.instances[0]
        assert t.target is dd._run_rule_defaults
        assert t.args == (7, "instr", 9, 4, None)

    def test_default_counts(self, fake_threading):
        dd.generate_rule_defaults(7, "instr")
        t = fake_threading.instances[0]
        # Defaults: target_count=100, calibration_count=50.
        assert t.args == (7, "instr", 100, 50, None)
