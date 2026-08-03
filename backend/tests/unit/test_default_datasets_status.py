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
# rule_defaults_status — failure reasons (generation_log surfaced as `error`)
# ---------------------------------------------------------------------------


class TestRuleDefaultsStatusErrorReason:
    def test_errored_bucket_carries_its_generation_log_reason(self, monkeypatch):
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "ready",
             "generation_log": "Completed: 100 positive conversations"},
            {"dataset_id": 2, "dataset_type": "negative", "status": "error",
             "generation_log": "negative config generation failed: LLM quota exceeded"},
            {"dataset_id": 3, "dataset_type": "positive_calibration", "status": "ready",
             "generation_log": None},
        ]
        _patch_query_dict(monkeypatch, rows)
        result = dd.rule_defaults_status(7)
        assert result["state"] == "error"
        # Rolled-up top-level reason names the bucket + its log message.
        assert result["error"] == (
            "negative: negative config generation failed: LLM quota exceeded"
        )
        # The errored entry carries the raw reason; ready entries don't.
        neg = next(d for d in result["datasets"] if d["dataset_type"] == "negative")
        assert neg["error"] == "negative config generation failed: LLM quota exceeded"
        for d in result["datasets"]:
            if d["dataset_type"] != "negative":
                assert "error" not in d

    def test_no_error_fields_when_nothing_errored(self, monkeypatch):
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "ready",
             "generation_log": "Completed: 100 positive conversations"},
            {"dataset_id": 2, "dataset_type": "negative", "status": "generating",
             "generation_log": "Generated 3/100 conversations"},
        ]
        _patch_query_dict(monkeypatch, rows)
        result = dd.rule_defaults_status(7)
        assert "error" not in result
        assert all("error" not in d for d in result["datasets"])

    def test_multiple_errored_buckets_join_reasons(self, monkeypatch):
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "error",
             "generation_log": "boom A"},
            {"dataset_id": 2, "dataset_type": "negative", "status": "error",
             "generation_log": "boom B"},
        ]
        _patch_query_dict(monkeypatch, rows)
        result = dd.rule_defaults_status(7)
        assert result["error"] == "positive: boom A; negative: boom B"

    def test_error_reason_falls_back_when_log_missing(self, monkeypatch):
        # An errored row with no generation_log still yields a usable message.
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "error",
             "generation_log": None},
        ]
        _patch_query_dict(monkeypatch, rows)
        result = dd.rule_defaults_status(7)
        assert result["error"] == "positive: generation failed"
        assert result["datasets"][0]["error"] == "generation failed"


# ---------------------------------------------------------------------------
# _run_rule_defaults — failure reveal (rule never stranded is_ready=FALSE)
# ---------------------------------------------------------------------------


class _RecordingSeams:
    """Bundles the DB/helper seams _run_rule_defaults writes through."""

    def __init__(self):
        self.errors = []      # (dataset_id, message) from _mark_row_error
        self.queries = []     # (sql, params) from execute_query

    def install(self, monkeypatch):
        monkeypatch.setattr(
            dd, "_upsert_default_row",
            lambda rule_id, dtype, cfg: {"positive": 1, "positive_calibration": 2, "negative": 3}[dtype],
        )
        monkeypatch.setattr(dd, "_mark_row_error", lambda did, msg: self.errors.append((did, msg)))
        monkeypatch.setattr(dd, "execute_query", lambda sql, params=None: self.queries.append((sql, params)))
        return self

    def rule_flipped(self):
        return any("UPDATE rules SET is_ready = TRUE" in sql for sql, _ in self.queries)

    def ces_flipped(self):
        return any("cognitive_elements SET is_ready = TRUE" in sql for sql, _ in self.queries)


@pytest.fixture
def fake_ai_pipeline(monkeypatch):
    """Inject a lightweight stand-in for routes.ai_pipeline so
    _run_rule_defaults' lazy `from routes.ai_pipeline import ...` never pulls
    the real (heavy, LLM-calling) module. Tests override the attributes."""
    import sys
    import types

    mod = types.ModuleType("routes.ai_pipeline")
    mod.build_positive_config = lambda scenario: {"scenario_instructions": scenario}
    mod.build_negative_config = lambda pos: ({"scenario_instructions": "neg"}, "reasoning")
    mod._run_test_generation = lambda dataset_id, config, count, dtype: None
    monkeypatch.setitem(sys.modules, "routes.ai_pipeline", mod)
    return mod


class TestRunRuleDefaultsFailureReveal:
    def test_positive_config_failure_marks_all_buckets_and_reveals(
        self, monkeypatch, fake_ai_pipeline
    ):
        def boom(scenario):
            raise RuntimeError("LLM down")

        fake_ai_pipeline.build_positive_config = boom
        seams = _RecordingSeams().install(monkeypatch)

        dd._run_rule_defaults(7, "scenario", 5, 3, finalize_ce_ids=[41, 42])

        # All three buckets carry the failure reason...
        assert {did for did, _ in seams.errors} == {1, 2, 3}
        assert all("LLM down" in msg for _, msg in seams.errors)
        # ...and the rule + its deferred CEs are revealed, not stranded hidden.
        assert seams.rule_flipped()
        assert seams.ces_flipped()

    def test_bucket_failure_reveals_rule_even_without_finalize_ids(
        self, monkeypatch, fake_ai_pipeline
    ):
        # Manual build-from-CEs path: finalize_ce_ids is None (the frontend
        # owns the success-finalize, but it never runs on failure). A bucket
        # error must still flip the rule visible.
        seams = _RecordingSeams().install(monkeypatch)
        monkeypatch.setattr(dd, "rule_defaults_status", lambda rid: {"state": "error"})

        dd._run_rule_defaults(7, "scenario", 5, 3, finalize_ce_ids=None)

        assert seams.rule_flipped()
        assert not seams.ces_flipped()  # no deferred CEs in this path

    def test_success_without_finalize_ids_does_not_flip(
        self, monkeypatch, fake_ai_pipeline
    ):
        # Contract preserved: on SUCCESS the manual path's reveal is owned by
        # the frontend finalize (embed + flip), not this thread.
        seams = _RecordingSeams().install(monkeypatch)
        monkeypatch.setattr(dd, "rule_defaults_status", lambda rid: {"state": "ready"})

        dd._run_rule_defaults(7, "scenario", 5, 3, finalize_ce_ids=None)

        assert not seams.rule_flipped()
        assert not seams.ces_flipped()

    def test_success_with_finalize_ids_flips_rule_and_ces(
        self, monkeypatch, fake_ai_pipeline
    ):
        seams = _RecordingSeams().install(monkeypatch)
        monkeypatch.setattr(dd, "rule_defaults_status", lambda rid: {"state": "ready"})

        dd._run_rule_defaults(7, "scenario", 5, 3, finalize_ce_ids=[9])

        assert seams.rule_flipped()
        assert seams.ces_flipped()


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
