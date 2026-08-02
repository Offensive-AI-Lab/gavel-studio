"""Pure unit tests for services/default_datasets.py — bucket-sizing math,
the daemon-thread body (`_run_rule_defaults`), and the status/ready helper
edge cases not already covered by test_default_datasets_status.py.

No database and no network: every DB call (execute_query / execute_query_dict
as imported into the target module) is monkeypatched, and the lazily-imported
generator helpers from `routes.ai_pipeline` (build_positive_config,
build_negative_config, _run_test_generation) are intercepted by injecting a
fake `routes.ai_pipeline` module into sys.modules. That fake is what the
`from routes.ai_pipeline import ...` inside `_run_rule_defaults` resolves to,
so the real (torch-heavy, LLM-calling) module is never imported.

Focus (complements, does not duplicate, test_default_datasets_status.py):
  * generate_rule_defaults bucket sizing: positive=target_count,
    negative=target_count, calibration=calibration_count, with defaults
    100/100/50 and overrides flowing through.
  * _run_rule_defaults end-to-end wiring: 3 rows upserted, configs persisted,
    generation called per bucket with the right counts/types.
  * Error branches: positive-config failure marks all 3 rows error and aborts;
    negative-config failure marks only the negative row and still runs the
    positive + calibration buckets (partial generation).
  * _upsert_default_row / _mark_row_error pure behaviour + boundary counts.
  * rule_defaults_status / rule_defaults_ready edge rows (malformed/None/zero).
"""
import json
import sys
import types

import pytest

import services.default_datasets as dd


# ---------------------------------------------------------------------------
# Fake `routes.ai_pipeline` — intercepts the lazy import in _run_rule_defaults
# ---------------------------------------------------------------------------


class _Recorder:
    """Captures every call to the stubbed generator helpers."""

    def __init__(self):
        self.positive_calls = []
        self.negative_calls = []
        self.gen_calls = []  # list of (dataset_id, config, count, dataset_type)


def _install_fake_ai_pipeline(
    monkeypatch,
    rec,
    *,
    positive=None,
    positive_exc=None,
    negative=None,
    negative_exc=None,
):
    """Put a fake `routes.ai_pipeline` module in sys.modules.

    `positive` / `negative` are the configs returned; `*_exc` (if set) is
    raised instead. The real module is never imported (avoids torch/LLM).
    """
    mod = types.ModuleType("routes.ai_pipeline")

    def build_positive_config(description):
        rec.positive_calls.append(description)
        if positive_exc is not None:
            raise positive_exc
        return dict(positive) if positive is not None else {"pos": True}

    def build_negative_config(pos_config):
        rec.negative_calls.append(pos_config)
        if negative_exc is not None:
            raise negative_exc
        cfg = dict(negative) if negative is not None else {"neg": True}
        return cfg, "reasoning text"

    def _run_test_generation(dataset_id, config, count, dataset_type):
        rec.gen_calls.append((dataset_id, config, count, dataset_type))

    mod.build_positive_config = build_positive_config
    mod.build_negative_config = build_negative_config
    mod._run_test_generation = _run_test_generation

    # Ensure the parent `routes` package object exists but don't import the
    # real ai_pipeline; the `from routes.ai_pipeline import ...` statement
    # resolves straight out of sys.modules.
    monkeypatch.setitem(sys.modules, "routes.ai_pipeline", mod)
    return mod


class _DBStub:
    """Records execute_query / execute_query_dict calls; upserts return ids."""

    def __init__(self, upsert_ids):
        # ids handed out by successive _upsert_default_row calls, in order:
        # positive, positive_calibration, negative.
        self._upsert_ids = list(upsert_ids)
        self.query_calls = []
        self.dict_calls = []

    def execute_query(self, sql, params=None):
        self.query_calls.append((sql, params))
        return None

    def execute_query_dict(self, sql, params=None):
        self.dict_calls.append((sql, params))
        if "INSERT INTO test_datasets" in sql:
            return [{"dataset_id": self._upsert_ids.pop(0)}]
        return []


def _install_db(monkeypatch, upsert_ids=(101, 102, 103)):
    db = _DBStub(upsert_ids)
    monkeypatch.setattr(dd, "execute_query", db.execute_query)
    monkeypatch.setattr(dd, "execute_query_dict", db.execute_query_dict)
    return db


# ---------------------------------------------------------------------------
# generate_rule_defaults — bucket sizing math via the synchronous run path
# ---------------------------------------------------------------------------


def _run_synchronously(monkeypatch):
    """Make threading.Thread.start() execute the target inline so we can
    assert on the full generation wiring deterministically."""

    class _InlineThread:
        def __init__(self, target=None, args=(), daemon=None, **kw):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            self.target(*self.args)

    fake_threading = types.SimpleNamespace(Thread=_InlineThread)
    monkeypatch.setattr(dd, "threading", fake_threading)


class TestBucketSizing:
    def test_default_counts_route_to_buckets(self, monkeypatch):
        # Defaults: positive=100, negative=100, calibration=50.
        rec = _Recorder()
        _install_fake_ai_pipeline(monkeypatch, rec)
        _install_db(monkeypatch, upsert_ids=(11, 12, 13))
        _run_synchronously(monkeypatch)

        dd.generate_rule_defaults(7, "judge politely")

        # gen_calls order in _run_rule_defaults: positive, calibration, negative.
        by_type = {c[3]: c for c in rec.gen_calls}
        assert by_type["positive"][2] == 100
        assert by_type["positive_calibration"][2] == 50
        assert by_type["negative"][2] == 100

    def test_override_counts_flow_through(self, monkeypatch):
        rec = _Recorder()
        _install_fake_ai_pipeline(monkeypatch, rec)
        _install_db(monkeypatch, upsert_ids=(11, 12, 13))
        _run_synchronously(monkeypatch)

        dd.generate_rule_defaults(7, "instr", target_count=8, calibration_count=3)

        by_type = {c[3]: c[2] for c in rec.gen_calls}
        assert by_type == {
            "positive": 8,
            "positive_calibration": 3,
            "negative": 8,
        }

    def test_zero_counts_still_dispatched(self, monkeypatch):
        # target=0 / calibration=0 are passed straight through (no clamping).
        rec = _Recorder()
        _install_fake_ai_pipeline(monkeypatch, rec)
        _install_db(monkeypatch, upsert_ids=(11, 12, 13))
        _run_synchronously(monkeypatch)

        dd.generate_rule_defaults(7, "instr", target_count=0, calibration_count=0)

        by_type = {c[3]: c[2] for c in rec.gen_calls}
        assert by_type == {"positive": 0, "positive_calibration": 0, "negative": 0}

    def test_each_bucket_gets_its_own_dataset_id(self, monkeypatch):
        rec = _Recorder()
        _install_fake_ai_pipeline(monkeypatch, rec)
        _install_db(monkeypatch, upsert_ids=(201, 202, 203))
        _run_synchronously(monkeypatch)

        dd.generate_rule_defaults(7, "instr")

        # Upsert order: positive(201), positive_calibration(202), negative(203).
        by_type = {c[3]: c[0] for c in rec.gen_calls}
        assert by_type == {
            "positive": 201,
            "positive_calibration": 202,
            "negative": 203,
        }

    def test_positive_and_calibration_share_positive_config(self, monkeypatch):
        rec = _Recorder()
        _install_fake_ai_pipeline(
            monkeypatch, rec, positive={"k": "POS"}, negative={"k": "NEG"}
        )
        _install_db(monkeypatch, upsert_ids=(1, 2, 3))
        _run_synchronously(monkeypatch)

        dd.generate_rule_defaults(7, "instr")

        by_type = {c[3]: c[1] for c in rec.gen_calls}
        assert by_type["positive"]["k"] == "POS"
        assert by_type["positive_calibration"]["k"] == "POS"
        assert by_type["negative"]["k"] == "NEG"


# ---------------------------------------------------------------------------
# _run_rule_defaults — row creation + config persistence wiring
# ---------------------------------------------------------------------------


class TestRunRuleDefaultsWiring:
    def test_three_rows_upserted_with_correct_types(self, monkeypatch):
        rec = _Recorder()
        _install_fake_ai_pipeline(monkeypatch, rec)
        db = _install_db(monkeypatch, upsert_ids=(1, 2, 3))

        dd._run_rule_defaults(7, "instr", 100, 50)

        inserts = [c for c in db.dict_calls if "INSERT INTO test_datasets" in c[0]]
        assert len(inserts) == 3
        # params: (rule_id, dataset_type, DEFAULT_TEST_SET_NAME, config_json)
        types_in_order = [c[1][1] for c in inserts]
        assert types_in_order == ["positive", "positive_calibration", "negative"]
        # rule_id threaded into every upsert.
        assert all(c[1][0] == 7 for c in inserts)
        # Reserved default name stamped on each row.
        assert all(c[1][2] == dd.DEFAULT_TEST_SET_NAME for c in inserts)

    def test_scenario_embedded_in_persisted_positive_config(self, monkeypatch):
        # build_positive_config returns a config WITHOUT scenario_instructions;
        # _run_rule_defaults must inject the original scenario for provenance.
        rec = _Recorder()
        _install_fake_ai_pipeline(monkeypatch, rec, positive={"foo": 1})
        _install_db(monkeypatch, upsert_ids=(1, 2, 3))

        dd._run_rule_defaults(7, "my scenario", 5, 2)

        pos_cfg = {c[3]: c[1] for c in rec.gen_calls}["positive"]
        assert pos_cfg["scenario_instructions"] == "my scenario"

    def test_existing_scenario_in_config_not_overwritten(self, monkeypatch):
        rec = _Recorder()
        _install_fake_ai_pipeline(
            monkeypatch, rec, positive={"scenario_instructions": "kept"}
        )
        _install_db(monkeypatch, upsert_ids=(1, 2, 3))

        dd._run_rule_defaults(7, "incoming", 5, 2)

        pos_cfg = {c[3]: c[1] for c in rec.gen_calls}["positive"]
        assert pos_cfg["scenario_instructions"] == "kept"

    def test_negative_config_derived_from_positive(self, monkeypatch):
        rec = _Recorder()
        _install_fake_ai_pipeline(
            monkeypatch, rec, positive={"p": 1}, negative={"n": 1}
        )
        _install_db(monkeypatch, upsert_ids=(1, 2, 3))

        dd._run_rule_defaults(7, "instr", 5, 2)

        # build_negative_config received the positive config.
        assert rec.negative_calls and rec.negative_calls[0]["p"] == 1
        neg_cfg = {c[3]: c[1] for c in rec.gen_calls}["negative"]
        assert neg_cfg["n"] == 1
        # Negative config inherits scenario_instructions from positive.
        assert neg_cfg["scenario_instructions"] == "instr"


# ---------------------------------------------------------------------------
# _run_rule_defaults — error branches (full failure vs partial generation)
# ---------------------------------------------------------------------------


class TestRunRuleDefaultsErrors:
    def test_positive_config_failure_marks_all_three_error_and_aborts(self, monkeypatch):
        rec = _Recorder()
        _install_fake_ai_pipeline(
            monkeypatch, rec, positive_exc=RuntimeError("LLM down")
        )
        db = _install_db(monkeypatch, upsert_ids=(11, 22, 33))

        dd._run_rule_defaults(7, "instr", 100, 50)

        # No dialogue generation should have happened.
        assert rec.gen_calls == []
        # Negative config never attempted.
        assert rec.negative_calls == []
        # All three rows flipped to error via execute_query UPDATE ... status='error'.
        error_updates = [
            c for c in db.query_calls
            if "status = 'error'" in c[0]
        ]
        marked_ids = {c[1][1] for c in error_updates}
        assert marked_ids == {11, 22, 33}
        # The error message references the positive-config failure.
        assert all("positive config generation failed" in c[1][0] for c in error_updates)

    def test_negative_config_failure_is_partial_not_total(self, monkeypatch):
        # Negative config blows up, but positive + calibration must still generate.
        rec = _Recorder()
        _install_fake_ai_pipeline(
            monkeypatch,
            rec,
            positive={"p": 1},
            negative_exc=RuntimeError("polar context failed"),
        )
        db = _install_db(monkeypatch, upsert_ids=(11, 22, 33))

        dd._run_rule_defaults(7, "instr", 9, 4)

        # positive (11) and calibration (22) generated; negative (33) did NOT.
        gen_types = {c[3] for c in rec.gen_calls}
        assert gen_types == {"positive", "positive_calibration"}
        assert all(c[3] != "negative" for c in rec.gen_calls)

        # Only the negative row (33) marked error.
        error_updates = [c for c in db.query_calls if "status = 'error'" in c[0]]
        assert len(error_updates) == 1
        assert error_updates[0][1][1] == 33
        assert "negative config generation failed" in error_updates[0][1][0]

    def test_mark_row_error_truncates_to_500_chars(self, monkeypatch):
        db = _install_db(monkeypatch)
        long_msg = "x" * 1000
        dd._mark_row_error(5, long_msg)
        assert len(db.query_calls) == 1
        sql, params = db.query_calls[0]
        assert "status = 'error'" in sql
        assert params == ("x" * 500, 5)

    def test_mark_row_error_swallows_db_exception(self, monkeypatch):
        def boom(sql, params=None):
            raise RuntimeError("db gone")

        monkeypatch.setattr(dd, "execute_query", boom)
        # Must not propagate — error-marking is best-effort.
        dd._mark_row_error(5, "msg")


# ---------------------------------------------------------------------------
# _upsert_default_row — pure SQL/params behaviour
# ---------------------------------------------------------------------------


class TestUpsertDefaultRow:
    def test_returns_dataset_id_from_returning_clause(self, monkeypatch):
        db = _install_db(monkeypatch, upsert_ids=(777,))
        out = dd._upsert_default_row(7, "positive", {"a": 1})
        assert out == 777

    def test_config_serialized_and_null_bytes_stripped(self, monkeypatch):
        db = _install_db(monkeypatch, upsert_ids=(1,))
        dd._upsert_default_row(7, "positive", {"k": "v"})
        sql, params = db.dict_calls[0]
        # params: (rule_id, dataset_type, name, config_json)
        assert params[0] == 7
        assert params[1] == "positive"
        assert params[2] == dd.DEFAULT_TEST_SET_NAME
        # Config is JSON-serialized; the literal escaped null sequence is removed.
        assert params[3] == json.dumps({"k": "v"})
        assert "\\u0000" not in params[3]

    def test_upsert_uses_conflict_on_default_partial_index(self, monkeypatch):
        db = _install_db(monkeypatch, upsert_ids=(1,))
        dd._upsert_default_row(7, "negative", {})
        sql = db.dict_calls[0][0]
        assert "ON CONFLICT (rule_id, dataset_type) WHERE is_default = TRUE" in sql
        assert "status = 'generating'" in sql


# ---------------------------------------------------------------------------
# Idempotent re-generate — re-running upserts onto the same rows, no dupes
# ---------------------------------------------------------------------------


class TestIdempotentRegenerate:
    def test_regenerate_reissues_three_upserts_each_run(self, monkeypatch):
        # Idempotency is enforced by the ON CONFLICT upsert, not a pre-check:
        # a second run issues the same three upserts (which the partial unique
        # index collapses onto the existing default rows), never more.
        rec = _Recorder()
        _install_fake_ai_pipeline(monkeypatch, rec)
        db = _install_db(monkeypatch, upsert_ids=(1, 2, 3, 4, 5, 6))

        dd._run_rule_defaults(7, "instr", 100, 50)
        dd._run_rule_defaults(7, "instr", 100, 50)

        inserts = [c for c in db.dict_calls if "INSERT INTO test_datasets" in c[0]]
        # Exactly three upserts per run, six total — no per-run growth.
        assert len(inserts) == 6
        # Every upsert carries the conflict clause that makes re-runs idempotent.
        assert all(
            "ON CONFLICT (rule_id, dataset_type) WHERE is_default = TRUE" in c[0]
            for c in inserts
        )

    def test_status_missing_before_first_generation(self, monkeypatch):
        # The "missing" state is the pre-generation guard signal the caller
        # checks: no default rows yet for the rule.
        _patch_query_dict(monkeypatch, [])
        assert dd.rule_defaults_status(7)["state"] == "missing"
        # ... and immediately after kickoff (rows now 'generating') it flips off
        # 'missing'.
        _patch_query_dict(
            monkeypatch,
            [
                {"dataset_id": 1, "dataset_type": "positive", "status": "generating"},
                {"dataset_id": 2, "dataset_type": "negative", "status": "generating"},
                {"dataset_id": 3, "dataset_type": "positive_calibration", "status": "generating"},
            ],
        )
        assert dd.rule_defaults_status(7)["state"] == "generating"


# ---------------------------------------------------------------------------
# Status / ready helpers — edge rows not covered by the existing status test
# ---------------------------------------------------------------------------


def _patch_query_dict(monkeypatch, rows):
    monkeypatch.setattr(dd, "execute_query_dict", lambda sql, params=None: rows)


class TestStatusEdgeCases:
    def test_all_three_error_rolls_up_to_error(self, monkeypatch):
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "error"},
            {"dataset_id": 2, "dataset_type": "negative", "status": "error"},
            {"dataset_id": 3, "dataset_type": "positive_calibration", "status": "error"},
        ]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_status(7)["state"] == "error"

    def test_error_with_only_one_bucket_present(self, monkeypatch):
        # A single errored row (others not yet created) still rolls up to error.
        rows = [{"dataset_id": 1, "dataset_type": "positive", "status": "error"}]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_status(7)["state"] == "error"

    def test_duplicate_type_last_value_wins_in_by_type(self, monkeypatch):
        # Malformed data: two rows share a dataset_type. The dict comprehension
        # keeps the LAST one for the ready/state computation, but the datasets
        # payload still lists every raw row.
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "ready"},
            {"dataset_id": 9, "dataset_type": "positive", "status": "generating"},
            {"dataset_id": 2, "dataset_type": "negative", "status": "ready"},
            {"dataset_id": 3, "dataset_type": "positive_calibration", "status": "ready"},
        ]
        _patch_query_dict(monkeypatch, rows)
        result = dd.rule_defaults_status(7)
        # Last 'positive' is 'generating' -> not all ready -> generating.
        assert result["state"] == "generating"
        # Payload preserves all four raw rows.
        assert len(result["datasets"]) == 4

    def test_unknown_status_string_is_not_ready(self, monkeypatch):
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "queued"},
            {"dataset_id": 2, "dataset_type": "negative", "status": "ready"},
            {"dataset_id": 3, "dataset_type": "positive_calibration", "status": "ready"},
        ]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_ready(7) is False
        assert dd.rule_defaults_status(7)["state"] == "generating"

    def test_ready_when_extra_bucket_errors_rolls_up_error(self, monkeypatch):
        # The three canonical buckets are ready, but a 4th unknown bucket
        # errored. `any(error)` scans ALL statuses, so state is 'error' even
        # though rule_defaults_ready (which only checks the canonical 3) is True.
        rows = [
            {"dataset_id": 1, "dataset_type": "positive", "status": "ready"},
            {"dataset_id": 2, "dataset_type": "negative", "status": "ready"},
            {"dataset_id": 3, "dataset_type": "positive_calibration", "status": "ready"},
            {"dataset_id": 4, "dataset_type": "bonus", "status": "error"},
        ]
        _patch_query_dict(monkeypatch, rows)
        assert dd.rule_defaults_ready(7) is True
        assert dd.rule_defaults_status(7)["state"] == "error"
