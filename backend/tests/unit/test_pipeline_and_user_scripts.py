"""Unit tests for sql_scripts.pipeline_run_scripts and sql_scripts.user_scripts.

These are pure-logic tests: the DB seam (execute_query / execute_query_dict,
imported INTO each module under test) is monkeypatched with small fakes that
record the SQL + params they're handed and return canned rows. No Postgres, no
network. We assert on:

  * the SQL/params the helpers build (composite WHERE clauses, ordering,
    jsonb_set path, RETURNING shape),
  * the step-id sequences and default-state map (the '1','2A','2B','2C','2D'
    rule sequence and friends),
  * validation / error branches (unknown pipeline_type, unknown step id,
    test_eval needing a classifier_id),
  * None / empty-row handling from the DB layer,
  * user lookup/create upsert helpers and the central-server fallback in
    ensure_creators_in_local (network stubbed).

Style mirrors tests/unit/test_services.py: small fakes + monkeypatch +
assertions on pure logic.
"""
import json

import pytest

import sql_scripts.pipeline_run_scripts as prs
import sql_scripts.user_scripts as us


# ---------------------------------------------------------------------------
# Recording fakes for the DB seam
# ---------------------------------------------------------------------------


class _Recorder:
    """Records (sql, params) tuples and returns a pre-seeded result.

    `result` may be a single value (returned for every call) or a list of
    values consumed one per call (so a helper that issues two queries can be
    given two distinct return values)."""

    def __init__(self, result=None, results=None):
        self.calls = []
        self._single = result
        self._queue = list(results) if results is not None else None

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        if self._queue is not None:
            return self._queue.pop(0)
        return self._single

    @property
    def last_sql(self):
        return self.calls[-1][0]

    @property
    def last_params(self):
        return self.calls[-1][1]


def _patch_dict(monkeypatch, recorder):
    """Patch execute_query_dict in the pipeline module."""
    monkeypatch.setattr(prs, "execute_query_dict", recorder)


def _patch_query(monkeypatch, recorder):
    """Patch execute_query in the pipeline module."""
    monkeypatch.setattr(prs, "execute_query", recorder)


# A canonical row shape the helpers return (mirrors _COLS column order, but the
# fake just echoes a dict so exact column values don't matter for logic tests).
def _row(**over):
    base = {
        "run_id": 7,
        "user_id": 3,
        "classifier_id": None,
        "rule_id": None,
        "pipeline_type": "rule",
        "current_step": "1",
        "steps": {},
        "completed": False,
        "created_at": "t0",
        "updated_at": "t1",
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Step-id constants and default-state map
# ---------------------------------------------------------------------------


class TestStepConstants:
    def test_rule_step_sequence_is_exact(self):
        # The canonical rule-generation step sequence.
        assert prs._STEP_IDS_RULE == ("1", "2A", "2B", "2C", "2D")

    def test_test_eval_and_ce_sequences(self):
        assert prs._STEP_IDS_TEST_EVAL == ("define", "cal", "eval")
        assert prs._STEP_IDS_CE == ("1", "2.1", "2.2", "2.3")

    def test_all_step_ids_is_the_union(self):
        expected = prs._STEP_IDS_RULE + prs._STEP_IDS_TEST_EVAL + prs._STEP_IDS_CE
        assert prs._ALL_STEP_IDS == expected

    def test_valid_pipeline_types(self):
        assert prs._VALID_PIPELINE_TYPES == ("rule", "test_eval", "ce")

    def test_first_step_map_matches_sequence_heads(self):
        # The landing step of each flavor must be the first id of its sequence.
        assert prs._FIRST_STEP["rule"] == prs._STEP_IDS_RULE[0]
        assert prs._FIRST_STEP["test_eval"] == prs._STEP_IDS_TEST_EVAL[0]
        assert prs._FIRST_STEP["ce"] == prs._STEP_IDS_CE[0]

    def test_step_ids_for_each_type(self):
        assert prs._step_ids_for("rule") == prs._STEP_IDS_RULE
        assert prs._step_ids_for("test_eval") == prs._STEP_IDS_TEST_EVAL
        assert prs._step_ids_for("ce") == prs._STEP_IDS_CE

    def test_step_ids_for_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline_type"):
            prs._step_ids_for("bogus")


class TestDefaultStepsState:
    def test_rule_default_state_keys_and_shape(self):
        state = prs._default_steps_state("rule")
        # One entry per rule step, in the right set.
        assert set(state.keys()) == set(prs._STEP_IDS_RULE)
        # Every step starts pending with empty data.
        for sid, entry in state.items():
            assert entry == {"status": "pending", "data": {}}

    def test_default_state_data_dicts_are_distinct_objects(self):
        # Each step's `data` must be its own dict — a shared reference would
        # mean mutating one step's data bleeds into all others.
        state = prs._default_steps_state("rule")
        state["1"]["data"]["x"] = 1
        assert state["2A"]["data"] == {}

    def test_default_state_for_test_eval(self):
        state = prs._default_steps_state("test_eval")
        assert set(state.keys()) == {"define", "cal", "eval"}


# ---------------------------------------------------------------------------
# create_pipeline_run
# ---------------------------------------------------------------------------


class TestCreatePipelineRun:
    def test_create_rule_run_seeds_first_step_and_default_state(self, monkeypatch):
        rec = _Recorder(result=[_row()])
        _patch_dict(monkeypatch, rec)

        out = prs.create_pipeline_run(user_id=3)

        assert out == _row()
        # Params: (user_id, classifier_id, rule_id, pipeline_type, first_step, steps_json)
        user_id, classifier_id, rule_id, ptype, first_step, steps_json = rec.last_params
        assert user_id == 3
        assert classifier_id is None
        assert rule_id is None
        assert ptype == "rule"
        assert first_step == "1"
        # steps_json is JSON-serialized default state.
        assert json.loads(steps_json) == prs._default_steps_state("rule")
        # The INSERT returns the canonical column set.
        assert "RETURNING" in rec.last_sql and "run_id" in rec.last_sql

    def test_create_test_eval_run_requires_classifier(self, monkeypatch):
        # No DB call should happen — the guard raises first. Patch with a
        # recorder that would blow up the assertion if it were ever invoked.
        rec = _Recorder(result=[_row()])
        _patch_dict(monkeypatch, rec)
        with pytest.raises(ValueError, match="require a classifier_id"):
            prs.create_pipeline_run(user_id=3, pipeline_type="test_eval")
        assert rec.calls == []

    def test_create_test_eval_with_classifier_uses_define_first_step(self, monkeypatch):
        rec = _Recorder(result=[_row(pipeline_type="test_eval", classifier_id=5,
                                     current_step="define")])
        _patch_dict(monkeypatch, rec)

        prs.create_pipeline_run(user_id=3, pipeline_type="test_eval",
                                classifier_id=5, rule_id=9)

        _u, classifier_id, rule_id, ptype, first_step, steps_json = rec.last_params
        assert classifier_id == 5
        assert rule_id == 9
        assert ptype == "test_eval"
        assert first_step == "define"
        assert json.loads(steps_json) == prs._default_steps_state("test_eval")

    def test_create_unknown_pipeline_type_raises(self, monkeypatch):
        rec = _Recorder(result=[_row()])
        _patch_dict(monkeypatch, rec)
        with pytest.raises(ValueError, match="Unknown pipeline_type"):
            prs.create_pipeline_run(user_id=1, pipeline_type="nope")
        assert rec.calls == []

    def test_create_returns_none_when_db_returns_empty(self, monkeypatch):
        # RETURNING yielded no rows (shouldn't happen for INSERT, but the helper
        # guards `rows[0] if rows else None`).
        _patch_dict(monkeypatch, _Recorder(result=[]))
        assert prs.create_pipeline_run(user_id=1) is None

    def test_create_returns_none_when_db_returns_none(self, monkeypatch):
        _patch_dict(monkeypatch, _Recorder(result=None))
        assert prs.create_pipeline_run(user_id=1) is None


# ---------------------------------------------------------------------------
# get_pipeline_run
# ---------------------------------------------------------------------------


class TestGetPipelineRun:
    def test_get_returns_first_row(self, monkeypatch):
        rec = _Recorder(result=[_row(run_id=42)])
        _patch_dict(monkeypatch, rec)
        out = prs.get_pipeline_run(42)
        assert out["run_id"] == 42
        assert rec.last_params == (42,)
        assert "WHERE run_id = %s" in rec.last_sql

    def test_get_returns_none_on_empty(self, monkeypatch):
        _patch_dict(monkeypatch, _Recorder(result=[]))
        assert prs.get_pipeline_run(99) is None

    def test_get_returns_none_on_none(self, monkeypatch):
        _patch_dict(monkeypatch, _Recorder(result=None))
        assert prs.get_pipeline_run(99) is None


# ---------------------------------------------------------------------------
# get_active_runs — composite WHERE / ordering / completed=FALSE
# ---------------------------------------------------------------------------


class TestGetActiveRuns:
    def test_only_user_filter_includes_completed_false_and_orders(self, monkeypatch):
        rec = _Recorder(result=[_row(), _row(run_id=8)])
        _patch_dict(monkeypatch, rec)

        out = prs.get_active_runs(user_id=3)

        assert len(out) == 2
        sql = rec.last_sql
        assert "user_id = %s" in sql
        assert "completed = FALSE" in sql
        assert "ORDER BY updated_at DESC" in sql
        # No optional filters -> only the user_id param.
        assert rec.last_params == (3,)

    def test_all_filters_compose_in_order(self, monkeypatch):
        rec = _Recorder(result=[])
        _patch_dict(monkeypatch, rec)

        prs.get_active_runs(user_id=3, classifier_id=5,
                            pipeline_type="test_eval", rule_id=9)

        sql = rec.last_sql
        for clause in ("user_id = %s", "completed = FALSE",
                       "classifier_id = %s", "pipeline_type = %s", "rule_id = %s"):
            assert clause in sql
        # Param order: user_id, then classifier, pipeline_type, rule_id.
        assert rec.last_params == (3, 5, "test_eval", 9)

    def test_partial_filters_skip_absent_clauses(self, monkeypatch):
        rec = _Recorder(result=[])
        _patch_dict(monkeypatch, rec)

        prs.get_active_runs(user_id=3, rule_id=9)

        sql = rec.last_sql
        assert "rule_id = %s" in sql
        assert "classifier_id = %s" not in sql
        assert "pipeline_type = %s" not in sql
        assert rec.last_params == (3, 9)

    def test_invalid_pipeline_type_raises(self, monkeypatch):
        rec = _Recorder(result=[])
        _patch_dict(monkeypatch, rec)
        with pytest.raises(ValueError, match="Unknown pipeline_type"):
            prs.get_active_runs(user_id=3, pipeline_type="garbage")
        assert rec.calls == []

    def test_empty_result_normalized_to_list(self, monkeypatch):
        # DB returning None -> helper coalesces to [].
        _patch_dict(monkeypatch, _Recorder(result=None))
        assert prs.get_active_runs(user_id=3) == []

    def test_classifier_zero_is_a_real_filter(self, monkeypatch):
        # Boundary: classifier_id=0 is "not None" so it must be applied (a `if
        # classifier_id` truthiness bug would silently drop it).
        rec = _Recorder(result=[])
        _patch_dict(monkeypatch, rec)
        prs.get_active_runs(user_id=3, classifier_id=0)
        assert "classifier_id = %s" in rec.last_sql
        assert rec.last_params == (3, 0)


# ---------------------------------------------------------------------------
# update_step — jsonb_set path, advance_to, validation
# ---------------------------------------------------------------------------


class TestUpdateStep:
    def test_update_without_advance_sets_step_value(self, monkeypatch):
        rec = _Recorder(result=[_row()])
        _patch_dict(monkeypatch, rec)

        prs.update_step(run_id=7, step_id="2A", status="completed",
                        data={"k": "v"})

        sql = rec.last_sql
        assert "json_set" in sql
        # No current_step bump when advance_to is omitted.
        assert "current_step = %s" not in sql
        path, value_json, run_id = rec.last_params
        # SQLite json_set takes a quoted JSON path, not a text[] array.
        assert path == '$."2A"'
        assert json.loads(value_json) == {"status": "completed", "data": {"k": "v"}}
        assert run_id == 7

    def test_update_with_advance_bumps_current_step(self, monkeypatch):
        rec = _Recorder(result=[_row(current_step="2B")])
        _patch_dict(monkeypatch, rec)

        prs.update_step(run_id=7, step_id="2A", status="completed",
                        advance_to="2B")

        sql = rec.last_sql
        assert "current_step = %s" in sql
        path, value_json, advance_to, run_id = rec.last_params
        assert path == '$."2A"'
        assert advance_to == "2B"
        assert run_id == 7
        # data defaults to {} when None.
        assert json.loads(value_json) == {"status": "completed", "data": {}}

    def test_update_none_data_serializes_empty_dict(self, monkeypatch):
        rec = _Recorder(result=[_row()])
        _patch_dict(monkeypatch, rec)
        prs.update_step(run_id=7, step_id="1", status="in_progress", data=None)
        _path, value_json, _run = rec.last_params
        assert json.loads(value_json)["data"] == {}

    def test_update_unknown_step_raises_before_db(self, monkeypatch):
        rec = _Recorder(result=[_row()])
        _patch_dict(monkeypatch, rec)
        with pytest.raises(ValueError, match="Unknown step id"):
            prs.update_step(run_id=7, step_id="ZZ", status="completed")
        assert rec.calls == []

    def test_update_unknown_advance_to_raises(self, monkeypatch):
        rec = _Recorder(result=[_row()])
        _patch_dict(monkeypatch, rec)
        with pytest.raises(ValueError, match="advance_to"):
            prs.update_step(run_id=7, step_id="1", status="completed",
                            advance_to="ZZ")
        assert rec.calls == []

    def test_update_accepts_every_valid_step_id(self, monkeypatch):
        rec = _Recorder(result=[_row()])
        _patch_dict(monkeypatch, rec)
        for sid in prs._ALL_STEP_IDS:
            prs.update_step(run_id=7, step_id=sid, status="pending")
        # One DB call per valid step id, none raised.
        assert len(rec.calls) == len(prs._ALL_STEP_IDS)

    def test_update_returns_none_when_no_row_matched(self, monkeypatch):
        # WHERE run_id didn't match -> RETURNING is empty -> None.
        _patch_dict(monkeypatch, _Recorder(result=[]))
        assert prs.update_step(run_id=123, step_id="1", status="completed") is None

    def test_update_status_state_machine_values_pass_through(self, monkeypatch):
        # The backend treats status opaquely; verify each legal status lands in
        # the serialized value verbatim.
        rec = _Recorder(result=[_row()])
        _patch_dict(monkeypatch, rec)
        for status in ("pending", "in_progress", "completed", "skipped", "error"):
            prs.update_step(run_id=7, step_id="1", status=status)
            _path, value_json, _run = rec.last_params
            assert json.loads(value_json)["status"] == status


# ---------------------------------------------------------------------------
# set_run_links
# ---------------------------------------------------------------------------


class TestSetRunLinks:
    def test_set_rule_id_updates(self, monkeypatch):
        rec = _Recorder(result=[_row(rule_id=11)])
        _patch_dict(monkeypatch, rec)
        out = prs.set_run_links(run_id=7, rule_id=11)
        assert out["rule_id"] == 11
        assert "rule_id = %s" in rec.last_sql
        # params end with run_id.
        assert rec.last_params == (11, 7)

    def test_no_changes_falls_back_to_get(self, monkeypatch):
        # rule_id=None -> no UPDATE parts -> delegates to get_pipeline_run, which
        # issues a SELECT (not an UPDATE).
        rec = _Recorder(result=[_row(run_id=7)])
        _patch_dict(monkeypatch, rec)
        out = prs.set_run_links(run_id=7, rule_id=None)
        assert out["run_id"] == 7
        assert "SELECT" in rec.last_sql
        assert "UPDATE" not in rec.last_sql
        assert rec.last_params == (7,)

    def test_returns_none_when_update_matches_nothing(self, monkeypatch):
        _patch_dict(monkeypatch, _Recorder(result=[]))
        assert prs.set_run_links(run_id=7, rule_id=11) is None


# ---------------------------------------------------------------------------
# complete_run — already-completed / missing run
# ---------------------------------------------------------------------------


class TestCompleteRun:
    def test_complete_sets_completed_true(self, monkeypatch):
        rec = _Recorder(result=[_row(completed=True)])
        _patch_dict(monkeypatch, rec)
        out = prs.complete_run(7)
        assert out["completed"] is True
        assert "completed = TRUE" in rec.last_sql
        assert rec.last_params == (7,)

    def test_complete_already_completed_run_still_returns_row(self, monkeypatch):
        # Completing an already-completed run is idempotent at the SQL level —
        # the UPDATE still matches the row and RETURNING yields it.
        rec = _Recorder(result=[_row(completed=True)])
        _patch_dict(monkeypatch, rec)
        assert prs.complete_run(7)["completed"] is True

    def test_complete_missing_run_returns_none(self, monkeypatch):
        _patch_dict(monkeypatch, _Recorder(result=[]))
        assert prs.complete_run(404) is None


# ---------------------------------------------------------------------------
# delete_pipeline_run — ownership + bool return
# ---------------------------------------------------------------------------


class TestDeletePipelineRun:
    def test_delete_success_returns_true(self, monkeypatch):
        # execute_query (not _dict) returns the RETURNING run_id rows.
        rec = _Recorder(result=[(7,)])
        _patch_query(monkeypatch, rec)
        assert prs.delete_pipeline_run(run_id=7, user_id=3) is True
        # Ownership enforced in WHERE.
        assert "user_id = %s" in rec.last_sql
        assert rec.last_params == (7, 3)

    def test_delete_not_owned_returns_false(self, monkeypatch):
        # No row matched (wrong user) -> empty -> False.
        _patch_query(monkeypatch, _Recorder(result=[]))
        assert prs.delete_pipeline_run(run_id=7, user_id=999) is False

    def test_delete_none_result_returns_false(self, monkeypatch):
        _patch_query(monkeypatch, _Recorder(result=None))
        assert prs.delete_pipeline_run(run_id=7, user_id=3) is False


# ===========================================================================
# user_scripts.py
# ===========================================================================


def _patch_user_query(monkeypatch, recorder):
    monkeypatch.setattr(us, "execute_query", recorder)


def _patch_user_dict(monkeypatch, recorder):
    monkeypatch.setattr(us, "execute_query_dict", recorder)



class TestGetUserById:
    def test_returns_first_row(self, monkeypatch):
        rec = _Recorder(result=[{"user_id": 5, "username": "alice",
                                 "email": "a@x.io", "tutorial_seen": False}])
        _patch_user_dict(monkeypatch, rec)
        out = us.get_user_by_id(5)
        assert out["user_id"] == 5
        assert rec.last_params == (5,)

    def test_returns_none_on_empty(self, monkeypatch):
        _patch_user_dict(monkeypatch, _Recorder(result=[]))
        assert us.get_user_by_id(999) is None

    def test_returns_none_on_none(self, monkeypatch):
        _patch_user_dict(monkeypatch, _Recorder(result=None))
        assert us.get_user_by_id(999) is None


class TestEnsureCreatorsInLocal:
    """After an HF sync, every `created_by_username` on the imported rules/CEs
    must exist locally so attribution JOINs and profile lookups resolve. With
    login (and the central user directory) gone, unknown authors get a local
    placeholder row instead of being fetched from the central server."""

    def test_empty_list_short_circuits(self, monkeypatch):
        rec = _Recorder(result=[])
        _patch_user_dict(monkeypatch, rec)
        us.ensure_creators_in_local([])
        assert rec.calls == []

    def test_only_blank_or_none_usernames_short_circuits(self, monkeypatch):
        rec = _Recorder(result=[])
        _patch_user_dict(monkeypatch, rec)
        us.ensure_creators_in_local([None, "", None])
        assert rec.calls == []

    def test_all_present_locally_writes_nothing(self, monkeypatch):
        # The existence probe finds both -> no INSERT is issued.
        _patch_user_dict(monkeypatch, _Recorder(result=[{"u": "alice"}, {"u": "bob"}]))
        writes = _Recorder(result=None)
        _patch_user_query(monkeypatch, writes)
        us.ensure_creators_in_local(["Alice", "BOB"])
        assert writes.calls == []

    def test_missing_authors_get_a_local_placeholder_row(self, monkeypatch):
        # Nobody is known locally -> one INSERT per author, lowercased.
        _patch_user_dict(monkeypatch, _Recorder(result=[]))
        writes = _Recorder(result=None)
        _patch_user_query(monkeypatch, writes)
        us.ensure_creators_in_local(["Alice", "BOB"])
        assert len(writes.calls) == 2
        inserted = sorted(c[1][0] for c in writes.calls)
        assert inserted == ["alice", "bob"]
        # Placeholder ids start above the local user so they can never collide
        # with LOCAL_USER_ID (scalar MAX(1000, ...) is SQLite's GREATEST).
        assert "MAX(1000" in writes.calls[0][0]
        assert "ON CONFLICT DO NOTHING" in writes.calls[0][0]

    def test_probe_is_lowercased_and_deduped(self, monkeypatch):
        probe = _Recorder(result=[])
        _patch_user_dict(monkeypatch, probe)
        _patch_user_query(monkeypatch, _Recorder(result=None))
        us.ensure_creators_in_local(["Alice", "alice", "ALICE"])
        assert len(probe.calls) == 1
        assert "LOWER(username)" in probe.calls[0][0]
        assert probe.calls[0][1][0] == ["alice"]

    def test_insert_failure_is_swallowed(self, monkeypatch):
        # A bad row must not abort the sync — attribution just degrades to a
        # bare username, which the UI already handles.
        _patch_user_dict(monkeypatch, _Recorder(result=[]))

        def _boom(*a, **k):
            raise RuntimeError("db down")

        monkeypatch.setattr(us, "execute_query", _boom)
        us.ensure_creators_in_local(["ghost"])  # no exception
