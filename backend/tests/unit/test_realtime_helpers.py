"""Pure unit tests for the browse/extract helpers in routes.realtime.

These cover the non-LLM, non-torch helpers that only touch the DB and a couple
of normalization utilities:

  * _usable_dialogue        — the user+assistant-content predicate
  * _extract_ce_conversations — table fallback, JSON decoding, item shapes
  * _list_sample_groups     — test-dataset catalog + per-CE calibration groups
  * _load_sample_group      — testds:/ce:/unknown key dispatch

Every DB call goes through `execute_query_dict`, which is imported at module
top into routes.realtime, so we monkeypatch `routes.realtime.execute_query_dict`.
The two helpers that lazy-import `_normalize_conversation` /
`get_classifier_labels` pull them from their source modules at call time, so we
patch those sources (`evaluation.inference._normalize_conversation`,
`evaluation.ruleset_builder.get_classifier_labels`). No DB, no torch, no LLM is
ever touched.
"""
import json

import pytest

import routes.realtime as rt


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

def _identity_normalize(conv):
    """A stand-in for evaluation.inference._normalize_conversation.

    For a list of message dicts it returns them coerced to {role, content};
    for anything else it returns a single empty user message. This is enough
    to drive the realtime helpers without importing the heavy real one.
    """
    if isinstance(conv, list):
        out = []
        for i, m in enumerate(conv):
            if isinstance(m, dict):
                role = m.get("role") or ("assistant" if i % 2 else "user")
                content = m.get("content") or ""
                out.append({"role": str(role), "content": content})
            elif isinstance(m, str):
                out.append({"role": "assistant" if i % 2 else "user", "content": m})
        return out
    if isinstance(conv, str):
        return [{"role": "assistant", "content": conv}]
    return [{"role": "user", "content": ""}]


def _patch_normalize(monkeypatch):
    """Patch _normalize_conversation in its source module (lazy-imported)."""
    import evaluation.inference as inf
    monkeypatch.setattr(inf, "_normalize_conversation", _identity_normalize)


def _make_eqd(monkeypatch, handler):
    """Install a fake execute_query_dict on routes.realtime.

    `handler(sql, params)` returns the canned rows (a list) for each call.
    Also records the calls for assertions.
    """
    calls = []

    def fake(sql, params=None):
        calls.append((sql, params))
        return handler(sql, params)

    monkeypatch.setattr(rt, "execute_query_dict", fake)
    return calls


# ===========================================================================
# _usable_dialogue
# ===========================================================================

class TestUsableDialogue:
    def test_happy_path_user_then_assistant(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert rt._usable_dialogue(msgs) is True

    def test_empty_list(self):
        assert rt._usable_dialogue([]) is False

    def test_only_user(self):
        assert rt._usable_dialogue([{"role": "user", "content": "hi"}]) is False

    def test_only_assistant(self):
        assert rt._usable_dialogue([{"role": "assistant", "content": "hi"}]) is False

    def test_assistant_with_empty_content_is_not_usable(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ]
        assert rt._usable_dialogue(msgs) is False

    def test_role_is_case_insensitive(self):
        msgs = [
            {"role": "User", "content": "hi"},
            {"role": "ASSISTANT", "content": "yo"},
        ]
        assert rt._usable_dialogue(msgs) is True

    def test_missing_role_key_does_not_crash(self):
        # m.get("role") -> None, ("" ) .lower() guards against None.
        msgs = [
            {"content": "hi"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        assert rt._usable_dialogue(msgs) is True

    def test_none_role_value(self):
        msgs = [
            {"role": None, "content": "x"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        assert rt._usable_dialogue(msgs) is True

    def test_assistant_content_none_not_usable(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None},
        ]
        assert rt._usable_dialogue(msgs) is False


# ===========================================================================
# _extract_ce_conversations
# ===========================================================================

class TestExtractCeConversations:
    def test_prefers_calibration_table(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [convo]}]
            raise AssertionError("should not query excitation when calibration hits")

        calls = _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(7)
        assert out == [convo]
        # Only the calibration table was queried.
        assert len(calls) == 1
        assert "calibration_datasets" in calls[0][0]
        assert calls[0][1] == (7,)

    def test_falls_back_to_excitation_when_calibration_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return []  # nothing in calibration
            if "excitation_datasets" in sql:
                return [{"dataset": [convo]}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(3)
        assert out == [convo]

    def test_falls_back_when_calibration_dataset_is_falsy(self, monkeypatch):
        # Row exists but dataset is None -> not usable, must fall through.
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": None}]
            return [{"dataset": [convo]}]

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(3)
        assert out == [convo]

    def test_no_rows_at_all_returns_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)
        _make_eqd(monkeypatch, lambda sql, params: [])
        assert rt._extract_ce_conversations(99) == []

    def test_execute_returns_none_handled(self, monkeypatch):
        # execute_query_dict may return None; the `or []` guards both reads.
        _patch_normalize(monkeypatch)
        _make_eqd(monkeypatch, lambda sql, params: None)
        assert rt._extract_ce_conversations(1) == []

    def test_dataset_as_json_string_is_decoded(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        raw = json.dumps([convo])

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": raw}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1)
        assert out == [convo]

    def test_invalid_json_string_returns_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": "{not valid json"}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == []

    def test_dict_dataset_uses_conversations_key(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": {"conversations": [convo]}}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == [convo]

    def test_dict_dataset_samples_key_fallback(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": {"samples": [convo]}}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == [convo]

    def test_dict_dataset_training_data_key_fallback(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": {"training_data": [convo]}}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == [convo]

    def test_dict_dataset_with_no_known_keys_yields_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": {"unrelated": [1, 2, 3]}}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == []

    def test_training_pair_item_input_output(self, monkeypatch):
        # Items shaped as {input, output} become a 2-message convo without
        # touching _normalize_conversation.
        import evaluation.inference as inf
        monkeypatch.setattr(inf, "_normalize_conversation",
                            lambda c: (_ for _ in ()).throw(AssertionError("should not be called")))

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [{"input": "hello", "output": "world"}]}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1)
        assert out == [[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]]

    def test_training_pair_prompt_response(self, monkeypatch):
        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [{"prompt": "p", "response": "r"}]}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1)
        assert out == [[
            {"role": "user", "content": "p"},
            {"role": "assistant", "content": "r"},
        ]]

    def test_training_pair_coerces_non_str_to_str(self, monkeypatch):
        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [{"input": 42, "output": {"a": 1}}]}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1)
        assert out[0][0]["content"] == "42"
        assert out[0][1]["content"] == str({"a": 1})

    def test_item_with_messages_key_uses_normalize(self, monkeypatch):
        _patch_normalize(monkeypatch)
        inner = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [{"messages": inner}]}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1)
        assert out == [inner]

    def test_unusable_dialogue_filtered_out(self, monkeypatch):
        # A pair whose assistant content is empty -> filtered (no assistant
        # content), so the result is empty.
        _patch_normalize(monkeypatch)

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": [{"input": "hi", "output": ""}]}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == []

    def test_limit_caps_output(self, monkeypatch):
        _patch_normalize(monkeypatch)
        pair = {"input": "q", "output": "a"}
        items = [pair] * 10

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": items}]
            return []

        _make_eqd(monkeypatch, handler)
        out = rt._extract_ce_conversations(1, limit=3)
        assert len(out) == 3

    def test_raw_is_unexpected_scalar_type_yields_empty(self, monkeypatch):
        # dataset is an int -> not str/dict/list -> items=[] -> []
        _patch_normalize(monkeypatch)

        def handler(sql, params):
            if "calibration_datasets" in sql:
                return [{"dataset": 12345}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._extract_ce_conversations(1) == []


# ===========================================================================
# _list_sample_groups
# ===========================================================================

def _patch_labels(monkeypatch, labels):
    import evaluation.ruleset_builder as rb
    monkeypatch.setattr(rb, "get_classifier_labels", lambda cid: labels)


class TestListSampleGroups:
    def test_test_datasets_mapped_with_type_labels(self, monkeypatch):
        _patch_labels(monkeypatch, {})  # no CEs

        def handler(sql, params):
            if "test_datasets" in sql:
                return [
                    {"dataset_id": 1, "dataset_type": "positive", "rule_name": "R1", "count": 5},
                    {"dataset_id": 2, "dataset_type": "negative", "rule_name": "R1", "count": 2},
                    {"dataset_id": 3, "dataset_type": "positive_calibration", "rule_name": "R2", "count": 4},
                ]
            return []

        _make_eqd(monkeypatch, handler)
        groups = rt._list_sample_groups(10)
        assert groups == [
            {"key": "testds:1", "label": "Test · positive · R1", "count": 5},
            {"key": "testds:2", "label": "Test · negative · R1", "count": 2},
            {"key": "testds:3", "label": "Test · calibration · R2", "count": 4},
        ]

    def test_zero_or_negative_count_test_datasets_skipped(self, monkeypatch):
        _patch_labels(monkeypatch, {})

        def handler(sql, params):
            if "test_datasets" in sql:
                return [
                    {"dataset_id": 1, "dataset_type": "positive", "rule_name": "R1", "count": 0},
                    {"dataset_id": 2, "dataset_type": "positive", "rule_name": "R2", "count": None},
                    {"dataset_id": 3, "dataset_type": "positive", "rule_name": "R3", "count": 1},
                ]
            return []

        _make_eqd(monkeypatch, handler)
        groups = rt._list_sample_groups(10)
        assert [g["key"] for g in groups] == ["testds:3"]

    def test_unknown_dataset_type_passes_through(self, monkeypatch):
        _patch_labels(monkeypatch, {})

        def handler(sql, params):
            if "test_datasets" in sql:
                return [{"dataset_id": 9, "dataset_type": "weird", "rule_name": "R", "count": 3}]
            return []

        _make_eqd(monkeypatch, handler)
        groups = rt._list_sample_groups(10)
        assert groups[0]["label"] == "Test · weird · R"

    def test_ce_groups_appended_when_conversations_present(self, monkeypatch):
        _patch_labels(monkeypatch, {"ce_alpha": 0, "ce_beta": 1})
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "test_datasets" in sql:
                return []
            if "cognitive_elements" in sql:
                # name = ANY(%s) -> params is (ce_names_list,)
                assert params == (["ce_alpha", "ce_beta"],)
                return [
                    {"ce_id": 100, "name": "ce_alpha"},
                    {"ce_id": 200, "name": "ce_beta"},
                ]
            if "calibration_datasets" in sql:
                ce_id = params[0]
                if ce_id == 100:
                    return [{"dataset": [convo, convo]}]  # 2 convos
                return []  # ce 200 calibration empty
            if "excitation_datasets" in sql:
                return []  # ce 200 has nothing -> skipped
            return []

        _make_eqd(monkeypatch, handler)
        groups = rt._list_sample_groups(10)
        # Only ce_alpha (count 2) survives; ce_beta has 0 convos.
        assert groups == [{"key": "ce:100", "label": "CE · ce_alpha", "count": 2}]

    def test_no_ce_labels_skips_ce_query(self, monkeypatch):
        _patch_labels(monkeypatch, {})  # falsy -> no CE branch

        seen = {"ce_query": False}

        def handler(sql, params):
            if "cognitive_elements" in sql:
                seen["ce_query"] = True
            if "test_datasets" in sql:
                return []
            return []

        _make_eqd(monkeypatch, handler)
        groups = rt._list_sample_groups(10)
        assert groups == []
        assert seen["ce_query"] is False

    def test_get_classifier_labels_returns_none(self, monkeypatch):
        # `(get_classifier_labels(...) or {})` guards a None return.
        _patch_labels(monkeypatch, None)

        def handler(sql, params):
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._list_sample_groups(10) == []

    def test_test_datasets_query_returns_none(self, monkeypatch):
        # The `or []` after the first query guards a None result.
        _patch_labels(monkeypatch, {})

        def handler(sql, params):
            return None

        _make_eqd(monkeypatch, handler)
        assert rt._list_sample_groups(10) == []


# ===========================================================================
# _load_sample_group
# ===========================================================================

class TestLoadSampleGroup:
    def test_unknown_key_returns_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)
        # No DB call expected.
        _make_eqd(monkeypatch, lambda sql, params: (_ for _ in ()).throw(AssertionError("no DB")))
        assert rt._load_sample_group("bogus:1") == []

    def test_ce_key_delegates_to_extract(self, monkeypatch):
        _patch_normalize(monkeypatch)
        convo = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            if "calibration_datasets" in sql:
                assert params == (55,)
                return [{"dataset": [convo]}]
            return []

        _make_eqd(monkeypatch, handler)
        assert rt._load_sample_group("ce:55") == [convo]

    def test_testds_key_normalizes_and_filters(self, monkeypatch):
        _patch_normalize(monkeypatch)
        good = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        bad = [{"role": "user", "content": "q"}]  # no assistant -> filtered

        def handler(sql, params):
            assert "test_datasets" in sql
            assert params == (77,)
            return [{"conversations": [good, bad]}]

        _make_eqd(monkeypatch, handler)
        out = rt._load_sample_group("testds:77")
        assert out == [good]

    def test_testds_conversations_as_json_string(self, monkeypatch):
        _patch_normalize(monkeypatch)
        good = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

        def handler(sql, params):
            return [{"conversations": json.dumps([good])}]

        _make_eqd(monkeypatch, handler)
        assert rt._load_sample_group("testds:1") == [good]

    def test_testds_no_rows_returns_empty(self, monkeypatch):
        _patch_normalize(monkeypatch)
        _make_eqd(monkeypatch, lambda sql, params: [])
        assert rt._load_sample_group("testds:1") == []

    def test_testds_null_conversations_returns_empty(self, monkeypatch):
        # Row present but conversations is None -> `(convos or [])` -> [].
        _patch_normalize(monkeypatch)
        _make_eqd(monkeypatch, lambda sql, params: [{"conversations": None}])
        assert rt._load_sample_group("testds:1") == []


# ===========================================================================
# rule_triggers — the v2 per-rule verdict shape the RealtimeViewer consumes
# ===========================================================================

class TestScoreSpanRuleTriggers:
    """_score_span emits one entry per enabled rule:
    {rule_name, fired, condition, groups: {gname: {members, hits, quantifier,
    satisfied}}, groups_unmet}. Torch is required (compute_triggers), but no
    DB, no LLM — the scoring context is built by hand."""

    def _ctx(self, unified_ruleset):
        torch = pytest.importorskip("torch")
        import classifier_engine.reference  # noqa: F401 — gavel.* alias
        from gavel.evaluation.metrics import build_ruleset_logic

        labels = {"ce_a": 0, "ce_b": 1, "ce_c": 2}
        return {
            "idx_to_label": {v: k for k, v in labels.items()},
            "num_topics": len(labels),
            "thr_vec": torch.tensor([0.5, 0.5, 0.5]),
            "patience_rate": 1,
            "unified_ruleset": unified_ruleset,
            "ruleset_logic": build_ruleset_logic(unified_ruleset, labels),
        }

    def _windows(self, fire_idxs):
        # One window; ±5 logits — sigmoid 0.993 / 0.007 around the 0.5 thr.
        logits = [-5.0, -5.0, -5.0]
        for i in fire_idxs:
            logits[i] = 5.0
        return [{"logits": list(logits)}]

    def test_fired_rule_reports_satisfied_groups(self):
        ruleset = {"r": {
            "enabled": True,
            "groups": {"req": ["ce_a"], "opt": ["ce_b", "ce_c"]},
            "condition": "all of req and 1 of opt",
        }}
        out = rt._score_span(self._ctx(ruleset), self._windows([0, 1]), [])
        assert out["triggered_ces"] == ["ce_a", "ce_b"]
        (rtg,) = out["rule_triggers"]
        assert rtg["rule_name"] == "r"
        assert rtg["fired"] is True
        assert rtg["condition"] == "all of req and 1 of opt"
        assert rtg["groups_unmet"] == []
        assert rtg["groups"]["req"] == {
            "members": ["ce_a"], "hits": ["ce_a"],
            "quantifier": "all", "satisfied": True,
        }
        assert rtg["groups"]["opt"] == {
            "members": ["ce_b", "ce_c"], "hits": ["ce_b"],
            "quantifier": 1, "satisfied": True,
        }

    def test_unfired_rule_reports_unmet_groups(self):
        ruleset = {"r": {
            "enabled": True,
            "groups": {"req": ["ce_a"], "opt": ["ce_b", "ce_c"]},
            "condition": "all of req and 1 of opt",
        }}
        out = rt._score_span(self._ctx(ruleset), self._windows([0]), [])
        (rtg,) = out["rule_triggers"]
        assert rtg["fired"] is False
        assert rtg["groups_unmet"] == ["opt"]
        assert rtg["groups"]["opt"]["hits"] == []
        assert rtg["groups"]["opt"]["satisfied"] is False

    def test_unreferenced_supporting_group_is_always_satisfied(self):
        ruleset = {"r": {
            "enabled": True,
            "groups": {"req": ["ce_a"], "supporting": ["ce_c"]},
            "condition": "all of req",
        }}
        out = rt._score_span(self._ctx(ruleset), self._windows([0]), [])
        (rtg,) = out["rule_triggers"]
        assert rtg["fired"] is True
        sup = rtg["groups"]["supporting"]
        assert sup["quantifier"] is None      # not part of the condition
        assert sup["satisfied"] is True       # can never block the rule
        assert rtg["groups_unmet"] == []

    def test_disabled_rules_are_omitted(self):
        ruleset = {
            "on": {"enabled": True, "groups": {"g": ["ce_a"]}, "condition": "all of g"},
            "off": {"enabled": False, "groups": {"g": ["ce_b"]}, "condition": "all of g"},
        }
        out = rt._score_span(self._ctx(ruleset), self._windows([0, 1]), [])
        assert [r["rule_name"] for r in out["rule_triggers"]] == ["on"]


class TestAggregateRuleTriggers:
    """_aggregate_rule_triggers — conversation-level rollup of per-turn
    verdicts (pure function, no torch)."""

    def _entry(self, name, fired, hits, satisfied, quantifier="all",
               members=("ce_a", "ce_b")):
        return {
            "rule_name": name,
            "fired": fired,
            "condition": "all of g",
            "groups": {"g": {"members": list(members), "hits": list(hits),
                             "quantifier": quantifier, "satisfied": satisfied}},
            "groups_unmet": [] if satisfied else ["g"],
        }

    def test_fired_on_any_turn_wins(self):
        turns = [
            [self._entry("r", False, [], False)],
            [self._entry("r", True, ["ce_a", "ce_b"], True)],
        ]
        (agg,) = rt._aggregate_rule_triggers(turns)
        assert agg["fired"] is True
        assert agg["groups"]["g"]["hits"] == ["ce_a", "ce_b"]
        assert agg["groups_unmet"] == []

    def test_unfired_turns_merge_hits_across_turns(self):
        # Neither turn fired, but different members hit on different turns —
        # the rollup unions the hits for the "how close" hint.
        turns = [
            [self._entry("r", False, ["ce_a"], False)],
            [self._entry("r", False, ["ce_b"], False)],
        ]
        (agg,) = rt._aggregate_rule_triggers(turns)
        assert agg["fired"] is False
        assert agg["groups"]["g"]["hits"] == ["ce_a", "ce_b"]
        assert agg["groups_unmet"] == ["g"]

    def test_satisfied_merge_clears_groups_unmet(self):
        turns = [
            [self._entry("r", False, [], False)],
            [self._entry("r", False, ["ce_a", "ce_b"], True)],
        ]
        (agg,) = rt._aggregate_rule_triggers(turns)
        assert agg["groups"]["g"]["satisfied"] is True
        assert agg["groups_unmet"] == []

    def test_order_is_first_seen(self):
        turns = [
            [self._entry("r1", False, [], False), self._entry("r2", True, ["ce_a"], True)],
            [self._entry("r3", False, [], False)],
        ]
        agg = rt._aggregate_rule_triggers(turns)
        assert [a["rule_name"] for a in agg] == ["r1", "r2", "r3"]

    def test_empty_input(self):
        assert rt._aggregate_rule_triggers([]) == []
        assert rt._aggregate_rule_triggers([None, []]) == []
