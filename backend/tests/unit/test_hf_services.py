"""Pure unit tests for the HuggingFace-facing services.

Targets the *pure logic* inside three modules that otherwise talk to HF over
the network:

  * services.hf_publish  — payload / manifest / path builders, race detection,
    stable JSON encoding, category-id translation.
  * services.hf_sync     — manifest content-hash probe, published_at staleness
    comparison, sha match-vs-mismatch short-circuit, crash-recovery manifest
    diff, SyncResult shape.

NO network and NO database are touched. Every DB seam (`execute_query` /
`execute_query_dict`, imported INTO each module at load time) and every HF
seam (`hf_hub_download`, `_fetch_head_sha_and_manifest`, `_resolve_token`,
`_get_api`, etc.) is monkeypatched on the module-under-test namespace.

Pure network I/O with no branching logic (the actual `hf_hub_download`
round-trips, `_push_atomic`, `central_server.*`) is deliberately NOT tested
here — there is no logic to assert on, only a forwarded call.
"""
import hashlib
import json

import pytest

from services import hf_publish
from services import hf_sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingWriter:
    """Records execute_query (write) calls; returns None like the real seam."""

    def __init__(self):
        self.calls = []

    def __call__(self, sql, params=None):
        self.calls.append((sql, params))
        return None


# ===========================================================================
# hf_publish — stable JSON encoding (_to_bytes) + race detection
# ===========================================================================


class TestToBytes:
    def test_deterministic_for_key_order(self):
        a = hf_publish._to_bytes({"b": 1, "a": 2})
        b = hf_publish._to_bytes({"a": 2, "b": 1})
        assert a == b  # sort_keys makes hash stable regardless of insert order

    def test_is_bytes_and_utf8_decodable(self):
        out = hf_publish._to_bytes({"x": 1})
        assert isinstance(out, bytes)
        assert json.loads(out.decode("utf-8")) == {"x": 1}

    def test_non_ascii_preserved(self):
        # ensure_ascii=False keeps unicode literal, not \uXXXX escapes.
        out = hf_publish._to_bytes({"name": "café"})
        assert "café" in out.decode("utf-8")

    def test_hash_stability_across_calls(self):
        payload = {"z": [3, 2, 1], "a": "x"}
        h1 = hashlib.sha256(hf_publish._to_bytes(payload)).hexdigest()
        h2 = hashlib.sha256(hf_publish._to_bytes(dict(payload))).hexdigest()
        assert h1 == h2


class TestIsRaceError:
    @pytest.mark.parametrize(
        "msg",
        [
            "HTTP 412 Precondition Failed",
            "precondition failed",
            "the ref is stale",
            "fetch first then push",
            "branch is out-of-date",
        ],
    )
    def test_race_messages_detected(self, msg):
        assert hf_publish._is_race_error(Exception(msg)) is True

    @pytest.mark.parametrize("msg", ["connection reset", "404 not found", "", "500 server error"])
    def test_non_race_messages_not_detected(self, msg):
        assert hf_publish._is_race_error(Exception(msg)) is False

    def test_case_insensitive(self):
        assert hf_publish._is_race_error(Exception("PRECONDITION")) is True
        assert hf_publish._is_race_error(Exception("STALE ref")) is True


# ===========================================================================
# hf_publish — _now_iso shape
# ===========================================================================


class TestNowIso:
    def test_iso_z_suffixed(self):
        s = hf_publish._now_iso()
        assert s.endswith("Z")
        assert "T" in s
        # Parses back as a real timestamp after swapping Z for +00:00.
        from datetime import datetime
        datetime.fromisoformat(s.replace("Z", "+00:00"))


# ===========================================================================
# hf_publish — _op operation builder
# ===========================================================================


class TestOp:
    def test_op_shape(self):
        op = hf_publish._op("public_ces/x.json", b"data")
        assert op == {"path": "public_ces/x.json", "content": b"data"}


# ===========================================================================
# hf_publish — path construction
# ===========================================================================


class TestRuleDatasetPath:
    def test_composite_path(self):
        assert (
            hf_publish._rule_dataset_path("rule_abc", "positive")
            == "public_rule_datasets/rule_abc_positive.json"
        )

    def test_matches_hf_sync_path(self):
        # The publish and sync sides MUST agree on the path or lazy-pull breaks.
        assert hf_publish._rule_dataset_path("rule_z", "negative") == \
            hf_sync._rule_dataset_path("rule_z", "negative")


# ===========================================================================
# hf_publish — _slim_dataset_config
# ===========================================================================


class TestSlimDatasetConfig:
    def test_keeps_only_consumer_keys(self):
        cfg = {
            "scenario_instructions": "do X",
            "necessary_labels": ["a"],
            "sufficient_labels": ["b"],
            "generator_model": "secret-model",
            "persona_pool": ["p1"],
            "seed_dialogues": [1, 2, 3],
        }
        out = hf_publish._slim_dataset_config(cfg)
        assert out == {
            "scenario_instructions": "do X",
            "necessary_labels": ["a"],
            "sufficient_labels": ["b"],
        }

    def test_empty_config_yields_empty(self):
        assert hf_publish._slim_dataset_config({}) == {}

    def test_missing_keys_skipped(self):
        assert hf_publish._slim_dataset_config({"generator_model": "x"}) == {}


# ===========================================================================
# hf_publish — payload builders (patch _category_names_from_ids' DB seam)
# ===========================================================================


class TestBuildCePayload:
    def _patch_cats(self, monkeypatch, names):
        monkeypatch.setattr(
            hf_publish, "execute_query_dict",
            lambda *a, **k: [{"name": n} for n in names],
        )

    def test_full_payload(self, monkeypatch):
        self._patch_cats(monkeypatch, ["Persuasion"])
        row = {
            "name": "Flattery",
            "definition": "praises the user",
            "category": "TACTIC",
            "categories": [3],
            "examples": [{"a": 1}],
            "created_by_username": "alice",
        }
        out = hf_publish._build_ce_payload(row, "ce_123", "2026-01-01T00:00:00Z")
        assert out["schema_version"] == 1
        assert out["public_id"] == "ce_123"
        assert out["name"] == "Flattery"
        assert out["definition"] == "praises the user"
        assert out["category"] == "TACTIC"
        assert out["categories"] == ["Persuasion"]
        assert out["examples"] == [{"a": 1}]
        assert out["published_at"] == "2026-01-01T00:00:00Z"
        assert out["created_by_username"] == "alice"

    def test_defaults_for_missing_optional_fields(self, monkeypatch):
        # No categories -> _category_names_from_ids short-circuits to [] without
        # any DB call; definition/category fall back to defaults.
        self._patch_cats(monkeypatch, ["unused"])
        row = {"name": "Bare"}
        out = hf_publish._build_ce_payload(row, "ce_x", "t")
        assert out["definition"] == ""
        assert out["category"] == "CONTEXT"
        assert out["categories"] == []
        assert out["examples"] == []

    def test_creator_omitted_when_absent(self, monkeypatch):
        self._patch_cats(monkeypatch, [])
        out = hf_publish._build_ce_payload({"name": "N"}, "ce_x", "t")
        assert "created_by_username" not in out

    def test_examples_string_is_parsed(self, monkeypatch):
        self._patch_cats(monkeypatch, [])
        row = {"name": "N", "examples": json.dumps([{"k": "v"}])}
        out = hf_publish._build_ce_payload(row, "ce_x", "t")
        assert out["examples"] == [{"k": "v"}]


class TestBuildExcitationPayload:
    def test_counts_samples(self):
        out = hf_publish._build_excitation_payload([1, 2, 3], "ce_1", "t")
        assert out["sample_count"] == 3
        assert out["samples"] == [1, 2, 3]
        assert out["ce_public_id"] == "ce_1"
        assert out["schema_version"] == 1

    def test_empty_samples(self):
        out = hf_publish._build_excitation_payload([], "ce_1", "t")
        assert out["sample_count"] == 0

    def test_non_list_sample_count_zero(self):
        # Defensive branch: non-list samples => count 0.
        out = hf_publish._build_excitation_payload("notalist", "ce_1", "t")
        assert out["sample_count"] == 0


class TestBuildCeCalibrationPayload:
    def test_same_envelope_as_excitation(self):
        out = hf_publish._build_ce_calibration_payload([1, 2], "ce_9", "t")
        assert out["sample_count"] == 2
        assert out["ce_public_id"] == "ce_9"
        assert out["samples"] == [1, 2]


class TestBuildRuleDatasetPayload:
    def test_slims_config_and_counts_convos(self):
        row = {
            "dataset_type": "positive",
            "conversations": [{"a": 1}, {"b": 2}],
            "config": {"scenario_instructions": "s", "generator_model": "drop-me"},
        }
        out = hf_publish._build_rule_dataset_payload(row, "rule_1", "t")
        assert out["rule_public_id"] == "rule_1"
        assert out["dataset_type"] == "positive"
        assert out["conversation_count"] == 2
        assert out["config"] == {"scenario_instructions": "s"}

    def test_string_conversations_parsed(self):
        row = {
            "dataset_type": "negative",
            "conversations": json.dumps([{"x": 1}]),
            "config": {},
        }
        out = hf_publish._build_rule_dataset_payload(row, "rule_1", "t")
        assert out["conversations"] == [{"x": 1}]
        assert out["conversation_count"] == 1

    def test_malformed_conversations_string_yields_empty(self):
        row = {"dataset_type": "negative", "conversations": "{not json", "config": {}}
        out = hf_publish._build_rule_dataset_payload(row, "rule_1", "t")
        assert out["conversations"] == []
        assert out["conversation_count"] == 0

    def test_malformed_config_string_yields_empty(self):
        row = {"dataset_type": "positive", "conversations": [], "config": "{bad"}
        out = hf_publish._build_rule_dataset_payload(row, "rule_1", "t")
        assert out["config"] == {}

    def test_none_conversations_and_config(self):
        row = {"dataset_type": "positive", "conversations": None, "config": None}
        out = hf_publish._build_rule_dataset_payload(row, "rule_1", "t")
        assert out["conversations"] == []
        assert out["config"] == {}


class TestBuildRulePayload:
    def _patch_cats(self, monkeypatch, names):
        monkeypatch.setattr(
            hf_publish, "execute_query_dict",
            lambda *a, **k: [{"name": n} for n in names],
        )

    def test_ce_dependencies_sorted_and_deduped(self, monkeypatch):
        self._patch_cats(monkeypatch, [])
        rule_row = {"name": "R", "predicate": "A AND B", "categories": []}
        necessary = [{"name": "Alpha"}]
        fallback = [[{"name": "Beta"}, {"name": "Gamma"}]]
        sufficient = [{"name": "Alpha"}]  # duplicate name across roles
        name_to_pid = {
            "Alpha": "ce_a", "Beta": "ce_b", "Gamma": "ce_g",
        }
        out = hf_publish._build_rule_payload(
            rule_row, necessary, fallback, sufficient, "rule_1", "t", name_to_pid
        )
        # Sorted, deduped union of all referenced CE public_ids.
        assert out["ce_dependencies"] == ["ce_a", "ce_b", "ce_g"]
        assert out["necessary"] == ["Alpha"]
        assert out["fallback"] == [["Beta", "Gamma"]]
        assert out["sufficient"] == ["Alpha"]

    def test_name_missing_from_map_excluded_from_deps(self, monkeypatch):
        self._patch_cats(monkeypatch, [])
        rule_row = {"name": "R", "categories": []}
        necessary = [{"name": "Known"}, {"name": "Unmapped"}]
        out = hf_publish._build_rule_payload(
            rule_row, necessary, [], [], "rule_1", "t", {"Known": "ce_k"}
        )
        # Only the mapped name contributes to ce_dependencies.
        assert out["ce_dependencies"] == ["ce_k"]

    def test_creator_omitted_when_absent(self, monkeypatch):
        self._patch_cats(monkeypatch, [])
        out = hf_publish._build_rule_payload(
            {"name": "R", "categories": []}, [], [], [], "rule_1", "t", {}
        )
        assert "created_by_username" not in out
        assert out["predicate"] == ""
        assert out["definition"] == ""


class TestCategoryNamesFromIds:
    def test_empty_ids_skips_db(self, monkeypatch):
        called = {"n": 0}

        def fake(*a, **k):
            called["n"] += 1
            return []

        monkeypatch.setattr(hf_publish, "execute_query_dict", fake)
        assert hf_publish._category_names_from_ids([]) == []
        assert called["n"] == 0  # short-circuit, no DB query

    def test_translates_ids_to_names(self, monkeypatch):
        monkeypatch.setattr(
            hf_publish, "execute_query_dict",
            lambda *a, **k: [{"name": "X"}, {"name": "Y"}],
        )
        assert hf_publish._category_names_from_ids([1, 2]) == ["X", "Y"]

    def test_none_rows_returns_empty(self, monkeypatch):
        monkeypatch.setattr(hf_publish, "execute_query_dict", lambda *a, **k: None)
        assert hf_publish._category_names_from_ids([5]) == []


# ===========================================================================
# hf_sync — content-hash + SyncResult
# ===========================================================================


class TestHashBytes:
    def test_matches_sha256(self):
        assert hf_sync._hash_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()

    def test_deterministic(self):
        assert hf_sync._hash_bytes(b"x") == hf_sync._hash_bytes(b"x")

    def test_different_bytes_differ(self):
        assert hf_sync._hash_bytes(b"a") != hf_sync._hash_bytes(b"b")


class TestSyncResultToDict:
    def test_to_dict_shape(self):
        r = hf_sync.SyncResult(changed=True, ces_added=2, rules_added=1,
                               rule_sets_added=7,
                               ces_refreshed=5, rules_refreshed=6,
                               rule_sets_refreshed=8,
                               categories_synced=3, neutral_synced=4,
                               skipped_records=["s"], errors=["e"])
        d = r.to_dict()
        assert d == {
            "changed": True,
            "ces_added": 2,
            "rules_added": 1,
            "rule_sets_added": 7,
            "ces_refreshed": 5,
            "rules_refreshed": 6,
            "rule_sets_refreshed": 8,
            "categories_synced": 3,
            "neutral_synced": 4,
            "skipped_records": ["s"],
            "errors": ["e"],
        }

    def test_defaults_zeroed(self):
        r = hf_sync.SyncResult(changed=False)
        assert r.ces_added == 0 and r.rules_added == 0
        assert r.skipped_records == [] and r.errors == []


# ===========================================================================
# hf_sync — _hf_pubat_is_newer (staleness comparison)
# ===========================================================================


class TestHfPubatIsNewer:
    def test_empty_manifest_pubat_false(self):
        assert hf_sync._hf_pubat_is_newer("", "2020-01-01T00:00:00Z") is False
        assert hf_sync._hf_pubat_is_newer(None, "2020-01-01T00:00:00Z") is False

    def test_local_none_forces_refresh(self):
        assert hf_sync._hf_pubat_is_newer("2020-01-01T00:00:00Z", None) is True

    def test_newer_manifest_is_true(self):
        assert hf_sync._hf_pubat_is_newer(
            "2021-01-01T00:00:00Z", "2020-01-01T00:00:00Z"
        ) is True

    def test_older_manifest_is_false(self):
        assert hf_sync._hf_pubat_is_newer(
            "2019-01-01T00:00:00Z", "2020-01-01T00:00:00Z"
        ) is False

    def test_equal_is_false(self):
        ts = "2020-01-01T00:00:00Z"
        assert hf_sync._hf_pubat_is_newer(ts, ts) is False

    def test_z_suffix_tolerated(self):
        # Both Z forms parse; newer should win.
        assert hf_sync._hf_pubat_is_newer(
            "2022-06-01T12:00:00Z", "2022-06-01T11:00:00Z"
        ) is True

    def test_datetime_local_value(self):
        from datetime import datetime, timezone
        local = datetime(2020, 1, 1, tzinfo=timezone.utc)
        assert hf_sync._hf_pubat_is_newer("2021-01-01T00:00:00+00:00", local) is True
        assert hf_sync._hf_pubat_is_newer("2019-01-01T00:00:00+00:00", local) is False

    def test_malformed_manifest_pubat_false(self):
        assert hf_sync._hf_pubat_is_newer("not-a-date", "2020-01-01T00:00:00Z") is False

    def test_malformed_local_pubat_false(self):
        assert hf_sync._hf_pubat_is_newer("2020-01-01T00:00:00Z", "garbage") is False


# ===========================================================================
# hf_sync — check_for_updates (sha match vs mismatch probe logic)
# ===========================================================================


class TestCheckForUpdates:
    def test_no_token_still_probes_anonymously(self, monkeypatch):
        # The public library repo is readable without auth, so a missing token
        # must NOT short-circuit the probe — it proceeds with token=None and
        # still reports checked=True.
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: None)
        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", lambda: b"manifest")
        cached = hf_sync._hash_bytes(b"manifest")
        monkeypatch.setattr(hf_sync, "_get_state", lambda key: cached)
        out = hf_sync.check_for_updates()
        assert out == {"available": False, "checked": True, "reason": None}

    def test_probe_exception_not_checked(self, monkeypatch):
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: "tok")

        def boom():
            raise RuntimeError("network down")

        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", boom)
        out = hf_sync.check_for_updates()
        assert out["available"] is False
        assert out["checked"] is False
        assert "network down" in out["reason"]

    def test_hash_match_no_update(self, monkeypatch):
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: "tok")
        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", lambda: b"manifest")
        cached = hf_sync._hash_bytes(b"manifest")
        monkeypatch.setattr(hf_sync, "_get_state", lambda key: cached)
        out = hf_sync.check_for_updates()
        assert out == {"available": False, "checked": True, "reason": None}

    def test_hash_mismatch_update_available(self, monkeypatch):
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: "tok")
        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", lambda: b"new-manifest")
        monkeypatch.setattr(hf_sync, "_get_state", lambda key: "stale-hash")
        out = hf_sync.check_for_updates()
        assert out == {"available": True, "checked": True, "reason": None}

    def test_no_cached_hash_means_available(self, monkeypatch):
        # First-ever probe: last_hash is None != current -> available True.
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: "tok")
        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", lambda: b"m")
        monkeypatch.setattr(hf_sync, "_get_state", lambda key: None)
        assert hf_sync.check_for_updates()["available"] is True


# ===========================================================================
# hf_sync — _sync_library_locked sha-match short-circuit (changed=False)
# ===========================================================================


class TestSyncLibraryShortCircuit:
    def test_no_token_syncs_anonymously(self, monkeypatch):
        # Missing token no longer aborts the sync — the public repo is read
        # anonymously. With a matching manifest hash it short-circuits cleanly
        # (changed=False, no errors), proving no HF_TOKEN gate remains.
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: None)
        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", lambda: b"manifest")
        cached = hf_sync._hash_bytes(b"manifest")
        monkeypatch.setattr(hf_sync, "_get_state", lambda key: cached)
        res = hf_sync._sync_library_locked(force=False)
        assert res.changed is False
        assert not res.errors

    def test_manifest_fetch_failure_returns_error(self, monkeypatch):
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: "tok")

        def boom():
            raise RuntimeError("download failed")

        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", boom)
        res = hf_sync._sync_library_locked(force=False)
        assert res.changed is False
        assert "Could not fetch manifest" in res.errors[0]

    def test_hash_match_short_circuits_changed_false(self, monkeypatch):
        # The core diff-detection property: when the manifest hash matches the
        # cached one and force=False, sync returns changed=False and does no
        # record pulls. recover_pending_publishes is stubbed to a no-op.
        manifest_bytes = json.dumps({"schema_version": 1}).encode("utf-8")
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: "tok")
        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", lambda: manifest_bytes)
        monkeypatch.setattr(hf_sync, "_get_state", lambda key: hf_sync._hash_bytes(manifest_bytes))

        recover_calls = {"n": 0}
        monkeypatch.setattr(
            hf_sync, "recover_pending_publishes",
            lambda manifest=None, token=None: recover_calls.__setitem__("n", recover_calls["n"] + 1) or {},
        )
        # Guard: a stray DB call would surface as an error rather than a silent pass.
        monkeypatch.setattr(hf_sync, "execute_query_dict", lambda *a, **k: pytest.fail("no DB on cache hit"))

        res = hf_sync._sync_library_locked(force=False)
        assert res.changed is False
        assert res.ces_added == 0 and res.rules_added == 0
        assert recover_calls["n"] == 1  # recovery still runs on cache hit

    def test_force_bypasses_hash_match(self, monkeypatch):
        # With force=True, an equal hash must NOT short-circuit; it proceeds to
        # manifest validation + diff. We stub the diff seams so no pulls happen.
        manifest_bytes = json.dumps(
            {"schema_version": 1, "ces": {}, "rules": {}}
        ).encode("utf-8")
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: "tok")
        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", lambda: manifest_bytes)
        monkeypatch.setattr(hf_sync, "_get_state", lambda key: hf_sync._hash_bytes(manifest_bytes))
        monkeypatch.setattr(hf_sync, "recover_pending_publishes", lambda **k: {})
        monkeypatch.setattr(hf_sync, "_local_public_ids", lambda: (set(), set()))
        writer = _RecordingWriter()
        monkeypatch.setattr(hf_sync, "execute_query", writer)

        res = hf_sync._sync_library_locked(force=True)
        # changed=True path taken (no short-circuit); empty manifest -> nothing
        # to pull, returns early after marking state seen.
        assert res.changed is True
        assert res.ces_added == 0 and res.rules_added == 0

    def test_manifest_validation_failure_returns_error(self, monkeypatch):
        # Bytes that aren't a valid manifest (missing required schema_version)
        # must surface as an error result, not crash, once past the hash check.
        bad_bytes = json.dumps({"not": "a manifest"}).encode("utf-8")
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: "tok")
        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", lambda: bad_bytes)
        monkeypatch.setattr(hf_sync, "_get_state", lambda key: "different-hash")
        monkeypatch.setattr(hf_sync, "recover_pending_publishes", lambda **k: {})
        res = hf_sync._sync_library_locked(force=False)
        assert res.changed is False
        assert "Manifest validation failed" in res.errors[0]

    def test_stale_record_is_refreshed_not_duplicated(self, monkeypatch):
        # H3 regression: a record already present locally but with a NEWER
        # published_at in the manifest must be RE-pulled (refreshed in place),
        # while a record with an equal/older timestamp must be left alone.
        # Without this, an upstream edit that keeps the same public_id (an admin
        # re-categorizing a seed CE) would silently never reach clients that
        # already hold the record.
        manifest_bytes = json.dumps({
            "schema_version": 1,
            "ces": {"ce-1": "2024-01-02T00:00:00Z"},      # newer than local -> stale
            "rules": {"rule-1": "2024-01-01T00:00:00Z"},  # equal to local -> fresh
        }).encode("utf-8")
        monkeypatch.setattr(hf_sync, "_resolve_token", lambda: "tok")
        monkeypatch.setattr(hf_sync, "_fetch_manifest_bytes", lambda: manifest_bytes)
        monkeypatch.setattr(hf_sync, "_get_state", lambda key: "old-hash")  # force the diff path
        monkeypatch.setattr(hf_sync, "recover_pending_publishes", lambda **k: {})
        # Both records already present locally.
        monkeypatch.setattr(hf_sync, "_local_public_ids", lambda: ({"ce-1"}, {"rule-1"}))
        monkeypatch.setattr(hf_sync, "_local_pubat_map", lambda: (
            {"ce-1": "2024-01-01T00:00:00Z"},    # older than manifest -> refresh
            {"rule-1": "2024-01-01T00:00:00Z"},  # same as manifest -> skip
        ))
        pulled = {"ce": [], "rule": []}
        monkeypatch.setattr(hf_sync, "_pull_ce",
                            lambda token, pid, result: (pulled["ce"].append(pid) or True))
        monkeypatch.setattr(hf_sync, "_pull_rule",
                            lambda token, pid, result: (pulled["rule"].append(pid) or True))
        monkeypatch.setattr(hf_sync, "_set_state", lambda *a, **k: None)
        # Creator-mirror lookup touches the DB; no rows -> ensure_creators skipped.
        monkeypatch.setattr(hf_sync, "execute_query_dict", lambda *a, **k: [])

        res = hf_sync._sync_library_locked(force=False)
        assert pulled["ce"] == ["ce-1"]   # stale CE re-pulled
        assert pulled["rule"] == []       # fresh rule left alone
        assert res.ces_refreshed == 1 and res.ces_added == 0
        assert res.rules_refreshed == 0 and res.rules_added == 0


# ===========================================================================
# hf_sync — recover_pending_publishes (manifest diff + heal/clear logic)
# ===========================================================================


class TestRecoverPendingPublishes:
    def test_nothing_pending_no_db_writes(self, monkeypatch):
        # All three pending lookups empty -> early return, manifest never fetched.
        monkeypatch.setattr(hf_sync, "execute_query_dict", lambda *a, **k: [])
        writer = _RecordingWriter()
        monkeypatch.setattr(hf_sync, "execute_query", writer)
        out = hf_sync.recover_pending_publishes(manifest={}, token="tok")
        assert out == {
            "healed_rules": 0, "healed_ces": 0, "healed_datasets": 0, "healed_rule_sets": 0,
            "cleared_rules": 0, "cleared_ces": 0, "cleared_datasets": 0, "cleared_rule_sets": 0,
        }
        assert writer.calls == []

    def test_pending_rule_in_registry_heals_forward(self, monkeypatch):
        # First lookup: pending rules. Second: pending ces. Third: pending
        # datasets. We queue those, then the manifest says the rule's
        # pending_public_id exists -> heal forward.
        queue = [
            [{"rule_id": 1, "pending_public_id": "rule_p"}],  # pending_rules
            [],  # pending_ces
            [],  # pending_datasets
        ]
        monkeypatch.setattr(
            hf_sync, "execute_query_dict",
            lambda *a, **k: queue.pop(0) if queue else [],
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(hf_sync, "execute_query", writer)
        manifest = {"rules": {"rule_p": "2026-01-01T00:00:00Z"}, "ces": {}}
        out = hf_sync.recover_pending_publishes(manifest=manifest, token="tok")
        assert out["healed_rules"] == 1
        assert out["cleared_rules"] == 0
        # The heal-forward UPDATE carries the public_id + published_at + rule_id.
        sql, params = writer.calls[0]
        assert "UPDATE rules" in sql
        assert params == ("rule_p", "2026-01-01T00:00:00Z", 1)

    def test_pending_rule_not_in_registry_cleared(self, monkeypatch):
        queue = [
            [{"rule_id": 2, "pending_public_id": "rule_missing"}],
            [],
            [],
        ]
        monkeypatch.setattr(
            hf_sync, "execute_query_dict",
            lambda *a, **k: queue.pop(0) if queue else [],
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(hf_sync, "execute_query", writer)
        manifest = {"rules": {}, "ces": {}}
        out = hf_sync.recover_pending_publishes(manifest=manifest, token="tok")
        assert out["cleared_rules"] == 1
        assert out["healed_rules"] == 0
        sql, params = writer.calls[0]
        assert "pending_public_id = NULL" in sql
        assert params == (2,)

    def test_pending_ce_in_registry_heals(self, monkeypatch):
        queue = [
            [],  # pending_rules
            [{"ce_id": 5, "pending_public_id": "ce_p"}],  # pending_ces
            [],  # pending_datasets
        ]
        monkeypatch.setattr(
            hf_sync, "execute_query_dict",
            lambda *a, **k: queue.pop(0) if queue else [],
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(hf_sync, "execute_query", writer)
        manifest = {"rules": {}, "ces": {"ce_p": "2026-02-02T00:00:00Z"}}
        out = hf_sync.recover_pending_publishes(manifest=manifest, token="tok")
        assert out["healed_ces"] == 1
        sql, params = writer.calls[0]
        assert "UPDATE cognitive_elements" in sql
        assert params == ("ce_p", "2026-02-02T00:00:00Z", 5)

    def test_pending_dataset_heals_when_rule_published(self, monkeypatch):
        # Dataset rows: their fate follows the rule. First three lookups are the
        # pending lists; a later lookup resolves the rule's public_id.
        queue = [
            [],  # pending_rules
            [],  # pending_ces
            [{"dataset_id": 9, "rule_id": 3, "pending_public_id": "rule_p_positive"}],
            [],  # pending_rule_sets
            [{"public_id": "rule_p"}],  # rule public_id lookup for dataset
        ]
        monkeypatch.setattr(
            hf_sync, "execute_query_dict",
            lambda *a, **k: queue.pop(0) if queue else [],
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(hf_sync, "execute_query", writer)
        manifest = {
            "rules": {}, "ces": {},
            "rule_datasets": {"rule_p": "2026-03-03T00:00:00Z"},
        }
        out = hf_sync.recover_pending_publishes(manifest=manifest, token="tok")
        assert out["healed_datasets"] == 1
        sql, params = writer.calls[0]
        assert "UPDATE test_datasets" in sql
        assert params == ("rule_p_positive", "2026-03-03T00:00:00Z", 9)

    def test_pending_dataset_cleared_when_rule_absent(self, monkeypatch):
        queue = [
            [],
            [],
            [{"dataset_id": 9, "rule_id": 3, "pending_public_id": "rule_p_positive"}],
            [],  # pending_rule_sets
            [{"public_id": "rule_p"}],
        ]
        monkeypatch.setattr(
            hf_sync, "execute_query_dict",
            lambda *a, **k: queue.pop(0) if queue else [],
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(hf_sync, "execute_query", writer)
        manifest = {"rules": {}, "ces": {}, "rule_datasets": {}}  # rule not present
        out = hf_sync.recover_pending_publishes(manifest=manifest, token="tok")
        assert out["cleared_datasets"] == 1
        sql, params = writer.calls[0]
        assert "pending_public_id = NULL" in sql
        assert params == (9,)

    def test_missing_manifest_sections_default_empty(self, monkeypatch):
        # Manifest with no 'rules'/'ces' keys at all -> .get(...) or {} -> a
        # pending row is cleared, not crashed.
        queue = [
            [{"rule_id": 1, "pending_public_id": "rule_x"}],
            [],
            [],
        ]
        monkeypatch.setattr(
            hf_sync, "execute_query_dict",
            lambda *a, **k: queue.pop(0) if queue else [],
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(hf_sync, "execute_query", writer)
        out = hf_sync.recover_pending_publishes(manifest={}, token="tok")
        assert out["cleared_rules"] == 1

    def test_pending_rule_set_in_registry_heals(self, monkeypatch):
        # 4th pending lookup is rule sets. When the manifest carries its
        # pending_public_id, heal forward (finalize the local rule_sets row).
        queue = [
            [],  # pending_rules
            [],  # pending_ces
            [],  # pending_datasets
            [{"rule_set_id": 7, "pending_public_id": "ruleset_p"}],  # pending_rule_sets
        ]
        monkeypatch.setattr(
            hf_sync, "execute_query_dict",
            lambda *a, **k: queue.pop(0) if queue else [],
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(hf_sync, "execute_query", writer)
        manifest = {"rules": {}, "ces": {}, "rule_sets": {"ruleset_p": "2026-04-04T00:00:00Z"}}
        out = hf_sync.recover_pending_publishes(manifest=manifest, token="tok")
        assert out["healed_rule_sets"] == 1
        assert out["cleared_rule_sets"] == 0
        sql, params = writer.calls[0]
        assert "UPDATE rule_sets" in sql
        assert params == ("ruleset_p", "2026-04-04T00:00:00Z", 7)

    def test_pending_rule_set_absent_is_deleted(self, monkeypatch):
        # A transient rule_sets row whose push didn't land is DELETED (not just
        # cleared) so its UNIQUE(name) slot is freed for a retry.
        queue = [
            [],
            [],
            [],
            [{"rule_set_id": 8, "pending_public_id": "ruleset_missing"}],
        ]
        monkeypatch.setattr(
            hf_sync, "execute_query_dict",
            lambda *a, **k: queue.pop(0) if queue else [],
        )
        writer = _RecordingWriter()
        monkeypatch.setattr(hf_sync, "execute_query", writer)
        manifest = {"rules": {}, "ces": {}, "rule_sets": {}}  # not present
        out = hf_sync.recover_pending_publishes(manifest=manifest, token="tok")
        assert out["cleared_rule_sets"] == 1
        assert out["healed_rule_sets"] == 0
        sql, params = writer.calls[0]
        assert "DELETE FROM rule_sets" in sql
        assert params == (8,)


class TestPublishRuleSetGate:
    """publish_rule_set's pre-flight guards — no network, all seams mocked."""

    def test_empty_set_errors(self, monkeypatch):
        monkeypatch.setattr(hf_publish, "_sync_is_fresh", lambda *a, **k: True)
        monkeypatch.setattr(hf_publish, "_resolve_username", lambda uid: "sht")
        monkeypatch.setattr(hf_publish, "_load_classifier_row",
                            lambda cid: {"classifier_id": cid, "name": "Empty"})
        monkeypatch.setattr(hf_publish, "_rule_set_members", lambda cid: [])
        r = hf_publish.publish_rule_set(5, publisher_user_id=1)
        assert r.status == hf_publish.PublishStatus.ERROR
        assert "no rules" in (r.error or "").lower()

    def test_refuses_unpublished_members(self, monkeypatch):
        # One published member + one manual (rule_id NULL / no public_id) ->
        # the members-published-first gate refuses and names the offender.
        monkeypatch.setattr(hf_publish, "_sync_is_fresh", lambda *a, **k: True)
        monkeypatch.setattr(hf_publish, "_resolve_username", lambda uid: "sht")
        monkeypatch.setattr(hf_publish, "_load_classifier_row",
                            lambda cid: {"classifier_id": cid, "name": "MySet"})
        monkeypatch.setattr(hf_publish, "_rule_set_members", lambda cid: [
            {"setup_id": 1, "rule_id": 10, "display_name": "pub_rule",
             "public_id": "rule_x", "is_local_draft": False, "categories": [1]},
            {"setup_id": 2, "rule_id": None, "display_name": "manual_rule",
             "public_id": None, "is_local_draft": None, "categories": None},
        ])
        r = hf_publish.publish_rule_set(1, publisher_user_id=1)
        assert r.status == hf_publish.PublishStatus.ERROR
        assert "published first" in (r.error or "")
        assert "manual_rule" in (r.error or "")
