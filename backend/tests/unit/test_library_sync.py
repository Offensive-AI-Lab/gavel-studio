"""Tests for services.library_sync — the gavel-rules registry ingest.

NO network: the RegistryReader singleton (`services.library_sync._READER`) is
replaced with an in-memory FakeReader serving an index.json + dataset files
modeled on the REAL gavel-rules repository shapes (a trimmed copy of the
upstream index entries — see gavel-rules/index.json).

The DB is real SQLite, but a THROWAWAY one: the `temp_db` fixture points
utils.sqlite_db.DB_PATH at a tmp file and runs init_database() there, so
these tests never touch the dev database.

Covered:
  * cold sync counts + persisted rows (CE role->category mirror, rule
    ce_groups/condition + derived predicate + membership links, ruleset
    members by name, categories, sync_state sha/hashes)
  * HEAD-SHA short-circuit (no index fetch when nothing changed) + force
  * content_hash-driven refresh, incl. invalidation of cached excitation/
    calibration datasets (CE) and default test rows (rule)
  * draft-guard skip (skipped_records + state NOT persisted)
  * foreign-namespace refs and invalid conditions -> errors, not imports
  * the lazy ensure_* fetchers' envelope mapping
"""
import json

import pytest

import services.library_sync as ls
from utils.sqlite_db import execute_query, execute_query_dict


# ---------------------------------------------------------------------------
# Throwaway DB
# ---------------------------------------------------------------------------


def _drain_boot_sync_thread():
    """If main.py was imported by an earlier test, its daemon boot thread may
    still be waiting to run a startup library sync. The root conftest's dead
    reader makes that sync a fast no-op, but it must FINISH before we repoint
    DB_PATH / _READER, or it would race this test's fixtures."""
    import threading
    for t in threading.enumerate():
        if t.name == "library-sync-bootstrap":
            t.join(timeout=60)


@pytest.fixture()
def temp_db(monkeypatch, tmp_path):
    """Point the sqlite layer at a fresh temp file and build the schema."""
    import utils.sqlite_db as sq
    from utils.DButils import init_database

    _drain_boot_sync_thread()
    sq.close_pool()
    monkeypatch.setattr(sq, "DB_PATH", tmp_path / "library_sync_test.sqlite3")
    monkeypatch.setattr(sq, "_db_dir_ready", False)
    init_database()
    yield
    sq.close_pool()
    sq._db_dir_ready = False


# ---------------------------------------------------------------------------
# FakeReader + index fixture (shapes copied from the real gavel-rules clone)
# ---------------------------------------------------------------------------


class FakeReader:
    name = "fake"

    def __init__(self, index, files=None, head="sha-one"):
        self.index = index
        self.files = dict(files or {})
        self.head = head
        self.head_calls = 0
        self.fetched = []          # every path handed to fetch_json

    def head_version(self):
        self.head_calls += 1
        return self.head

    def fetch_json(self, path, *, revision=None):
        self.fetched.append((path, revision))
        if path == "index.json":
            return json.loads(json.dumps(self.index))  # deep copy
        if path in self.files:
            return json.loads(json.dumps(self.files[path]))
        from services.registry_sync.reader import RegistryNotFound
        raise RegistryNotFound(path)


def _ce_entry(name, public_id, role="topic", content_hash="h", definition=None):
    """One index.json CE entry — field set mirrors the real registry."""
    return {
        "title": name.replace("_", " ").title(),
        "role": role,
        "public_id": public_id,
        "tags": [],
        "definition": definition or f"{name}: a test CE.",
        "examples": [{"input": "sample", "output": "YES"}],
        "published_at": "2026-05-07T09:27:59Z",
        "by": "gavel",
        "path": f"ces/{name}",
        "calibration": f"ces/{name}/calibration.json",
        "excitation": f"ces/{name}/excitation.json",
        "calibration_samples": 2,
        "excitation_samples": 3,
        "used_by": [],
        "content_hash": content_hash,
    }


def _rule_entry(name, public_id, groups, condition, content_hash="h",
                categories=None):
    members = sorted({m for ms in groups.values() for m in ms})
    return {
        "title": name.replace("_", " ").title(),
        "public_id": public_id,
        "categories": categories if categories is not None else ["Fairness & Ethics"],
        "tags": [],
        "definition": f"Detects {name}.",
        "groups": groups,
        "condition": condition,
        "predicate": "(ignored — re-derived locally)",
        "ces": members,
        "published_at": "2026-06-11T09:59:26Z",
        "by": "gavel",
        "path": f"rules/{name}",
        "tests": {
            t: f"rules/{name}/tests/{t}.json"
            for t in ("positive", "negative", "positive_calibration")
        },
        "content_hash": content_hash,
    }


def _index():
    """A minimal 2-CE / 1-rule / 1-ruleset index modeled on the real repo."""
    return {
        "schema_version": 2,
        "generated_by": "tools/build_index.py",
        "namespace": "gavel",
        "global_signature": "v2:feedface",
        "counts": {"ces": 2, "rules": 1, "rulesets": 1},
        "categories": [
            {"name": "Fairness & Ethics", "description": "Standards ensuring fairness."},
        ],
        "ces": {
            "content_creation": _ce_entry(
                "content_creation", "ce_cc", role="llm_task", content_hash="cc-1"),
            "electoral_politics": _ce_entry(
                "electoral_politics", "ce_ep", role="topic", content_hash="ep-1"),
        },
        "rules": {
            "electoral_political_content_creation": _rule_entry(
                "electoral_political_content_creation", "rule_epcc",
                {"electoral_content_creation": ["content_creation", "electoral_politics"]},
                "all of electoral_content_creation",
                content_hash="rule-1"),
        },
        "rulesets": {
            "default": {
                "title": "Default",
                "public_id": "ruleset_def",
                "description": "Every rule in the library.",
                "rules": ["electoral_political_content_creation"],
                "categories": ["Fairness & Ethics"],
                "published_at": "2026-06-23T14:39:31Z",
                "by": "seansh",
                "path": "rulesets/default.yaml",
                "content_hash": "rs-1",
            },
        },
    }


@pytest.fixture()
def reader(monkeypatch, temp_db):
    """Install a FakeReader serving the standard index; return it for
    per-test mutation."""
    fake = FakeReader(_index())
    monkeypatch.setattr(ls, "_READER", fake)
    return fake


def _rule_row(name):
    rows = execute_query_dict("SELECT * FROM rules WHERE name = %s", (name,)) or []
    return rows[0] if rows else None


def _ce_row(name):
    rows = execute_query_dict(
        "SELECT * FROM cognitive_elements WHERE name = %s", (name,)) or []
    return rows[0] if rows else None


# ===========================================================================
# Cold sync
# ===========================================================================


class TestColdSync:
    def test_counts_and_changed_flag(self, reader):
        result = ls.sync_library()
        assert result.changed is True
        assert result.ces_added == 2
        assert result.rules_added == 1
        assert result.rule_sets_added == 1
        assert result.categories_synced == 1
        assert result.ces_refreshed == result.rules_refreshed == 0
        assert result.skipped_records == []
        assert result.errors == []

    def test_result_dict_has_no_neutral_key(self, reader):
        # neutral_synced is not a sync output (the corpus has no upstream source).
        d = ls.sync_library().to_dict()
        assert "neutral_synced" not in d
        assert set(d) == {
            "changed", "ces_added", "rules_added", "rule_sets_added",
            "ces_refreshed", "rules_refreshed", "rule_sets_refreshed",
            "categories_synced", "skipped_records", "errors",
        }

    def test_ce_row_mirrors_role_into_category_facets(self, reader):
        ls.sync_library()
        ce = _ce_row("content_creation")
        assert ce["public_id"] == "ce_cc"
        assert ce["role"] == "llm_task"
        # Browse facets: category = role, categories = [role].
        assert ce["category"] == "llm_task"
        assert ce["categories"] == ["llm_task"]
        assert bool(ce["is_local_draft"]) is False

    def test_rule_row_stores_groups_condition_and_derived_predicate(self, reader):
        ls.sync_library()
        rule = _rule_row("electoral_political_content_creation")
        assert rule["public_id"] == "rule_epcc"
        assert rule["ce_groups"] == {
            "electoral_content_creation": ["content_creation", "electoral_politics"],
        }
        assert rule["condition"] == "all of electoral_content_creation"
        # predicate is re-rendered locally, never trusted from the registry.
        assert rule["predicate"] == "(content_creation AND electoral_politics)"

    def test_rule_membership_links_written(self, reader):
        ls.sync_library()
        rule = _rule_row("electoral_political_content_creation")
        links = execute_query_dict(
            """
            SELECT ce.name FROM rule_ce_link l
            JOIN cognitive_elements ce ON ce.ce_id = l.ce_id
            WHERE l.rule_id = %s ORDER BY ce.name
            """,
            (rule["rule_id"],),
        )
        assert [r["name"] for r in links] == ["content_creation", "electoral_politics"]

    def test_ruleset_members_resolved_by_name(self, reader):
        ls.sync_library()
        rows = execute_query_dict(
            """
            SELECT r.name FROM rule_sets rs
            JOIN rule_set_member m ON m.rule_set_id = rs.rule_set_id
            JOIN rules r ON r.rule_id = m.rule_id
            WHERE rs.name = 'default'
            """,
        )
        assert [r["name"] for r in rows] == ["electoral_political_content_creation"]

    def test_sha_and_hashes_persisted_on_clean_pass(self, reader):
        ls.sync_library()
        assert ls._get_state("last_synced_sha") == "sha-one"
        hashes = json.loads(ls._get_state("record_hashes"))
        assert hashes["ces"]["content_creation"] == "cc-1"
        assert hashes["rules"]["electoral_political_content_creation"] == "rule-1"
        assert hashes["rulesets"]["default"] == "rs-1"

    def test_index_fetch_pinned_to_probed_sha(self, reader):
        ls.sync_library()
        assert ("index.json", "sha-one") in reader.fetched


# ===========================================================================
# Freshness: SHA short-circuit, force, check_for_updates
# ===========================================================================


class TestFreshness:
    def test_sha_short_circuit_skips_index_fetch(self, reader):
        ls.sync_library()
        fetched_before = len(reader.fetched)
        result = ls.sync_library()
        assert result.changed is False
        assert result.ces_added == 0 and result.rules_added == 0
        assert len(reader.fetched) == fetched_before  # only the HEAD probe ran

    def test_force_resyncs_but_unchanged_hashes_pull_nothing(self, reader):
        ls.sync_library()
        result = ls.sync_library(force=True)
        assert result.changed is True
        # Same content_hash everywhere -> nothing re-pulled.
        assert result.ces_added == result.ces_refreshed == 0
        assert result.rules_added == result.rules_refreshed == 0

    def test_check_for_updates_flags_new_head(self, reader):
        ls.sync_library()
        assert ls.check_for_updates() == {
            "available": False, "checked": True, "reason": None}
        reader.head = "sha-two"
        out = ls.check_for_updates()
        assert out["available"] is True and out["checked"] is True

    def test_check_for_updates_probe_failure_is_unchecked(self, reader, monkeypatch):
        def boom():
            raise RuntimeError("api down")
        monkeypatch.setattr(reader, "head_version", boom)
        out = ls.check_for_updates()
        assert out["checked"] is False
        assert out["available"] is False  # outage must not pin the badge


# ===========================================================================
# content_hash refresh
# ===========================================================================


class TestRefresh:
    def test_changed_ce_hash_refreshes_and_clears_cached_datasets(self, reader):
        ls.sync_library()
        ce = _ce_row("content_creation")
        # Seed cached lazy datasets THE SYNC WROTE (marked with `source`) —
        # a content change must invalidate those so they re-pull. Datasets the
        # user generated locally carry no marker and are preserved instead
        # (see TestRefreshPreservesLocalDatasets).
        execute_query(
            "INSERT INTO excitation_datasets (ce_id, dataset) VALUES (%s, %s)",
            (ce["ce_id"], json.dumps({"samples": ["old"], "source": ls._DATASET_SOURCE})))
        execute_query(
            "INSERT INTO calibration_datasets (ce_id, dataset) VALUES (%s, %s)",
            (ce["ce_id"], json.dumps({"conversations": ["old"], "source": ls._DATASET_SOURCE})))

        reader.head = "sha-two"
        reader.index["ces"]["content_creation"]["content_hash"] = "cc-2"
        reader.index["ces"]["content_creation"]["definition"] = "updated definition"
        result = ls.sync_library()

        assert result.ces_refreshed == 1
        assert result.ces_added == 0
        assert _ce_row("content_creation")["definition"] == "updated definition"
        assert execute_query_dict(
            "SELECT 1 FROM excitation_datasets WHERE ce_id = %s", (ce["ce_id"],)) == []
        assert execute_query_dict(
            "SELECT 1 FROM calibration_datasets WHERE ce_id = %s", (ce["ce_id"],)) == []

    def test_changed_rule_hash_refreshes_and_clears_default_test_rows(self, reader):
        ls.sync_library()
        rule = _rule_row("electoral_political_content_creation")
        execute_query(
            """
            INSERT INTO test_datasets (rule_id, user_id, is_default, dataset_type,
                                       conversations, status)
            VALUES (%s, NULL, TRUE, 'positive', '[]', 'ready')
            """,
            (rule["rule_id"],))
        # A user's own custom set must survive the refresh.
        execute_query(
            """
            INSERT INTO test_datasets (rule_id, user_id, is_default, dataset_type,
                                       conversations, status)
            VALUES (%s, 1, FALSE, 'positive', '[]', 'ready')
            """,
            (rule["rule_id"],))

        reader.head = "sha-two"
        entry = reader.index["rules"]["electoral_political_content_creation"]
        entry["content_hash"] = "rule-2"
        entry["condition"] = "1 of electoral_content_creation"
        result = ls.sync_library()

        assert result.rules_refreshed == 1
        refreshed = _rule_row("electoral_political_content_creation")
        assert refreshed["condition"] == "1 of electoral_content_creation"
        assert refreshed["predicate"] == "(content_creation OR electoral_politics)"
        rows = execute_query_dict(
            "SELECT is_default FROM test_datasets WHERE rule_id = %s",
            (rule["rule_id"],))
        assert [bool(r["is_default"]) for r in rows] == [False]

    def test_unchanged_records_not_repulled(self, reader):
        ls.sync_library()
        reader.head = "sha-two"
        reader.index["ces"]["content_creation"]["content_hash"] = "cc-2"
        result = ls.sync_library()
        assert result.ces_refreshed == 1
        assert result.rules_refreshed == 0
        assert result.rule_sets_refreshed == 0


# ===========================================================================
# Draft guard — local drafts are sacred
# ===========================================================================


class TestDraftGuard:
    def test_same_name_local_draft_blocks_pull_and_state_persist(self, reader):
        execute_query(
            "INSERT INTO rules (name, predicate, is_local_draft) VALUES (%s, %s, TRUE)",
            ("electoral_political_content_creation", "MY DRAFT"))
        result = ls.sync_library()

        assert "rules/electoral_political_content_creation" in result.skipped_records
        # The draft row is untouched.
        row = _rule_row("electoral_political_content_creation")
        assert row["predicate"] == "MY DRAFT"
        assert bool(row["is_local_draft"]) is True
        # A skipped record must NOT persist the sha — the next sync retries.
        assert ls._get_state("last_synced_sha") is None

    def test_ce_draft_guard(self, reader):
        execute_query(
            "INSERT INTO cognitive_elements (name, definition, is_local_draft) "
            "VALUES (%s, %s, TRUE)",
            ("content_creation", "my draft ce"))
        result = ls.sync_library()
        assert "ces/content_creation" in result.skipped_records
        assert _ce_row("content_creation")["definition"] == "my draft ce"


# ===========================================================================
# Validation failures — bad records error instead of importing
# ===========================================================================


class TestValidation:
    def test_foreign_namespace_member_errors(self, reader):
        entry = reader.index["rules"]["electoral_political_content_creation"]
        entry["groups"] = {"g": ["other_ns/some_ce"]}
        entry["condition"] = "all of g"
        result = ls.sync_library()
        assert any("foreign namespace" in e for e in result.errors)
        assert _rule_row("electoral_political_content_creation") is None
        assert ls._get_state("last_synced_sha") is None  # errored pass: retry

    def test_invalid_condition_errors(self, reader):
        entry = reader.index["rules"]["electoral_political_content_creation"]
        entry["condition"] = "all of undefined_group"
        result = ls.sync_library()
        assert any("invalid logic" in e for e in result.errors)
        assert _rule_row("electoral_political_content_creation") is None

    def test_unknown_member_ce_errors(self, reader):
        entry = reader.index["rules"]["electoral_political_content_creation"]
        entry["groups"] = {"creation": ["content_creation", "ghost_ce"]}
        entry["condition"] = "all of creation"
        result = ls.sync_library()
        assert any("unknown CE" in e for e in result.errors)

    def test_uppercase_ce_names_import_unlowered(self, reader):
        # Real registry precedent: the 'LGBTQ' CE. Names must round-trip
        # verbatim — no lowercasing anywhere in the import path.
        reader.index["ces"]["LGBTQ"] = _ce_entry("LGBTQ", "ce_lgbtq", content_hash="lg-1")
        result = ls.sync_library()
        assert result.errors == []
        assert _ce_row("LGBTQ") is not None


# ===========================================================================
# Lazy dataset fetchers
# ===========================================================================


@pytest.fixture()
def synced(reader):
    ls.sync_library()
    return reader


class TestEnsureExcitation:
    def test_fetches_and_stores_samples_envelope(self, synced):
        ce = _ce_row("content_creation")
        synced.files["ces/content_creation/excitation.json"] = {
            "schema_version": 2, "ce_public_id": "ce_cc", "ce": "content_creation",
            "type": "excitation",
            "samples": [[{"role": "user", "content": "hi"}]], "sample_count": 1,
        }
        assert ls.ensure_excitation(ce["ce_id"]) is True
        row = execute_query_dict(
            "SELECT dataset FROM excitation_datasets WHERE ce_id = %s",
            (ce["ce_id"],))[0]
        envelope = json.loads(row["dataset"])
        assert envelope == {
            "samples": [[{"role": "user", "content": "hi"}]], "sample_count": 1,
            "source": ls._DATASET_SOURCE}

    def test_fast_path_skips_fetch_when_row_exists(self, synced):
        ce = _ce_row("content_creation")
        execute_query(
            "INSERT INTO excitation_datasets (ce_id, dataset) VALUES (%s, %s)",
            (ce["ce_id"], json.dumps({"samples": []})))
        before = len(synced.fetched)
        assert ls.ensure_excitation(ce["ce_id"]) is True
        assert len(synced.fetched) == before

    def test_local_only_ce_returns_false_without_fetch(self, synced):
        execute_query(
            "INSERT INTO cognitive_elements (name, is_local_draft) VALUES (%s, TRUE)",
            ("my_local_ce",))
        ce = _ce_row("my_local_ce")
        before = len(synced.fetched)
        assert ls.ensure_excitation(ce["ce_id"]) is False
        assert len(synced.fetched) == before

    def test_fetch_failure_returns_false(self, synced):
        ce = _ce_row("content_creation")   # no file staged in the FakeReader
        assert ls.ensure_excitation(ce["ce_id"]) is False


class TestEnsureCeCalibration:
    def test_remaps_samples_to_conversations_envelope(self, synced):
        ce = _ce_row("electoral_politics")
        synced.files["ces/electoral_politics/calibration.json"] = {
            "schema_version": 2, "ce_public_id": "ce_ep",
            "samples": [[{"role": "user", "content": "vote"}]], "sample_count": 1,
        }
        assert ls.ensure_ce_calibration(ce["ce_id"]) is True
        row = execute_query_dict(
            "SELECT dataset FROM calibration_datasets WHERE ce_id = %s",
            (ce["ce_id"],))[0]
        envelope = json.loads(row["dataset"])
        # The calibration readers expect the legacy {conversations,...} shape.
        assert envelope == {
            "conversations": [[{"role": "user", "content": "vote"}]],
            "sample_count": 1, "source": ls._DATASET_SOURCE}


class TestEnsureRuleDefaults:
    def _stage_tests(self, reader, name="electoral_political_content_creation"):
        for dtype in ("positive", "negative", "positive_calibration"):
            reader.files[f"rules/{name}/tests/{dtype}.json"] = {
                "schema_version": 2, "rule_public_id": "rule_epcc",
                "type": dtype,
                "config": {"scenario_instructions": f"{dtype} scenario"},
                "conversations": [[{"role": "user", "content": dtype}]],
                "conversation_count": 1,
                "published_at": "2026-06-11T09:59:26Z",
            }

    def test_pulls_all_three_buckets_as_ready_default_rows(self, synced):
        self._stage_tests(synced)
        rule = _rule_row("electoral_political_content_creation")
        assert ls.ensure_rule_defaults(rule["rule_id"]) is True
        rows = execute_query_dict(
            "SELECT dataset_type, status, is_default, public_id, scenario_name "
            "FROM test_datasets WHERE rule_id = %s ORDER BY dataset_type",
            (rule["rule_id"],))
        assert [r["dataset_type"] for r in rows] == [
            "negative", "positive", "positive_calibration"]
        for r in rows:
            assert r["status"] == "ready"
            assert bool(r["is_default"]) is True
            assert r["public_id"] == f"rule_epcc_{r['dataset_type']}"

    def test_fast_path_when_all_ready(self, synced):
        self._stage_tests(synced)
        rule = _rule_row("electoral_political_content_creation")
        ls.ensure_rule_defaults(rule["rule_id"])
        before = len(synced.fetched)
        assert ls.ensure_rule_defaults(rule["rule_id"]) is True
        assert len(synced.fetched) == before

    def test_local_only_rule_returns_false(self, synced):
        execute_query(
            "INSERT INTO rules (name, predicate, is_local_draft) VALUES (%s, %s, TRUE)",
            ("local_rule", "X"))
        rule = _rule_row("local_rule")
        assert ls.ensure_rule_defaults(rule["rule_id"]) is False

    def test_partial_failure_returns_false_but_keeps_what_landed(self, synced):
        # Only the positive file exists upstream.
        synced.files["rules/electoral_political_content_creation/tests/positive.json"] = {
            "schema_version": 2, "rule_public_id": "rule_epcc", "type": "positive",
            "config": {}, "conversations": [], "conversation_count": 0,
        }
        rule = _rule_row("electoral_political_content_creation")
        assert ls.ensure_rule_defaults(rule["rule_id"]) is False
        rows = execute_query_dict(
            "SELECT dataset_type FROM test_datasets WHERE rule_id = %s",
            (rule["rule_id"],))
        assert [r["dataset_type"] for r in rows] == ["positive"]

    def test_ensure_rule_calibration_delegates_to_defaults(self, synced):
        self._stage_tests(synced)
        rule = _rule_row("electoral_political_content_creation")
        assert ls.ensure_rule_calibration(rule["rule_id"]) is True
        rows = execute_query_dict(
            """
            SELECT 1 FROM test_datasets
            WHERE rule_id = %s AND dataset_type = 'positive_calibration'
              AND is_default = TRUE AND status = 'ready'
            """,
            (rule["rule_id"],))
        assert rows


# ===========================================================================
# Upstream renames (same public_id, new name) — the `go` ->
# physical_movement_solicitation case from the real registry
# ===========================================================================


class TestUpstreamRenames:
    def test_ce_rename_updates_row_in_place_and_patches_groups(self, reader):
        ls.sync_library()
        old = _ce_row("electoral_politics")
        assert old is not None

        # Upstream renames the CE, and the rule's groups now use the new name.
        idx = reader.index
        entry = idx["ces"].pop("electoral_politics")
        entry["content_hash"] = "ep-2"
        idx["ces"]["voting_politics"] = entry
        rule = idx["rules"]["electoral_political_content_creation"]
        rule["groups"] = {
            "electoral_content_creation": ["content_creation", "voting_politics"]
        }
        rule["ces"] = sorted(["content_creation", "voting_politics"])
        rule["content_hash"] = "rule-2"
        reader.head = "sha-two"

        result = ls.sync_library()
        assert not result.errors, result.errors

        renamed = _ce_row("voting_politics")
        assert renamed is not None
        assert renamed["ce_id"] == old["ce_id"], "rename must keep the ce_id"
        assert renamed["public_id"] == "ce_ep"
        assert _ce_row("electoral_politics") is None, "old name must be gone"

        # The refreshed rule's groups reference the new name.
        row = _rule_row("electoral_political_content_creation")
        groups = row["ce_groups"]
        assert "voting_politics" in groups["electoral_content_creation"]

    def test_ce_rename_patches_local_rule_groups(self, reader):
        ls.sync_library()
        # A LOCAL rule (not in the registry) references the soon-renamed CE.
        from gavel_pipeline.db_access import upsert_rule_with_links
        upsert_rule_with_links({
            "rule_name": "my_local_rule",
            "description": "local",
            "ce_groups": {"topic_watch": ["electoral_politics"]},
            "condition": "all of topic_watch",
        })

        idx = reader.index
        entry = idx["ces"].pop("electoral_politics")
        entry["content_hash"] = "ep-3"
        idx["ces"]["voting_politics"] = entry
        rule = idx["rules"]["electoral_political_content_creation"]
        rule["groups"] = {
            "electoral_content_creation": ["content_creation", "voting_politics"]
        }
        rule["content_hash"] = "rule-3"
        reader.head = "sha-three"

        result = ls.sync_library()
        assert not result.errors, result.errors

        local = _rule_row("my_local_rule")
        assert local["ce_groups"]["topic_watch"] == ["voting_politics"]
        assert "voting_politics" in local["predicate"]

    def test_ruleset_rename_keeps_public_id_row(self, reader):
        ls.sync_library()
        rows = execute_query_dict(
            "SELECT rule_set_id FROM rule_sets WHERE public_id = 'ruleset_def'")
        old_id = rows[0]["rule_set_id"]

        idx = reader.index
        entry = idx["rulesets"].pop("default")
        entry["content_hash"] = "rs-2"
        idx["rulesets"]["starter"] = entry
        reader.head = "sha-four"

        result = ls.sync_library()
        assert not result.errors, result.errors
        rows = execute_query_dict(
            "SELECT rule_set_id, name FROM rule_sets WHERE public_id = 'ruleset_def'")
        assert rows[0]["name"] == "starter"
        assert rows[0]["rule_set_id"] == old_id


# ===========================================================================
# Legacy records: a local public_id does not mean the record exists in
# gavel-rules (it may have been curated out or renamed). Those must never be
# fetched (every attempt is a guaranteed 404 on every boot).
# ===========================================================================


class TestLegacyRecordsAreNotFetched:
    @staticmethod
    def _seed_legacy_ce(name="self_harm", public_id="ce_legacy_orphan"):
        """A legacy-shaped row: published (public_id set, not a draft) but
        absent from the registry."""
        execute_query(
            "INSERT INTO cognitive_elements (name, definition, public_id, "
            "is_local_draft) VALUES (%s, 'legacy', %s, FALSE)",
            (name, public_id),
        )
        return execute_query_dict(
            "SELECT ce_id FROM cognitive_elements WHERE name = %s", (name,)
        )[0]["ce_id"]

    @staticmethod
    def _stage_calibration(reader, name, public_id):
        reader.files[f"ces/{name}/calibration.json"] = {
            "schema_version": 2, "ce_public_id": public_id,
            "samples": [[{"role": "user", "content": "hello"}]], "sample_count": 1,
        }

    def test_lazy_ce_fetchers_skip_legacy_without_touching_the_registry(self, reader):
        ls.sync_library()
        ce_id = self._seed_legacy_ce()
        before = len(reader.fetched)

        assert ls.ensure_ce_calibration(ce_id) is False
        assert ls.ensure_excitation(ce_id) is False
        assert reader.fetched[before:] == [], "legacy CE must not hit the registry"

    def test_lazy_rule_defaults_skip_legacy(self, reader):
        ls.sync_library()
        execute_query(
            "INSERT INTO rules (name, predicate, public_id, is_local_draft) "
            "VALUES ('legacy_rule', 'x', 'rule_legacy_hf', FALSE)"
        )
        rule_id = execute_query_dict(
            "SELECT rule_id FROM rules WHERE name = 'legacy_rule'")[0]["rule_id"]
        before = len(reader.fetched)

        assert ls.ensure_rule_defaults(rule_id) is False
        assert reader.fetched[before:] == [], "legacy rule must not hit the registry"

    def test_warm_pass_skips_legacy_but_still_warms_registry_ces(self, reader):
        ls.sync_library()
        legacy_id = self._seed_legacy_ce()
        # Drop a registry CE's calibration so the warm pass has real work.
        registry_id = _ce_row("content_creation")["ce_id"]
        execute_query("DELETE FROM calibration_datasets WHERE ce_id = %s", (registry_id,))
        self._stage_calibration(reader, "content_creation", "ce_cc")

        ls.pull_all_aux_datasets()

        assert execute_query_dict(
            "SELECT 1 FROM calibration_datasets WHERE ce_id = %s", (registry_id,)
        ), "registry CE should have been warmed"
        assert not execute_query_dict(
            "SELECT 1 FROM calibration_datasets WHERE ce_id = %s", (legacy_id,)
        ), "legacy CE must be left alone"
        assert not any(
            "self_harm" in path for path, _ in reader.fetched
        ), "warm pass must not request legacy paths"

    def test_registry_records_still_fetch_normally(self, reader):
        ls.sync_library()
        ce_id = _ce_row("electoral_politics")["ce_id"]
        execute_query("DELETE FROM calibration_datasets WHERE ce_id = %s", (ce_id,))
        self._stage_calibration(reader, "electoral_politics", "ce_ep")
        assert ls.ensure_ce_calibration(ce_id) is True

    def test_no_completed_sync_means_nothing_is_treated_as_legacy(self, temp_db, monkeypatch):
        """Before the first successful sync the registry contents are unknown,
        so the fetchers must still TRY (a fresh DB restored from a backup
        must be able to lazily pull)."""
        fake = FakeReader(_index())
        monkeypatch.setattr(ls, "_READER", fake)
        ce_id = self._seed_legacy_ce(name="content_creation", public_id="ce_cc")
        self._stage_calibration(fake, "content_creation", "ce_cc")
        # sync_state has no record_hashes yet -> _registry_names returns None
        assert ls._registry_names("ces") is None
        assert ls.ensure_ce_calibration(ce_id) is True
        assert any("content_creation" in path for path, _ in fake.fetched)


# ===========================================================================
# Refresh must never destroy LOCAL work. Sync is additive: it replaces its own
# cached copies, but a dataset the user generated locally for a library CE is
# theirs and survives an upstream refresh.
# ===========================================================================


class TestRefreshPreservesLocalDatasets:
    @staticmethod
    def _bump_ce_upstream(reader, name="content_creation", h="cc-changed"):
        reader.index["ces"][name]["content_hash"] = h
        reader.head = "sha-refresh"

    def test_sync_written_dataset_is_replaced_on_refresh(self, reader):
        ls.sync_library()
        ce_id = _ce_row("content_creation")["ce_id"]
        reader.files["ces/content_creation/excitation.json"] = {
            "schema_version": 2, "ce_public_id": "ce_cc",
            "samples": [[{"role": "user", "content": "x"}]], "sample_count": 1,
        }
        assert ls.ensure_excitation(ce_id) is True          # writes a marked row
        stored = json.loads(execute_query_dict(
            "SELECT dataset FROM excitation_datasets WHERE ce_id = %s", (ce_id,))[0]["dataset"])
        assert stored["source"] == ls._DATASET_SOURCE

        self._bump_ce_upstream(reader)
        assert not ls.sync_library().errors
        # Our own cached copy is dropped so the next ensure_* re-pulls it.
        assert not execute_query_dict(
            "SELECT 1 FROM excitation_datasets WHERE ce_id = %s", (ce_id,))

    def test_locally_generated_dataset_survives_refresh(self, reader):
        ls.sync_library()
        ce_id = _ce_row("content_creation")["ce_id"]
        # What routes/ai_pipeline.py writes: no `source` marker.
        local = {"samples": [[{"role": "user", "content": "hand-made"}]]}
        execute_query(
            "INSERT INTO excitation_datasets (ce_id, dataset) VALUES (%s, %s)",
            (ce_id, json.dumps(local)))

        self._bump_ce_upstream(reader)
        assert not ls.sync_library().errors

        rows = execute_query_dict(
            "SELECT dataset FROM excitation_datasets WHERE ce_id = %s", (ce_id,))
        assert rows, "locally generated excitation must NOT be deleted by a refresh"
        assert json.loads(rows[0]["dataset"]) == local

    def test_local_calibration_also_survives(self, reader):
        ls.sync_library()
        ce_id = _ce_row("electoral_politics")["ce_id"]
        execute_query("DELETE FROM calibration_datasets WHERE ce_id = %s", (ce_id,))
        local = {"conversations": [[{"role": "user", "content": "mine"}]]}
        execute_query(
            "INSERT INTO calibration_datasets (ce_id, dataset) VALUES (%s, %s)",
            (ce_id, json.dumps(local)))

        self._bump_ce_upstream(reader, "electoral_politics", "ep-changed")
        assert not ls.sync_library().errors
        rows = execute_query_dict(
            "SELECT dataset FROM calibration_datasets WHERE ce_id = %s", (ce_id,))
        assert rows and json.loads(rows[0]["dataset"]) == local

    def test_sync_never_deletes_library_records(self, reader):
        """The whole-library invariant: a sync only adds/updates rows."""
        ls.sync_library()
        before_ces = {r["name"] for r in execute_query_dict("SELECT name FROM cognitive_elements")}
        before_rules = {r["name"] for r in execute_query_dict("SELECT name FROM rules")}

        # Upstream drops an artifact entirely.
        reader.index["ces"].pop("electoral_politics")
        reader.index["rules"].pop("electoral_political_content_creation")
        reader.head = "sha-shrunk"
        assert not ls.sync_library().errors

        after_ces = {r["name"] for r in execute_query_dict("SELECT name FROM cognitive_elements")}
        after_rules = {r["name"] for r in execute_query_dict("SELECT name FROM rules")}
        assert after_ces == before_ces, "sync must not prune CEs that vanished upstream"
        assert after_rules == before_rules, "sync must not prune rules that vanished upstream"
