"""Migration test: a v23-shaped database (role/fallback_group junctions,
role-era fingerprints) upgraded to the v24 groups+condition model by
utils.DButils.init_database.

The v23 database is built BY HAND with raw sqlite3 (the exact pre-migration
DDL shapes), then utils.sqlite_db.DB_PATH is pointed at it and
init_database() runs the real migration. Asserted:

  * role links convert per the documented mapping (necessary -> 'required'
    all-of; fallback group i -> 'option_i' 1-of, joined with 'and';
    sufficient -> 'supporting', present in ce_groups but NOT in the
    condition) — on rules AND rule_setup;
  * the predicate column is re-rendered from the converted logic;
  * the junction tables become pure membership (no role column, deduped
    (parent, ce_id) PK) with no rows lost;
  * pending_public_id is gone; ce_groups/condition/title/tags exist;
  * a DRIFT-FREE trained guardrail keeps its status: its
    trained_policy_fingerprint is rewritten to the v2 value so
    reconcile_classifier_status still reports 'active';
  * a guardrail that was ALREADY drifted under the old fingerprint stays
    drifted (its stored fingerprint is left alone);
  * re-running init_database is a no-op (idempotent, stamps v24).
"""
import hashlib
import sqlite3

import pytest


# ---------------------------------------------------------------------------
# The retired v1 fingerprint math, inlined (the production code that computed
# it is deleted; the migration snapshots drift state with its own inline copy
# and this test must agree with the OLD stored values byte-for-byte).
# ---------------------------------------------------------------------------


def _old_rule_fp(links):
    necessary, sufficient = [], []
    fallback = {}
    for link in links:
        role = (link.get("role") or "necessary").lower()
        cid = link["ce_id"]
        if role == "sufficient":
            sufficient.append(cid)
        elif role == "fallback":
            fallback.setdefault(int(link.get("fallback_group") or 0), []).append(cid)
        else:
            necessary.append(cid)
    fb_norm = sorted(tuple(sorted(g)) for g in fallback.values())
    return f"N:{tuple(sorted(necessary))}|F:{fb_norm}|S:{tuple(sorted(sufficient))}"


def _old_policy_fp(per_setup_links):
    fps = sorted(_old_rule_fp(links) for links in per_setup_links)
    canonical = ";".join(fps)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest() if canonical else ""


# ---------------------------------------------------------------------------
# Hand-built v23 database
# ---------------------------------------------------------------------------

# The 5-CE legacy rule: necessary + two fallback groups + sufficient.
_LEGACY_LINKS = [
    (1, "hook_ce", "necessary", 0),
    (2, "opt_a", "fallback", 1),
    (3, "opt_b", "fallback", 1),
    (4, "opt_c", "fallback", 2),
    (5, "helper", "sufficient", 0),
]

_EXPECTED_GROUPS = {
    "required": ["hook_ce"],
    "option_1": ["opt_a", "opt_b"],
    "option_2": ["opt_c"],
    "supporting": ["helper"],
}
_EXPECTED_CONDITION = "all of required and 1 of option_1 and 1 of option_2"
_EXPECTED_PREDICATE = "hook_ce AND (opt_a OR opt_b) AND opt_c"


def _build_v23_db(path):
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE users (
            user_id       INTEGER PRIMARY KEY,
            username      TEXT NOT NULL UNIQUE,
            password      TEXT NOT NULL DEFAULT '',
            email         TEXT NOT NULL UNIQUE,
            display_name  TEXT,
            bio           TEXT,
            is_team       BOOLEAN NOT NULL DEFAULT FALSE,
            tutorial_seen BOOLEAN DEFAULT FALSE,
            created_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE cognitive_elements (
            ce_id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT NOT NULL UNIQUE,
            definition          TEXT,
            category            TEXT DEFAULT 'CONTEXT',
            categories          JSONB DEFAULT '[]',
            note                TEXT,
            examples            JSONB DEFAULT '[]',
            embedding           BLOB,
            type                TEXT,
            public_id           TEXT,
            pending_public_id   TEXT,
            published_at        TIMESTAMPTZ,
            is_local_draft      BOOLEAN NOT NULL DEFAULT TRUE,
            is_ready            BOOLEAN NOT NULL DEFAULT TRUE,
            created_by_username TEXT COLLATE NOCASE,
            created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE rules (
            rule_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT NOT NULL UNIQUE,
            predicate           TEXT NOT NULL,
            description         TEXT,
            categories          JSONB DEFAULT '[]',
            embedding           BLOB,
            type                TEXT,
            public_id           TEXT,
            pending_public_id   TEXT,
            published_at        TIMESTAMPTZ,
            is_local_draft      BOOLEAN NOT NULL DEFAULT TRUE,
            is_ready            BOOLEAN NOT NULL DEFAULT TRUE,
            created_by_username TEXT COLLATE NOCASE,
            created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE rule_ce_link (
            rule_id        INTEGER NOT NULL REFERENCES rules(rule_id) ON DELETE CASCADE,
            ce_id          INTEGER NOT NULL REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE,
            role           TEXT NOT NULL DEFAULT 'necessary'
                           CHECK (role IN ('necessary','fallback','sufficient')),
            fallback_group INTEGER NOT NULL DEFAULT 0,
            created_at     TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (rule_id, ce_id, role, fallback_group)
        );
        CREATE TABLE classifiers (
            classifier_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id                    INTEGER NOT NULL,
            model_id                   INTEGER,
            folder_id                  INTEGER,
            name                       TEXT NOT NULL,
            status                     TEXT DEFAULT 'untrained',
            model_path                 TEXT,
            training_log               TEXT,
            training_config            JSONB DEFAULT '{}',
            training_phase             TEXT,
            training_phase_detail      TEXT,
            trained_rule_setup_ids     JSONB,
            trained_rule_names         JSONB,
            trained_at                 TIMESTAMPTZ,
            trained_policy_fingerprint TEXT,
            created_at                 TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE rule_setup (
            setup_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            classifier_id INTEGER REFERENCES classifiers(classifier_id) ON DELETE CASCADE,
            rule_id       INTEGER REFERENCES rules(rule_id) ON DELETE SET NULL,
            custom_name   TEXT,
            predicate     TEXT NOT NULL,
            is_active     BOOLEAN DEFAULT TRUE
        );
        CREATE TABLE setup_ce_link (
            setup_id       INTEGER NOT NULL REFERENCES rule_setup(setup_id) ON DELETE CASCADE,
            ce_id          INTEGER NOT NULL REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE,
            role           TEXT NOT NULL DEFAULT 'necessary'
                           CHECK (role IN ('necessary','fallback','sufficient')),
            fallback_group INTEGER NOT NULL DEFAULT 0,
            created_at     TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (setup_id, ce_id, role, fallback_group)
        );
        CREATE TABLE rule_sets (
            rule_set_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT NOT NULL UNIQUE,
            description         TEXT,
            categories          JSONB DEFAULT '[]',
            public_id           TEXT,
            pending_public_id   TEXT,
            published_at        TIMESTAMPTZ,
            is_local_draft      BOOLEAN NOT NULL DEFAULT TRUE,
            is_ready            BOOLEAN NOT NULL DEFAULT TRUE,
            created_by_username TEXT COLLATE NOCASE,
            created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE test_datasets (
            dataset_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id           INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE,
            user_id           INTEGER,
            dataset_type      TEXT NOT NULL,
            scenario_name     TEXT,
            config            JSONB,
            conversations     JSONB,
            status            TEXT DEFAULT 'pending',
            generation_log    TEXT,
            is_default        BOOLEAN NOT NULL DEFAULT FALSE,
            public_id         TEXT,
            pending_public_id TEXT,
            published_at      TIMESTAMPTZ,
            created_at        TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE _app_meta (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("INSERT INTO _app_meta (key, value) VALUES ('schema_version', '23')")
    cur.execute(
        "INSERT INTO users (user_id, username, email) VALUES (1, 'local', 'l@x')")

    for ce_id, name, _, _ in _LEGACY_LINKS:
        cur.execute(
            "INSERT INTO cognitive_elements (ce_id, name, definition, is_local_draft) "
            "VALUES (?, ?, ?, 0)", (ce_id, name, f"def {name}"))

    # Rule 1: the full legacy shape (necessary + 2 fallback groups + sufficient).
    cur.execute(
        "INSERT INTO rules (rule_id, name, predicate, is_local_draft) "
        "VALUES (1, 'legacy_rule', 'OLD PREDICATE', 0)")
    for ce_id, _, role, fb in _LEGACY_LINKS:
        cur.execute(
            "INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group) "
            "VALUES (1, ?, ?, ?)", (ce_id, role, fb))

    # Rule 2: necessary-only (the single-group edge case).
    cur.execute(
        "INSERT INTO rules (rule_id, name, predicate, is_local_draft) "
        "VALUES (2, 'necessary_only', 'hook_ce', 0)")
    cur.execute(
        "INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group) "
        "VALUES (2, 1, 'necessary', 0)")

    # Guardrail 1: trained on setup 1 (mirror of rule 1's links) and DRIFT-FREE
    # under the OLD fingerprint (stored == recomputed at migration time).
    setup1_links = [
        {"ce_id": cid, "role": role, "fallback_group": fb}
        for cid, _, role, fb in _LEGACY_LINKS
    ]
    old_fp = _old_policy_fp([setup1_links])
    cur.execute(
        "INSERT INTO classifiers (classifier_id, user_id, name, status, "
        "trained_policy_fingerprint) VALUES (1, 1, 'in-sync guard', 'active', ?)",
        (old_fp,))
    cur.execute(
        "INSERT INTO rule_setup (setup_id, classifier_id, rule_id, custom_name, "
        "predicate, is_active) VALUES (1, 1, 1, 'legacy_rule', 'OLD PREDICATE', 1)")
    for cid, _, role, fb in _LEGACY_LINKS:
        cur.execute(
            "INSERT INTO setup_ce_link (setup_id, ce_id, role, fallback_group) "
            "VALUES (1, ?, ?, ?)", (cid, role, fb))

    # Guardrail 2: ALREADY drifted — stored fingerprint doesn't match its links.
    cur.execute(
        "INSERT INTO classifiers (classifier_id, user_id, name, status, "
        "trained_policy_fingerprint) VALUES (2, 1, 'drifted guard', "
        "'needs_retraining', 'stale-old-fp')")
    cur.execute(
        "INSERT INTO rule_setup (setup_id, classifier_id, rule_id, custom_name, "
        "predicate, is_active) VALUES (2, 2, 2, 'necessary_only', 'hook_ce', 1)")
    cur.execute(
        "INSERT INTO setup_ce_link (setup_id, ce_id, role, fallback_group) "
        "VALUES (2, 1, 'necessary', 0)")

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixture: point the sqlite layer at the handmade DB and run the migration
# ---------------------------------------------------------------------------


@pytest.fixture()
def migrated_db(monkeypatch, tmp_path):
    import threading

    import utils.sqlite_db as sq
    from utils.DButils import init_database

    db_file = tmp_path / "v23.sqlite3"
    _build_v23_db(db_file)

    # If main.py was imported by an earlier test, let its (dead-reader,
    # no-op) boot-sync daemon finish before repointing DB_PATH.
    for t in threading.enumerate():
        if t.name == "library-sync-bootstrap":
            t.join(timeout=60)

    sq.close_pool()
    monkeypatch.setattr(sq, "DB_PATH", db_file)
    monkeypatch.setattr(sq, "_db_dir_ready", False)
    init_database()          # runs _migrate_v23_to_v24 + full DDL + v24 stamp
    yield
    sq.close_pool()
    sq._db_dir_ready = False


def _rows(sql, params=None):
    from utils.sqlite_db import execute_query_dict
    return execute_query_dict(sql, params) or []


# ===========================================================================
# Assertions
# ===========================================================================


class TestRoleConversion:
    def test_rule_groups_and_condition(self, migrated_db):
        rule = _rows("SELECT * FROM rules WHERE name = 'legacy_rule'")[0]
        assert rule["ce_groups"] == _EXPECTED_GROUPS
        assert rule["condition"] == _EXPECTED_CONDITION

    def test_rule_predicate_rerendered(self, migrated_db):
        rule = _rows("SELECT * FROM rules WHERE name = 'legacy_rule'")[0]
        assert rule["predicate"] == _EXPECTED_PREDICATE

    def test_supporting_group_not_in_condition(self, migrated_db):
        rule = _rows("SELECT * FROM rules WHERE name = 'legacy_rule'")[0]
        assert "supporting" in rule["ce_groups"]
        assert "supporting" not in rule["condition"]
        assert "helper" not in rule["predicate"]

    def test_necessary_only_rule_single_group_edge(self, migrated_db):
        rule = _rows("SELECT * FROM rules WHERE name = 'necessary_only'")[0]
        assert rule["ce_groups"] == {"required": ["hook_ce"]}
        assert rule["condition"] == "all of required"
        assert rule["predicate"] == "hook_ce"

    def test_setup_converted_like_its_rule(self, migrated_db):
        setup = _rows("SELECT * FROM rule_setup WHERE setup_id = 1")[0]
        assert setup["ce_groups"] == _EXPECTED_GROUPS
        assert setup["condition"] == _EXPECTED_CONDITION
        assert setup["predicate"] == _EXPECTED_PREDICATE


class TestMembershipJunctions:
    def test_rule_ce_link_is_membership_only(self, migrated_db):
        from utils.sqlite_db import execute_query_dict
        ddl = _rows(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='rule_ce_link'"
        )[0]["sql"]
        assert "role" not in ddl
        assert "fallback_group" not in ddl
        assert "PRIMARY KEY (rule_id, ce_id)" in ddl

    def test_setup_ce_link_is_membership_only(self, migrated_db):
        ddl = _rows(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='setup_ce_link'"
        )[0]["sql"]
        assert "role" not in ddl
        assert "PRIMARY KEY (setup_id, ce_id)" in ddl

    def test_membership_rows_preserved_and_deduped(self, migrated_db):
        links = _rows(
            "SELECT ce_id FROM rule_ce_link WHERE rule_id = 1 ORDER BY ce_id")
        assert [r["ce_id"] for r in links] == [1, 2, 3, 4, 5]
        setup_links = _rows(
            "SELECT ce_id FROM setup_ce_link WHERE setup_id = 1 ORDER BY ce_id")
        assert [r["ce_id"] for r in setup_links] == [1, 2, 3, 4, 5]


class TestSchemaShape:
    def test_pending_public_id_dropped_everywhere(self, migrated_db):
        for table in ("rules", "cognitive_elements", "rule_sets", "test_datasets"):
            ddl = _rows(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name = %s",
                (table,))[0]["sql"]
            assert "pending_public_id" not in ddl, table

    def test_new_metadata_columns_exist(self, migrated_db):
        rules_ddl = _rows(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='rules'"
        )[0]["sql"]
        for col in ("title", "tags", "ce_groups", "condition"):
            assert col in rules_ddl
        ces_ddl = _rows(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='cognitive_elements'"
        )[0]["sql"]
        for col in ("title", "role", "tags"):
            assert col in ces_ddl

    def test_stamped_v24(self, migrated_db):
        from utils.DButils import SCHEMA_VERSION, _stored_schema_version
        assert _stored_schema_version() == SCHEMA_VERSION == 24


class TestFingerprintRewrite:
    def test_drift_free_guardrail_keeps_active_status(self, migrated_db):
        """The heart of migration step E: a guardrail whose policy was
        drift-free under the OLD fingerprint must NOT flip to
        needs_retraining just because the fingerprint algorithm changed."""
        from sql_scripts.model_scripts import (
            compute_classifier_policy_fingerprint_v2,
            reconcile_classifier_status,
        )
        row = _rows(
            "SELECT trained_policy_fingerprint, status FROM classifiers "
            "WHERE classifier_id = 1")[0]
        # Rewritten to exactly the live v2 fingerprint...
        assert row["trained_policy_fingerprint"] == \
            compute_classifier_policy_fingerprint_v2(1)
        assert row["trained_policy_fingerprint"] != ""
        # ...so reconcile still reports no drift.
        assert reconcile_classifier_status(1) == "active"

    def test_already_drifted_guardrail_stays_drifted(self, migrated_db):
        from sql_scripts.model_scripts import reconcile_classifier_status
        row = _rows(
            "SELECT trained_policy_fingerprint FROM classifiers "
            "WHERE classifier_id = 2")[0]
        # Not snapshotted as in-sync -> the stale fingerprint is left alone...
        assert row["trained_policy_fingerprint"] == "stale-old-fp"
        # ...and the guardrail still reads as needing retraining.
        assert reconcile_classifier_status(2) == "needs_retraining"


class TestIdempotency:
    def test_rerun_is_a_noop(self, migrated_db):
        from utils.DButils import init_database, _needs_v24_rebuild

        before_rule = _rows("SELECT * FROM rules WHERE name = 'legacy_rule'")[0]
        before_fp = _rows(
            "SELECT trained_policy_fingerprint FROM classifiers "
            "WHERE classifier_id = 1")[0]

        assert _needs_v24_rebuild() is False
        init_database()   # second run: fast path, nothing rebuilt

        after_rule = _rows("SELECT * FROM rules WHERE name = 'legacy_rule'")[0]
        after_fp = _rows(
            "SELECT trained_policy_fingerprint FROM classifiers "
            "WHERE classifier_id = 1")[0]
        assert after_rule["ce_groups"] == before_rule["ce_groups"]
        assert after_rule["condition"] == before_rule["condition"]
        assert after_rule["predicate"] == before_rule["predicate"]
        assert after_fp == before_fp
        links = _rows("SELECT ce_id FROM rule_ce_link WHERE rule_id = 1")
        assert len(links) == 5
