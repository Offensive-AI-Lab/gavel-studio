# backend/utils/DButils.py
"""Schema bootstrap for the SQLite storage engine (utils/sqlite_db.py).

The whole schema is one consolidated DDL block: every table is created
directly in its final shape. The version-sentinel mechanism (`_app_meta` +
SCHEMA_VERSION) exists so future schema changes can ship as incremental,
idempotent migration steps below the CREATEs.

The storage layer speaks a Postgres-flavoured dialect mapped onto SQLite
(see utils/sqlite_db.py for the value/dialect layer):
  SERIAL              -> INTEGER PRIMARY KEY AUTOINCREMENT (ids never reused)
  JSONB / arrays      -> JSONB-declared TEXT columns holding JSON (converters
                         hand dicts/lists back, matching psycopg2)
  TIMESTAMPTZ         -> TIMESTAMPTZ-declared TEXT (UTC ISO, converter -> datetime)
  CITEXT              -> TEXT COLLATE NOCASE
  vector(384) + HNSW  -> BLOB of float32 (brute-force cosine in
                         services/library_search.py — trivial at library scale)
  tsvector + GIN      -> FTS5 tables (rules_fts / ces_fts) kept in sync by
                         triggers; bm25() with a high name weight replaces
                         setweight A/B + ts_rank_cd
  pg_trgm             -> Python trigram scoring in services/library_search.py
"""
from typing import Iterable, List, Set

from utils.sqlite_db import execute_query

# Bump this constant whenever init_database adds / removes / alters a
# table, column, index, or trigger. Mismatch with the value stored in
# _app_meta triggers a full re-run of the DDL block; a match
# short-circuits init_database.
#
# WARNING for future bumps: the DDL block below is CREATE ... IF NOT EXISTS
# only — it NEVER changes an existing table. Any column/default/constraint
# change needs an explicit migration step (rebuild-and-copy) keyed on the
# stored version, or existing databases will be stamped with the new version
# while silently keeping the old shape.
SCHEMA_VERSION = 1

# Baseline taxonomy kept general and shared across rules/CEs.
# NOT seeded locally — the registry (categories.json) is the source of truth;
# this list exists only for the maintainer bootstrap script that publishes the
# seed taxonomy to the registry.
DEFAULT_CATEGORIES = [
    ("Security & Defense", "Mechanisms designed to detect and prevent malicious attacks, jailbreaks, unauthorized access, system manipulation, and adversarial inputs that threaten the integrity of the AI."),
    ("Privacy & Data Protection", "Measures that identify, redact, or protect sensitive personal information, financial data, medical records, and secrets to ensure confidentiality and compliance with data laws."),
    ("Safety & Harm Prevention", "Filters aimed at preventing the generation of harmful, illegal, dangerous, or toxic content, including hate speech, violence, self-harm, and sexual material."),
    ("Fairness & Ethics", "Standards ensuring the AI remains unbiased, politically neutral, respectful, and free from discrimination against any group or individual."),
    ("Output Quality & Operational", "Controls that ensure the AI's output is accurate, coherent, on-topic, grammatically correct, and follows the user's formatting instructions."),
    ("Domain & Business Logic", "Rules specific to professional or industry contexts, handling specialized topics like legal contracts, medical advice, financial consulting, or coding standards."),
    ("Legal & Compliance", "Strict adherence to organizational policies, copyright laws, terms of service, and regulatory constraints that dictate what the AI is legally allowed to say."),
    ("Utility & Tools", "Operational capabilities that detect specific data structures, extract useful entities, or trigger external tools and workflows."),
    ("Resource & Cost Management", "Constraints related to token usage, latency limits, rate limiting, context window management, and cost control to ensure efficient operation."),
    ("Tone & Style", "Enforces specific communication styles, personas (e.g., helpful, empathetic), or restricts manipulative behaviors like sycophancy.")
]


def exec_query(query, params=None):
    """Historical alias used throughout this module and a few callers."""
    return execute_query(query, params)


# Tables in children-first order. Used by drop_all_tables (FK enforcement is
# ON, so parents must go last) and mirrored by the test-suite cleanup.
_ALL_TABLES_CHILDREN_FIRST = [
    # Bookmarks (FK to users + rules/CEs/rule_sets)
    "ce_bookmarks", "rule_bookmarks", "rule_set_bookmarks",
    # Wizard state
    "pipeline_runs",
    # Background jobs
    "bundle_jobs",
    # Classifier-attached data
    "evaluation_results", "test_datasets",
    "setup_ce_link", "rule_ce_link",
    "rule_setup", "classifiers",
    "guardrail_folders",
    "calibration_datasets",
    "excitation_datasets",
    # Public rule sets (membership FK to rule_sets + rules — drop first)
    "rule_set_member", "rule_sets",
    # Search side-tables (no FKs, but tied to rules/CEs by triggers)
    "rules_fts", "ces_fts",
    # Top-level definitions
    "rules", "cognitive_elements",
    # User-owned resources
    "target_models", "users",
    # Shared taxonomies + state
    "categories",
    "sync_state",
    "neutral_corpus",
    # Schema sentinel — MUST be dropped or fast-skip will think
    # the DB is up-to-date next boot.
    "_app_meta",
]


def drop_all_tables():
    """Drops all GAVEL tables to ensure a clean state.

    Order matters: children with FKs first (SQLite has no DROP ... CASCADE).
    `_app_meta` is the schema-version sentinel — if it's left in place after a
    drop, the next init_database() will fast-skip and leave the DB empty.
    Always drop it together with the rest. Triggers drop with their tables."""
    print("--- Dropping All Tables ---")
    for table in _ALL_TABLES_CHILDREN_FIRST:
        exec_query(f"DROP TABLE IF EXISTS {table};")
    print("[OK]All tables dropped.")


def _schema_version_table_exists() -> bool:
    """One-shot probe: does the `_app_meta` sentinel table itself exist?
    On a fresh DB the answer is no — short-circuit to "needs full init"."""
    try:
        rows = exec_query(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_app_meta'"
        )
        return bool(rows)
    except Exception:
        return False


def _stored_schema_version() -> int:
    """Read the schema version recorded by the last successful init_database.
    Returns 0 if the row is missing, malformed, or the table doesn't exist
    (callers treat 0 as "needs init")."""
    try:
        rows = exec_query("SELECT value FROM _app_meta WHERE key = 'schema_version'")
        if not rows:
            return 0
        return int(rows[0][0])
    except Exception:
        return 0


# Tables we sanity-check before trusting the schema-version sentinel.
# If _app_meta says v=current BUT any of these don't exist (partial wipe,
# corrupted state), we treat the sentinel as stale and run a full init.
_CRITICAL_TABLES = (
    "rules",
    "cognitive_elements",
    "classifiers",
    "rule_sets",
    "sync_state",
    "categories",
    "neutral_corpus",
    "bundle_jobs",
    "rule_bookmarks",
    "ce_bookmarks",
    "rule_set_bookmarks",
    # The FTS5 side-tables are load-bearing for /library/search.
    "rules_fts",
    "ces_fts",
)


# Indexes that future table rebuilds can drop — the fast path must not be
# taken while any of them is missing, or they never come back (only the full
# DDL block's CREATE INDEX IF NOT EXISTS recreates them).
_CRITICAL_INDEXES = (
    "idx_rule_ce_link_rule",
    "idx_rule_ce_link_ce",
    "idx_rule_set_member_rule",
    "uq_rules_public_id",
    "uq_ces_public_id",
    "uq_default_per_rule_type",
)


def _critical_tables_present() -> bool:
    """Belt-and-braces check that the schema-version sentinel matches
    physical reality — tables AND rebuild-droppable indexes."""
    try:
        rows = exec_query(
            "SELECT name FROM sqlite_master WHERE name = ANY(%s)",
            (list(_CRITICAL_TABLES) + list(_CRITICAL_INDEXES),),
        ) or []
        found = {r[0] for r in rows}
        missing = (set(_CRITICAL_TABLES) | set(_CRITICAL_INDEXES)) - found
        if missing:
            print(
                f"[init_database] sentinel says schema is up-to-date but "
                f"these critical tables/indexes are missing: {sorted(missing)}. "
                f"Falling through to full init."
            )
            return False
        return True
    except Exception:
        return False


def ensure_local_user():
    """Guarantee the single local user row exists.

    There is no register/login flow, so nothing else ever creates a `users`
    row — yet six tables FK to it with ON DELETE CASCADE. Called on EVERY boot
    so a database whose user row was manually deleted self-heals on restart.
    Idempotent: ON CONFLICT DO NOTHING leaves an existing row — including any
    display_name/bio the user edited — untouched."""
    from utils.auth import (
        LOCAL_USER_ID, LOCAL_USERNAME, LOCAL_EMAIL, LOCAL_DISPLAY_NAME,
    )
    exec_query(
        """
        INSERT INTO users (user_id, username, email, display_name, is_team)
        VALUES (%s, %s, %s, %s, FALSE)
        ON CONFLICT (user_id) DO NOTHING
        """,
        (LOCAL_USER_ID, LOCAL_USERNAME, LOCAL_EMAIL, LOCAL_DISPLAY_NAME),
    )


def init_database():
    """Idempotent schema bootstrap. Runs the full DDL block only when
    `SCHEMA_VERSION` differs from the version recorded in `_app_meta`.

    Fast-path (warm restart, schema already at SCHEMA_VERSION): two SELECTs.
    Slow-path (fresh DB, post-wipe, or after a SCHEMA_VERSION bump): full DDL
    pass + a final UPSERT that records the new version.

    If you change anything in the DDL block below, bump SCHEMA_VERSION at the
    top of the file. Otherwise existing deployments will skip your change on
    warm restarts."""
    if _schema_version_table_exists():
        stored = _stored_schema_version()
        if stored == SCHEMA_VERSION and _critical_tables_present():
            # Still (cheaply) re-assert the local user on the fast path — it is
            # the one row the whole schema's ownership FKs depend on.
            ensure_local_user()
            print(f"[init_database] schema up-to-date (v{SCHEMA_VERSION}), skipping setup.")
            return
        if stored != 0 and stored != SCHEMA_VERSION:
            print(f"[init_database] schema v{stored} -> v{SCHEMA_VERSION}, running upgrade...")

    print("--- Initializing GAVEL Studio Database (SQLite) ---")

    # 1. USERS — the single local operator (seeded by ensure_local_user) plus
    # placeholder rows (user_id >= 1000) for registry authors so
    # attribution JOINs resolve.
    exec_query("""
        CREATE TABLE IF NOT EXISTS users (
            user_id       INTEGER PRIMARY KEY,
            username      TEXT NOT NULL UNIQUE,
            email         TEXT NOT NULL UNIQUE,
            display_name  TEXT,
            bio           TEXT,
            is_team       BOOLEAN NOT NULL DEFAULT FALSE,
            tutorial_seen BOOLEAN DEFAULT FALSE,
            created_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 2. TARGET MODELS (Private). hf_token: gated/private HF repos.
    # selected_layers: the [start, end) transformer-layer range whose
    # activations feed the guardrail RNN (JSON two-element array).
    exec_query("""
        CREATE TABLE IF NOT EXISTS target_models (
            model_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            name            TEXT NOT NULL,
            storage_path    TEXT NOT NULL,
            hf_token        TEXT,
            num_layers      INTEGER,
            selected_layers JSONB,
            created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 3. GLOBAL COGNITIVE ELEMENTS (Public library, synced from the
    # gavel-rules registry). embedding: float32 BLOB (MiniLM, 384 dims) —
    # consumed by services/library_search.py. Library-state columns:
    #   public_id      — permanent registry identity (partial UNIQUE below)
    #   is_local_draft — TRUE = the user's own unsynced draft
    #   is_ready       — FALSE while an AI pipeline is generating; boot
    #                    recovery deletes FALSE rows
    #   title/role/tags — v2 discovery metadata (role is the closed enum
    #                    directive_to_user/llm_task/llm_behavior/topic)
    exec_query("""
        CREATE TABLE IF NOT EXISTS cognitive_elements (
            ce_id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT NOT NULL UNIQUE,
            title               TEXT,
            definition          TEXT,
            role                TEXT,
            category            TEXT DEFAULT 'CONTEXT',
            categories          JSONB DEFAULT '[]',
            tags                JSONB DEFAULT '[]',
            note                TEXT,
            examples            JSONB DEFAULT '[]',
            embedding           BLOB,
            type                TEXT,
            public_id           TEXT,
            published_at        TIMESTAMPTZ,
            is_local_draft      BOOLEAN NOT NULL DEFAULT TRUE,
            is_ready            BOOLEAN NOT NULL DEFAULT TRUE,
            created_by_username TEXT COLLATE NOCASE,
            created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    exec_query(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_ces_public_id "
        "ON cognitive_elements (public_id) WHERE public_id IS NOT NULL"
    )
    exec_query("CREATE INDEX IF NOT EXISTS idx_ces_created_by ON cognitive_elements (created_by_username);")

    # 3A. CATEGORY TAXONOMY (Shared; populated by the first library sync).
    exec_query("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            description TEXT,
            active      BOOLEAN DEFAULT TRUE,
            created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 3B/3C. Per-CE datasets (one row per CE; dataset TEXT holds JSON).
    exec_query("""
        CREATE TABLE IF NOT EXISTS excitation_datasets (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ce_id      INTEGER UNIQUE REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE,
            dataset    TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    exec_query("""
        CREATE TABLE IF NOT EXISTS calibration_datasets (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ce_id      INTEGER UNIQUE REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE,
            dataset    TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 4. GLOBAL RULES (Public library, synced from the gavel-rules registry).
    # Same library-state column set as cognitive_elements. The rule's LOGIC is
    # (ce_groups, condition): ce_groups is a JSON object mapping group name ->
    # [CE names]; condition is the condition-grammar string over those group names
    # (utils/rule_condition.py). predicate is a DERIVED human-readable
    # rendering (render_predicate) kept for FTS/search/display — never the
    # source of truth.
    exec_query("""
        CREATE TABLE IF NOT EXISTS rules (
            rule_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT NOT NULL UNIQUE,
            title               TEXT,
            predicate           TEXT NOT NULL,
            description         TEXT,
            categories          JSONB DEFAULT '[]',
            tags                JSONB DEFAULT '[]',
            ce_groups           JSONB DEFAULT '{}',
            condition           TEXT,
            embedding           BLOB,
            type                TEXT,
            public_id           TEXT,
            published_at        TIMESTAMPTZ,
            is_local_draft      BOOLEAN NOT NULL DEFAULT TRUE,
            is_ready            BOOLEAN NOT NULL DEFAULT TRUE,
            created_by_username TEXT COLLATE NOCASE,
            created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    exec_query(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_rules_public_id "
        "ON rules (public_id) WHERE public_id IS NOT NULL"
    )
    exec_query("CREATE INDEX IF NOT EXISTS idx_rules_created_by ON rules (created_by_username);")

    # 4S. FULL-TEXT SEARCH side-tables (replace Postgres tsvector + GIN).
    # Plain (self-contained) FTS5 tables keyed by the source row's id as
    # rowid, kept in sync by AFTER INSERT/UPDATE/DELETE triggers. Query with
    # `MATCH` + bm25(fts, 10.0, 1.0) — the 10x name weight replaces
    # setweight('A'/'B') + ts_rank_cd (name hits outrank body hits).
    exec_query("CREATE VIRTUAL TABLE IF NOT EXISTS rules_fts USING fts5(name, body);")
    exec_query("CREATE VIRTUAL TABLE IF NOT EXISTS ces_fts USING fts5(name, body);")
    exec_query("""
        CREATE TRIGGER IF NOT EXISTS rules_fts_ai AFTER INSERT ON rules BEGIN
            INSERT INTO rules_fts(rowid, name, body)
            VALUES (new.rule_id, new.name,
                    COALESCE(new.predicate, '') || ' ' || COALESCE(new.description, ''));
        END;
    """)
    exec_query("""
        CREATE TRIGGER IF NOT EXISTS rules_fts_ad AFTER DELETE ON rules BEGIN
            DELETE FROM rules_fts WHERE rowid = old.rule_id;
        END;
    """)
    exec_query("""
        CREATE TRIGGER IF NOT EXISTS rules_fts_au AFTER UPDATE ON rules BEGIN
            DELETE FROM rules_fts WHERE rowid = old.rule_id;
            INSERT INTO rules_fts(rowid, name, body)
            VALUES (new.rule_id, new.name,
                    COALESCE(new.predicate, '') || ' ' || COALESCE(new.description, ''));
        END;
    """)
    exec_query("""
        CREATE TRIGGER IF NOT EXISTS ces_fts_ai AFTER INSERT ON cognitive_elements BEGIN
            INSERT INTO ces_fts(rowid, name, body)
            VALUES (new.ce_id, new.name, COALESCE(new.definition, ''));
        END;
    """)
    exec_query("""
        CREATE TRIGGER IF NOT EXISTS ces_fts_ad AFTER DELETE ON cognitive_elements BEGIN
            DELETE FROM ces_fts WHERE rowid = old.ce_id;
        END;
    """)
    exec_query("""
        CREATE TRIGGER IF NOT EXISTS ces_fts_au AFTER UPDATE ON cognitive_elements BEGIN
            DELETE FROM ces_fts WHERE rowid = old.ce_id;
            INSERT INTO ces_fts(rowid, name, body)
            VALUES (new.ce_id, new.name, COALESCE(new.definition, ''));
        END;
    """)
    # Repair pass: (re)index any base rows the FTS tables don't know about —
    # covers databases whose FTS tables were wiped or created after content.
    exec_query("""
        INSERT INTO rules_fts(rowid, name, body)
        SELECT rule_id, name, COALESCE(predicate, '') || ' ' || COALESCE(description, '')
        FROM rules WHERE rule_id NOT IN (SELECT rowid FROM rules_fts);
    """)
    exec_query("""
        INSERT INTO ces_fts(rowid, name, body)
        SELECT ce_id, name, COALESCE(definition, '')
        FROM cognitive_elements WHERE ce_id NOT IN (SELECT rowid FROM ces_fts);
    """)

    # 5. GUARDRAIL FOLDERS — manual grouping over classifiers. Deleting a
    # folder CASCADE-deletes the guardrails inside it (deliberate, v17).
    exec_query("""
        CREATE TABLE IF NOT EXISTS guardrail_folders (
            folder_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            name       TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS guardrail_folders_user_idx ON guardrail_folders (user_id);")

    # 6. CLASSIFIERS (Private; UI "guardrails"). user_id is the direct owner
    # (v15); model_id stays nullable so an unattached guardrail can hold a
    # rule set until train time. trained_* columns are the post-training
    # snapshot driving drift detection ('needs_retraining').
    exec_query("""
        CREATE TABLE IF NOT EXISTS classifiers (
            classifier_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id                    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            model_id                   INTEGER REFERENCES target_models(model_id) ON DELETE CASCADE,
            folder_id                  INTEGER REFERENCES guardrail_folders(folder_id) ON DELETE CASCADE,
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
    """)
    exec_query("CREATE INDEX IF NOT EXISTS classifiers_user_idx ON classifiers (user_id);")
    exec_query("CREATE INDEX IF NOT EXISTS classifiers_folder_idx ON classifiers (folder_id);")

    # 7. RULE SETUP (Private per-guardrail copy of a rule). Carries its own
    # (ce_groups, condition) copy so per-guardrail edits never mutate the
    # global rule; predicate is the derived rendering, as on rules.
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_setup (
            setup_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            classifier_id INTEGER REFERENCES classifiers(classifier_id) ON DELETE CASCADE,
            rule_id       INTEGER REFERENCES rules(rule_id) ON DELETE SET NULL,
            custom_name   TEXT,
            predicate     TEXT NOT NULL,
            ce_groups     JSONB DEFAULT '{}',
            condition     TEXT,
            is_active     BOOLEAN DEFAULT TRUE
        );
    """)

    # 8. JUNCTIONS: pure CE membership (the union of a rule's/setup's group
    # members) for FK cascades and role-agnostic queries — training labels,
    # counts, dependency lookups, lazy dataset walkers. The logic itself
    # lives in ce_groups/condition on the parent row. NOT NULL is explicit on
    # the key columns: SQLite's rowid tables (legacy quirk) accept NULLs in
    # PK columns — and NULLs also bypass PK uniqueness AND FK enforcement.
    exec_query("""
        CREATE TABLE IF NOT EXISTS setup_ce_link (
            setup_id   INTEGER NOT NULL REFERENCES rule_setup(setup_id) ON DELETE CASCADE,
            ce_id      INTEGER NOT NULL REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (setup_id, ce_id)
        );
    """)
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_ce_link (
            rule_id    INTEGER NOT NULL REFERENCES rules(rule_id) ON DELETE CASCADE,
            ce_id      INTEGER NOT NULL REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (rule_id, ce_id)
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_ce_link_rule ON rule_ce_link(rule_id);")
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_ce_link_ce ON rule_ce_link(ce_id);")

    # 9. BOOKMARKS (local; drafts are bookmarkable).
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_bookmarks (
            user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            rule_id    INTEGER NOT NULL REFERENCES rules(rule_id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, rule_id)
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS rule_bookmarks_user_idx ON rule_bookmarks (user_id, created_at DESC);")
    exec_query("""
        CREATE TABLE IF NOT EXISTS ce_bookmarks (
            user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            ce_id      INTEGER NOT NULL REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, ce_id)
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS ce_bookmarks_user_idx ON ce_bookmarks (user_id, created_at DESC);")

    # 10. SYNC STATE — key/value store for the gavel-rules registry sync
    # (last synced commit SHA, per-record content hashes).
    exec_query("""
        CREATE TABLE IF NOT EXISTS sync_state (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 10b. NEUTRAL CORPUS (global, registry-synced) — benign conversations the
    # classifier must NEVER fire on; the evaluation's false-positive split.
    exec_query("""
        CREATE TABLE IF NOT EXISTS neutral_corpus (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            category     TEXT NOT NULL,
            conversation JSONB NOT NULL,
            published_at TIMESTAMPTZ,
            created_at   TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 11. EVALUATION RESULTS (per classifier). eval_type also encodes in-flight
    # markers (calibration_running / evaluation_running) used by crash recovery.
    # created_at carries MILLISECOND precision (strftime %f) — CURRENT_TIMESTAMP
    # is whole-second, and a '*_error' row written in the same second as its
    # '*_running' marker must still sort strictly after it in the "latest
    # result" queries.
    exec_query("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            eval_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            classifier_id INTEGER REFERENCES classifiers(classifier_id) ON DELETE CASCADE,
            eval_type     TEXT NOT NULL,
            thresholds    JSONB,
            metrics       JSONB,
            plots         JSONB,
            created_at    TIMESTAMPTZ DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now'))
        );
    """)

    # 11b. BUNDLE JOBS (server-side export/import background jobs).
    exec_query("""
        CREATE TABLE IF NOT EXISTS bundle_jobs (
            job_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            job_type      TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'running',
            phase         TEXT,
            error         TEXT,
            classifier_id INTEGER,
            tier          TEXT,
            artifact_path TEXT,
            filename      TEXT,
            result        JSONB,
            created_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS bundle_jobs_user_idx ON bundle_jobs (user_id, created_at DESC);")
    exec_query("CREATE INDEX IF NOT EXISTS bundle_jobs_running_idx ON bundle_jobs (status) WHERE status = 'running';")

    # 12. TEST DATASETS — purely rule-scoped. Default rows (is_default=TRUE,
    # user_id NULL, 3 per published rule) or private custom sets (user_id set).
    exec_query("""
        CREATE TABLE IF NOT EXISTS test_datasets (
            dataset_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id           INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE,
            user_id           INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            dataset_type      TEXT NOT NULL,
            scenario_name     TEXT,
            config            JSONB,
            conversations     JSONB,
            status            TEXT DEFAULT 'pending',
            generation_log    TEXT,
            is_default        BOOLEAN NOT NULL DEFAULT FALSE,
            public_id         TEXT,
            published_at      TIMESTAMPTZ,
            created_at        TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # One default per (rule, dataset_type) so regeneration is an idempotent
    # UPSERT. Partial unique on public_id mirrors uq_rules_public_id.
    exec_query(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_default_per_rule_type "
        "ON test_datasets (rule_id, dataset_type) WHERE is_default = TRUE"
    )
    exec_query(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_test_datasets_public_id "
        "ON test_datasets (public_id) WHERE public_id IS NOT NULL"
    )
    exec_query(
        "CREATE INDEX IF NOT EXISTS idx_test_datasets_rule "
        "ON test_datasets (rule_id, is_default)"
    )

    # 13. PIPELINE RUNS — wizard-state persistence. `steps` is a per-step
    # state map keyed by step id; classifier_id nullable (Pipeline A is
    # guardrail-agnostic); rule FK SET NULL so the run survives a rollback of
    # its outputs.
    exec_query("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            classifier_id INTEGER REFERENCES classifiers(classifier_id) ON DELETE CASCADE,
            rule_id       INTEGER REFERENCES rules(rule_id) ON DELETE SET NULL,
            pipeline_type TEXT NOT NULL DEFAULT 'rule',
            current_step  TEXT NOT NULL DEFAULT '1',
            steps         JSONB NOT NULL DEFAULT '{}',
            completed     BOOLEAN NOT NULL DEFAULT FALSE,
            created_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at    TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS pipeline_runs_user_idx ON pipeline_runs (user_id);")
    exec_query(
        "CREATE INDEX IF NOT EXISTS pipeline_runs_active_idx "
        "ON pipeline_runs (user_id, classifier_id) WHERE completed = FALSE"
    )

    # 14. PUBLIC RULE SETS (v19) — third public-library peer of rules/CEs; a
    # thin pointer-collection over already-published global rules. The private
    # `classifiers` row stays the authoring source and is never itself
    # published.
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_sets (
            rule_set_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT NOT NULL UNIQUE,
            title               TEXT,
            description         TEXT,
            categories          JSONB DEFAULT '[]',
            public_id           TEXT,
            published_at        TIMESTAMPTZ,
            is_local_draft      BOOLEAN NOT NULL DEFAULT TRUE,
            is_ready            BOOLEAN NOT NULL DEFAULT TRUE,
            created_by_username TEXT COLLATE NOCASE,
            created_at          TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    exec_query(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_rule_sets_public_id "
        "ON rule_sets (public_id) WHERE public_id IS NOT NULL"
    )
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_sets_created_by ON rule_sets (created_by_username);")
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_set_member (
            rule_set_id INTEGER NOT NULL REFERENCES rule_sets(rule_set_id) ON DELETE CASCADE,
            rule_id     INTEGER NOT NULL REFERENCES rules(rule_id) ON DELETE CASCADE,
            position    INTEGER NOT NULL DEFAULT 0,
            created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (rule_set_id, rule_id)
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_set_member_rule ON rule_set_member (rule_id);")
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_set_bookmarks (
            user_id     INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            rule_set_id INTEGER NOT NULL REFERENCES rule_sets(rule_set_id) ON DELETE CASCADE,
            created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, rule_set_id)
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS rule_set_bookmarks_user_idx ON rule_set_bookmarks (user_id, created_at DESC);")

    # Schema version sentinel. Created last so a partial init (process killed
    # mid-DDL) leaves the version row absent, forcing the next boot to re-run
    # the full DDL block.
    exec_query("""
        CREATE TABLE IF NOT EXISTS _app_meta (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    exec_query(
        """
        INSERT INTO _app_meta (key, value) VALUES ('schema_version', %s)
        ON CONFLICT (key) DO UPDATE
        SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
        """,
        (str(SCHEMA_VERSION),),
    )
    # The single local user. Must exist before any model / rule set / folder
    # can be created, since all of those FK to users(user_id).
    ensure_local_user()
    print(f"[OK]Database Initialized: Public Library + Private Setup Overrides (schema v{SCHEMA_VERSION}).")


def normalize_and_upsert_categories(
    categories: Iterable,
    max_len: int = 3,
    allow_new: bool = False,
) -> List[int]:
    """
    Resolves category names (str) or IDs (int) to a list of Category IDs.
    Returns: List[int]
    """
    final_ids: Set[int] = set()

    # Pre-fetch all active categories map: name -> id, id -> id
    rows = exec_query("SELECT category_id, name FROM categories WHERE active = TRUE") or []
    name_map = {r[1].lower().strip(): r[0] for r in rows}
    id_map = {r[0]: r[0] for r in rows}

    unique_inputs = []
    seen = set()
    # Deduplicate input order-preserving logic roughly
    for item in categories or []:
        if item not in seen:
            unique_inputs.append(item)
            seen.add(item)

    for item in unique_inputs:
        # 1. Handle Integer ID
        if isinstance(item, int):
            if item in id_map:
                final_ids.add(item)
            continue

        # 2. Handle String ID
        if isinstance(item, str) and item.isdigit():
            i_val = int(item)
            if i_val in id_map:
                final_ids.add(i_val)
            continue

        # 3. Handle String Name
        s_item = str(item).strip()
        if not s_item:
            continue

        key = s_item.lower()
        if key in name_map:
            final_ids.add(name_map[key])
        elif allow_new:
            # Create new
            try:
                res = exec_query(
                    """
                    INSERT INTO categories (name, active)
                    VALUES (%s, TRUE)
                    ON CONFLICT (name) DO UPDATE SET active = TRUE
                    RETURNING category_id
                    """,
                    (s_item,)
                )
                if res:
                    new_id = res[0][0]
                    final_ids.add(new_id)
                    name_map[key] = new_id  # Update cache
            except Exception as e:
                print(f"Error creating category {s_item}: {e}")

    # Convert to list
    result_list = sorted(list(final_ids))
    if max_len and len(result_list) > max_len:
        result_list = result_list[:max_len]

    return result_list
