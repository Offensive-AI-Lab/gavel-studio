# backend/utils/DButils.py
from typing import Iterable, List, Set

from utils.PostgreSQL import get_connection, release_connection

# Bump this constant whenever init_database adds / removes / alters a
# table, column, index, trigger, or function. Mismatch with the value
# stored in _app_meta triggers a full re-run of the DDL block; a match
# short-circuits init_database in <50ms instead of hitting Postgres
# for every CREATE TABLE IF NOT EXISTS.
#
# Version log (informal — the actual source of truth is git history):
#   1  initial schema (users, rules, ces, classifiers, ...)
#   2  bookmarks tables + sync_state
#   3  registry-link columns + crash-recovery flags
#   4  rule_calibration_datasets + categories
#   5  Phase 1 of artist feature: CITEXT username, ratings, summaries
#   6  scenarios table + scenario_id FK on rules and test_datasets
#   7  pipeline_runs table (wizard state persistence for the 8-step
#      rule-creation flow)
#   8  pipeline_runs split into two flavors — Pipeline A (rule generation,
#      classifier-agnostic) and Pipeline B (test + evaluation, per-rule).
#      classifier_id is now nullable; new pipeline_type column.
#   9  Rule-level default test/calibration datasets. test_datasets gains
#      rule_id / user_id / is_default + HF registry columns (public_id,
#      published_at, pending_public_id); classifier_id becomes nullable.
#      The `scenarios` table and all scenario_id FKs are removed — the
#      scenario now lives inside test_datasets.config (scenario_instructions)
#      and, while editing, in pipeline_runs.steps.
#  10  test_datasets fully decoupled from classifiers: classifier_id column
#      dropped. A test set is now purely rule-scoped — a public default
#      (is_default=TRUE) or a user's private custom set (user_id set). Test
#      dialogues are classifier-agnostic, so the classifier link added
#      nothing.
#  11  target_models gains an optional hf_token (some models are gated /
#      private and need a Hugging Face token to download).
#  12  classifiers gain trained_policy_fingerprint — a content hash of the
#      policy a model was trained on, so 'needs_retraining' is computed from
#      real drift (current policy != trained policy) instead of a sticky flag.
#  13  neutral_corpus table (shared benign-conversation corpus for neutral-set
#      evaluation, keyed by content_hash + category). It was added to the DDL
#      block below WITHOUT a version bump, so DBs already at v12 took the
#      fast-path skip and never created it ("relation neutral_corpus does not
#      exist"). This bump forces the idempotent CREATE TABLE to run on next boot.
#   v14: added `bundle_jobs` (server-side export/import background jobs) — the
#      fast-path skip would otherwise never create it on existing DBs.
#   v15: classifiers gain a direct `user_id` owner column. Guardrails (the UI
#      name for classifiers) can now exist BEFORE a model is chosen, so the
#      old "owner derived through model_id" chain no longer holds. user_id is
#      backfilled from target_models for existing rows; model_id stays nullable
#      so an unattached guardrail can hold a rule set until train time.
#   v16: guardrail folders — an optional "library" grouping over guardrails.
#      New `guardrail_folders` table (per-user, named); `folder_id` on
#      classifiers. (v16 originally shipped an auto-arrange experiment —
#      `policy_sig` + a `guardrail_auto_arrange` flag — dropped in v17.)
#   v17: folders are purely MANUAL. Auto-arrange removed (policy_sig +
#      guardrail_auto_arrange columns dropped). Deleting a folder now
#      CASCADE-deletes the guardrails inside it (was ON DELETE SET NULL) — the
#      product decision is that removing a folder removes its guardrails.
#   v18: per-model LLM layer selection — target_models gains `num_layers` (total
#      transformer layers, the picker bound) and `selected_layers` (the [start,
#      end) range whose activations train the guardrail). Lets the user pick/edit
#      layers per model; training reads this instead of the global default.
#   v19: public Rule Sets — a model-less rule set (UI "guardrail") can be SHARED
#      to the community. New `rule_sets` table: a named, attributed, model-agnostic
#      collection of already-published rules, carrying the SAME publish-state
#      columns as rules/CEs (public_id, published_at, is_local_draft,
#      pending_public_id, is_ready, created_by_username) so it publishes to + syncs
#      from HF exactly like a rule. New `rule_set_member` join table (which global
#      rules belong to a set, ordered by `position`). The private `classifiers`
#      row stays the authoring source / fork target and is NEVER itself published —
#      publishing mints a separate public record (public_rule_sets/<pid>.json),
#      leaving the private workspace fully editable/deletable.
#   v20: LOGIN REMOVED — single local user. Two changes:
#      (a) a fixed `users` row (utils.auth.LOCAL_USER_ID) is seeded, because
#          six tables FK to users(user_id) ON DELETE CASCADE and nothing
#          creates that row anymore now that register/login are gone;
#      (b) `rule_bookmarks` / `ce_bookmarks` / `rule_set_bookmarks` come BACK
#          as local tables. They had been moved to the central server so they
#          could follow an account between machines; with one local user there
#          is no account to follow, so they live here again. This also repairs
#          /library/bookmarks/search, whose category-only browse path still
#          JOINed these tables after they were dropped (it was dead code).
SCHEMA_VERSION = 20

# Baseline taxonomy kept general and shared across rules/CEs
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
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, params) if params else cursor.execute(query)
        conn.commit()
        result = cursor.fetchall() if cursor.description else None
        cursor.close()
        return result
    except Exception as err:
        conn.rollback()
        print(f"[X]ROLLBACK: {err}")
        raise err
    finally:
        release_connection(conn)

def drop_all_tables():
    """Drops all GAVEL tables to ensure a clean state.

    Order matters: children with FKs first. `_app_meta` is the schema-
    version sentinel — if it's left in place after a drop, the next
    init_database() will fast-skip and leave the DB empty. Always drop
    it together with the rest."""
    print("--- Dropping All Tables ---")
    tables = [
        # Bookmarks (FK to users + rules/CEs/rule_sets)
        "ce_bookmarks", "rule_bookmarks", "rule_set_bookmarks",
        # Ratings + aggregates (Phase 1 of artist feature)
        "ratings", "asset_ratings_summary", "user_ratings_summary",
        # Wizard state (FK to users, classifiers, scenarios, rules — drop
        # first so the FK targets can go down cleanly afterwards)
        "pipeline_runs",
        # Classifier-attached data
        "evaluation_results", "test_datasets",
        "setup_ce_link", "rule_ce_link",
        "rule_setup", "classifiers",
        "calibration_datasets",
        "excitation_datasets",
        # Public rule sets (membership FK to rule_sets + rules — drop first)
        "rule_set_member", "rule_sets",
        # Top-level definitions
        "rules", "cognitive_elements",
        # Per-user scenarios (referenced by rules + test_datasets; FK to users)
        "scenarios",
        # User-owned resources
        "target_models", "users",
        # Shared taxonomies + state
        "categories",
        "sync_state",
        # Schema sentinel — MUST be dropped or fast-skip will think
        # the DB is up-to-date next boot.
        "_app_meta",
    ]
    for table in tables:
        exec_query(f"DROP TABLE IF EXISTS {table} CASCADE;")
    print("[OK]All tables dropped.")

def _schema_version_table_exists() -> bool:
    """One-shot probe: does the `_app_meta` sentinel table itself exist?
    On a fresh DB the answer is no — short-circuit to "needs full init".
    """
    try:
        rows = exec_query(
            "SELECT 1 FROM information_schema.tables WHERE table_name = '_app_meta'"
        )
        return bool(rows)
    except Exception:
        # Connection failure or weird state — fall through to full init.
        return False


def _stored_schema_version() -> int:
    """Read the schema version recorded by the last successful
    init_database. Returns 0 if the row is missing, malformed, or the
    table doesn't exist (callers treat 0 as "needs init")."""
    try:
        rows = exec_query(
            "SELECT value FROM _app_meta WHERE key = 'schema_version'"
        )
        if not rows:
            return 0
        return int(rows[0][0])
    except Exception:
        return 0


# Tables we sanity-check before trusting the schema-version sentinel.
# If _app_meta says v=current BUT any of these don't exist (because
# someone partially-wiped the DB, ran an old drop_all_tables that
# missed _app_meta, or otherwise corrupted state), we treat the
# sentinel as stale and run a full init. Pick a handful from different
# functional areas so a partial wipe in any zone surfaces.
_CRITICAL_TABLES = (
    "rules",
    "cognitive_elements",
    "classifiers",
    "rule_sets",
    "sync_state",
    "categories",
    "neutral_corpus",
    "bundle_jobs",
    # v20: bookmarks are local again (no accounts, nothing to sync to).
    "rule_bookmarks",
    "ce_bookmarks",
    "rule_set_bookmarks",
)


def _critical_tables_present() -> bool:
    """Belt-and-braces check that the schema-version sentinel matches
    physical reality. One query against information_schema is cheaper
    than a stack of failing SELECTs at runtime."""
    try:
        rows = exec_query(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = ANY(%s)
            """,
            (list(_CRITICAL_TABLES),),
        ) or []
        found = {r[0] for r in rows}
        missing = set(_CRITICAL_TABLES) - found
        if missing:
            print(
                f"[init_database] sentinel says schema is up-to-date but "
                f"these critical tables are missing: {sorted(missing)}. "
                f"Falling through to full init."
            )
            return False
        return True
    except Exception:
        # If even the probe fails, definitely run the full init.
        return False


def ensure_local_user():
    """Guarantee the single local user row exists.

    There is no register/login flow anymore, so nothing else ever creates a
    `users` row — yet six tables FK to it with ON DELETE CASCADE
    (target_models, classifiers, guardrail_folders, bundle_jobs, test_datasets,
    pipeline_runs). Without this row every "create a model / rule set / folder"
    insert would fail on the foreign key.

    Called on EVERY boot (not just when the DDL block runs), so a database whose
    user row was manually deleted self-heals on restart instead of failing every
    write. Idempotent: ON CONFLICT DO NOTHING leaves an existing row — including
    any display_name/bio the user edited — untouched.
    """
    from utils.auth import (
        LOCAL_USER_ID, LOCAL_USERNAME, LOCAL_EMAIL, LOCAL_DISPLAY_NAME,
    )
    exec_query(
        """
        INSERT INTO users (user_id, username, email, password, display_name, is_team)
        VALUES (%s, %s, %s, '', %s, FALSE)
        ON CONFLICT (user_id) DO NOTHING
        """,
        (LOCAL_USER_ID, LOCAL_USERNAME, LOCAL_EMAIL, LOCAL_DISPLAY_NAME),
    )


def init_database():
    """Idempotent schema bootstrap. Runs the full DDL block only when
    `SCHEMA_VERSION` (the source-of-truth constant in this file) differs
    from the version recorded in `_app_meta` on the live DB.

    Fast-path (warm restart, schema already at SCHEMA_VERSION):
        ~30ms — two SELECTs and we're out.

    Slow-path (fresh DB, post-wipe, or after a SCHEMA_VERSION bump):
        full DDL pass + a final UPSERT that records the new version.

    If you change anything in the DDL block below, bump SCHEMA_VERSION
    at the top of the file. Otherwise existing deployments will skip
    your change on warm restarts."""
    # Fast path: skip DDL entirely if the live schema is already at
    # the expected version. The _app_meta probe is cheap and safe even
    # on a brand-new DB (returns False, falls through to full init).
    #
    # Sanity check: we ALSO probe for a handful of critical tables
    # (see _CRITICAL_TABLES). If the version says "up to date" but
    # those tables don't actually exist (because someone ran an old
    # drop_all_tables that missed _app_meta, or did a partial wipe),
    # we fall through to a full init rather than launching with an
    # empty schema. This makes the fast path safe under
    # "_app_meta is stale" failure modes.
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

    print("--- Initializing GAVEL-Web Database ---")

    # Enable pg_trgm for fast fuzzy search
    exec_query("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    
    try:
        exec_query("CREATE EXTENSION IF NOT EXISTS vector;")
        print("[OK]Enabled 'vector' extension.")
    except Exception as e:
        print(f"Warning: Could not enable 'vector' extension (requires pgvector installed on host). Error: {e}")
    
    # 1. USERS — local mirror of the remote (Neon) users table.
    # The remote DB is the source of truth for auth/identity. This local
    # copy is populated by sync_user_to_local() on every login/register
    # and exists solely so FK constraints and triggers can resolve user_id.
    exec_query("""
        CREATE TABLE IF NOT EXISTS users (
            user_id      INTEGER PRIMARY KEY,
            username     VARCHAR(255) NOT NULL UNIQUE,
            password     VARCHAR(255) NOT NULL DEFAULT '',
            email        VARCHAR(255) NOT NULL UNIQUE,
            display_name VARCHAR(255),
            bio          TEXT,
            is_team      BOOLEAN NOT NULL DEFAULT FALSE,
            tutorial_seen BOOLEAN DEFAULT FALSE,
            created_at   TIMESTAMPTZ DEFAULT now()
        );
    """)
    exec_query("ALTER TABLE users DROP COLUMN IF EXISTS firstname;")
    exec_query("ALTER TABLE users DROP COLUMN IF EXISTS lastname;")
    exec_query("ALTER TABLE users ADD COLUMN IF NOT EXISTS tutorial_seen BOOLEAN DEFAULT FALSE;")
    exec_query("ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(255);")
    exec_query("ALTER TABLE users ADD COLUMN IF NOT EXISTS bio TEXT;")
    exec_query("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_team BOOLEAN NOT NULL DEFAULT FALSE;")

    # 2. TARGET MODELS (Private)
    exec_query("""
        CREATE TABLE IF NOT EXISTS target_models (
            model_id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            storage_path TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)
    # v11: optional HF token for gated / private models. Stored so training
    # and inference can pass it to transformers' from_pretrained.
    exec_query("ALTER TABLE target_models ADD COLUMN IF NOT EXISTS hf_token TEXT;")
    # v18: per-model LLM layer selection. `num_layers` is the model's total
    # transformer layer count (the picker's upper bound; NULL if unknown);
    # `selected_layers` is the [start, end) range whose activations feed the
    # guardrail RNN. Training uses this range instead of the global default.
    exec_query("ALTER TABLE target_models ADD COLUMN IF NOT EXISTS num_layers INTEGER;")
    exec_query("ALTER TABLE target_models ADD COLUMN IF NOT EXISTS selected_layers INTEGER[];")

    # 3. GLOBAL COGNITIVE ELEMENTS (Public)
    exec_query("""
        CREATE TABLE IF NOT EXISTS cognitive_elements (
            ce_id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            definition TEXT,
            category VARCHAR(50) DEFAULT 'CONTEXT',
            categories INTEGER[] DEFAULT ARRAY[]::INTEGER[],
            note TEXT,
            examples JSONB DEFAULT '[]'::jsonb,
            embedding vector(384),
            search_vector tsvector,
            type VARCHAR(100),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        -- Indexes for High-Performance Search
        CREATE INDEX IF NOT EXISTS ce_name_trgm_idx ON cognitive_elements USING gin (name gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS ce_search_idx ON cognitive_elements USING gin (search_vector);
        -- HNSW Index for Semantic Search (Requires pgvector)
        CREATE INDEX IF NOT EXISTS ce_embedding_hnsw_idx ON cognitive_elements USING hnsw (embedding vector_cosine_ops);
    """)

    # 3A. CATEGORY TAXONOMY (Shared)
    exec_query(
        """
        CREATE TABLE IF NOT EXISTS categories (
            category_id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL UNIQUE,
            description TEXT,
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        """
    )
    
    # 3B. EXCITATION DATASETS (Training Data for CEs)
    exec_query("""
        CREATE TABLE IF NOT EXISTS excitation_datasets (
            dataset_id SERIAL PRIMARY KEY,
            ce_id INTEGER REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE UNIQUE,
            dataset TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)

    # 3C. CALIBRATION DATASETS (Per-CE calibration conversations)
    exec_query("""
        CREATE TABLE IF NOT EXISTS calibration_datasets (
            dataset_id SERIAL PRIMARY KEY,
            ce_id INTEGER REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE UNIQUE,
            dataset TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)

    # 3D. SCENARIOS — REMOVED in schema v9.
    #
    # The scenario (misuse description) used to live in its own per-user
    # table, reused to pre-fill test-set generation. That role is now
    # redundant: the final scenario is embedded inside each generated
    # set's config (test_datasets.config.scenario_instructions), and the
    # in-progress scenario lives in pipeline_runs.steps while the wizard
    # is open. The table + its scenario_id FKs are torn down below (see
    # the "v9 scenario teardown" block after the test_datasets section).

    # 4. GLOBAL RULES (Public Library)
    exec_query("""
        CREATE TABLE IF NOT EXISTS rules (
            rule_id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            predicate TEXT NOT NULL,
            categories INTEGER[] DEFAULT ARRAY[]::INTEGER[],
            embedding vector(384),
            search_vector tsvector,
            type VARCHAR(100),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        -- Indexes
        CREATE INDEX IF NOT EXISTS rule_name_trgm_idx ON rules USING gin (name gin_trgm_ops);
        CREATE INDEX IF NOT EXISTS rule_search_idx ON rules USING gin (search_vector);
        CREATE INDEX IF NOT EXISTS rule_embedding_hnsw_idx ON rules USING hnsw (embedding vector_cosine_ops);
    """)

    # 4B. Legacy table cleanup. `rule_calibration_datasets` was merged into
    # `test_datasets` long ago; as of v10 rule calibration is rule-scoped
    # (is_default / user-owned) and classifier-scoped migration no longer
    # applies, so just drop the old table if a previous deploy left it.
    # Same for the even older `rule_evaluation_datasets`. Idempotent.
    exec_query("DROP TABLE IF EXISTS rule_calibration_datasets;")
    exec_query("DROP TABLE IF EXISTS rule_evaluation_datasets;")

    # 5. CLASSIFIERS (Private)
    exec_query("""
        CREATE TABLE IF NOT EXISTS classifiers (
            classifier_id SERIAL PRIMARY KEY,
            model_id INTEGER REFERENCES target_models(model_id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            status VARCHAR(50) DEFAULT 'untrained',
            model_path TEXT,
            training_log TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)
    # Backfill columns for existing deployments
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS model_path TEXT;")
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS training_log TEXT;")
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS training_config JSONB DEFAULT '{}'::jsonb;")
    # Snapshot of which rule_setup rows the classifier was trained against,
    # frozen at the moment training finished. The user can keep editing the
    # live rule_setup selection afterwards, but evaluation, calibration, and
    # the realtime classifier all reference this snapshot — that's the only
    # rule set the trained weights actually understand. Cleared when a
    # retrain fails or the classifier is reset to 'untrained'.
    #
    # `trained_rule_setup_ids` stores volatile local PKs and is kept around
    # for backward compat. `trained_rule_names` is the durable identity
    # used by drift detection: a rule deleted and re-added with the same
    # name should NOT register as drift, even though setup_id changes.
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS trained_rule_setup_ids INTEGER[] DEFAULT NULL;")
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS trained_rule_names TEXT[] DEFAULT NULL;")
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS trained_at TIMESTAMPTZ DEFAULT NULL;")
    # Content fingerprint of the policy the model was trained on (CE ids +
    # roles + fallback grouping per rule, order/setup_id-independent). Compared
    # against the live policy fingerprint to decide 'needs_retraining' from REAL
    # drift instead of a sticky flag. NULL for classifiers trained before this.
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS trained_policy_fingerprint TEXT DEFAULT NULL;")

    # Live-training progress signal. Updated by run_training()'s
    # progress_callback on every stage boundary so the UI can show
    # something more informative than "Training..." while the multi-
    # minute pipeline runs. Cleared back to NULL on completion or error.
    #   training_phase        — short user-facing label ("Loading model")
    #   training_phase_detail — optional sub-status ("Epoch 3 of 10")
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS training_phase TEXT DEFAULT NULL;")
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS training_phase_detail TEXT DEFAULT NULL;")

    # v15: a classifier (UI "guardrail") is now owned directly by a user, not
    # only via its model. This lets a guardrail hold a rule set before a model
    # is picked (model_id stays NULL until the user attaches one at train time).
    # Backfill existing rows from their model's owner, then enforce NOT NULL —
    # but only once every row has an owner (a freshly added NULL column on an
    # empty/legacy table would otherwise make SET NOT NULL fail).
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE;")
    exec_query("""
        UPDATE classifiers c SET user_id = tm.user_id
        FROM target_models tm
        WHERE c.model_id = tm.model_id AND c.user_id IS NULL;
    """)
    exec_query("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM classifiers WHERE user_id IS NULL) THEN
                ALTER TABLE classifiers ALTER COLUMN user_id SET NOT NULL;
            END IF;
        END $$;
    """)
    exec_query("CREATE INDEX IF NOT EXISTS classifiers_user_idx ON classifiers (user_id);")

    exec_query("""
        UPDATE classifiers SET status = 'untrained'
        WHERE status = 'training' AND model_path IS NULL;
    """)

    # GUARDRAIL FOLDERS — a MANUAL "library" grouping over the classifiers (UI
    # "guardrails"). A guardrail belongs to at most one folder (folder_id NULL =
    # ungrouped). The user decides what goes in each folder; there is no
    # auto-arrange. Deleting a folder CASCADE-deletes the guardrails inside it.
    exec_query("""
        CREATE TABLE IF NOT EXISTS guardrail_folders (
            folder_id  SERIAL PRIMARY KEY,
            user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            name       VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS guardrail_folders_user_idx ON guardrail_folders (user_id);")
    exec_query("ALTER TABLE classifiers ADD COLUMN IF NOT EXISTS folder_id INTEGER;")
    exec_query("CREATE INDEX IF NOT EXISTS classifiers_folder_idx ON classifiers (folder_id);")
    # Folder deletion takes its guardrails with it (ON DELETE CASCADE). Re-assert
    # the FK so DBs created under the earlier SET NULL design are corrected.
    exec_query("ALTER TABLE classifiers DROP CONSTRAINT IF EXISTS classifiers_folder_id_fkey;")
    exec_query("""
        ALTER TABLE classifiers ADD CONSTRAINT classifiers_folder_id_fkey
            FOREIGN KEY (folder_id) REFERENCES guardrail_folders(folder_id) ON DELETE CASCADE;
    """)
    # v17: auto-arrange was removed — folders are purely manual. Drop its leftovers.
    exec_query("ALTER TABLE guardrail_folders DROP COLUMN IF EXISTS policy_sig;")
    exec_query("ALTER TABLE users DROP COLUMN IF EXISTS guardrail_auto_arrange;")

    # 6. RULE SETUP (Private Override)
    # This stores the specific logic (predicate) for this specific guardrail.
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_setup (
            setup_id SERIAL PRIMARY KEY,
            classifier_id INTEGER REFERENCES classifiers(classifier_id) ON DELETE CASCADE,
            rule_id INTEGER REFERENCES rules(rule_id) ON DELETE SET NULL, -- Reference to public origin
            custom_name VARCHAR(255), -- If user renames the rule locally
            predicate TEXT NOT NULL,   -- The Boolean Logic (e.g., CE1 AND CE3)
            is_active BOOLEAN DEFAULT TRUE
        );
    """)

    # 7. MANY-TO-MANY: SETUP <-> COGNITIVE ELEMENTS (Private) 
    # Tracks which CEs are currently part of this specific setup (mirrors rule_ce_link metadata).
    exec_query("""
        CREATE TABLE IF NOT EXISTS setup_ce_link (
            setup_id INTEGER REFERENCES rule_setup(setup_id) ON DELETE CASCADE,
            ce_id INTEGER REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL DEFAULT 'necessary',
            fallback_group INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            PRIMARY KEY (setup_id, ce_id, role, fallback_group)
        );
    """)

    # Normalize setup_ce_link to support role + fallback grouping
    exec_query("ALTER TABLE setup_ce_link ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'necessary';")
    exec_query("ALTER TABLE setup_ce_link ADD COLUMN IF NOT EXISTS fallback_group INTEGER NOT NULL DEFAULT 0;")
    exec_query("ALTER TABLE setup_ce_link ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT now();")
    exec_query("""
        ALTER TABLE setup_ce_link
        DROP CONSTRAINT IF EXISTS setup_ce_link_pkey;
    """)
    exec_query("""
        ALTER TABLE setup_ce_link
        ADD CONSTRAINT setup_ce_link_pkey PRIMARY KEY (setup_id, ce_id, role, fallback_group);
    """)
    exec_query("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'setup_ce_link_role_check'
            ) THEN
                ALTER TABLE setup_ce_link
                ADD CONSTRAINT setup_ce_link_role_check
                CHECK (role IN ('necessary','fallback','sufficient'));
            END IF;
        END $$;
    """)

    # 8. MANY-TO-MANY: RULE <-> COGNITIVE ELEMENTS (Public)
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_ce_link (
            rule_id INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE,
            ce_id INTEGER REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL DEFAULT 'necessary',
            fallback_group INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)

    # Normalize rule_ce_link to support role + fallback grouping (mirrors JSON structure)
    # - role ∈ {necessary, fallback, sufficient}
    # - fallback_group groups OR-sets for fallback; 0 for non-fallback roles
    exec_query("ALTER TABLE rule_ce_link ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'necessary';")
    exec_query("ALTER TABLE rule_ce_link ADD COLUMN IF NOT EXISTS fallback_group INTEGER NOT NULL DEFAULT 0;")
    exec_query("ALTER TABLE rule_ce_link ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT now();")
    exec_query("""
        ALTER TABLE rule_ce_link
        DROP CONSTRAINT IF EXISTS rule_ce_link_pkey;
    """)
    exec_query("""
        ALTER TABLE rule_ce_link
        ADD CONSTRAINT rule_ce_link_pkey PRIMARY KEY (rule_id, ce_id, role, fallback_group);
    """)
    exec_query("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'rule_ce_link_role_check'
            ) THEN
                ALTER TABLE rule_ce_link
                ADD CONSTRAINT rule_ce_link_role_check
                CHECK (role IN ('necessary','fallback','sufficient'));
            END IF;
        END$$;
    """)
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_ce_link_rule ON rule_ce_link(rule_id);")
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_ce_link_ce ON rule_ce_link(ce_id);")
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_ce_link_role ON rule_ce_link(role);")

    # 8b. (removed) REALTIME ANALYSIS CACHE — the realtime viewer now computes CE
    # logits live each session and persists nothing, matching the reference project
    # (the reference realtime monitor holds logits only in session memory and discards them).
    # Drop the legacy cache table on existing deployments.
    exec_query("DROP TABLE IF EXISTS realtime_analysis_cache CASCADE;")

    # 9. BOOKMARKS (v20 — local again).
    #
    # These lived on the central server for a while so a user's bookmarks would
    # follow their ACCOUNT between machines, keyed by the HF `public_id`. With
    # login removed there is no account and no second machine, so they are plain
    # local rows keyed by the local SERIAL id — which also means you can bookmark
    # a DRAFT (the central version couldn't: an unpublished rule has no
    # public_id). `rule_set_bookmarks` is created further down, after the
    # `rule_sets` table it references exists.
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_bookmarks (
            user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            rule_id    INTEGER NOT NULL REFERENCES rules(rule_id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (user_id, rule_id)
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS rule_bookmarks_user_idx ON rule_bookmarks (user_id, created_at DESC);")
    exec_query("""
        CREATE TABLE IF NOT EXISTS ce_bookmarks (
            user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            ce_id      INTEGER NOT NULL REFERENCES cognitive_elements(ce_id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (user_id, ce_id)
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS ce_bookmarks_user_idx ON ce_bookmarks (user_id, created_at DESC);")

    # Backfill hybrid search columns for existing deployments
    column_alterations = [
        "ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS categories TEXT[] DEFAULT ARRAY[]::TEXT[]",
        "ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS embedding DOUBLE PRECISION[]",
        "ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS type VARCHAR(100)",
        "ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS note TEXT",
        "ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS examples JSONB DEFAULT '[]'::jsonb",
        "ALTER TABLE rules ADD COLUMN IF NOT EXISTS categories TEXT[] DEFAULT ARRAY[]::TEXT[]",
        "ALTER TABLE rules ADD COLUMN IF NOT EXISTS embedding DOUBLE PRECISION[]",
        "ALTER TABLE rules ADD COLUMN IF NOT EXISTS type VARCHAR(100)",
        "ALTER TABLE rules ADD COLUMN IF NOT EXISTS description TEXT",
        # Registry-link columns. Local rows that came from the public HF
        # registry carry the public_id of their source record. NULL means
        # the row is local-only (a draft or a pre-registry seed). UNIQUE so
        # the next sync can dedup against the manifest by public_id.
        "ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS public_id TEXT",
        "ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ",
        "ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS is_local_draft BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE rules ADD COLUMN IF NOT EXISTS public_id TEXT",
        "ALTER TABLE rules ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ",
        "ALTER TABLE rules ADD COLUMN IF NOT EXISTS is_local_draft BOOLEAN NOT NULL DEFAULT TRUE",
        # parent_public_id was a placeholder for a Fork & Edit feature that
        # never shipped — every row has it as NULL. Drop it on existing
        # databases so the schema reflects what the code actually uses.
        "ALTER TABLE cognitive_elements DROP COLUMN IF EXISTS parent_public_id",
        "ALTER TABLE rules DROP COLUMN IF EXISTS parent_public_id",
        # pending_public_id is the "intent stamp" used by the publish flow:
        # it's set right before the HF push and cleared on either success or
        # failure. If a row carries this stamp into the next session, a
        # crash happened mid-publish — boot-time recovery decides whether
        # to heal forward (HF has the record) or clear (HF doesn't).
        "ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS pending_public_id TEXT",
        "ALTER TABLE rules ADD COLUMN IF NOT EXISTS pending_public_id TEXT",
        # is_ready is the "creation complete" flag. AI-pipeline writes set
        # this to FALSE up-front and flip to TRUE only after training data +
        # embeddings have actually landed. The /library/drafts query and all
        # user-facing list endpoints filter on is_ready = TRUE so half-baked
        # rows are invisible. Boot-time IncompletePipelineRecovery wipes any
        # is_ready = FALSE rows so a crash, network drop, or closed tab
        # mid-pipeline cleanly looks like the user never created the row.
        # DEFAULT TRUE keeps every legacy row + every HF-synced row visible.
        "ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS is_ready BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE rules ADD COLUMN IF NOT EXISTS is_ready BOOLEAN NOT NULL DEFAULT TRUE",
        # parked_at + parked_proposal carried "user X-dismissed the AI
        # proposal review" state for the old AIChatModal flow. Phase 3
        # replaced that mechanism with the `pipeline_runs` table — the
        # columns are no longer written or read, so drop them. Idempotent:
        # fresh installs that never had them just no-op the DROP.
        "ALTER TABLE rules DROP COLUMN IF EXISTS parked_at",
        "ALTER TABLE rules DROP COLUMN IF EXISTS parked_proposal",
        # scenario_id (rules) removed in v9 — see the scenario teardown
        # block after the test_datasets section.
    ]
    for ddl in column_alterations:
        exec_query(ddl)

    # Unique-when-present index on public_id (a partial unique index). Lets
    # multiple drafts coexist with public_id NULL while still preventing two
    # rows from claiming the same registry identity.
    exec_query(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_ces_public_id "
        "ON cognitive_elements (public_id) WHERE public_id IS NOT NULL"
    )
    exec_query(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_rules_public_id "
        "ON rules (public_id) WHERE public_id IS NOT NULL"
    )

    # Categories are NOT seeded here. HF (categories.json at the registry
    # root) is the single source of truth, and the local categories table is
    # populated by the first /library/sync call on login. DEFAULT_CATEGORIES
    # is kept around for the one-time bootstrap_hf_categories.py script that
    # publishes the seed taxonomy to HF on a maintainer machine.

    # Hybrid search indexes (trigram + category filtering)
    exec_query("CREATE INDEX IF NOT EXISTS idx_rules_name_trgm ON rules USING gin (name gin_trgm_ops);")
    exec_query("CREATE INDEX IF NOT EXISTS idx_ces_name_trgm ON cognitive_elements USING gin (name gin_trgm_ops);")
    exec_query("CREATE INDEX IF NOT EXISTS idx_rules_categories_gin ON rules USING GIN (categories);")
    exec_query("CREATE INDEX IF NOT EXISTS idx_ces_categories_gin ON cognitive_elements USING GIN (categories);")

    # Backfill: rebuild any existing search_vector entries with weighted A/B form so
    # ts_rank_cd favors name matches. New writes already go through the weighted path
    # in embedding_utils.py; this catches rows seeded before the upgrade. Skipped for
    # rows whose text representation already shows a weight letter (A/B/C/D after the
    # position) — that means they're already in the new format.
    exec_query("""
        UPDATE rules
        SET search_vector = setweight(to_tsvector('english', COALESCE(name, '')), 'A')
                         || setweight(to_tsvector('english', COALESCE(predicate, '')), 'B')
        WHERE search_vector IS NOT NULL
          AND search_vector::text !~ ':[0-9]+[ABCD]'
    """)
    exec_query("""
        UPDATE cognitive_elements
        SET search_vector = setweight(to_tsvector('english', COALESCE(name, '')), 'A')
                         || setweight(to_tsvector('english', COALESCE(definition, '')), 'B')
        WHERE search_vector IS NOT NULL
          AND search_vector::text !~ ':[0-9]+[ABCD]'
    """)

    # Library sync state. Single-row-per-key store used by services/hf_sync.py
    # to remember the last manifest hash we pulled, so repeat syncs can
    # short-circuit when nothing has changed in the registry.
    exec_query("""
        CREATE TABLE IF NOT EXISTS sync_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMPTZ DEFAULT now()
        )
    """)

    # 9b. NEUTRAL CORPUS (global, registry-synced)
    #
    # Domain-agnostic benign conversations the classifier must NEVER fire on —
    # the evaluation's third split for measuring false-positive rate against
    # everyday content. Mirrors the reference neutral set, split into its two
    # pseudo use-cases:
    #   - conversational : small-talk / opinions / chit-chat
    #   - instructive    : how-to / factual / informational Q&A
    # Global (not per-rule/per-CE): one shared pool, pulled from HF
    # (neutral/<category>/conversations.json) and deduped by content_hash.
    exec_query("""
        CREATE TABLE IF NOT EXISTS neutral_corpus (
            id SERIAL PRIMARY KEY,
            content_hash TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            conversation JSONB NOT NULL,
            published_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT now()
        )
    """)

    # 10. EVALUATION RESULTS (per classifier)
    exec_query("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            eval_id SERIAL PRIMARY KEY,
            classifier_id INTEGER REFERENCES classifiers(classifier_id) ON DELETE CASCADE,
            eval_type VARCHAR(50) NOT NULL,
            thresholds JSONB,
            metrics JSONB,
            plots JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)

    # 10b. BUNDLE JOBS (server-side export/import background jobs)
    #
    # Export/import of a classifier bundle runs as a detached background task so
    # it survives the user closing the modal; only a backend crash ends it. This
    # table is the durable job record the frontend polls, AND the breadcrumb
    # crash recovery uses to clean up a partially-imported classifier (it records
    # the created classifier_id as soon as the row exists, so a crash mid-import
    # can roll it back). `artifact_path`/`filename` hold a finished export zip on
    # disk for download-when-ready.
    exec_query("""
        CREATE TABLE IF NOT EXISTS bundle_jobs (
            job_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            job_type VARCHAR(16) NOT NULL,                  -- 'export' | 'import'
            status VARCHAR(16) NOT NULL DEFAULT 'running',  -- running | done | error
            phase TEXT,
            error TEXT,
            classifier_id INTEGER,                          -- export: source; import: created
            tier VARCHAR(32),
            artifact_path TEXT,                             -- finished export zip on disk
            filename TEXT,                                  -- export download filename
            result JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS bundle_jobs_user_idx ON bundle_jobs (user_id, created_at DESC);")
    exec_query("CREATE INDEX IF NOT EXISTS bundle_jobs_running_idx ON bundle_jobs (status) WHERE status = 'running';")

    # 11. TEST DATASETS
    #
    # Purely rule-scoped (v10 — classifier_id dropped). A test set is one of:
    #   - default      : rule_id set, is_default=TRUE, user_id NULL. The
    #                    canonical set born with the rule, pushed to HF when
    #                    the rule is published (3 rows per rule: positive /
    #                    negative / positive_calibration).
    #   - private custom: rule_id set, is_default=FALSE, user_id set. A user's
    #                     own test set for a rule; never published.
    # Test dialogues are classifier-agnostic, so there is no classifier link.
    exec_query("""
        CREATE TABLE IF NOT EXISTS test_datasets (
            dataset_id SERIAL PRIMARY KEY,
            dataset_type VARCHAR(50) NOT NULL,
            scenario_name VARCHAR(255),
            config JSONB,
            conversations JSONB,
            status VARCHAR(50) DEFAULT 'pending',
            generation_log TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)
    exec_query("ALTER TABLE test_datasets ADD COLUMN IF NOT EXISTS scenario_name VARCHAR(255);")
    # v9/v10: rule-level ownership + HF columns; classifier link removed.
    test_dataset_alterations = [
        "ALTER TABLE test_datasets ADD COLUMN IF NOT EXISTS rule_id INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE",
        "ALTER TABLE test_datasets ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE",
        "ALTER TABLE test_datasets ADD COLUMN IF NOT EXISTS is_default BOOLEAN NOT NULL DEFAULT FALSE",
        # HF registry columns, mirroring rules / cognitive_elements. Only the
        # is_default rows of a *published* rule ever get a public_id.
        "ALTER TABLE test_datasets ADD COLUMN IF NOT EXISTS public_id TEXT",
        "ALTER TABLE test_datasets ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ",
        "ALTER TABLE test_datasets ADD COLUMN IF NOT EXISTS pending_public_id TEXT",
        # v10: drop the classifier link. Pre-v10 rows that were classifier-
        # scoped (rule_id NULL, not a default) are now unreachable — purge
        # them before dropping the column so no orphans linger.
        "DELETE FROM test_datasets WHERE rule_id IS NULL AND is_default = FALSE",
        "ALTER TABLE test_datasets DROP COLUMN IF EXISTS classifier_id",
    ]
    for ddl in test_dataset_alterations:
        exec_query(ddl)
    # One default per (rule, dataset_type) so regeneration is an idempotent
    # UPSERT (ON CONFLICT). Partial unique on public_id mirrors uq_rules_public_id.
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

    # 12. PIPELINE RUNS — wizard-state persistence for the 8-step
    # rule-creation flow (Phase 3).
    #
    # Each row is one in-progress (or completed) walk through the
    # reference 1 → 2A → 2B → 2C → 3A → 3B → 3C → 3D pipeline. The
    # user can close their browser and resume: GET /pipeline-runs/active
    # surfaces unfinished runs scoped to their classifier.
    #
    # `current_step` is the step the user is *on* (not necessarily the
    # last completed one — a user can rewind). `steps` is a per-step
    # state map keyed by step id ("1", "2A", ..., "3D"); each value
    # carries a `status` (pending/in_progress/completed/skipped/error)
    # plus step-specific `data` (scenario_id, rule_id, ce_ids,
    # dataset_ids, ...). Skip-able steps move on without producing
    # output; the wizard ignores them downstream.
    #
    # FKs cascade-delete from the obvious owners (user, classifier) but
    # SET NULL from rule so the run record survives a rollback of its
    # outputs (useful for "user discarded a step" debugging). The
    # in-progress scenario lives in `steps` (v9 removed the scenarios FK).
    exec_query("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            classifier_id INTEGER REFERENCES classifiers(classifier_id) ON DELETE CASCADE,
            rule_id INTEGER REFERENCES rules(rule_id) ON DELETE SET NULL,
            pipeline_type VARCHAR(20) NOT NULL DEFAULT 'rule',
            current_step VARCHAR(10) NOT NULL DEFAULT '1',
            steps JSONB NOT NULL DEFAULT '{}'::jsonb,
            completed BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS pipeline_runs_user_idx
            ON pipeline_runs (user_id);
        CREATE INDEX IF NOT EXISTS pipeline_runs_active_idx
            ON pipeline_runs (user_id, classifier_id)
            WHERE completed = FALSE;
    """)
    # Phase 7 migrations: classifier_id was NOT NULL until v8. Pipeline A
    # (rule generation) creates rows with no classifier context, so drop
    # the NOT NULL. The pipeline_type column tracks which flavor a row
    # belongs to ('rule' or 'test_eval') — defaults to 'rule' so a fresh
    # row from an older client at least lands in the rule flow.
    exec_query("ALTER TABLE pipeline_runs ALTER COLUMN classifier_id DROP NOT NULL")
    exec_query("ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS pipeline_type VARCHAR(20) NOT NULL DEFAULT 'rule'")
    # Best-effort retag of pre-Phase-7 rows: anything that already had a
    # classifier_id was the old unified 8-step wizard, which is closer to
    # test_eval than rule. Pre-Phase-7 rows are typically abandoned so
    # this mostly affects diagnostic queries, not user-visible state.
    exec_query(
        "UPDATE pipeline_runs SET pipeline_type = 'test_eval' "
        "WHERE pipeline_type = 'rule' AND classifier_id IS NOT NULL AND created_at < now() - interval '1 minute'"
    )

    # --- v9 scenario teardown ------------------------------------------------
    # The scenarios table is gone (see "3D. SCENARIOS — REMOVED"). Drop the
    # scenario_id FK columns from every dependant BEFORE dropping the table so
    # nothing still references it. Placed here, after rules / test_datasets /
    # pipeline_runs all exist, so the ALTERs never hit a missing table. All
    # idempotent: fresh v9 installs never created these, so the DROPs no-op;
    # databases converging from v8 or earlier get cleaned exactly once.
    exec_query("ALTER TABLE rules DROP COLUMN IF EXISTS scenario_id;")
    exec_query("ALTER TABLE test_datasets DROP COLUMN IF EXISTS scenario_id;")
    exec_query("ALTER TABLE pipeline_runs DROP COLUMN IF EXISTS scenario_id;")
    exec_query("DROP TABLE IF EXISTS scenarios CASCADE;")

    # -- COMMUNITY FEATURES: CREATOR ATTRIBUTION, RATINGS -----------------------

    try:
        exec_query("CREATE EXTENSION IF NOT EXISTS citext;")
    except Exception as e:
        print(f"Warning: could not enable 'citext' extension: {e}")

    exec_query("ALTER TABLE rules ADD COLUMN IF NOT EXISTS created_by_username CITEXT;")
    exec_query("ALTER TABLE cognitive_elements ADD COLUMN IF NOT EXISTS created_by_username CITEXT;")
    exec_query("CREATE INDEX IF NOT EXISTS idx_rules_created_by ON rules (created_by_username);")
    exec_query("CREATE INDEX IF NOT EXISTS idx_ces_created_by ON cognitive_elements (created_by_username);")

    # -- COMMUNITY FEATURES: PUBLIC RULE SETS (v19) ---------------------------
    #
    # A "rule set" (UI: a model-less guardrail/classifier — rules+CEs picked
    # BEFORE a model) can be SHARED to the public library. The shared artifact
    # is a SEPARATE, HF-published record — NOT the private `classifiers` row,
    # which stays fully editable/deletable. `rule_sets` is therefore a third
    # public-library peer of `rules`/`cognitive_elements`: it carries the same
    # publish-state column set (public_id / published_at / is_local_draft /
    # pending_public_id / is_ready / created_by_username) so it publishes to and
    # syncs from HF through the exact same machinery. A public rule set is a
    # thin pointer-collection: `rule_set_member` references already-published
    # GLOBAL rules by rule_id (like rule_ce_link references CEs), so no rule/CE
    # content is duplicated. Created after the citext extension + the rule/CE
    # created_by_username columns so the CITEXT column type resolves.
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_sets (
            rule_set_id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT,
            categories INTEGER[] DEFAULT ARRAY[]::INTEGER[],
            public_id TEXT,
            published_at TIMESTAMPTZ,
            is_local_draft BOOLEAN NOT NULL DEFAULT TRUE,
            pending_public_id TEXT,
            is_ready BOOLEAN NOT NULL DEFAULT TRUE,
            created_by_username CITEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );
    """)
    # Partial unique on public_id (mirrors uq_rules_public_id): many NULL drafts
    # coexist, but exactly one row per registry identity. Never a plain UNIQUE.
    exec_query(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_rule_sets_public_id "
        "ON rule_sets (public_id) WHERE public_id IS NOT NULL"
    )
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_sets_created_by ON rule_sets (created_by_username);")
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_sets_name_trgm ON rule_sets USING gin (name gin_trgm_ops);")
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_sets_categories_gin ON rule_sets USING GIN (categories);")

    # Membership: which (already-published) rules make up a public rule set,
    # referenced by global rule_id. `position` preserves the author's ordering.
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_set_member (
            rule_set_id INTEGER REFERENCES rule_sets(rule_set_id) ON DELETE CASCADE,
            rule_id INTEGER REFERENCES rules(rule_id) ON DELETE CASCADE,
            position INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            PRIMARY KEY (rule_set_id, rule_id)
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS idx_rule_set_member_rule ON rule_set_member (rule_id);")

    # Rule-set bookmarks (v20). Created here rather than with the other two
    # bookmark tables because it FKs `rule_sets`, which is only defined above.
    exec_query("""
        CREATE TABLE IF NOT EXISTS rule_set_bookmarks (
            user_id     INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            rule_set_id INTEGER NOT NULL REFERENCES rule_sets(rule_set_id) ON DELETE CASCADE,
            created_at  TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (user_id, rule_set_id)
        );
    """)
    exec_query("CREATE INDEX IF NOT EXISTS rule_set_bookmarks_user_idx ON rule_set_bookmarks (user_id, created_at DESC);")

    # Ratings: one row per (user, asset). UNIQUE means re-rating becomes
    # an UPDATE. CHECK enforces 1-5 score. Self-rating is blocked in
    # application code (the relevant row's created_by_username must !=
    # rater's username) rather than via a CHECK constraint that would
    # need to look across tables.
    # RATINGS + summaries — moved to central server (see central-server/app/utils/schema.py).
    # Drop legacy local tables and triggers on existing deployments.
    exec_query("DROP TRIGGER IF EXISTS trg_ratings_summary ON ratings;")
    exec_query("DROP TRIGGER IF EXISTS trg_rules_contribution_count ON rules;")
    exec_query("DROP TRIGGER IF EXISTS trg_ces_contribution_count ON cognitive_elements;")
    exec_query("DROP FUNCTION IF EXISTS update_ratings_summaries() CASCADE;")
    exec_query("DROP FUNCTION IF EXISTS update_user_rules_contribution_count() CASCADE;")
    exec_query("DROP FUNCTION IF EXISTS update_user_ces_contribution_count() CASCADE;")
    exec_query("DROP TABLE IF EXISTS asset_ratings_summary CASCADE;")
    exec_query("DROP TABLE IF EXISTS user_ratings_summary CASCADE;")
    exec_query("DROP TABLE IF EXISTS ratings CASCADE;")

    # Schema version sentinel. Created last so a partial init (process
    # killed mid-DDL) leaves the version row absent, forcing the next
    # boot to re-run the full DDL block. _app_meta is a generic
    # key/value store so future bootstrap state can sit here too.
    exec_query("""
        CREATE TABLE IF NOT EXISTS _app_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)
    exec_query(
        """
        INSERT INTO _app_meta (key, value) VALUES ('schema_version', %s)
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value, updated_at = now()
        """,
        (str(SCHEMA_VERSION),),
    )
    # The single local user (v20). Must exist before any model / rule set /
    # folder can be created, since all of those FK to users(user_id).
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
                    name_map[key] = new_id # Update cache
            except Exception as e:
                print(f"Error creating category {s_item}: {e}")

    # Convert to list
    result_list = sorted(list(final_ids))
    if max_len and len(result_list) > max_len:
        result_list = result_list[:max_len]
        
    return result_list




