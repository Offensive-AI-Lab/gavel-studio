"""Schema initialization for the central server.

The central server is now purely the HuggingFace write proxy plus the registry
control plane, so it owns exactly ONE table:

    registry_version — the watcher's persisted {commit, global_signature,
                       namespaces} map, so a restart doesn't re-announce a
                       version_update for a commit it already reconciled.

Everything else this server used to store — `users`, `ratings`,
`asset_ratings_summary`, `user_ratings_summary`, and the three `*_bookmarks`
tables — existed to support ACCOUNTS. Login was removed project-wide, so they
are dropped below: nothing reads or writes them anymore, and leaving stale
password hashes sitting in a database is worse than removing them.

Bookmarks did NOT disappear as a feature — they moved into the local backend's
own database, keyed by local id (see backend/services/bookmarks.py). Ratings
did disappear: they were only meaningful across many accounts.
"""
from .db import execute

# Tables from the account era. Dropped on boot — idempotent, and a no-op on a
# fresh install. CASCADE handles the FKs between them.
_LEGACY_ACCOUNT_TABLES = (
    "rule_bookmarks",
    "ce_bookmarks",
    "rule_set_bookmarks",
    "ratings",
    "asset_ratings_summary",
    "user_ratings_summary",
    "users",
)


def init_schema() -> None:
    print("--- Initializing Central Server Schema ---")

    # The ratings trigger references `ratings`; drop it before the table so the
    # DROP isn't blocked by a dependency. Wrapped because DROP TRIGGER ... ON
    # <missing table> is an error in Postgres (unlike DROP TABLE IF EXISTS),
    # which is exactly the fresh-install case.
    try:
        execute("DROP TRIGGER IF EXISTS trg_asset_ratings_summary ON ratings;")
    except Exception:
        pass
    execute("DROP FUNCTION IF EXISTS update_asset_ratings_summary() CASCADE;")

    for table in _LEGACY_ACCOUNT_TABLES:
        execute(f"DROP TABLE IF EXISTS {table} CASCADE;")

    # The control plane's persisted version map. `source_watcher` also creates
    # this lazily on first use; doing it here means a fresh deployment has the
    # table before the first webhook lands.
    execute("""
        CREATE TABLE IF NOT EXISTS registry_version (
            repo             TEXT PRIMARY KEY,
            commit_sha       TEXT,
            global_signature TEXT,
            namespaces       JSONB,
            updated_at       TIMESTAMPTZ DEFAULT now()
        );
    """)

    print("[OK] Central schema initialized (HF proxy + control plane only)")
