"""DB-dependent environment checks.

These verify that the running database has the schema and extensions the
application expects.
"""
import pytest


class TestDatabaseConnectivity:
    """Verify env vars resolve to a working DB connection."""

    def test_can_get_connection(self):
        from utils.sqlite_db import get_connection, release_connection
        conn = get_connection()
        try:
            assert conn is not None
        finally:
            release_connection(conn)


class TestDatabaseInitialization:
    """Verify schema initialization is idempotent and complete."""

    def test_init_database_idempotent(self):
        """Running init_database twice should not error or duplicate data."""
        from utils.DButils import init_database
        from utils.sqlite_db import execute_query_dict

        before = execute_query_dict("SELECT COUNT(*) AS c FROM categories")
        count_before = before[0]["c"]

        init_database()

        after = execute_query_dict("SELECT COUNT(*) AS c FROM categories")
        count_after = after[0]["c"]
        assert count_after == count_before

    def test_required_tables_exist(self):
        """Critical tables must exist after init."""
        from utils.sqlite_db import execute_query_dict
        # Bookmarks + ratings tables moved to the earlier multi-user server when we
        # migrated users to a shared identity service; only model/training
        # data lives locally now.
        required = [
            "users", "target_models", "classifiers",
            "cognitive_elements", "rules", "categories",
            "rule_setup", "setup_ce_link", "rule_ce_link",
            "excitation_datasets", "calibration_datasets",
            "evaluation_results", "test_datasets",
        ]
        result = execute_query_dict(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        existing = {r["name"] for r in result or []}
        missing = [t for t in required if t not in existing]
        assert not missing, f"Missing tables: {missing}"

    def test_logic_columns_present(self):
        """The groups+condition model: rules and rule_setup carry
        ce_groups/condition, and the junction tables are membership-only."""
        from utils.sqlite_db import execute_query_dict
        for table in ("rules", "rule_setup"):
            cols = {r["name"] for r in execute_query_dict(
                f"SELECT name FROM pragma_table_info('{table}')") or []}
            assert {"ce_groups", "condition"}.issubset(cols), table
        for table in ("rule_ce_link", "setup_ce_link"):
            cols = {r["name"] for r in execute_query_dict(
                f"SELECT name FROM pragma_table_info('{table}')") or []}
            assert "role" not in cols and "fallback_group" not in cols, table

    def test_search_infrastructure_present(self):
        """The SQLite replacements for the old Postgres extensions must exist:
        FTS5 side-tables (replaced tsvector/pg_trgm search) and enforced
        foreign keys (replaced Postgres' always-on FK enforcement)."""
        from utils.sqlite_db import execute_query_dict, get_connection
        result = execute_query_dict(
            "SELECT name FROM sqlite_master WHERE name IN ('rules_fts', 'ces_fts')"
        )
        fts = {r["name"] for r in result or []}
        assert "rules_fts" in fts, "rules_fts FTS5 table required for keyword search"
        assert "ces_fts" in fts, "ces_fts FTS5 table required for keyword search"
        # Cascades depend on this pragma being ON for every connection.
        fk = get_connection().execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1, "foreign_keys pragma must be ON (cascade chains depend on it)"
