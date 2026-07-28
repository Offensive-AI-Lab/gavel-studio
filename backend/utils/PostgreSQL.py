from psycopg2 import pool
from psycopg2 import extras
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from backend directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# ---------------------------------------------------------------------------
# Local database (models, guardrails, training data, rules, CEs, etc.)
# ---------------------------------------------------------------------------
config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": os.getenv("DB_PORT")
}

_connection_pool = None


# Max pooled connections. ThreadedConnectionPool (NOT SimpleConnectionPool) is
# required because the server runs DB work on many threads at once — Starlette's
# sync-handler threadpool, BackgroundTasks, and several ThreadPoolExecutors.
# SimpleConnectionPool isn't thread-safe and can hand one connection to two
# threads under load. Optional DB_POOL_MAX override; sized for concurrency.
_POOL_MAX = int(os.getenv("DB_POOL_MAX", "30"))


def _get_pool():
    """Return the singleton local connection pool, creating it on first call."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(1, _POOL_MAX, **config)
    return _connection_pool


def get_connection():
    """Get connection from local pool"""
    return _get_pool().getconn()

def release_connection(conn):
    """Release connection back to local pool"""
    if conn:
        _get_pool().putconn(conn)


# ---------------------------------------------------------------------------
# Local-only helpers
# Auth and ratings were removed with multi-user support; bookmarks live in
# this local database (services/bookmarks.py). Direct PostgreSQL remote-pool access
# is no longer needed here.
# ---------------------------------------------------------------------------

def execute_query(query, params=None):
    """Execute query against the local database."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        if cursor.description:
            results = cursor.fetchall()
        else:
            results = None

        conn.commit()
        cursor.close()
        return results
    except Exception as e:
        conn.rollback()
        print(f"Error executing query: {e}")
        raise e
    finally:
        release_connection(conn)

def execute_update(query, params=None):
    """Execute an INSERT/UPDATE/DELETE and return the number of affected rows.

    Unlike execute_query (which returns None for non-SELECT statements), this
    exposes cursor.rowcount so callers can tell whether a conditional write
    actually hit a row — e.g. detecting that a guardrail was deleted out from
    under an in-flight write (a guarded `UPDATE ... WHERE classifier_id = %s`
    that affects 0 rows means the row is gone)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        n = cursor.rowcount
        conn.commit()
        cursor.close()
        return n
    except Exception as e:
        conn.rollback()
        print(f"Error executing update: {e}")
        raise e
    finally:
        release_connection(conn)


def execute_query_dict(query, params=None):
    """Execute query against the local database, returning dicts."""
    conn = get_connection()
    try:
        cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        if cursor.description:
            results = cursor.fetchall()
        else:
            results = None

        conn.commit()
        cursor.close()
        return results
    except Exception as e:
        conn.rollback()
        print(f"Error executing query: {e}")
        raise e
    finally:
        release_connection(conn)

def close_pool():
    """Close the local connection pool."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
        print("Local PostgreSQL connection pool closed")
