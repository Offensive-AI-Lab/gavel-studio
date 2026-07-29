"""SQLite storage engine for GAVEL Studio.

Replaces the PostgreSQL server with a single database file (default:
backend/db/gavel.sqlite3, override with DB_PATH). The public API is
call-compatible with the old psycopg2-era helpers so the ~40 modules
that import execute_query / execute_query_dict / execute_update keep working:

  * SQL written in the psycopg2 dialect is translated on the fly:
      - %s / %(name)s placeholders          -> ? / :name
      - `col = ANY(%s)` with a list param   -> `col IN (?, ?, ...)`
      - now()                               -> CURRENT_TIMESTAMP
      - now() +/- interval 'N unit'         -> datetime('now', '+/-N unit')
      - ILIKE                               -> LIKE (case-insensitive in SQLite)
      - ::jsonb / ::text / ... casts        -> stripped (SQLite is dynamically typed)
  * Values round-trip like psycopg2:
      - dict / list / tuple params are stored as JSON text; columns declared
        JSONB come back as Python dicts/lists (PARSE_DECLTYPES converter)
      - datetime params are stored as UTC ISO text; TIMESTAMP/TIMESTAMPTZ
        columns come back as datetime objects
      - BOOLEAN columns come back as real Python bools
  * Concurrency: one connection per thread (SQLite connections are not
    thread-safe), WAL journal mode so readers never block behind the single
    writer, busy_timeout so concurrent writers queue instead of erroring.
  * PRAGMA foreign_keys=ON is set on EVERY connection — SQLite ships with FK
    enforcement off, and the schema's ON DELETE CASCADE chains depend on it.

Autocommit per statement (isolation_level=None) mirrors the old pool's
commit-per-call behavior; there were never multi-statement transactions.
"""
import json
import os
import re
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the backend directory (same file the old module loaded).
_BACKEND_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=_BACKEND_DIR / ".env")


def _resolve_db_path() -> Path:
    """DB file location. DB_PATH may be absolute or relative to backend/.
    Default keeps everything in a repo-local `db/` folder."""
    raw = os.getenv("DB_PATH", "db/gavel.sqlite3")
    p = Path(raw)
    if not p.is_absolute():
        p = _BACKEND_DIR / p
    return p


DB_PATH = _resolve_db_path()


# ---------------------------------------------------------------------------
# Value adapters (Python -> SQLite) and converters (SQLite -> Python)
# ---------------------------------------------------------------------------

def _adapt_datetime(dt: datetime) -> str:
    # Store UTC, naive, space-separated — the same shape CURRENT_TIMESTAMP
    # produces, so string comparison between stored values and
    # CURRENT_TIMESTAMP expressions stays correct.
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat(sep=" ")


def _convert_timestamp(raw: bytes):
    text = raw.decode("utf-8")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return text
    # Return AWARE UTC datetimes, like psycopg2 did for timestamptz columns.
    # This keeps every DB datetime comparable with every other (all aware) AND
    # keeps FastAPI's JSON serialization emitting the '+00:00' suffix —
    # without it, a browser's `new Date('2026-07-29T10:00:05')` interprets the
    # string in LOCAL time and every displayed timestamp shifts by the user's
    # UTC offset. Stored text is UTC whether it carries an offset or not.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _convert_json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return raw.decode("utf-8", errors="replace")


def _convert_bool(raw: bytes) -> bool:
    text = raw.decode("utf-8").strip().lower()
    if text in ("true", "t"):
        return True
    if text in ("false", "f", ""):
        return False
    try:
        return bool(int(text))
    except ValueError:
        return bool(text)


sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_adapter(date, lambda d: d.isoformat())
sqlite3.register_adapter(dict, lambda v: json.dumps(v))
sqlite3.register_adapter(list, lambda v: json.dumps(v))
sqlite3.register_adapter(tuple, lambda v: json.dumps(list(v)))

# Declared-type -> converter. sqlite3 matches on the first word of the
# column's declared type, case-insensitively.
sqlite3.register_converter("TIMESTAMP", _convert_timestamp)
sqlite3.register_converter("TIMESTAMPTZ", _convert_timestamp)
sqlite3.register_converter("DATE", _convert_timestamp)
sqlite3.register_converter("JSONB", _convert_json)
sqlite3.register_converter("JSON", _convert_json)
sqlite3.register_converter("BOOLEAN", _convert_bool)


# ---------------------------------------------------------------------------
# Connections — one per thread, autocommit, WAL, FKs on
# ---------------------------------------------------------------------------

_local = threading.local()
_init_lock = threading.Lock()
_db_dir_ready = False


def _connect() -> sqlite3.Connection:
    global _db_dir_ready
    if not _db_dir_ready:
        with _init_lock:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            _db_dir_ready = True
    conn = sqlite3.connect(
        str(DB_PATH),
        # PARSE_DECLTYPES: converters keyed on the column's declared type
        # (JSONB/TIMESTAMPTZ/BOOLEAN). PARSE_COLNAMES additionally lets a
        # computed column opt into a converter via its alias, e.g.
        #   json_group_array(c.name) AS "categories [JSONB]"
        # — the suffix is stripped from the returned column name.
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        timeout=30.0,           # block up to 30s on a locked database
        isolation_level=None,   # autocommit — one implicit txn per statement
    )
    conn.execute("PRAGMA foreign_keys = ON")       # load-bearing: cascades
    # WAL lets readers proceed while the single writer works. On network
    # filesystems (NFS/SMB — common for lab home dirs) WAL's shared-memory
    # file doesn't work; SQLite then keeps the rollback journal instead of
    # erroring. Surface that so "database is locked" reports are explainable,
    # and point DB_PATH at local disk for best behavior.
    mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if str(mode).lower() != "wal":
        print(
            f"[sqlite] WAL unavailable on this filesystem (journal_mode={mode}) — "
            f"using rollback journal. If {DB_PATH} is on network storage, set "
            f"DB_PATH to a local-disk path for better concurrency."
        )
    conn.execute("PRAGMA synchronous = NORMAL")    # safe with WAL, much faster
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def get_connection() -> sqlite3.Connection:
    """Return this thread's connection, creating it on first use."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


def release_connection(conn) -> None:
    """No-op kept for pool-API compatibility (connections are thread-local
    and reused; they close when the process exits)."""


def close_pool() -> None:
    """Close the calling thread's connection (compat shim)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        finally:
            _local.conn = None


# ---------------------------------------------------------------------------
# psycopg2-dialect translation
# ---------------------------------------------------------------------------

_NAMED_PARAM_RE = re.compile(r"%\(([A-Za-z_][A-Za-z0-9_]*)\)s")
_PLACEHOLDER_RE = re.compile(r"%%|%s|%\(([A-Za-z_][A-Za-z0-9_]*)\)s")
_NOW_RE = re.compile(r"\bnow\(\)", re.IGNORECASE)
_ILIKE_RE = re.compile(r"\bILIKE\b", re.IGNORECASE)
_CAST_RE = re.compile(r"::[A-Za-z_]+(?:\s+precision)?(?:\(\d+\))?(?:\[\])?", re.IGNORECASE)
_INTERVAL_RE = re.compile(
    r"CURRENT_TIMESTAMP\s*([+-])\s*interval\s*'([^']+)'", re.IGNORECASE
)
_ANY_RE = re.compile(r"(=|!=|<>)\s*ANY\s*\(\s*\?\s*\)", re.IGNORECASE)
def _split_literals(sql: str):
    """Split SQL into (is_protected, text) segments so transforms can skip
    quoted string literals and comments.

    A literal '%s', '?', 'ILIKE' or '::' inside a string must never be
    rewritten — and neither must anything inside a comment (an apostrophe in
    a comment like `-- the rule's flag` must NOT open a string literal, and a
    '?' in a comment is not a bind slot). Proper state machine: normal code /
    'literal' (with '' escapes) / -- line comment / block comment."""
    segments = []
    n = len(sql)
    pos = 0          # start of the current segment
    i = 0
    state = "code"   # code | literal | line_comment | block_comment

    def flush(upto, protected):
        nonlocal pos
        if upto > pos:
            segments.append((protected, sql[pos:upto]))
        pos = upto

    while i < n:
        ch = sql[i]
        if state == "code":
            if ch == "'":
                flush(i, False)
                state = "literal"
                i += 1
            elif ch == "-" and sql.startswith("--", i):
                flush(i, False)
                state = "line_comment"
                i += 2
            elif ch == "/" and sql.startswith("/*", i):
                flush(i, False)
                state = "block_comment"
                i += 2
            else:
                i += 1
        elif state == "literal":
            if ch == "'":
                if sql.startswith("''", i):  # escaped quote inside literal
                    i += 2
                else:
                    flush(i + 1, True)
                    state = "code"
                    i += 1
            else:
                i += 1
        elif state == "line_comment":
            if ch == "\n":
                flush(i + 1, True)
                state = "code"
            i += 1
        else:  # block_comment
            if ch == "*" and sql.startswith("*/", i):
                i += 2
                flush(i, True)
                state = "code"
            else:
                i += 1
    flush(n, state != "code")
    return segments


def _sub_outside_literals(sql: str, fn):
    """Apply `fn` to the non-literal segments of `sql` only."""
    return "".join(
        text if is_lit else fn(text) for is_lit, text in _split_literals(sql)
    )


def _replace_placeholders(text: str) -> str:
    # One left-to-right pass so '%%' always wins over '%s' ('%%s' is the
    # psycopg2 escape for a literal '%s' and must become '%s', not '%?').
    def repl(m):
        tok = m.group(0)
        if tok == "%%":
            return "%"
        if tok == "%s":
            return "?"
        return f":{m.group(1)}"  # %(name)s
    return _PLACEHOLDER_RE.sub(repl, text)


def _translate(sql: str, params):
    """Translate psycopg2-flavored SQL + params into sqlite3 form.

    Every rewrite skips single-quoted string literals. (SQL '--' comments are
    NOT skipped — don't put placeholder-like or cast-like text in comments.)
    """
    if params is not None:
        sql = _sub_outside_literals(sql, _replace_placeholders)

    sql = _sub_outside_literals(sql, lambda t: _NOW_RE.sub("CURRENT_TIMESTAMP", t))
    sql = _sub_outside_literals(sql, lambda t: _ILIKE_RE.sub("LIKE", t))
    sql = _INTERVAL_RE.sub(lambda m: f"datetime('now', '{m.group(1)}{m.group(2)}')", sql)
    sql = _sub_outside_literals(sql, lambda t: _CAST_RE.sub("", t))

    if params is not None and not isinstance(params, dict):
        params = list(params)
        sql, params = _expand_any(sql, params)
        params = [_normalize_param(p) for p in params]

    return sql, params


def _placeholder_count_before(sql: str, offset: int) -> int:
    """Number of REAL bind placeholders ('?' outside string literals) in
    sql[:offset]."""
    count = 0
    pos = 0
    for is_lit, text in _split_literals(sql):
        end = pos + len(text)
        if not is_lit:
            count += text.count("?", 0, min(len(text), max(0, offset - pos)))
        if end >= offset:
            break
        pos = end
    return count


def _in_literal(sql: str, offset: int) -> bool:
    pos = 0
    for is_lit, text in _split_literals(sql):
        end = pos + len(text)
        if pos <= offset < end:
            return is_lit
        pos = end
    return False


def _expand_any(sql: str, params: list):
    """Rewrite `col = ANY(?)` (list param) as `col IN (?, ?, ...)`.

    Postgres `= ANY(array)` with an empty array matches nothing; `IN (NULL)`
    reproduces that. `!= ANY` / `<> ANY` has different semantics than NOT IN
    and is unused in this codebase — refuse loudly rather than mistranslate.
    The parameter index counts only real placeholders (a '?' inside a string
    literal is data, not a bind slot).
    """
    search_from = 0
    while True:
        m = _ANY_RE.search(sql, search_from)
        if m is None:
            return sql, params
        if _in_literal(sql, m.start()):
            search_from = m.end()
            continue
        if m.group(1) != "=":
            raise ValueError("'!= ANY(...)' has no direct SQLite translation")
        idx = _placeholder_count_before(sql, m.start())
        values = params[idx]
        if values is None:
            values = []
        if not isinstance(values, (list, tuple, set)):
            raise ValueError("= ANY(%s) expects a list/tuple/set parameter")
        values = list(values)
        placeholders = ", ".join("?" for _ in values) if values else "NULL"
        sql = sql[: m.start()] + f"IN ({placeholders})" + sql[m.end():]
        params = params[:idx] + values + params[idx + 1:]
        search_from = 0


def _normalize_param(value):
    if isinstance(value, set):
        return json.dumps(sorted(value))
    return value


_MULTI_STMT_MSG = "one statement at a time"


def _run(cursor, sql: str, params):
    sql, params = _translate(sql, params)
    try:
        if params is not None:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
    except sqlite3.ProgrammingError as e:
        # psycopg2 allowed several ';'-separated statements per execute (the
        # DDL blocks use this). sqlite3 doesn't — fall back to executescript,
        # which is fine because multi-statement strings never carry params
        # or expect results.
        if _MULTI_STMT_MSG in str(e) and params is None:
            cursor.executescript(sql)
        else:
            raise


# ---------------------------------------------------------------------------
# Public helpers (same signatures/behavior as the old psycopg2-era module)
# ---------------------------------------------------------------------------

def execute_query(query, params=None):
    """Execute a statement; return list-of-tuples for SELECTs, None otherwise."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        _run(cursor, query, params)
        if cursor.description:
            results = [tuple(row) for row in cursor.fetchall()]
        else:
            results = None
        cursor.close()
        return results
    except Exception as e:
        print(f"Error executing query: {e}")
        raise


def execute_update(query, params=None):
    """Execute an INSERT/UPDATE/DELETE and return the number of affected rows.

    Exposes rowcount so callers can tell whether a conditional write actually
    hit a row — e.g. detecting that a guardrail was deleted out from under an
    in-flight write (a guarded `UPDATE ... WHERE classifier_id = %s` that
    affects 0 rows means the row is gone)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        _run(cursor, query, params)
        n = cursor.rowcount
        cursor.close()
        return n
    except Exception as e:
        print(f"Error executing update: {e}")
        raise


def execute_query_dict(query, params=None):
    """Execute a statement; return list-of-dicts for SELECTs, None otherwise."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.row_factory = sqlite3.Row
        _run(cursor, query, params)
        if cursor.description:
            results = [dict(row) for row in cursor.fetchall()]
        else:
            results = None
        cursor.close()
        return results
    except Exception as e:
        print(f"Error executing query: {e}")
        raise
