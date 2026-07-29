"""Local user table helpers.

There is no login and no central user directory. The `users` table now holds:

  * the single local user (`utils.auth.LOCAL_USER_ID`), seeded by
    `DButils.ensure_local_user()` — every ownership FK points at it;
  * placeholder rows for OTHER authors whose rules/CEs arrive from the public
    HuggingFace registry, created by `ensure_creators_in_local()` so
    attribution JOINs ("by @someone") and their profile pages resolve.

Those placeholder rows carry a username and nothing else — display name, bio
and email used to come from the central server, which no longer exists. They
are purely a local index of "who authored the content I've synced".
"""
from utils.sqlite_db import execute_query, execute_query_dict


def get_user_by_id(user_id: int):
    """Read a user from the local table."""
    result = execute_query_dict(
        "SELECT user_id, username, email, tutorial_seen FROM users WHERE user_id = %s",
        (user_id,),
    )
    return result[0] if result else None


def ensure_creators_in_local(usernames: list):
    """After an HF sync imports rules/CEs, make sure every `created_by_username`
    they reference exists locally, so attribution JOINs and profile lookups
    resolve.

    Placeholder rows get a synthetic negative-free id from a dedicated sequence
    range: we allocate ids ABOVE the local user rather than reusing the central
    server's ids (which no longer exist). Nothing FKs to these rows — they are
    only ever read by name — so the id is an implementation detail.

    Idempotent and best-effort: a name that already exists is skipped, and any
    failure is swallowed (attribution degrades to a bare username, which the UI
    already handles).
    """
    if not usernames:
        return

    unique = sorted({u.strip().lower() for u in usernames if u and str(u).strip()})
    if not unique:
        return

    existing = execute_query_dict(
        "SELECT LOWER(username) AS u FROM users WHERE LOWER(username) = ANY(%s)",
        (unique,),
    ) or []
    have = {r["u"] for r in existing}
    missing = [u for u in unique if u not in have]
    if not missing:
        return

    for name in missing:
        try:
            # user_id is generated locally: MAX(user_id)+1, floored at 1000 so
            # placeholders never collide with LOCAL_USER_ID (1) or look like a
            # "real" account. email is synthesized because the column is UNIQUE
            # NOT NULL and we have no real address for a remote author.
            # MAX(a, b) is SQLite's scalar GREATEST; the inner MAX(user_id) is
            # the aggregate. `WHERE TRUE` is required: SQLite refuses an upsert
            # clause after a SELECT source unless the SELECT has a WHERE
            # (parser-ambiguity rule).
            execute_query(
                """
                INSERT INTO users (user_id, username, email, password, is_team)
                SELECT MAX(1000, COALESCE(MAX(user_id), 0) + 1), %s, %s, '', FALSE
                FROM users
                WHERE TRUE
                ON CONFLICT DO NOTHING
                """,
                (name, f"{name}@registry.local"),
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Could not register library author {name!r} locally: {e}"
            )
