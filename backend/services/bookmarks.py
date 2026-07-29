"""Bookmark CRUD service — LOCAL tables.

Bookmarks used to be stored on the central server, keyed by the HuggingFace
`public_id`, so they would follow a user's ACCOUNT between machines. With login
removed there is no account and no second machine, so they are plain rows in the
local database keyed by the local SERIAL id.

Two consequences of that, both improvements for a single-user install:

  * You can bookmark a DRAFT. The central version couldn't — an unpublished
    rule/CE has no `public_id`, so `add()` used to reject it outright.
  * No network hop. Listing bookmarks is one SQL join instead of an HTTP
    round-trip plus a join.
"""
from dataclasses import dataclass
from typing import List

from utils.auth import LOCAL_USER_ID
from utils.sqlite_db import execute_query, execute_query_dict


class BookmarkLookupError(Exception):
    """Raised when a local asset id doesn't resolve to a row. Routes turn this
    into a 400. (It no longer fires for "unpublished draft" — that is now
    allowed — only for an id that genuinely doesn't exist.)"""


@dataclass(frozen=True)
class BookmarkSpec:
    asset_type: str         # "rule" | "ce" | "rule_set"
    asset_table: str        # "rules" | "cognitive_elements" | "rule_sets"
    asset_id_col: str       # "rule_id" | "ce_id" | "rule_set_id"
    bookmark_table: str     # "rule_bookmarks" | "ce_bookmarks" | "rule_set_bookmarks"
    list_columns: str       # comma-separated columns to surface in list view


BOOKMARK_REGISTRY: dict = {
    "rule": BookmarkSpec(
        asset_type="rule",
        asset_table="rules",
        asset_id_col="rule_id",
        bookmark_table="rule_bookmarks",
        list_columns="a.name, a.predicate, a.description",
    ),
    "ce": BookmarkSpec(
        asset_type="ce",
        asset_table="cognitive_elements",
        asset_id_col="ce_id",
        bookmark_table="ce_bookmarks",
        list_columns="a.name, a.definition, a.category",
    ),
    "rule_set": BookmarkSpec(
        asset_type="rule_set",
        asset_table="rule_sets",
        asset_id_col="rule_set_id",
        bookmark_table="rule_set_bookmarks",
        list_columns="a.name, a.description",
    ),
}


def _spec(asset_type: str) -> BookmarkSpec:
    spec = BOOKMARK_REGISTRY.get(asset_type)
    if not spec:
        raise ValueError(f"Unknown bookmark asset type: {asset_type!r}")
    return spec


def _assert_exists(spec: BookmarkSpec, asset_id: int) -> None:
    rows = execute_query_dict(
        f"SELECT 1 AS ok FROM {spec.asset_table} WHERE {spec.asset_id_col} = %s",
        (asset_id,),
    ) or []
    if not rows:
        raise BookmarkLookupError(f"{spec.asset_type} #{asset_id} not found")


class BookmarkService:
    """Bookmark API for the single local user.

    Every method takes the asset id only — there is one user, so there is
    nothing to disambiguate.
    """

    @staticmethod
    def add(asset_type: str, asset_id: int) -> bool:
        spec = _spec(asset_type)
        _assert_exists(spec, asset_id)
        # Idempotent: re-bookmarking is a no-op rather than a PK violation.
        execute_query(
            f"INSERT INTO {spec.bookmark_table} (user_id, {spec.asset_id_col}) "
            f"VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (LOCAL_USER_ID, asset_id),
        )
        return True

    @staticmethod
    def remove(asset_type: str, asset_id: int) -> bool:
        spec = _spec(asset_type)
        execute_query(
            f"DELETE FROM {spec.bookmark_table} "
            f"WHERE user_id = %s AND {spec.asset_id_col} = %s",
            (LOCAL_USER_ID, asset_id),
        )
        return True

    @staticmethod
    def list(asset_type: str) -> List[dict]:
        """Bookmarked rows, most recently bookmarked first.

        `public_id` is still surfaced (NULL for drafts) because the UI uses it
        to decide whether an item can be shared / forked.
        """
        spec = _spec(asset_type)
        return execute_query_dict(
            f"""
            SELECT a.{spec.asset_id_col}, a.public_id, {spec.list_columns}
            FROM {spec.bookmark_table} b
            JOIN {spec.asset_table} a ON a.{spec.asset_id_col} = b.{spec.asset_id_col}
            WHERE b.user_id = %s
            ORDER BY b.created_at DESC
            """,
            (LOCAL_USER_ID,),
        ) or []
