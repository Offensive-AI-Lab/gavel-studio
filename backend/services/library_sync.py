"""gavel-rules registry sync — the read/ingest side of the public library.

Replaces the retired HuggingFace sync service. The registry is
the gavel-rules GitHub repository (Offensive-AI-Lab/gavel-rules): a CI-
validated library of schema-v2 artifacts. The studio consumes exactly two
kinds of files, both JSON — **no YAML parsing**:

  * ``index.json``  — the generated catalog (name-keyed rules/ces/rulesets
    with groups, condition, role, definitions, categories, per-record
    content_hash and the repo-wide global_signature). CI rebuilds it on every
    merge to main, so it is always fresh and always consistent with the tree.
  * dataset files — ``ces/<name>/{excitation,calibration}.json`` and
    ``rules/<name>/tests/{positive,negative,positive_calibration}.json`` —
    fetched LAZILY at the moment training/calibration/evaluation first needs
    them (same contract as before: ``ensure_*`` fast-path on the local row).

Freshness probe = repo HEAD commit SHA (one API call) vs the
``last_synced_sha`` cached in sync_state; per-record staleness = the
``content_hash`` stored per artifact at last sync. All reads go through the
RegistryReader port (services/registry_sync/reader.py — GitHubReader), pinned
to the probed SHA so one sync run sees a consistent snapshot.

Sync NEVER writes to the repo — contributions to gavel-rules go by pull
request outside the studio. Local drafts are sacred: an incoming record whose
name collides with a local draft is skipped, never clobbered.
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

from utils.sqlite_db import execute_query, execute_query_dict
from utils.DButils import normalize_and_upsert_categories
from utils.embedding_utils import trigger_embedding
from utils.rule_condition import ConditionError, parse, render_predicate, validate_condition

logger = logging.getLogger(__name__)

_sync_lock = threading.Lock()

_STATE_SHA_KEY = "last_synced_sha"
_STATE_HASHES_KEY = "record_hashes"

_RULE_DEFAULT_TYPES = ("positive", "negative", "positive_calibration")
# Stamped into dataset envelopes the sync writes, so a later refresh can
# tell its own cached copy apart from one the user generated locally.
_DATASET_SOURCE = "gavel-rules"
_PULL_WORKERS = 8


# --- Result type -----------------------------------------------------------


@dataclass
class SyncResult:
    """What sync_library() reports back. `changed` is False when the cheap
    HEAD-SHA probe found nothing new — all counters stay zero.

    `*_added` counts records missing locally that got pulled; `*_refreshed`
    counts records that existed locally but changed upstream (content_hash
    mismatch) and were re-pulled in place."""
    changed: bool
    ces_added: int = 0
    rules_added: int = 0
    rule_sets_added: int = 0
    ces_refreshed: int = 0
    rules_refreshed: int = 0
    rule_sets_refreshed: int = 0
    categories_synced: int = 0
    skipped_records: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "changed": self.changed,
            "ces_added": self.ces_added,
            "rules_added": self.rules_added,
            "rule_sets_added": self.rule_sets_added,
            "ces_refreshed": self.ces_refreshed,
            "rules_refreshed": self.rules_refreshed,
            "rule_sets_refreshed": self.rule_sets_refreshed,
            "categories_synced": self.categories_synced,
            "skipped_records": self.skipped_records,
            "errors": self.errors,
        }


# --- Reader + sync_state helpers -------------------------------------------


_READER = None


def _reader():
    """Lazily-built RegistryReader (the GitHub adapter). The backend reads the
    library THROUGH this port, never an HTTP client directly."""
    global _READER
    if _READER is None:
        from services.registry_sync.reader import build_reader
        _READER = build_reader()
    return _READER


def _get_state(key: str) -> Optional[str]:
    rows = execute_query_dict(
        "SELECT value FROM sync_state WHERE key = %s", (key,)
    ) or []
    return rows[0]["value"] if rows else None


def _set_state(key: str, value: str) -> None:
    execute_query(
        """
        INSERT INTO sync_state (key, value, updated_at)
        VALUES (%s, %s, now())
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value, updated_at = now()
        """,
        (key, value),
    )


def _stored_hashes() -> dict:
    try:
        data = json.loads(_get_state(_STATE_HASHES_KEY) or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# --- Freshness probe --------------------------------------------------------


def check_for_updates() -> dict:
    """Cheap "is there new content in the registry?" probe.

    Compares the repo's HEAD commit SHA against the locally-cached
    `last_synced_sha`. One API call, nothing pulled — safe on a poller tick.

    Return shape (consumed by the sidebar badge via registry_sync.wiring and
    GET /library/check-updates):
      {"available": bool, "checked": bool, "reason": str | None}

    `checked=False` paths return `available=False` so a transient outage
    doesn't pin the badge in its "updates" state."""
    try:
        head = _reader().head_version()
    except Exception as exc:
        return {"available": False, "checked": False, "reason": f"registry probe failed: {exc}"}
    available = _get_state(_STATE_SHA_KEY) != head
    return {"available": available, "checked": True, "reason": None}


# --- Reference hygiene ------------------------------------------------------


def _assert_local_refs(refs, what: str) -> None:
    """The v2 format allows namespace-qualified references (gavel/ce/x). The
    current registry uses only bare names; per FORMAT.md the INGESTING tool
    must hard-error on external-namespace dependencies rather than silently
    skip them at resolution time."""
    for ref in refs or []:
        if "/" in str(ref):
            raise ValueError(
                f"{what} references '{ref}' in a foreign namespace — "
                f"cross-namespace dependencies are not supported by this studio yet"
            )


# --- Upserts (record row + wiring) ------------------------------------------


def _rename_holder(table: str, id_col: str, public_id: str, new_name: str):
    """Upstream renames keep the permanent public_id but change the name —
    the name-keyed upsert would then INSERT a second row and trip the
    public_id partial-unique index. Detect the case up front: return the
    existing row (id + old name) when this public_id is already held under a
    DIFFERENT name, so the caller can UPDATE that row in place instead."""
    if not public_id:
        return None
    rows = execute_query_dict(
        f"SELECT {id_col} AS row_id, name FROM {table} WHERE public_id = %s",
        (public_id,),
    ) or []
    if rows and rows[0]["name"] != new_name:
        return rows[0]
    return None


def _propagate_ce_rename(old_name: str, new_name: str) -> None:
    """A renamed CE keeps its ce_id, so membership links survive — but
    ce_groups store member NAMES. Patch every local rule/setup whose groups
    reference the old name and re-render its predicate, so legacy local
    rules stay coherent (registry rules get fully refreshed this same sync
    anyway)."""
    for table, id_col in (("rules", "rule_id"), ("rule_setup", "setup_id")):
        rows = execute_query_dict(
            f"SELECT {id_col} AS row_id, ce_groups, condition FROM {table} "
            f"WHERE ce_groups IS NOT NULL"
        ) or []
        for row in rows:
            groups = row["ce_groups"] if isinstance(row["ce_groups"], dict) else None
            if not groups or not any(old_name in (m or []) for m in groups.values()):
                continue
            patched = {
                g: [new_name if m == old_name else m for m in (members or [])]
                for g, members in groups.items()
            }
            predicate = None
            condition = (row.get("condition") or "").strip()
            if condition:
                try:
                    predicate = render_predicate(parse(condition), patched)
                except (ConditionError, KeyError):
                    predicate = None
            if predicate is not None:
                execute_query(
                    f"UPDATE {table} SET ce_groups = %s, predicate = %s WHERE {id_col} = %s",
                    (patched, predicate, row["row_id"]),
                )
            else:
                execute_query(
                    f"UPDATE {table} SET ce_groups = %s WHERE {id_col} = %s",
                    (patched, row["row_id"]),
                )


def _upsert_ce_from_index(name: str, entry: dict) -> Optional[int]:
    """Insert/update one CE row from its index.json entry. Returns the local
    ce_id, or None when a same-named local draft blocks the pull (drafts are
    sacred). The v2 CE `role` (closed enum) lands in its own column AND in
    the legacy category/categories columns so Browse facets stay populated.
    An upstream rename (same public_id, new name) renames the existing row
    in place — ce_id is stable, and local groups referencing the old name
    are patched via _propagate_ce_rename."""
    draft = execute_query_dict(
        "SELECT ce_id FROM cognitive_elements WHERE name = %s AND is_local_draft = TRUE",
        (name,),
    )
    if draft:
        logger.warning(
            "[library_sync] CE '%s' collides with a local draft (ce_id=%d); "
            "draft left alone, registry record not pulled",
            name, draft[0]["ce_id"],
        )
        return None

    renamed = _rename_holder("cognitive_elements", "ce_id", entry.get("public_id"), name)
    if renamed:
        logger.info(
            "[library_sync] CE '%s' renamed upstream to '%s' (same public_id) — "
            "renaming in place", renamed["name"], name,
        )
        execute_query(
            "UPDATE cognitive_elements SET name = %s WHERE ce_id = %s",
            (name, renamed["row_id"]),
        )
        _propagate_ce_rename(renamed["name"], name)

    role = entry.get("role")
    rows = execute_query_dict(
        """
        INSERT INTO cognitive_elements (
            name, title, definition, role, category, categories, tags,
            examples, public_id, published_at, is_local_draft, created_by_username
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s)
        ON CONFLICT (name) DO UPDATE
        SET title            = EXCLUDED.title,
            definition       = EXCLUDED.definition,
            role             = EXCLUDED.role,
            category         = EXCLUDED.category,
            categories       = EXCLUDED.categories,
            tags             = EXCLUDED.tags,
            examples         = EXCLUDED.examples,
            public_id        = EXCLUDED.public_id,
            published_at     = EXCLUDED.published_at,
            is_local_draft   = FALSE,
            created_by_username = COALESCE(cognitive_elements.created_by_username, EXCLUDED.created_by_username)
        RETURNING ce_id
        """,
        (
            name,
            entry.get("title"),
            entry.get("definition"),
            role,
            role or "CONTEXT",
            json.dumps([role] if role else []),
            json.dumps(entry.get("tags") or []),
            json.dumps(entry.get("examples") or []),
            entry.get("public_id"),
            entry.get("published_at"),
            entry.get("by"),
        ),
    )
    return rows[0]["ce_id"]


def _is_registry_sourced(table: str, ce_id: int) -> bool:
    """Was this CE's stored dataset written by the sync, or by the user?

    excitation_datasets / calibration_datasets hold ONE row per CE with no
    owner column: a registry pull and a locally-generated dataset (the AI
    pipeline's CE training-data step) land in the same slot. Sync-written
    envelopes carry a `source` marker, so anything without it is treated as
    the user's — never discarded. Rows written before the marker existed are
    therefore also preserved (fail-safe direction)."""
    rows = execute_query_dict(
        f"SELECT dataset FROM {table} WHERE ce_id = %s", (ce_id,)) or []
    if not rows:
        return False
    try:
        payload = rows[0]["dataset"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return (payload or {}).get("source") == _DATASET_SOURCE
    except Exception:
        return False


def _pull_ce(name: str, entry: dict, refreshed: bool) -> Optional[int]:
    ce_id = _upsert_ce_from_index(name, entry)
    if ce_id is None:
        return None
    if refreshed:
        # An artifact's content_hash covers its dataset files too, so an
        # upstream edit should invalidate the cached copy and let the next
        # ensure_* re-pull it. Only drop copies WE fetched — a dataset the
        # user generated locally for this CE is their work and stays put
        # (sync is additive; it never destroys local content).
        for table in ("excitation_datasets", "calibration_datasets"):
            if _is_registry_sourced(table, ce_id):
                execute_query(f"DELETE FROM {table} WHERE ce_id = %s", (ce_id,))
            else:
                rows = execute_query_dict(
                    f"SELECT 1 FROM {table} WHERE ce_id = %s", (ce_id,))
                if rows:
                    logger.info(
                        "[library_sync] CE '%s' changed upstream but its local "
                        "%s was not written by sync — keeping the local copy",
                        name, table,
                    )
    try:
        trigger_embedding("ce", ce_id, name, entry.get("definition") or "")
    except Exception as embed_err:
        logger.warning("[library_sync] embedding failed for CE %s: %s", name, embed_err)
    return ce_id


def _upsert_rule_from_index(name: str, entry: dict) -> Optional[int]:
    """Insert/update one rule row (+ membership links) from its index.json
    entry. The stored logic is (ce_groups, condition); the predicate column
    is re-derived locally with render_predicate — never trusted from the
    registry. Raises on invalid logic or unresolvable CE names (CEs sync
    first, so a miss means a genuinely broken record)."""
    draft = execute_query_dict(
        "SELECT rule_id FROM rules WHERE name = %s AND is_local_draft = TRUE",
        (name,),
    )
    if draft:
        logger.warning(
            "[library_sync] rule '%s' collides with a local draft (rule_id=%d); "
            "draft left alone, registry record not pulled",
            name, draft[0]["rule_id"],
        )
        return None

    renamed = _rename_holder("rules", "rule_id", entry.get("public_id"), name)
    if renamed:
        logger.info(
            "[library_sync] rule '%s' renamed upstream to '%s' (same public_id) — "
            "renaming in place", renamed["name"], name,
        )
        execute_query(
            "UPDATE rules SET name = %s WHERE rule_id = %s",
            (name, renamed["row_id"]),
        )

    groups = entry.get("groups") or {}
    condition = entry.get("condition") or ""
    for gname, members in groups.items():
        _assert_local_refs(members, f"rule '{name}' group '{gname}'")
    problems = validate_condition(condition, groups)
    if problems:
        raise ValueError(f"rule '{name}' has invalid logic: {'; '.join(problems)}")
    predicate = render_predicate(parse(condition), groups)

    member_names = sorted({m for members in groups.values() for m in members})
    rows = execute_query_dict(
        "SELECT ce_id, name FROM cognitive_elements WHERE name = ANY(%s)",
        (member_names,),
    ) or []
    name_to_id = {r["name"]: r["ce_id"] for r in rows}
    missing = [m for m in member_names if m not in name_to_id]
    if missing:
        raise ValueError(f"rule '{name}' references unknown CE(s): {missing}")

    final_categories = normalize_and_upsert_categories(
        list(entry.get("categories") or []), allow_new=True
    )
    rule_rows = execute_query_dict(
        """
        INSERT INTO rules (
            name, title, predicate, description, categories, tags,
            ce_groups, condition, public_id, published_at,
            is_local_draft, created_by_username
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s)
        ON CONFLICT (name) DO UPDATE
        SET title            = EXCLUDED.title,
            predicate        = EXCLUDED.predicate,
            description      = EXCLUDED.description,
            categories       = EXCLUDED.categories,
            tags             = EXCLUDED.tags,
            ce_groups        = EXCLUDED.ce_groups,
            condition        = EXCLUDED.condition,
            public_id        = EXCLUDED.public_id,
            published_at     = EXCLUDED.published_at,
            is_local_draft   = FALSE,
            created_by_username = COALESCE(rules.created_by_username, EXCLUDED.created_by_username)
        RETURNING rule_id
        """,
        (
            name,
            entry.get("title"),
            predicate,
            entry.get("definition"),
            final_categories,
            json.dumps(entry.get("tags") or []),
            json.dumps(groups),
            condition,
            entry.get("public_id"),
            entry.get("published_at"),
            entry.get("by"),
        ),
    )
    rule_id = rule_rows[0]["rule_id"]

    execute_query("DELETE FROM rule_ce_link WHERE rule_id = %s", (rule_id,))
    for member in member_names:
        execute_query(
            "INSERT INTO rule_ce_link (rule_id, ce_id) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (rule_id, name_to_id[member]),
        )
    return rule_id


def _pull_rule(name: str, entry: dict, refreshed: bool) -> Optional[int]:
    rule_id = _upsert_rule_from_index(name, entry)
    if rule_id is None:
        return None
    if refreshed:
        # Invalidate lazily-cached default test sets — the content_hash
        # change may have come from an edited tests/*.json.
        execute_query(
            "DELETE FROM test_datasets WHERE rule_id = %s AND is_default = TRUE",
            (rule_id,),
        )
    try:
        member_names = sorted({m for ms in (entry.get("groups") or {}).values() for m in ms})
        trigger_embedding(
            "rule", rule_id, name,
            entry.get("predicate") or "", " ".join(member_names),
        )
    except Exception as embed_err:
        logger.warning("[library_sync] embedding failed for rule %s: %s", name, embed_err)
    return rule_id


def _upsert_rule_set_from_index(name: str, entry: dict) -> Optional[int]:
    """Insert/update one rule set + its membership. v2 rulesets reference
    member rules by NAME; members not yet present locally are skipped and
    heal on the next sync."""
    draft = execute_query_dict(
        "SELECT rule_set_id FROM rule_sets WHERE name = %s AND is_local_draft = TRUE",
        (name,),
    )
    if draft:
        logger.warning(
            "[library_sync] rule set '%s' collides with a local draft; skipped", name
        )
        return None

    renamed = _rename_holder("rule_sets", "rule_set_id", entry.get("public_id"), name)
    if renamed:
        logger.info(
            "[library_sync] rule set '%s' renamed upstream to '%s' (same public_id) — "
            "renaming in place", renamed["name"], name,
        )
        execute_query(
            "UPDATE rule_sets SET name = %s WHERE rule_set_id = %s",
            (name, renamed["row_id"]),
        )

    member_names = list(entry.get("rules") or [])
    _assert_local_refs(member_names, f"rule set '{name}'")

    rows = execute_query_dict(
        """
        INSERT INTO rule_sets (
            name, title, description, categories, public_id, published_at,
            is_local_draft, created_by_username
        )
        VALUES (%s, %s, %s, %s, %s, %s, FALSE, %s)
        ON CONFLICT (name) DO UPDATE
        SET title            = EXCLUDED.title,
            description      = EXCLUDED.description,
            categories       = EXCLUDED.categories,
            public_id        = EXCLUDED.public_id,
            published_at     = EXCLUDED.published_at,
            is_local_draft   = FALSE,
            created_by_username = COALESCE(rule_sets.created_by_username, EXCLUDED.created_by_username)
        RETURNING rule_set_id
        """,
        (
            name,
            entry.get("title"),
            entry.get("description"),
            json.dumps(entry.get("categories") or []),
            entry.get("public_id"),
            entry.get("published_at"),
            entry.get("by"),
        ),
    )
    rule_set_id = rows[0]["rule_set_id"]

    resolved = execute_query_dict(
        "SELECT rule_id, name FROM rules WHERE name = ANY(%s) AND is_local_draft = FALSE",
        (member_names,),
    ) or []
    by_name = {r["name"]: r["rule_id"] for r in resolved}
    execute_query("DELETE FROM rule_set_member WHERE rule_set_id = %s", (rule_set_id,))
    for position, member in enumerate(member_names):
        rid = by_name.get(member)
        if rid is None:
            logger.warning(
                "[library_sync] rule set '%s' member '%s' not present locally; "
                "will heal on a future sync", name, member,
            )
            continue
        execute_query(
            "INSERT INTO rule_set_member (rule_set_id, rule_id, position) "
            "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (rule_set_id, rid, position),
        )
    return rule_set_id


# --- The full sync ----------------------------------------------------------


def sync_library(force: bool = False) -> SyncResult:
    """Bring the local library up to date with the gavel-rules registry.

    Serialized by a module lock (the boot thread and a user-clicked sync must
    not double-pull). Cheap path: HEAD SHA unchanged -> no reads beyond one
    API call. Full path: fetch index.json pinned at that SHA, diff every
    artifact by content_hash, pull CEs -> rules -> rule sets (in that order:
    rules resolve CE names, sets resolve rule names), upsert categories, and
    mirror creators into the local users table. The SHA + per-record hashes
    are persisted only when nothing was skipped and nothing errored, so
    partial failures retry on the next sync instead of being masked."""
    with _sync_lock:
        return _sync_library_locked(force=force)


def _sync_library_locked(force: bool = False) -> SyncResult:
    result = SyncResult(changed=False)
    try:
        head = _reader().head_version()
    except Exception as exc:
        result.errors.append(f"registry unreachable: {exc}")
        return result

    if not force and _get_state(_STATE_SHA_KEY) == head:
        return result

    try:
        index = _reader().fetch_json("index.json", revision=head)
    except Exception as exc:
        result.errors.append(f"index.json fetch failed: {exc}")
        return result

    result.changed = True
    stored = _stored_hashes()
    new_hashes = {"ces": {}, "rules": {}, "rulesets": {}}

    # Categories first — rules attach to them. Cheap full upsert.
    for cat in index.get("categories") or []:
        cat_name = (cat.get("name") or "").strip()
        if not cat_name:
            continue
        execute_query(
            """
            INSERT INTO categories (name, description) VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE SET description = EXCLUDED.description
            """,
            (cat_name, cat.get("description")),
        )
        result.categories_synced += 1

    def _sync_family(kind: str, entries: dict, local_names: set, pull) -> tuple:
        """Diff one artifact family by content_hash and pull what changed.
        Returns (added, refreshed)."""
        added = refreshed = 0
        todo = []
        for name, entry in (entries or {}).items():
            known_hash = (stored.get(kind) or {}).get(name)
            is_local = name in local_names
            if is_local and known_hash == entry.get("content_hash"):
                new_hashes[kind][name] = known_hash  # unchanged
                continue
            todo.append((name, entry, is_local))
        if not todo:
            return added, refreshed

        def _one(item):
            name, entry, is_local = item
            try:
                row_id = pull(name, entry, is_local)
            except Exception as exc:
                result.errors.append(f"{kind}/{name}: {exc}")
                logger.warning("[library_sync] %s '%s' failed: %s", kind, name, exc)
                return name, None, is_local
            return name, row_id, is_local

        with ThreadPoolExecutor(max_workers=_PULL_WORKERS) as pool:
            outcomes = list(pool.map(_one, todo))
        for name, row_id, is_local in outcomes:
            if row_id is None:
                entry = entries[name]
                if not any(f"{kind}/{name}:" in e for e in result.errors):
                    result.skipped_records.append(f"{kind}/{name}")
                continue
            new_hashes[kind][name] = entries[name].get("content_hash")
            if is_local:
                refreshed += 1
            else:
                added += 1
        return added, refreshed

    def _local_names(table: str) -> set:
        rows = execute_query_dict(
            f"SELECT name FROM {table} WHERE public_id IS NOT NULL"
        ) or []
        return {r["name"] for r in rows}

    # CEs -> rules -> rule sets (dependency order).
    result.ces_added, result.ces_refreshed = _sync_family(
        "ces", index.get("ces"), _local_names("cognitive_elements"), _pull_ce
    )
    result.rules_added, result.rules_refreshed = _sync_family(
        "rules", index.get("rules"), _local_names("rules"), _pull_rule
    )
    result.rule_sets_added, result.rule_sets_refreshed = _sync_family(
        "rulesets", index.get("rulesets"), _local_names("rule_sets"),
        lambda n, e, r: _upsert_rule_set_from_index(n, e),
    )

    # Mirror registry authors into the local users table so attribution
    # JOINs resolve (placeholder rows, user_id >= 1000).
    try:
        creators = sorted({
            e.get("by")
            for fam in ("ces", "rules", "rulesets")
            for e in (index.get(fam) or {}).values()
            if e.get("by")
        })
        if creators:
            from sql_scripts.user_scripts import ensure_creators_in_local
            ensure_creators_in_local(creators)
    except Exception as exc:
        logger.warning("[library_sync] creator mirror failed: %s", exc)

    # Persist state only on a fully clean pass — a skip/error must make the
    # next sync retry rather than short-circuit on the SHA.
    if not result.errors and not result.skipped_records:
        _set_state(_STATE_HASHES_KEY, json.dumps(new_hashes))
        _set_state(_STATE_SHA_KEY, head)

    logger.info(
        "[library_sync] sync done: +%d/%d CEs, +%d/%d rules, +%d/%d sets, "
        "%d categories, %d skipped, %d errors",
        result.ces_added, result.ces_refreshed,
        result.rules_added, result.rules_refreshed,
        result.rule_sets_added, result.rule_sets_refreshed,
        result.categories_synced, len(result.skipped_records), len(result.errors),
    )
    return result


# --- Lazy dataset fetchers --------------------------------------------------
#
# Registry paths are NAME-addressed (ces/<name>/excitation.json). The fast
# path is always "local row exists". Signatures preserved from the retired
# HF sync module so the ~15 inline call sites only changed their import.
#
# A public_id alone does NOT mean the record is fetchable: databases carried
# over from the retired HuggingFace library hold records that were public
# THERE but do not exist in gavel-rules (different curation). Fetching those
# is a guaranteed 404 on every boot, so the gate is "present in the registry
# index we last synced", read from sync_state — no network call to find out.


def _registry_names(kind: str) -> Optional[set]:
    """Names present in the last-synced registry index for 'ces' / 'rules' /
    'rulesets'. None when we have never completed a sync (unknown, so callers
    must not treat anything as an orphan)."""
    hashes = _stored_hashes()
    family = hashes.get(kind)
    if not isinstance(family, dict) or not family:
        return None
    return set(family)


def _library_name(table: str, id_col: str, row_id: int, kind: str) -> Optional[str]:
    """The registry-fetchable name for a local row, or None when the row has
    nothing upstream — either a local-only draft (no public_id) or a legacy
    record absent from the current registry."""
    rows = execute_query_dict(
        f"SELECT name, public_id FROM {table} WHERE {id_col} = %s", (row_id,)
    ) or []
    if not rows or not rows[0].get("public_id"):
        return None
    name = rows[0]["name"]
    known = _registry_names(kind)
    if known is not None and name not in known:
        logger.debug(
            "[library_sync] %s '%s' is not in the gavel-rules registry "
            "(legacy record) — no upstream dataset to fetch", kind, name,
        )
        return None
    return name


def ensure_excitation(ce_id: int) -> bool:
    """Guarantee a local excitation row for this CE, lazily fetching
    ces/<name>/excitation.json from the registry on first need. False when
    the CE is local-only (nothing upstream) or the fetch failed."""
    if execute_query_dict(
        "SELECT 1 FROM excitation_datasets WHERE ce_id = %s", (ce_id,)
    ):
        return True
    name = _library_name("cognitive_elements", "ce_id", ce_id, "ces")
    if not name:
        return False
    try:
        payload = _reader().fetch_json(f"ces/{name}/excitation.json")
        samples = payload.get("samples") or []
        envelope = {"samples": samples,
                    "sample_count": payload.get("sample_count") or len(samples),
                    "source": _DATASET_SOURCE}
        execute_query(
            """
            INSERT INTO excitation_datasets (ce_id, dataset) VALUES (%s, %s)
            ON CONFLICT (ce_id) DO UPDATE SET dataset = EXCLUDED.dataset
            """,
            (ce_id, json.dumps(envelope)),
        )
        return True
    except Exception as e:
        logger.warning("[library_sync] lazy excitation fetch failed for ce_id=%s (%s): %s", ce_id, name, e)
        return False


def ensure_ce_calibration(ce_id: int) -> bool:
    """Same contract as ensure_excitation, for ces/<name>/calibration.json.
    Stored under the legacy {conversations, sample_count} envelope the
    calibration readers expect."""
    if execute_query_dict(
        "SELECT 1 FROM calibration_datasets WHERE ce_id = %s", (ce_id,)
    ):
        return True
    name = _library_name("cognitive_elements", "ce_id", ce_id, "ces")
    if not name:
        return False
    try:
        payload = _reader().fetch_json(f"ces/{name}/calibration.json")
        samples = payload.get("samples") or []
        envelope = {"conversations": samples,
                    "sample_count": payload.get("sample_count") or len(samples),
                    "source": _DATASET_SOURCE}
        execute_query(
            """
            INSERT INTO calibration_datasets (ce_id, dataset) VALUES (%s, %s)
            ON CONFLICT (ce_id) DO UPDATE SET dataset = EXCLUDED.dataset
            """,
            (ce_id, json.dumps(envelope)),
        )
        return True
    except Exception as e:
        logger.warning("[library_sync] lazy CE-calibration fetch failed for ce_id=%s (%s): %s", ce_id, name, e)
        return False


def _classifier_ce_ids(classifier_id: int) -> List[int]:
    rows = execute_query_dict(
        """
        SELECT DISTINCT scl.ce_id
        FROM rule_setup rs
        JOIN setup_ce_link scl ON rs.setup_id = scl.setup_id
        WHERE rs.classifier_id = %s AND rs.is_active = TRUE
        """,
        (classifier_id,),
    ) or []
    return [r["ce_id"] for r in rows]


def ensure_excitations_for_classifier(classifier_id: int) -> None:
    """Bulk parallel excitation prefetch for every CE in a guardrail's active
    policy — called right before training starts."""
    ce_ids = _classifier_ce_ids(classifier_id)
    if not ce_ids:
        return
    with ThreadPoolExecutor(max_workers=_PULL_WORKERS) as pool:
        list(pool.map(ensure_excitation, ce_ids))


def ensure_ce_calibrations_for_classifier(classifier_id: int) -> None:
    """Bulk parallel CE-calibration prefetch (calibration/evaluation runs)."""
    ce_ids = _classifier_ce_ids(classifier_id)
    if not ce_ids:
        return
    with ThreadPoolExecutor(max_workers=_PULL_WORKERS) as pool:
        list(pool.map(ensure_ce_calibration, ce_ids))


def _upsert_rule_dataset(rule_id: int, dataset_type: str, payload: dict, rule_public_id: str) -> None:
    """Upsert one default dataset bucket as a local is_default row keyed by
    (rule_id, dataset_type). Marked 'ready' — the dialogues arrive fully
    generated. Registry files carry the bucket in `type`; config is
    {scenario_instructions} only (v2 contract)."""
    from services.default_datasets import DEFAULT_TEST_SET_NAME
    convos = payload.get("conversations") or []
    config = payload.get("config") or {}
    public_id = f"{rule_public_id}_{dataset_type}"
    execute_query(
        """
        INSERT INTO test_datasets
            (rule_id, user_id, is_default, dataset_type, scenario_name,
             config, conversations, status, public_id, published_at)
        VALUES (%s, NULL, TRUE, %s, %s, %s::jsonb, %s::jsonb, 'ready', %s, %s)
        ON CONFLICT (rule_id, dataset_type) WHERE is_default = TRUE
        DO UPDATE SET config = EXCLUDED.config,
                      scenario_name = EXCLUDED.scenario_name,
                      conversations = EXCLUDED.conversations,
                      status = 'ready',
                      public_id = EXCLUDED.public_id,
                      published_at = EXCLUDED.published_at
        """,
        (rule_id, dataset_type, DEFAULT_TEST_SET_NAME,
         json.dumps(config).replace("\\u0000", ""),
         json.dumps(convos).replace("\\u0000", ""),
         public_id, payload.get("published_at")),
    )


def ensure_rule_defaults(rule_id: int) -> bool:
    """Lazy-fetch a rule's DEFAULT test/calibration sets
    (rules/<name>/tests/{positive,negative,positive_calibration}.json).

    Fast path: all three local is_default rows present + 'ready' -> True.
    Rules with nothing upstream (local-only, or a legacy record absent from
    the current registry) -> False; their defaults are generated locally by
    services/default_datasets.py instead."""
    rows = execute_query_dict(
        "SELECT dataset_type, status FROM test_datasets WHERE rule_id = %s AND is_default = TRUE",
        (rule_id,),
    ) or []
    by_type = {r["dataset_type"]: r["status"] for r in rows}
    if all(by_type.get(t) == "ready" for t in _RULE_DEFAULT_TYPES):
        return True

    name = _library_name("rules", "rule_id", rule_id, "rules")
    if not name:
        return False
    public_id = execute_query_dict(
        "SELECT public_id FROM rules WHERE rule_id = %s", (rule_id,)
    )[0]["public_id"]

    ok = True
    for dtype in _RULE_DEFAULT_TYPES:
        if by_type.get(dtype) == "ready":
            continue
        try:
            payload = _reader().fetch_json(f"rules/{name}/tests/{dtype}.json")
            _upsert_rule_dataset(rule_id, dtype, payload, public_id)
        except Exception as e:
            logger.warning(
                "[library_sync] lazy rule-default fetch failed for rule_id=%s (%s/%s): %s",
                rule_id, name, dtype, e,
            )
            ok = False
    return ok


def ensure_rule_calibration(rule_id: int) -> bool:
    """A rule's calibration bucket is the positive_calibration default row;
    ensure_rule_defaults pulls all three buckets together."""
    rows = execute_query_dict(
        """
        SELECT 1 FROM test_datasets
        WHERE rule_id = %s AND is_default = TRUE
          AND dataset_type = 'positive_calibration' AND status = 'ready'
        """,
        (rule_id,),
    )
    if rows:
        return True
    return ensure_rule_defaults(rule_id)


def ensure_rule_aux_for_classifier(classifier_id: int) -> None:
    """Bulk prefetch of default test sets for every library rule in a
    guardrail's active policy."""
    rows = execute_query_dict(
        """
        SELECT DISTINCT rs.rule_id
        FROM rule_setup rs
        WHERE rs.classifier_id = %s AND rs.is_active = TRUE AND rs.rule_id IS NOT NULL
        """,
        (classifier_id,),
    ) or []
    rule_ids = [r["rule_id"] for r in rows]
    if not rule_ids:
        return
    with ThreadPoolExecutor(max_workers=_PULL_WORKERS) as pool:
        list(pool.map(ensure_rule_defaults, rule_ids))


def pull_all_aux_datasets() -> None:
    """Startup warm pass: eagerly fetch CE calibration sets for registry CEs
    that don't have one locally yet (excitation and rule test sets stay
    purely lazy — they are bigger and only needed for training/evaluation).

    Only CEs that exist in the CURRENT registry are candidates. Databases
    carried over from the retired HuggingFace library also hold records with
    a public_id that gavel-rules never had; those are reported once here
    rather than 404-ing on every boot."""
    rows = execute_query_dict(
        """
        SELECT ce.ce_id, ce.name FROM cognitive_elements ce
        LEFT JOIN calibration_datasets cd ON cd.ce_id = ce.ce_id
        WHERE ce.public_id IS NOT NULL AND cd.ce_id IS NULL
        """
    ) or []
    if not rows:
        return

    known = _registry_names("ces")
    if known is None:
        candidates = [r["ce_id"] for r in rows]      # never synced — try all
        legacy = []
    else:
        candidates = [r["ce_id"] for r in rows if r["name"] in known]
        legacy = sorted(r["name"] for r in rows if r["name"] not in known)

    if legacy:
        logger.info(
            "[library_sync] %d local library CE(s) are not in the gavel-rules "
            "registry (legacy records from the retired HuggingFace library) — "
            "skipping their dataset fetches; they keep working from whatever "
            "data is already stored locally. Names: %s",
            len(legacy), ", ".join(legacy),
        )
    if not candidates:
        return
    logger.info("[library_sync] warming %d CE calibration sets…", len(candidates))
    with ThreadPoolExecutor(max_workers=_PULL_WORKERS) as pool:
        list(pool.map(ensure_ce_calibration, candidates))
