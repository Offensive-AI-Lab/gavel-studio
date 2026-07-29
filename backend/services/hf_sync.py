"""HuggingFace registry sync service.

Pulls new records from the public library on demand and inserts them into
the local DB. Designed to run on user login (or on an explicit "Sync now"
click): a fast probe short-circuits when nothing has changed; otherwise we
fetch only the records the local DB is missing.

Architecture per GAVEL_HF_Sync_Plan.docx:
    section 6.1 — cheap probe (manifest content hash)
    section 6.2 — manifest diff (one-direction: registry is append-only)
    section 6.3 — pull and validate (per-record atomicity, one bad record
                  never blocks the rest)

SOLID intent:
    Single responsibility — this module owns "what to pull and how to
    insert it". Routes own HTTP I/O. Pydantic models in
    services/library_schemas.py own validation.
"""
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from utils.sqlite_db import execute_query, execute_query_dict
from utils.DButils import normalize_and_upsert_categories
from utils.embedding_utils import trigger_embedding
from services.library_schemas import (
    Manifest,
    CERecord,
    RuleRecord,
    RuleSetRecord,
    ExcitationRecord,
    CECalibrationRecord,
    CategoriesFile,
    NeutralCorpusFile,
)

logger = logging.getLogger(__name__)


# --- Configuration ---

# The registry repo, for BOTH reads (sync/Browse) and writes (publish via
# services/hf_write). Overridable so a local-only deployment can point its
# publishes at its own dataset repo instead of the shared public library.
REPO_ID = os.getenv("HF_REPO_ID", "GavelPublicData/public-library")
REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")

# sync_state keys
_LAST_MANIFEST_HASH_KEY = "last_manifest_hash"

# Module-level lock that coalesces concurrent sync_library() calls. The
# server-startup background thread and a same-instant user login both want
# to sync; serializing them prevents double-fetching the same records and
# stops the second caller from racing the manifest_hash short-circuit.
# Threading.Lock is enough — sync runs in <1s on the warm path, and a
# blocked caller just waits for the result of the first call to land.
_sync_lock = threading.Lock()


# --- Result type ---


@dataclass
class SyncResult:
    """What sync_library() reports back to its caller. `changed` is False
    when the cheap probe found nothing new — in that case all counters
    stay zero and errors is empty.

    `*_added` counts records that were missing locally and just got
    pulled. `*_refreshed` counts records that already existed locally
    but had a newer published_at on HF and got re-pulled in place.
    Splitting the two helps the user (and the test suite) tell apart
    "you onboarded new content" from "an admin edited existing content".
    """
    changed: bool
    ces_added: int = 0
    rules_added: int = 0
    rule_sets_added: int = 0
    ces_refreshed: int = 0
    rules_refreshed: int = 0
    rule_sets_refreshed: int = 0
    categories_synced: int = 0
    neutral_synced: int = 0
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
            "neutral_synced": self.neutral_synced,
            "skipped_records": self.skipped_records,
            "errors": self.errors,
        }


# --- Helpers: HF token + client ---


def _resolve_token() -> Optional[str]:
    """Read HF_TOKEN from backend/.env or the live environment.

    Returns None when no usable token is set. An EMPTY/whitespace value is
    coerced to None: huggingface_hub builds an ``Authorization: Bearer <tok>``
    header, and a blank token yields ``Bearer `` (no value), which the HTTP
    layer rejects with "Illegal header value b'Bearer '" — breaking even the
    anonymous public-repo reads that a token-less client relies on. Treating
    "" as None makes those reads fall back to unauthenticated, as intended."""
    backend_dir = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=backend_dir / ".env")
    return (os.environ.get("HF_TOKEN") or "").strip() or None


_READER = None


def _reader():
    """Lazily-built RegistryReader — the read-side port. The backend reads the
    public library THROUGH this, never a vendor SDK directly, so the storage
    backend (HuggingFace today, GitHub tomorrow) is one swappable adapter. Bulk
    bytes still stream straight from the vendor's CDN to this backend."""
    global _READER
    if _READER is None:
        from services.registry_sync.reader import build_reader
        _READER = build_reader()
    return _READER


# --- Helpers: sync_state key/value store ---


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


# --- Step 1: cheap probe via manifest content hash ---


def _fetch_manifest_bytes(api=None, token: str = None) -> bytes:
    """Download manifest.json from the registry and return its raw bytes. The
    content hash is a stable signal for "did anything change in the registry".
    Reads go through the RegistryReader port (vendor-agnostic); the api/token
    args are kept for call-site compatibility but the reader resolves its own."""
    return _reader().fetch_bytes("manifest.json")


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_for_updates() -> dict:
    """Cheap "is there new content on HF?" probe.

    Compares the current HF manifest hash against the locally-cached
    `last_manifest_hash`. Returns immediately without pulling anything,
    so it's safe to call on a 90s frontend timer without burning HF
    rate-limit on full diffs.

    Return shape:
      {
        "available": bool,    # True iff HF has changed since last pull
        "checked": bool,      # False if we couldn't reach HF or HF_TOKEN missing
        "reason": str | None, # populated when checked=False
      }

    The frontend uses `available` to decide whether to surface a "pull
    updates" indicator on the sidebar. `checked=False` paths return
    `available=False` so a transient HF outage doesn't pin the badge in
    its "updates" state forever.
    """
    # Public repo → the update probe works anonymously (the reader handles auth).
    try:
        manifest_bytes = _fetch_manifest_bytes()
    except Exception as exc:
        return {"available": False, "checked": False, "reason": f"HF probe failed: {exc}"}

    current_hash = _hash_bytes(manifest_bytes)
    last_hash = _get_state(_LAST_MANIFEST_HASH_KEY)
    available = (last_hash != current_hash)
    return {"available": available, "checked": True, "reason": None}


# --- Step 2: manifest diff ---


def _local_public_ids() -> Tuple[set, set]:
    """Set of (CE public_ids, rule public_ids) currently in the local DB."""
    ce_rows = execute_query_dict(
        "SELECT public_id FROM cognitive_elements WHERE public_id IS NOT NULL"
    ) or []
    rule_rows = execute_query_dict(
        "SELECT public_id FROM rules WHERE public_id IS NOT NULL"
    ) or []
    return (
        {r["public_id"] for r in ce_rows},
        {r["public_id"] for r in rule_rows},
    )


def _local_pubat_map() -> Tuple[dict, dict]:
    """Map of {public_id -> published_at} per CE and per rule.

    Used by sync_library's stale-detection: a record whose manifest
    published_at is strictly newer than its local published_at is
    re-fetched, even though the public_id is already in the local DB.
    Without this, content edits on HF that keep the same public_id
    (e.g. re-categorizing seed records) would silently never propagate
    to clients — exactly the bug that motivated this helper.
    """
    ce_rows = execute_query_dict(
        "SELECT public_id, published_at FROM cognitive_elements WHERE public_id IS NOT NULL"
    ) or []
    rule_rows = execute_query_dict(
        "SELECT public_id, published_at FROM rules WHERE public_id IS NOT NULL"
    ) or []
    return (
        {r["public_id"]: r["published_at"] for r in ce_rows},
        {r["public_id"]: r["published_at"] for r in rule_rows},
    )


def _hf_pubat_is_newer(manifest_pubat: str, local_pubat) -> bool:
    """Decide whether an HF manifest entry's published_at means we
    should re-fetch a record we already have locally.

    Returns True for "registry has a newer version than we cached", False
    otherwise (equal, older, or unparseable).

    Tolerant of three input quirks:
      * `local_pubat` is None — treat as "no timestamp on file", force
        a refresh so the local row gets a real timestamp on the next pull.
      * Z suffix vs +00:00 — Python <3.11 datetime.fromisoformat() chokes
        on the Z form; we replace it before parsing.
      * Either side malformed — log nothing, return False. A bad timestamp
        shouldn't trigger an infinite re-pull loop on every sync.
    """
    if not manifest_pubat:
        return False
    if local_pubat is None:
        return True

    from datetime import datetime, timezone

    def _naive_utc(dt: datetime) -> datetime:
        # The DB hands back naive-UTC datetimes; manifest strings usually
        # carry a Z/offset. Normalize both sides to naive UTC so the
        # comparison never mixes aware and naive (that raises TypeError).
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    try:
        manifest_dt = _naive_utc(datetime.fromisoformat(str(manifest_pubat).replace('Z', '+00:00')))
    except Exception:
        return False

    if isinstance(local_pubat, datetime):
        local_dt = _naive_utc(local_pubat)
    else:
        try:
            local_dt = _naive_utc(datetime.fromisoformat(str(local_pubat).replace('Z', '+00:00')))
        except Exception:
            return False

    return manifest_dt > local_dt


# --- Categories ---


def _pull_and_upsert_categories(token: str, result: SyncResult) -> int:
    """Fetch categories.json from the registry root and upsert each entry
    into the local categories table by name.

    Local-only categories that aren't in the registry are NOT deleted —
    users can have their own private categories which the sync should
    leave alone. We only ADD or UPDATE descriptions for matched names.

    Returns the number of categories synced. Counts include both new
    inserts and existing rows whose descriptions were updated.
    """
    try:
        payload = _fetch_record(token, "categories.json")
        cat_file = CategoriesFile.model_validate(payload)
    except Exception as e:
        msg = f"categories.json: {e}"
        logger.error(f"[hf_sync] {msg}")
        result.errors.append(msg)
        return 0

    synced = 0
    for cat in cat_file.categories:
        try:
            execute_query(
                """
                INSERT INTO categories (name, description, active)
                VALUES (%s, %s, TRUE)
                ON CONFLICT (name) DO UPDATE
                SET description = EXCLUDED.description, active = TRUE
                """,
                (cat.name, cat.description),
            )
            synced += 1
        except Exception as e:
            msg = f"category '{cat.name}': {e}"
            logger.error(f"[hf_sync] {msg}")
            result.errors.append(msg)
    return synced


_NEUTRAL_HASHES_KEY = "neutral_hashes"


def _fetch_and_upsert_neutral_category(token: str, category: str, result: SyncResult) -> int:
    """Download neutral/<category>/conversations.json and upsert every
    conversation into the neutral_corpus table (dedup by content_hash).
    Returns how many were upserted. Idempotent — safe to call repeatedly."""
    from evaluation.neutral_corpus import content_hash as _conv_hash
    try:
        payload = _fetch_record(token, f"neutral/{category}/conversations.json")
        parsed = NeutralCorpusFile.model_validate(payload)
    except Exception as e:
        msg = f"neutral/{category}: {e}"
        logger.error(f"[hf_sync] {msg}")
        result.errors.append(msg)
        return 0
    count = 0
    for conv in parsed.conversations:
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        try:
            execute_query(
                """
                INSERT INTO neutral_corpus (content_hash, category, conversation, published_at)
                VALUES (%s, %s, %s::jsonb, now())
                ON CONFLICT (content_hash) DO UPDATE SET category = EXCLUDED.category
                """,
                (_conv_hash(conv), category, json.dumps(conv)),
            )
            count += 1
        except Exception as e:
            result.errors.append(f"neutral/{category} conversation: {e}")
    return count


def ensure_neutral_corpus() -> int:
    """Guarantee the local neutral_corpus table holds the FULL registry corpus.

    Called right before an evaluation so the neutral split is complete even on a
    machine that hasn't run a full sync yet. Unlike the gated sync pull, this
    fetches each category and upserts unconditionally — idempotent, so it just
    tops up whatever is missing. Best-effort: returns 0 if HF is unreachable, in
    which case the evaluation hard-fails unless the corpus was already synced
    (the corpus is HF/DB-only — there is no bundled fallback).
    """
    token = _resolve_token()  # None is fine — public-repo reads work anonymously
    try:
        from services.hf_publish import _fetch_head_sha_and_manifest
        _sha, manifest = _fetch_head_sha_and_manifest()
    except Exception as e:
        logger.warning(f"[hf_sync] ensure_neutral_corpus: manifest fetch failed: {e}")
        return 0
    neutral = manifest.get("neutral") or {}
    if not neutral:
        return 0
    from evaluation.neutral_corpus import CATEGORIES
    result = SyncResult(changed=True)
    synced = 0
    for category in neutral:
        if category in CATEGORIES:
            synced += _fetch_and_upsert_neutral_category(token, category, result)
    # Record the hashes so the routine background sync won't re-pull them.
    try:
        cached = json.loads(_get_state(_NEUTRAL_HASHES_KEY) or "{}")
        cached.update({c: h for c, h in neutral.items() if c in CATEGORIES})
        _set_state(_NEUTRAL_HASHES_KEY, json.dumps(cached))
    except Exception:
        pass
    return synced


def _pull_neutral_corpus(token: str, manifest_neutral: Dict[str, str], result: SyncResult) -> int:
    """Pull the global neutral corpus into the local `neutral_corpus` table.

    `manifest_neutral` maps category -> the sha256 of that category's
    neutral/<category>/conversations.json. We only re-fetch a category whose
    hash moved since our last pull (cached in sync_state), so an unrelated
    manifest change (e.g. a newly published CE) doesn't drag the whole corpus
    down again. Upserts are idempotent (dedup by content_hash), so a partial
    pull is always safe to retry.

    Returns the number of conversations upserted this pass.
    """
    if not manifest_neutral:
        return 0
    from evaluation.neutral_corpus import CATEGORIES

    try:
        cached = json.loads(_get_state(_NEUTRAL_HASHES_KEY) or "{}")
    except Exception:
        cached = {}
    new_cached = dict(cached)
    synced = 0

    for category, h in manifest_neutral.items():
        if category not in CATEGORIES:
            continue
        if cached.get(category) == h:
            continue  # unchanged since last pull
        synced += _fetch_and_upsert_neutral_category(token, category, result)
        new_cached[category] = h

    if new_cached != cached and not result.errors:
        _set_state(_NEUTRAL_HASHES_KEY, json.dumps(new_cached))
    return synced


# --- Step 3: per-record fetch + insert ---


def _fetch_record(token: str, path_in_repo: str) -> dict:
    """Fetch a single JSON file from the registry, parse it, return the dict.
    Pydantic validation happens at the call site so the caller can route
    ValidationErrors through the per-record skip logic. Reads go through the
    RegistryReader port (vendor-agnostic); `token` is kept for call-site
    compatibility but the reader resolves its own."""
    return _reader().fetch_json(path_in_repo)


def _upsert_ce(record: CERecord) -> Optional[int]:
    """Insert (or update) a CE row from a validated record. Returns the
    local ce_id, or None if the record was skipped because a local draft
    with the same name exists.

    Drafts are sacred: if the user has an in-progress local CE with the
    same name as something incoming from the registry, we do NOT clobber
    their work. The name collision is left unresolved here and surfaced
    properly the next time the user tries to publish their draft (the
    publish service does its own name-index check against the registry
    manifest).
    """
    # Skip-overwrite guard: bail out if a local draft with this name exists.
    existing_draft = execute_query_dict(
        "SELECT ce_id FROM cognitive_elements WHERE name = %s AND is_local_draft = TRUE",
        (record.name,),
    )
    if existing_draft:
        logger.warning(
            "[hf_sync] CE '%s' incoming from registry collides with a local "
            "draft (ce_id=%d); leaving the draft alone, registry record not pulled",
            record.name, existing_draft[0]["ce_id"],
        )
        return None

    final_categories = normalize_and_upsert_categories(
        list(record.categories), allow_new=True
    )
    # Every HF record now carries created_by_username — legacy seed
    # records were backfilled by scripts/backfill_hf_creator.py.
    creator = record.created_by_username
    rows = execute_query_dict(
        """
        INSERT INTO cognitive_elements (
            name, definition, category, categories, examples,
            public_id, published_at, is_local_draft, created_by_username
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE, %s)
        ON CONFLICT (name) DO UPDATE
        SET definition       = EXCLUDED.definition,
            category         = EXCLUDED.category,
            categories       = EXCLUDED.categories,
            examples         = EXCLUDED.examples,
            public_id        = EXCLUDED.public_id,
            published_at     = EXCLUDED.published_at,
            is_local_draft   = FALSE,
            -- Only overwrite a NULL creator. Once a row has an author
            -- recorded (either from a prior sync or from a publish), we
            -- never silently rename them. Same-author replays are no-ops.
            created_by_username = COALESCE(cognitive_elements.created_by_username, EXCLUDED.created_by_username)
        RETURNING ce_id
        """,
        (
            record.name,
            record.definition,
            record.category,
            final_categories,
            json.dumps(record.examples),
            record.public_id,
            record.published_at,
            creator,
        ),
    )
    return rows[0]["ce_id"]


def _upsert_excitation(ce_id: int, record: ExcitationRecord) -> None:
    """Replace any prior excitation row for this CE with the registry one."""
    payload = {
        "samples": record.samples,
        "sample_count": record.sample_count or len(record.samples),
    }
    execute_query(
        """
        INSERT INTO excitation_datasets (ce_id, dataset)
        VALUES (%s, %s)
        ON CONFLICT (ce_id) DO UPDATE SET dataset = EXCLUDED.dataset
        """,
        (ce_id, json.dumps(payload)),
    )


def _upsert_ce_calibration(ce_id: int, record: CECalibrationRecord) -> None:
    """Replace any prior calibration row for this CE with the registry one.

    The local calibration_datasets table holds a `dataset` JSON blob with
    the same {samples, sample_count} envelope used for excitation, so the
    routes that read it (Evaluation page, calibration endpoint) don't have
    to special-case the source."""
    payload = {
        "conversations": record.samples,
        "sample_count": record.sample_count or len(record.samples),
    }
    execute_query(
        """
        INSERT INTO calibration_datasets (ce_id, dataset)
        VALUES (%s, %s)
        ON CONFLICT (ce_id) DO UPDATE SET dataset = EXCLUDED.dataset
        """,
        (ce_id, json.dumps(payload)),
    )


def _upsert_rule(record: RuleRecord) -> Optional[int]:
    """Insert (or update) a rule row + its rule_ce_link rows. Returns the
    local rule_id, or None if the record was skipped due to a local-draft
    name collision (same skip-overwrite guard as _upsert_ce).

    Caller must have already pulled every CE this rule depends on (resolved
    by name from the local DB).
    """
    # Skip-overwrite guard for drafts (see _upsert_ce for rationale).
    existing_draft = execute_query_dict(
        "SELECT rule_id FROM rules WHERE name = %s AND is_local_draft = TRUE",
        (record.name,),
    )
    if existing_draft:
        logger.warning(
            "[hf_sync] rule '%s' incoming from registry collides with a local "
            "draft (rule_id=%d); leaving the draft alone, registry record not pulled",
            record.name, existing_draft[0]["rule_id"],
        )
        return None

    final_categories = normalize_and_upsert_categories(
        list(record.categories), allow_new=True
    )
    # Derive the boolean-logic predicate from the record's ROLE LISTS rather than
    # trusting record.predicate. The role lists are the source of truth (we
    # rebuild rule_ce_link from them below); the published predicate string can be
    # stale/buggy (e.g. older publishes OR'd in 'sufficient'/helpful CEs, which
    # never fire a rule). Deriving here means every client computes the correct
    # predicate regardless of what string sits in the HF record. Fall back to the
    # record's predicate only for a degenerate record with no necessary/fallback
    # CEs (nothing to derive from).
    from sql_scripts.model_scripts import predicate_from_role_lists
    derived_predicate = predicate_from_role_lists(record.necessary, record.fallback)
    predicate_to_store = derived_predicate if (record.necessary or record.fallback) else record.predicate

    # Every HF record now carries created_by_username — legacy seed
    # records were backfilled by scripts/backfill_hf_creator.py.
    creator = record.created_by_username
    rows = execute_query_dict(
        """
        INSERT INTO rules (
            name, predicate, categories, description,
            public_id, published_at, is_local_draft, created_by_username
        )
        VALUES (%s, %s, %s, %s, %s, %s, FALSE, %s)
        ON CONFLICT (name) DO UPDATE
        SET predicate        = EXCLUDED.predicate,
            categories       = EXCLUDED.categories,
            -- Don't clobber an existing local explanation with an empty one
            -- coming from HF (seed rules predate descriptions). A non-empty
            -- incoming value still wins.
            description      = COALESCE(NULLIF(EXCLUDED.description, ''), rules.description),
            public_id        = EXCLUDED.public_id,
            published_at     = EXCLUDED.published_at,
            is_local_draft   = FALSE,
            created_by_username = COALESCE(rules.created_by_username, EXCLUDED.created_by_username)
        RETURNING rule_id
        """,
        (
            record.name,
            predicate_to_store,
            final_categories,
            record.definition,
            record.public_id,
            record.published_at,
            creator,
        ),
    )
    rule_id = rows[0]["rule_id"]

    # Rebuild role-aware links from scratch — the registry record is the
    # source of truth for which CEs play which roles.
    execute_query("DELETE FROM rule_ce_link WHERE rule_id = %s", (rule_id,))

    def _local_ce_id(ce_name: str) -> int:
        result = execute_query_dict(
            "SELECT ce_id FROM cognitive_elements WHERE name = %s", (ce_name,)
        )
        if not result:
            raise RuntimeError(
                f"rule '{record.name}' references unknown local CE '{ce_name}'"
            )
        return result[0]["ce_id"]

    # fallback_group convention: 0 for non-fallback roles, group index for fallback rows.
    for ce_name in record.necessary:
        execute_query(
            "INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group) "
            "VALUES (%s, %s, 'necessary', 0)",
            (rule_id, _local_ce_id(ce_name)),
        )
    for group_idx, group in enumerate(record.fallback):
        for ce_name in group:
            execute_query(
                "INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group) "
                "VALUES (%s, %s, 'fallback', %s)",
                (rule_id, _local_ce_id(ce_name), group_idx),
            )
    for ce_name in record.sufficient:
        execute_query(
            "INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group) "
            "VALUES (%s, %s, 'sufficient', 0)",
            (rule_id, _local_ce_id(ce_name)),
        )

    return rule_id


def _pull_ce(token: str, ce_public_id: str, result: SyncResult) -> bool:
    """Fetch a CE record, validate, insert. Returns True on success.

    Excitation datasets are NOT fetched here — they're loaded lazily by
    ensure_excitation() at the moment of first read (UI preview, guardrail
    training prep). This halves the per-CE HTTP cost of a cold sync because
    the excitation is the larger of the two files and most CEs are never
    opened by a given user. See ensure_excitation() below for the read-side
    counterpart.
    """
    try:
        ce_payload = _fetch_record(token, f"public_ces/{ce_public_id}.json")
        ce_record = CERecord.model_validate(ce_payload)

        local_ce_id = _upsert_ce(ce_record)
        if local_ce_id is None:
            # Skipped because a local draft of the same name exists. Not an
            # error — the conflict will be resolved when the user tries to
            # publish their draft.
            result.skipped_records.append(f"{ce_public_id} (local draft has same name)")
            return False
        try:
            trigger_embedding("ce", local_ce_id, ce_record.name, ce_record.definition)
        except Exception as embed_err:
            logger.warning(
                f"[hf_sync] embedding failed for CE {ce_record.name}: {embed_err}"
            )
        return True
    except Exception as e:
        msg = f"CE {ce_public_id}: {e}"
        logger.error(f"[hf_sync] {msg}")
        result.errors.append(msg)
        result.skipped_records.append(ce_public_id)
        return False


def ensure_excitation(ce_id: int) -> bool:
    """Lazy-load a CE's excitation dataset from HF if it isn't already in
    the local DB. Returns True if the local DB has the dataset after this
    call (whether we fetched it or it was already there), False otherwise.

    Designed for the read side of the lazy-excitation contract: sync only
    pulls CE metadata, and any path that actually needs the training samples
    (UI preview, guardrail training prep) calls this just-in-time.

    Cheap when nothing's missing: a single SELECT COUNT(*) and we return.
    On a miss, one HF round-trip + one INSERT.
    """
    # Fast path — excitation already cached locally.
    rows = execute_query_dict(
        "SELECT 1 FROM excitation_datasets WHERE ce_id = %s LIMIT 1", (ce_id,)
    ) or []
    if rows:
        return True

    # No local row. We can only fetch from HF if the CE was synced from
    # HF (has a public_id). User-created drafts that never published won't
    # have one, and there's nothing to lazy-load — they should already
    # carry an excitation row from the local generation pipeline.
    ce_rows = execute_query_dict(
        "SELECT public_id FROM cognitive_elements WHERE ce_id = %s", (ce_id,)
    ) or []
    if not ce_rows or not ce_rows[0].get("public_id"):
        return False

    public_id = ce_rows[0]["public_id"]
    token = _resolve_token()  # may be None — public-repo reads work anonymously

    try:
        payload = _fetch_record(token, f"public_excitation/excitation_{public_id}.json")
        record = ExcitationRecord.model_validate(payload)
        _upsert_excitation(ce_id, record)
        return True
    except Exception as e:
        logger.warning(f"[hf_sync] lazy excitation fetch failed for ce_id={ce_id} ({public_id}): {e}")
        return False


def ensure_ce_calibration(ce_id: int) -> bool:
    """Lazy-fetch a CE's calibration record from HF if not already cached.

    Mirrors `ensure_excitation` exactly — same idempotency, same skip-when-
    no-public_id behaviour. Returns True if the local row was populated
    (either freshly fetched or already present), False otherwise.
    """
    rows = execute_query_dict(
        "SELECT 1 FROM calibration_datasets WHERE ce_id = %s LIMIT 1", (ce_id,)
    ) or []
    if rows:
        return True

    ce_rows = execute_query_dict(
        "SELECT public_id FROM cognitive_elements WHERE ce_id = %s", (ce_id,)
    ) or []
    if not ce_rows or not ce_rows[0].get("public_id"):
        return False

    public_id = ce_rows[0]["public_id"]
    token = _resolve_token()  # may be None — public-repo reads work anonymously

    try:
        payload = _fetch_record(token, f"public_calibration/ce_{public_id}.json")
        record = CECalibrationRecord.model_validate(payload)
        _upsert_ce_calibration(ce_id, record)
        return True
    except Exception as e:
        logger.warning(
            f"[hf_sync] lazy CE-calibration fetch failed for ce_id={ce_id} ({public_id}): {e}"
        )
        return False


def ensure_ce_calibrations_for_classifier(classifier_id: int) -> dict:
    """Bulk lazy-fetch every CE-level calibration needed to calibrate a
    guardrail. Walks the same setup → CE chain as
    `ensure_excitations_for_classifier`. Called from the calibration route
    so the UI doesn't show "missing calibration data" for CEs that are
    sitting on HF and just haven't been pulled yet.

    Returns {fetched, missing, already_present}.
    """
    ce_rows = execute_query_dict(
        """
        SELECT DISTINCT ce.ce_id, ce.public_id
        FROM rule_setup rs
        JOIN setup_ce_link scl ON rs.setup_id = scl.setup_id
        JOIN cognitive_elements ce ON scl.ce_id = ce.ce_id
        WHERE rs.classifier_id = %s
        """,
        (classifier_id,),
    ) or []

    summary = {"fetched": 0, "missing": 0, "already_present": 0}
    if not ce_rows:
        return summary

    from concurrent.futures import ThreadPoolExecutor

    def _one(ce_id: int) -> str:
        rows = execute_query_dict(
            "SELECT 1 FROM calibration_datasets WHERE ce_id = %s LIMIT 1", (ce_id,)
        ) or []
        if rows:
            return "already_present"
        return "fetched" if ensure_ce_calibration(ce_id) else "missing"

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="hf-sync-cal") as pool:
        for outcome in pool.map(lambda r: _one(r["ce_id"]), ce_rows):
            summary[outcome] = summary.get(outcome, 0) + 1
    return summary


# Default test/calibration buckets for a rule (schema v9).
_RULE_DEFAULT_TYPES = ("positive", "negative", "positive_calibration")


def _rule_dataset_path(rule_public_id: str, dataset_type: str) -> str:
    """HF path for a rule's default dataset file (must match hf_publish)."""
    return f"public_rule_datasets/{rule_public_id}_{dataset_type}.json"


def _upsert_rule_dataset(rule_id: int, record) -> None:
    """Upsert one default dataset bucket pulled from HF as a local
    is_default row keyed by (rule_id, dataset_type). Marked 'ready' since
    the dialogues arrive fully generated."""
    from services.default_datasets import DEFAULT_TEST_SET_NAME
    convos = record.conversations or []
    public_id = f"{record.rule_public_id}_{record.dataset_type}"
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
        (rule_id, record.dataset_type, DEFAULT_TEST_SET_NAME,
         json.dumps(record.config).replace("\\u0000", ""),
         json.dumps(convos).replace("\\u0000", ""),
         public_id, record.published_at),
    )


def ensure_rule_defaults(rule_id: int) -> bool:
    """Lazy-fetch a rule's DEFAULT test/calibration set from HF.

    Fast path: all three local is_default rows present + 'ready' → True.
    If the rule isn't published (no public_id) there's nothing on HF → False.
    Otherwise pull the missing files, upsert them, return True on full
    success. Mirrors `ensure_excitation` / `ensure_ce_calibration`."""
    rows = execute_query_dict(
        "SELECT dataset_type, status FROM test_datasets WHERE rule_id = %s AND is_default = TRUE",
        (rule_id,),
    ) or []
    by_type = {r["dataset_type"]: r["status"] for r in rows}
    if all(by_type.get(t) == "ready" for t in _RULE_DEFAULT_TYPES):
        return True

    rule_rows = execute_query_dict(
        "SELECT public_id FROM rules WHERE rule_id = %s", (rule_id,)
    ) or []
    if not rule_rows or not rule_rows[0].get("public_id"):
        return False
    public_id = rule_rows[0]["public_id"]

    token = _resolve_token()  # may be None — public-repo reads work anonymously

    from services.library_schemas import RuleDatasetRecord
    ok = True
    for dtype in _RULE_DEFAULT_TYPES:
        if by_type.get(dtype) == "ready":
            continue
        try:
            payload = _fetch_record(token, _rule_dataset_path(public_id, dtype))
            record = RuleDatasetRecord.model_validate(payload)
            _upsert_rule_dataset(rule_id, record)
        except Exception as e:
            logger.warning(
                f"[hf_sync] lazy rule-default fetch failed for rule_id={rule_id} "
                f"({public_id}/{dtype}): {e}"
            )
            ok = False
    return ok


def ensure_rule_calibration(rule_id: int) -> bool:
    """True if the rule has a usable positive_calibration set locally.

    Backed by the rule's DEFAULT set (lazy-pulled from HF when the rule is
    published) or the requesting user's private custom set — both keyed by
    rule_id (v10: test sets are rule-scoped). Attempts the HF pull first,
    then checks presence."""
    try:
        ensure_rule_defaults(rule_id)
    except Exception:
        pass
    rows = execute_query_dict(
        """
        SELECT 1 FROM test_datasets td
        WHERE td.rule_id = %s
          AND td.dataset_type = 'positive_calibration'
          AND td.status = 'ready'
        LIMIT 1
        """,
        (rule_id,),
    ) or []
    return bool(rows)


def ensure_rule_aux_for_classifier(classifier_id: int) -> dict:
    """Lazy-pull the DEFAULT test/calibration set for every published rule
    on a guardrail so calibration/evaluation can use them. Returns a
    summary of which rules' calibration is now present."""
    rule_rows = execute_query_dict(
        """
        SELECT DISTINCT rs.rule_id
        FROM rule_setup rs
        WHERE rs.classifier_id = %s AND rs.rule_id IS NOT NULL
        """,
        (classifier_id,),
    ) or []

    summary = {
        "calibration": {"already_present": 0, "missing": 0, "fetched": 0},
    }
    if not rule_rows:
        return summary

    for r in rule_rows:
        present = ensure_rule_calibration(r["rule_id"])
        key = "already_present" if present else "missing"
        summary["calibration"][key] += 1
    return summary


def pull_all_aux_datasets() -> dict:
    """Eager-pull every CE / rule auxiliary dataset listed in the HF
    manifest into local tables. Idempotent — anything already present
    is skipped.

    Designed for backend startup: runs in a background thread so login,
    library browsing, and rule editing don't wait on it. Routes that
    actually need the data (Evaluation, Calibration) call the per-record
    `ensure_*` helpers, which transparently fetch on demand if the
    background pull hasn't reached that record yet.

    Returns a summary dict suitable for logging.
    """
    # Only CE-level calibration is pulled in this bulk aux pass. Rule-level
    # test/calibration data now lives in rule DEFAULT datasets
    # (public_rule_datasets/...), pulled on demand by ensure_rule_defaults.
    # The old `rule_calibration` manifest section is gone.
    summary = {
        "ce_calibration": {"fetched": 0, "missing": 0, "already_present": 0},
    }

    token = _resolve_token()  # may be None — public-repo reads work anonymously

    # Read the local view of every CE's public_id; we'll match those
    # against the manifest to figure out what to fetch.
    ce_rows = execute_query_dict(
        "SELECT ce_id, public_id FROM cognitive_elements WHERE public_id IS NOT NULL"
    ) or []

    try:
        from services.hf_publish import _fetch_head_sha_and_manifest
        _sha, manifest = _fetch_head_sha_and_manifest()
    except Exception as e:
        logger.warning(f"[hf_sync] aux pull: manifest fetch failed: {e}")
        return summary

    ce_cal_index = manifest.get("ce_calibration", {}) or {}

    from concurrent.futures import ThreadPoolExecutor

    def _do(family: str, rows, public_index, ensure_fn, table: str, fk: str):
        def _one(row):
            local_id = row[fk]
            if row["public_id"] not in public_index:
                return None  # registry doesn't have this dataset
            existing = execute_query_dict(
                f"SELECT 1 FROM {table} WHERE {fk} = %s LIMIT 1", (local_id,)
            ) or []
            if existing:
                return "already_present"
            return "fetched" if ensure_fn(local_id) else "missing"

        with ThreadPoolExecutor(max_workers=8, thread_name_prefix=f"hf-aux-{family}") as pool:
            for outcome in pool.map(_one, rows):
                if outcome is None:
                    continue
                summary[family][outcome] = summary[family].get(outcome, 0) + 1

    _do("ce_calibration", ce_rows, ce_cal_index, ensure_ce_calibration,
        "calibration_datasets", "ce_id")

    logger.info(f"[hf_sync] aux pull complete: {summary}")
    return summary


def ensure_excitations_for_classifier(classifier_id: int) -> dict:
    """Bulk lazy-fetch every excitation needed to train a guardrail.

    Walks rule_setup → setup_ce_link → cognitive_elements for the given
    guardrail and ensures each referenced CE has its excitation present
    locally. Done in parallel because a guardrail with many CEs would
    otherwise pay HTTP latency serially right before training.

    Returns {fetched: int, missing: int, already_present: int}.
    """
    ce_rows = execute_query_dict(
        """
        SELECT DISTINCT ce.ce_id, ce.public_id
        FROM rule_setup rs
        JOIN setup_ce_link scl ON rs.setup_id = scl.setup_id
        JOIN cognitive_elements ce ON scl.ce_id = ce.ce_id
        WHERE rs.classifier_id = %s
        """,
        (classifier_id,),
    ) or []

    summary = {"fetched": 0, "missing": 0, "already_present": 0}
    if not ce_rows:
        return summary

    from concurrent.futures import ThreadPoolExecutor

    def _one(ce_id: int) -> str:
        rows = execute_query_dict(
            "SELECT 1 FROM excitation_datasets WHERE ce_id = %s LIMIT 1", (ce_id,)
        ) or []
        if rows:
            return "already_present"
        return "fetched" if ensure_excitation(ce_id) else "missing"

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="hf-sync-exc") as pool:
        for outcome in pool.map(lambda r: _one(r["ce_id"]), ce_rows):
            summary[outcome] = summary.get(outcome, 0) + 1
    return summary


def _pull_rule(token: str, rule_public_id: str, result: SyncResult) -> bool:
    """Fetch a rule, validate, insert. Same per-record atomicity as CEs."""
    try:
        rule_payload = _fetch_record(token, f"public_rules/{rule_public_id}.json")
        rule_record = RuleRecord.model_validate(rule_payload)

        local_rule_id = _upsert_rule(rule_record)
        if local_rule_id is None:
            result.skipped_records.append(f"{rule_public_id} (local draft has same name)")
            return False
        try:
            ce_definitions = " ".join(rule_record.ce_dependencies)
            trigger_embedding(
                "rule",
                local_rule_id,
                rule_record.name,
                rule_record.predicate,
                ce_definitions,
            )
        except Exception as embed_err:
            logger.warning(
                f"[hf_sync] embedding failed for rule {rule_record.name}: {embed_err}"
            )
        return True
    except Exception as e:
        msg = f"rule {rule_public_id}: {e}"
        logger.error(f"[hf_sync] {msg}")
        result.errors.append(msg)
        result.skipped_records.append(rule_public_id)
        return False


def _upsert_rule_set(record: RuleSetRecord) -> Optional[int]:
    """Insert (or update) a rule_sets row + its rule_set_member rows. Returns
    the local rule_set_id, or None if skipped due to a local-draft name
    collision (same skip-overwrite guard as _upsert_rule).

    A rule set is a thin pointer-collection: its member rules are referenced by
    public_id and MUST already be present locally — sync_library pulls rules
    BEFORE rule sets so the public_ids resolve. A member whose rule hasn't
    synced yet (e.g. it was skipped for a local-draft name clash) is skipped
    here; the set still resolves its available members and heals on the next
    sync once the missing rule lands.
    """
    # Skip-overwrite guard for an in-flight local publish (see _upsert_rule).
    existing_draft = execute_query_dict(
        "SELECT rule_set_id FROM rule_sets WHERE name = %s AND is_local_draft = TRUE",
        (record.name,),
    )
    if existing_draft:
        logger.warning(
            "[hf_sync] rule set '%s' incoming from registry collides with a local "
            "draft (rule_set_id=%d); leaving the draft alone, registry record not pulled",
            record.name, existing_draft[0]["rule_set_id"],
        )
        return None

    final_categories = normalize_and_upsert_categories(
        list(record.categories), allow_new=True
    )
    creator = record.created_by_username
    rows = execute_query_dict(
        """
        INSERT INTO rule_sets (
            name, description, categories,
            public_id, published_at, is_local_draft, is_ready, created_by_username
        )
        VALUES (%s, %s, %s, %s, %s, FALSE, TRUE, %s)
        ON CONFLICT (name) DO UPDATE
        SET description         = COALESCE(NULLIF(EXCLUDED.description, ''), rule_sets.description),
            categories          = EXCLUDED.categories,
            public_id           = EXCLUDED.public_id,
            published_at        = EXCLUDED.published_at,
            is_local_draft      = FALSE,
            created_by_username = COALESCE(rule_sets.created_by_username, EXCLUDED.created_by_username)
        RETURNING rule_set_id
        """,
        (
            record.name,
            record.description,
            final_categories,
            record.public_id,
            record.published_at,
            creator,
        ),
    )
    rule_set_id = rows[0]["rule_set_id"]

    # Rebuild membership from the record (the registry is the source of truth
    # for which rules are in the set, and in what order).
    execute_query("DELETE FROM rule_set_member WHERE rule_set_id = %s", (rule_set_id,))
    for pos, rule_pid in enumerate(record.member_rules):
        member = execute_query_dict(
            "SELECT rule_id FROM rules WHERE public_id = %s", (rule_pid,)
        )
        if not member:
            logger.warning(
                "[hf_sync] rule set '%s' references rule %s not synced locally yet; "
                "skipping that member for now (heals on next sync)",
                record.name, rule_pid,
            )
            continue
        execute_query(
            "INSERT INTO rule_set_member (rule_set_id, rule_id, position) "
            "VALUES (%s, %s, %s) ON CONFLICT (rule_set_id, rule_id) DO NOTHING",
            (rule_set_id, member[0]["rule_id"], pos),
        )

    return rule_set_id


def _pull_rule_set(token: str, rule_set_public_id: str, result: SyncResult) -> bool:
    """Fetch a rule set record, validate, upsert. Same per-record atomicity as
    rules/CEs. MUST run after rules are pulled so member public_ids resolve."""
    try:
        payload = _fetch_record(token, f"public_rule_sets/{rule_set_public_id}.json")
        record = RuleSetRecord.model_validate(payload)
        local_id = _upsert_rule_set(record)
        if local_id is None:
            result.skipped_records.append(f"{rule_set_public_id} (local draft has same name)")
            return False
        return True
    except Exception as e:
        msg = f"rule set {rule_set_public_id}: {e}"
        logger.error(f"[hf_sync] {msg}")
        result.errors.append(msg)
        result.skipped_records.append(rule_set_public_id)
        return False


# --- Public entry point ---


def recover_pending_publishes(manifest: Optional[dict] = None, token: Optional[str] = None) -> dict:
    """Boot-time / sync-time crash recovery for the publish flow.

    The publish service stamps `pending_public_id` on a row right before the
    HF push and clears it on success/failure. If a process is killed between
    a successful HF push and the local finalize step, the row carries this
    stamp into the next session.

    For each such row, we ask the registry: does this public_id exist?
        - Yes → push succeeded; heal forward (set public_id, flip is_local_draft).
        - No  → push didn't land; clear the stamp, leave row as a regular draft.

    Returns counts: {healed_rules, healed_ces, healed_datasets, cleared_rules,
    cleared_ces, cleared_datasets}. Safe to call when nothing is pending — no
    DB writes occur.
    """
    out = {
        "healed_rules": 0, "healed_ces": 0, "healed_datasets": 0, "healed_rule_sets": 0,
        "cleared_rules": 0, "cleared_ces": 0, "cleared_datasets": 0, "cleared_rule_sets": 0,
    }

    pending_rules = execute_query_dict(
        "SELECT rule_id, pending_public_id FROM rules WHERE pending_public_id IS NOT NULL"
    ) or []
    pending_ces = execute_query_dict(
        "SELECT ce_id, pending_public_id FROM cognitive_elements WHERE pending_public_id IS NOT NULL"
    ) or []
    pending_datasets = execute_query_dict(
        "SELECT dataset_id, rule_id, pending_public_id FROM test_datasets WHERE pending_public_id IS NOT NULL"
    ) or []
    pending_rule_sets = execute_query_dict(
        "SELECT rule_set_id, pending_public_id FROM rule_sets WHERE pending_public_id IS NOT NULL"
    ) or []

    if not pending_rules and not pending_ces and not pending_datasets and not pending_rule_sets:
        return out  # nothing to do, no manifest fetch needed

    # We need the registry's manifest to decide each pending row's fate.
    # Reuse a passed-in manifest if the caller already has one (sync_library
    # always does), otherwise fetch fresh.
    if manifest is None or token is None:
        from services.hf_publish import _resolve_token, _fetch_head_sha_and_manifest
        token = token or _resolve_token()
        if not token:
            logger.warning("[hf_sync] recover_pending_publishes: no HF_TOKEN, skipping")
            return out
        try:
            _sha, manifest = _fetch_head_sha_and_manifest()
        except Exception as e:
            logger.warning(f"[hf_sync] recover_pending_publishes: manifest fetch failed: {e}")
            return out

    rules_in_registry = manifest.get("rules", {}) or {}
    ces_in_registry = manifest.get("ces", {}) or {}

    for row in pending_rules:
        pid = row["pending_public_id"]
        if pid in rules_in_registry:
            # Heal forward: HF has the record, finalize the local row.
            published_at = rules_in_registry[pid]
            execute_query(
                """
                UPDATE rules
                SET public_id = %s, published_at = %s, is_local_draft = FALSE,
                    pending_public_id = NULL
                WHERE rule_id = %s
                """,
                (pid, published_at, row["rule_id"]),
            )
            out["healed_rules"] += 1
            logger.info(f"[hf_sync] healed pending rule {row['rule_id']} -> {pid}")
        else:
            # Push didn't land; clear the stamp so the row goes back to a
            # normal draft state (next pass of cleanup will delete it).
            execute_query(
                "UPDATE rules SET pending_public_id = NULL WHERE rule_id = %s",
                (row["rule_id"],),
            )
            out["cleared_rules"] += 1

    for row in pending_ces:
        pid = row["pending_public_id"]
        if pid in ces_in_registry:
            published_at = ces_in_registry[pid]
            execute_query(
                """
                UPDATE cognitive_elements
                SET public_id = %s, published_at = %s, is_local_draft = FALSE,
                    pending_public_id = NULL
                WHERE ce_id = %s
                """,
                (pid, published_at, row["ce_id"]),
            )
            out["healed_ces"] += 1
            logger.info(f"[hf_sync] healed pending CE {row['ce_id']} -> {pid}")
        else:
            execute_query(
                "UPDATE cognitive_elements SET pending_public_id = NULL WHERE ce_id = %s",
                (row["ce_id"],),
            )
            out["cleared_ces"] += 1

    # Default test/calibration rows (schema v9). Their fate follows the rule
    # they belong to: rule_datasets[rule_public_id] in the manifest means the
    # push landed. Rules are healed above first, so the rule's public_id is
    # already finalized by the time we look it up here.
    rule_datasets_registry = manifest.get("rule_datasets", {}) or {}
    for row in pending_datasets:
        rule_pid_rows = execute_query_dict(
            "SELECT public_id FROM rules WHERE rule_id = %s", (row["rule_id"],)
        ) or []
        rule_pid = rule_pid_rows[0]["public_id"] if rule_pid_rows else None
        if rule_pid and rule_pid in rule_datasets_registry:
            execute_query(
                """
                UPDATE test_datasets
                SET public_id = %s, published_at = %s, pending_public_id = NULL
                WHERE dataset_id = %s
                """,
                (row["pending_public_id"], rule_datasets_registry[rule_pid], row["dataset_id"]),
            )
            out["healed_datasets"] += 1
        else:
            execute_query(
                "UPDATE test_datasets SET pending_public_id = NULL WHERE dataset_id = %s",
                (row["dataset_id"],),
            )
            out["cleared_datasets"] += 1

    # Rule sets. Unlike rules/CEs (durable user drafts), a rule_sets row with a
    # pending stamp and no public_id is a TRANSIENT publish artifact — the
    # private classifiers row is the durable source. So if the push didn't
    # land, DELETE the row (rather than just clearing the stamp), freeing its
    # UNIQUE(name) slot so the user can re-publish. If it DID land, heal forward.
    rule_sets_in_registry = manifest.get("rule_sets", {}) or {}
    for row in pending_rule_sets:
        pid = row["pending_public_id"]
        if pid in rule_sets_in_registry:
            published_at = rule_sets_in_registry[pid]
            execute_query(
                """
                UPDATE rule_sets
                SET public_id = %s, published_at = %s, is_local_draft = FALSE,
                    pending_public_id = NULL
                WHERE rule_set_id = %s
                """,
                (pid, published_at, row["rule_set_id"]),
            )
            out["healed_rule_sets"] += 1
            logger.info(f"[hf_sync] healed pending rule set {row['rule_set_id']} -> {pid}")
        else:
            execute_query(
                "DELETE FROM rule_sets WHERE rule_set_id = %s", (row["rule_set_id"],)
            )
            out["cleared_rule_sets"] += 1

    return out


def sync_library(force: bool = False) -> SyncResult:
    """Bring the local DB up to date with the public HF registry.

    Steps:
        1. Read HF_TOKEN. Abort with an error result if missing.
        2. Fetch manifest.json. If its content hash matches the last
           successful sync (and force is False), return changed=False
           immediately — no further work, no DB writes.
        3. Otherwise, diff the manifest against local public_ids. Pull
           every missing CE first (with its excitation), then every
           missing rule (which can now resolve its CE dependencies).
        4. Persist the new manifest hash so the next call short-circuits.

    Concurrency: serialized by `_sync_lock` so the server-startup background
    sync and a simultaneous user-login sync don't double-fetch / race the
    manifest hash check. The second caller blocks until the first finishes
    and then re-runs against the now-warm cache (which short-circuits via
    last_hash == current_hash and returns immediately).
    """
    with _sync_lock:
        return _sync_library_locked(force)


def _sync_library_locked(force: bool) -> SyncResult:
    # The public library repo (GavelPublicData/public-library) is readable
    # ANONYMOUSLY — a token is only needed for PUBLISHING (writes), which runs
    # on the central server. So a missing HF_TOKEN must NOT block the read-sync;
    # token stays None and huggingface_hub reads the public repo without auth.
    token = _resolve_token()   # kept — still passed to _fetch_record() below

    # Step 1 — cheap probe.
    try:
        manifest_bytes = _fetch_manifest_bytes()
    except Exception as e:
        return SyncResult(changed=False, errors=[f"Could not fetch manifest: {e}"])

    current_hash = _hash_bytes(manifest_bytes)
    last_hash = _get_state(_LAST_MANIFEST_HASH_KEY)

    if not force and last_hash == current_hash:
        logger.info("[hf_sync] manifest unchanged since last sync — no-op")
        # Even on a cache hit, run pending-publish recovery — there might
        # be a stamp left over from a publish that finished on HF before
        # this client was around to see the manifest change. Cheap when
        # nothing is pending (one DB lookup).
        try:
            recover_pending_publishes(manifest=json.loads(manifest_bytes), token=token)
        except Exception as e:
            logger.warning(f"[hf_sync] pending-publish recovery on cache-hit failed: {e}")
        return SyncResult(changed=False)

    # Step 2 — parse manifest.
    try:
        manifest = Manifest.model_validate(json.loads(manifest_bytes))
        manifest_dict = json.loads(manifest_bytes)
    except Exception as e:
        return SyncResult(changed=False, errors=[f"Manifest validation failed: {e}"])

    # Step 2-pre — boot-time crash recovery for the publish flow. Any row
    # carrying a pending_public_id from a killed-mid-publish gets healed
    # forward (if HF has the record) or cleared back to a draft. Cheap
    # when nothing is pending. See recover_pending_publishes for details.
    try:
        recover_pending_publishes(manifest=manifest_dict, token=token)
    except Exception as e:
        logger.warning(f"[hf_sync] pending-publish recovery failed: {e}")

    result = SyncResult(changed=True)

    # Step 2a — sync categories. Categories live in a single file at the
    # registry root; the manifest's categories_hash field signals when
    # they've changed. We always pull on a manifest change because:
    # (a) the file is small, (b) the upsert is idempotent against the
    # local table by name, and (c) skipping it would leave stale
    # category descriptions on every machine.
    if manifest.categories_hash:
        result.categories_synced = _pull_and_upsert_categories(token, result)

    # Step 2a' — sync the global neutral corpus (the evaluation's third split).
    # Like categories it's a root-level asset keyed by per-category hashes in
    # the manifest; pulled here so a neutral-only update still lands even when
    # the rules/CEs lists didn't move (the early-return below).
    if manifest.neutral:
        try:
            result.neutral_synced = _pull_neutral_corpus(token, manifest.neutral, result)
        except Exception as e:
            logger.warning(f"[hf_sync] neutral corpus pull failed: {e}")

    # Step 2b — diff rules and CEs. Two distinct kinds of work:
    #   * MISSING — public_id is in the manifest but not in our local DB → pull.
    #   * STALE   — public_id is present locally, but the manifest's
    #               published_at is strictly newer than our local copy →
    #               re-pull in place. This is the ONLY path by which an upstream
    #               EDIT that keeps the same public_id (an admin re-categorizing
    #               a seed CE, fixing a definition/predicate) reaches a client
    #               that already holds the record. Without it, edits to records
    #               we already have would silently never propagate — divergence.
    #               Upserts are keyed by name (ON CONFLICT DO UPDATE), so a
    #               refresh updates the existing row, never duplicates it.
    local_ce_ids, local_rule_ids = _local_public_ids()
    local_ce_pubat, local_rule_pubat = _local_pubat_map()
    missing_ce_ids = [pid for pid in manifest.ces if pid not in local_ce_ids]
    missing_rule_ids = [pid for pid in manifest.rules if pid not in local_rule_ids]
    stale_ce_ids = [
        pid for pid in manifest.ces
        if pid in local_ce_ids
        and _hf_pubat_is_newer(manifest.ces[pid], local_ce_pubat.get(pid))
    ]
    stale_rule_ids = [
        pid for pid in manifest.rules
        if pid in local_rule_ids
        and _hf_pubat_is_newer(manifest.rules[pid], local_rule_pubat.get(pid))
    ]

    # Pull order within each family: MISSING first (so freshly-arrived CEs are
    # available before rules try to resolve their dependencies by name), then
    # STALE refreshes appended after.
    ce_pull_ids = missing_ce_ids + stale_ce_ids
    rule_pull_ids = missing_rule_ids + stale_rule_ids

    # Rule sets diff (MISSING + STALE), same logic as rules/CEs. Computed here
    # (before the early-return) so a publish that adds ONLY a rule set — its
    # member rules already synced — still triggers a pull instead of being
    # short-circuited away. The actual pull runs AFTER rules below.
    manifest_rule_sets = manifest.rule_sets or {}
    local_rs_rows = execute_query_dict(
        "SELECT public_id, published_at FROM rule_sets WHERE public_id IS NOT NULL"
    ) or []
    local_rs_ids = {r["public_id"] for r in local_rs_rows}
    local_rs_pubat = {r["public_id"]: r["published_at"] for r in local_rs_rows}
    missing_rs_ids = [pid for pid in manifest_rule_sets if pid not in local_rs_ids]
    stale_rs_ids = [
        pid for pid in manifest_rule_sets
        if pid in local_rs_ids
        and _hf_pubat_is_newer(manifest_rule_sets[pid], local_rs_pubat.get(pid))
    ]
    rs_pull_ids = missing_rs_ids + stale_rs_ids

    if not ce_pull_ids and not rule_pull_ids and not rs_pull_ids:
        # Manifest hash changed but our local content is already complete AND
        # current. The common case after a categories-only / neutral-only
        # update (both handled above): the rules/ces/rule-sets lists didn't move
        # and no record was edited. Mark this manifest as seen so the next call
        # short-circuits.
        if not result.errors:
            _set_state(_LAST_MANIFEST_HASH_KEY, current_hash)
        return result

    # Step 3 — pull CEs and rules in parallel. Each pull is a few HTTP
    # round-trips bottlenecked on latency, not CPU, so a ThreadPoolExecutor
    # gives a 4–8× wall-clock speedup on a cold sync. CEs go first so rules
    # can resolve their CE dependencies; the two phases stay sequential.
    # Worker count is conservative — SQLite serializes writers (WAL +
    # busy_timeout make queued writes cheap), embedding model load is shared
    # (PyTorch forward passes are thread-safe for inference), and 8 saturates
    # HF's HTTPS keep-alive comfortably.
    from concurrent.futures import ThreadPoolExecutor
    _SYNC_WORKERS = 8

    # pool.map preserves input order, so an index below len(missing) was a
    # newly-added record; at/after that boundary it was a stale refresh.
    _n_missing_ce = len(missing_ce_ids)
    if ce_pull_ids:
        with ThreadPoolExecutor(max_workers=_SYNC_WORKERS, thread_name_prefix="hf-sync-ce") as pool:
            ce_outcomes = list(pool.map(lambda pid: _pull_ce(token, pid, result), ce_pull_ids))
        for idx, ok in enumerate(ce_outcomes):
            if ok:
                if idx < _n_missing_ce:
                    result.ces_added += 1
                else:
                    result.ces_refreshed += 1

    _n_missing_rule = len(missing_rule_ids)
    if rule_pull_ids:
        with ThreadPoolExecutor(max_workers=_SYNC_WORKERS, thread_name_prefix="hf-sync-rule") as pool:
            rule_outcomes = list(pool.map(lambda pid: _pull_rule(token, pid, result), rule_pull_ids))
        for idx, ok in enumerate(rule_outcomes):
            if ok:
                if idx < _n_missing_rule:
                    result.rules_added += 1
                else:
                    result.rules_refreshed += 1

    # Step 3c — rule sets, pulled LAST so their member rule public_ids resolve
    # to local rule_ids (rules were pulled just above). A member rule that
    # didn't make it (e.g. skipped for a local-draft name clash) is simply
    # omitted from the set and heals on a later sync.
    _n_missing_rs = len(missing_rs_ids)
    if rs_pull_ids:
        with ThreadPoolExecutor(max_workers=_SYNC_WORKERS, thread_name_prefix="hf-sync-ruleset") as pool:
            rs_outcomes = list(pool.map(lambda pid: _pull_rule_set(token, pid, result), rs_pull_ids))
        for idx, ok in enumerate(rs_outcomes):
            if ok:
                if idx < _n_missing_rs:
                    result.rule_sets_added += 1
                else:
                    result.rule_sets_refreshed += 1

    # Step 4 — ensure creators of synced content exist in the local users
    # mirror so FK triggers (ratings, contribution counts) and profile
    # JOINs resolve for users who have never logged in on this machine.
    if (result.ces_added or result.rules_added or result.rule_sets_added
            or result.ces_refreshed or result.rules_refreshed or result.rule_sets_refreshed):
        try:
            from sql_scripts.user_scripts import ensure_creators_in_local
            creator_rows = execute_query_dict(
                "SELECT DISTINCT created_by_username FROM ("
                "  SELECT created_by_username FROM rules WHERE created_by_username IS NOT NULL"
                "  UNION"
                "  SELECT created_by_username FROM cognitive_elements WHERE created_by_username IS NOT NULL"
                "  UNION"
                "  SELECT created_by_username FROM rule_sets WHERE created_by_username IS NOT NULL"
                ") AS creators"
            )
            if creator_rows:
                ensure_creators_in_local([r["created_by_username"] for r in creator_rows])
        except Exception as e:
            logger.warning(f"Creator user sync failed (non-fatal): {e}")

    # Step 5 — only mark this manifest as "seen" if no record was skipped.
    # If any pull failed, we want the next sync to retry the missing ones
    # rather than short-circuit on a stale hash.
    if not result.skipped_records and not result.errors:
        _set_state(_LAST_MANIFEST_HASH_KEY, current_hash)

    return result
