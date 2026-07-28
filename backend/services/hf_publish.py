"""Publish service: push local drafts to the HF registry atomically.

Two entry points:
  * publish_ce(ce_id)   — push one CE + its excitation in a single atomic commit.
  * publish_rule(rule_id) — push a rule and any of its CE dependencies that are
                            still local-only drafts, all together.

Every publish runs the same five-step contract:
  1. sync_library(): pull any new records from the registry first, so step 2
     dedup-checks against the current state of the world.
  2. Dedup check: if a *published* row with the same name already exists in
     the local DB after the sync, return CONFLICT — do not push.
  3. Build the commit operations + a fresh manifest with the new public_ids.
  4. Race-checked push: pass the registry HEAD's commit SHA as parent_commit.
     If anyone pushed between our sync and our push, HF rejects, we re-sync,
     and we report RACE so the caller can prompt the user.
  5. On success, finalize the local rows: set public_id, published_at, flip
     is_local_draft to false. Update sync_state so the next sync_library()
     short-circuits.

Failure semantics (from user requirements):
  * SUCCESS  — local row is published, public_id stamped.
  * CONFLICT — name already taken by another published record. Local draft
               kept so the user can rename or fork.
  * RACE     — registry HEAD moved during push. Local draft kept so the user
               can retry; the new content from the racing pusher is now in
               the local DB via the re-sync.
  * ERROR    — unrecoverable failure (validation, network, schema). The
               local draft is deleted per the "if something goes wrong we
               delete it" requirement.
"""
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from utils.PostgreSQL import execute_query, execute_query_dict
from services.hf_sync import REPO_ID, REPO_TYPE, sync_library

logger = logging.getLogger(__name__)


# --- Result types ---


class PublishStatus(str, Enum):
    SUCCESS = "success"
    CONFLICT = "conflict"  # name already taken
    RACE = "race"          # HEAD moved during push
    ERROR = "error"


@dataclass
class PublishResult:
    status: PublishStatus
    public_id: Optional[str] = None
    name: Optional[str] = None
    conflict_with: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "public_id": self.public_id,
            "name": self.name,
            "conflict_with": self.conflict_with,
            "error": self.error,
        }


# --- Helpers ---


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_bytes(payload: dict) -> bytes:
    """Stable JSON encoding so manifest hashes match across runs."""
    return json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")


def _resolve_token() -> Optional[str]:
    """HF token (optional for reads). Public repos read fine without a
    token. Writes use the same HF_TOKEN via services/hf_write. Returns
    None if HF_TOKEN is unset OR empty — a blank token would yield an
    illegal ``Bearer `` header that breaks even the anonymous reads (see
    hf_sync._resolve_token for the full rationale)."""
    backend_dir = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=backend_dir / ".env")
    return (os.environ.get("HF_TOKEN") or "").strip() or None


def _is_race_error(exc: Exception) -> bool:
    """HF returns 412 (Precondition Failed) when parent_commit is stale.
    Detection is by string match because the SDK doesn't expose a typed
    exception for this case."""
    msg = str(exc)
    return (
        "412" in msg
        or "precondition" in msg.lower()
        or "stale" in msg.lower()
        or "fetch first" in msg.lower()
        or "out-of-date" in msg.lower()
    )


# --- Local row loaders ---


def _resolve_username(publisher_user_id: Optional[int]) -> Optional[str]:
    """Look up the publishing user's canonical (lowercase) username so it
    can be stamped on the HF artifact + local row.

    Reads from the LOCAL users table, where the single local user is seeded
    at boot by DButils.ensure_local_user().

    Returns None if no user_id was provided (back-compat path for internal
    callers that don't have user context). Raises if a user_id was given
    but doesn't resolve — that's a programming error, not a soft case."""
    if publisher_user_id is None:
        return None
    from utils.PostgreSQL import execute_query_dict
    rows = execute_query_dict(
        "SELECT username FROM users WHERE user_id = %s", (publisher_user_id,)
    )
    if not rows:
        raise RuntimeError(
            f"publish flow received user_id={publisher_user_id} but no matching user row."
        )
    return rows[0]["username"]


def _load_ce_row(ce_id: int) -> Optional[dict]:
    rows = execute_query_dict(
        """
        SELECT ce_id, name, definition, category, categories, examples,
               public_id, is_local_draft, created_by_username
        FROM cognitive_elements WHERE ce_id = %s
        """,
        (ce_id,),
    )
    return rows[0] if rows else None


def _load_rule_row(rule_id: int) -> Optional[dict]:
    rows = execute_query_dict(
        """
        SELECT rule_id, name, predicate, categories, description,
               public_id, is_local_draft, is_ready, created_by_username
        FROM rules WHERE rule_id = %s
        """,
        (rule_id,),
    )
    return rows[0] if rows else None


def _load_excitation_samples(ce_id: int) -> Optional[list]:
    """Return the raw conversation list for a CE's excitation dataset, or
    None if no excitation row exists. Handles both the new
    {"samples": [...], "sample_count": N} wrapper and the legacy raw-array
    format."""
    rows = execute_query_dict(
        "SELECT dataset FROM excitation_datasets WHERE ce_id = %s", (ce_id,)
    )
    if not rows:
        return None
    raw = rows[0]["dataset"]
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(payload, list):
        return payload  # legacy
    return payload.get("samples", [])


def _load_calibration_samples(ce_id: int) -> Optional[list]:
    """Return the raw conversation list for a CE's calibration dataset, or
    None if no row exists.

    The calibration table uses a `conversations` key (vs excitation's
    `samples`) because that's the local naming used by the calibration
    pipeline. We translate it to `samples` on the way out so the HF
    record schema is consistent with excitation."""
    rows = execute_query_dict(
        "SELECT dataset FROM calibration_datasets WHERE ce_id = %s", (ce_id,)
    )
    if not rows:
        return None
    raw = rows[0]["dataset"]
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(payload, list):
        return payload  # legacy
    return payload.get("conversations", payload.get("samples", []))


# Rule-level calibration is no longer published or stored in a separate
# table — it lives in `test_datasets` (dataset_type='positive_calibration')
# and stays private to the local backend.


def _category_names_from_ids(cat_ids: List[int]) -> List[str]:
    """Translate categories-int-array into category names (the HF schema
    stores names, the local DB stores IDs)."""
    if not cat_ids:
        return []
    rows = execute_query_dict(
        "SELECT name FROM categories WHERE category_id = ANY(%s)", (list(cat_ids),)
    ) or []
    return [r["name"] for r in rows]


def _ce_role_lists_for_rule(rule_id: int) -> Tuple[List[dict], List[List[dict]], List[dict]]:
    """Return (necessary, fallback, sufficient) where each entry is a
    {ce_id, name, public_id, is_local_draft} dict. Fallback is grouped by
    fallback_group."""
    rows = execute_query_dict(
        """
        SELECT rcl.ce_id, rcl.role, rcl.fallback_group,
               ce.name, ce.public_id, ce.is_local_draft
        FROM rule_ce_link rcl
        JOIN cognitive_elements ce ON rcl.ce_id = ce.ce_id
        WHERE rcl.rule_id = %s
        ORDER BY rcl.role, rcl.fallback_group, ce.name
        """,
        (rule_id,),
    ) or []

    necessary: List[dict] = []
    sufficient: List[dict] = []
    fallback_groups: Dict[int, List[dict]] = {}

    for row in rows:
        info = {
            "ce_id": row["ce_id"],
            "name": row["name"],
            "public_id": row["public_id"],
            "is_local_draft": row["is_local_draft"],
        }
        if row["role"] == "necessary":
            necessary.append(info)
        elif row["role"] == "sufficient":
            sufficient.append(info)
        elif row["role"] == "fallback":
            fallback_groups.setdefault(row["fallback_group"], []).append(info)

    fallback = [fallback_groups[k] for k in sorted(fallback_groups.keys())]
    return necessary, fallback, sufficient


# --- Payload builders ---


def _build_ce_payload(ce_row: dict, public_id: str, published_at: str) -> dict:
    examples = ce_row.get("examples") or []
    if isinstance(examples, str):
        examples = json.loads(examples)
    payload = {
        "schema_version": 1,
        "public_id": public_id,
        "name": ce_row["name"],
        "definition": ce_row.get("definition") or "",
        "category": ce_row.get("category") or "CONTEXT",
        "categories": _category_names_from_ids(ce_row.get("categories") or []),
        "examples": examples,
        "published_at": published_at,
    }
    # Stamp creator only when we have one. Pre-feature artifacts in the
    # registry have no created_by_username; the sync flow defaults those
    # to the configured seed-team user, so omitting the field for clearly
    # anonymous publishes (e.g., legacy backfill) is safe.
    creator = ce_row.get("created_by_username")
    if creator:
        payload["created_by_username"] = creator
    return payload


def _build_excitation_payload(samples: list, ce_public_id: str, published_at: str) -> dict:
    return {
        "schema_version": 1,
        "ce_public_id": ce_public_id,
        "samples": samples,
        "sample_count": len(samples) if isinstance(samples, list) else 0,
        "published_at": published_at,
    }


def _build_ce_calibration_payload(samples: list, ce_public_id: str, published_at: str) -> dict:
    """Same envelope as excitation — calibration records ride along with
    the CE definition + excitation in a single atomic publish."""
    return {
        "schema_version": 1,
        "ce_public_id": ce_public_id,
        "samples": samples,
        "sample_count": len(samples) if isinstance(samples, list) else 0,
        "published_at": published_at,
    }


def _rule_dataset_path(rule_public_id: str, dataset_type: str) -> str:
    """HF path for a rule's default dataset file (one per dataset_type)."""
    return f"public_rule_datasets/{rule_public_id}_{dataset_type}.json"


# Only these config keys are relevant to a *consumer* of a published default
# set: the scenario it tests (provenance + the pre-fill / regenerate source)
# and the labels that describe what the dialogues contain. Everything else in
# the generation config (persona/style pools, seed dialogues, generator/judge
# model names, ideation + dialogue controls) is generation machinery the
# consumer never needs, so we strip it before publishing.
_PUBLISHED_CONFIG_KEYS = ("scenario_instructions", "necessary_labels", "sufficient_labels")


def _slim_dataset_config(config: dict) -> dict:
    """Keep only the consumer-relevant config keys for HF."""
    return {k: config[k] for k in _PUBLISHED_CONFIG_KEYS if k in config}


def _build_rule_dataset_payload(row: dict, rule_public_id: str, published_at: str) -> dict:
    """Payload for one default test/calibration bucket. DIALOGUES + a slimmed
    config only — never generation internals, thresholds, or metrics."""
    convos = row.get("conversations") or []
    if isinstance(convos, str):
        try:
            convos = json.loads(convos)
        except Exception:
            convos = []
    config = row.get("config") or {}
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except Exception:
            config = {}
    return {
        "schema_version": 1,
        "rule_public_id": rule_public_id,
        "dataset_type": row["dataset_type"],
        "config": _slim_dataset_config(config),
        "conversations": convos,
        "conversation_count": len(convos) if isinstance(convos, list) else 0,
        "published_at": published_at,
    }


def _load_default_dataset_rows(rule_id: int) -> list:
    """The rule's three default (is_default=TRUE) test_datasets rows."""
    return execute_query_dict(
        """SELECT dataset_id, dataset_type, status, config, conversations
           FROM test_datasets
           WHERE rule_id = %s AND is_default = TRUE""",
        (rule_id,),
    ) or []


def _build_rule_payload(
    rule_row: dict,
    necessary: List[dict],
    fallback: List[List[dict]],
    sufficient: List[dict],
    public_id: str,
    published_at: str,
    name_to_public_id: Dict[str, str],
) -> dict:
    """The rule's role lists are NAMES (matching local DB); ce_dependencies
    is the union of public_ids of every CE referenced. name_to_public_id
    must include every CE in the role lists (either pre-existing public_id
    or freshly minted in this commit)."""
    all_ces = set()
    for c in necessary:
        all_ces.add(c["name"])
    for group in fallback:
        for c in group:
            all_ces.add(c["name"])
    for c in sufficient:
        all_ces.add(c["name"])

    ce_dependencies = sorted({name_to_public_id[n] for n in all_ces if n in name_to_public_id})

    # Publish the predicate DERIVED from the role lists, so the canonical HF data
    # is always consistent with the roles and never re-publishes a stale string
    # that OR'd in helpful/'sufficient' CEs (which never fire a rule).
    from sql_scripts.model_scripts import predicate_from_role_lists
    derived_predicate = predicate_from_role_lists(
        [c["name"] for c in necessary],
        [[c["name"] for c in group] for group in fallback],
    )

    payload = {
        "schema_version": 1,
        "public_id": public_id,
        "name": rule_row["name"],
        "predicate": derived_predicate or (rule_row.get("predicate") or ""),
        "necessary": [c["name"] for c in necessary],
        "fallback": [[c["name"] for c in group] for group in fallback],
        "sufficient": [c["name"] for c in sufficient],
        "categories": _category_names_from_ids(rule_row.get("categories") or []),
        "definition": rule_row.get("description") or "",
        "ce_dependencies": ce_dependencies,
        "published_at": published_at,
    }
    creator = rule_row.get("created_by_username")
    if creator:
        payload["created_by_username"] = creator
    return payload


# --- HF I/O ---


def _fetch_head_sha_and_manifest() -> Tuple[str, dict]:
    """Return (HEAD commit SHA, parsed manifest).

    HEAD SHA is read straight from HF with the write token (see
    services/hf_write). The manifest is downloaded directly from HF as a
    public asset — anonymous access works for the public registry, and we
    use the local HF_TOKEN as a fallback for private repos / rate limiting.
    """
    from services import hf_write
    from huggingface_hub import hf_hub_download

    head_sha = hf_write.hf_head_sha()
    if not head_sha:
        raise RuntimeError("HF returned no HEAD SHA")

    local_read_token = _resolve_token()  # OK if None — public repos don't need it
    path = hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, filename="manifest.json",
        token=local_read_token, force_download=True,
    )
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return head_sha, manifest


def _push_atomic(operations: list, parent_sha: str, message: str):
    """Commit a batch of file ops directly to HF (services/hf_write).

    `operations` is a list of plain dicts `{"path": str, "content": bytes}`.

    Raises on race or any other HF error so the caller can distinguish
    via _is_race_error(). On race, hf_write returns status='race' — we
    translate that to a raised exception so the existing catch logic in
    publish_ce / publish_rule works unchanged.
    """
    from services import hf_write
    from services.hf_write import HFWriteError

    try:
        resp = hf_write.hf_commit(
            operations=operations,
            commit_message=message,
            parent_commit=parent_sha,
        )
    except HFWriteError as err:
        raise RuntimeError(f"HF commit failed: {err}")

    status = resp.get("status") if isinstance(resp, dict) else None
    if status == "race":
        raise RuntimeError(f"412 race: {resp.get('error', '')}")
    if status != "success":
        raise RuntimeError(resp.get("error") or "Unknown HF commit error")
    return resp


def _op(path: str, content: bytes) -> dict:
    """Build one commit operation in the dict format hf_write.hf_commit
    expects."""
    return {"path": path, "content": content}


def _sync_is_fresh(window_seconds: int = 30) -> bool:
    """Return True when sync_state shows a successful sync within the
    last `window_seconds`. Used to short-circuit the upfront sync that
    publish_ce / publish_rule normally do.

    Safety: skipping the pre-publish sync only means we *might* attempt
    a publish against a slightly stale manifest. The race-checked push
    (`_push_atomic` pins `parent_commit` to the HEAD SHA we read at
    build time) still rejects the commit if the registry moved
    underneath us — at which point the caller retries with a fresh
    HEAD. Worst case: one extra HF round-trip in a rare race window.
    No data corruption is possible.

    `window_seconds = 30` matches the frontend's 90s background poller
    closely enough that an interactive publish (user clicks Publish a
    second or two after a sync completed) will hit this fast path.
    Background tasks that publish far from a sync still fall through
    and run the full sync — they always pay the cost regardless."""
    try:
        rows = execute_query_dict(
            """
            SELECT EXTRACT(EPOCH FROM (now() - updated_at)) AS age_seconds
            FROM sync_state
            WHERE key = 'last_manifest_hash'
            """
        )
        if not rows or rows[0]["age_seconds"] is None:
            return False
        return float(rows[0]["age_seconds"]) < window_seconds
    except Exception:
        # If the freshness check itself fails (e.g., sync_state table
        # missing in a partial init), fall back to "not fresh" — i.e.,
        # we'll do the sync. Safer to be slower than to skip.
        return False


def _set_sync_state(manifest: dict) -> None:
    """Persist the new manifest hash so the next sync_library short-circuits
    instead of redundantly pulling our own commit."""
    new_hash = hashlib.sha256(_to_bytes(manifest)).hexdigest()
    execute_query(
        """
        INSERT INTO sync_state (key, value, updated_at)
        VALUES ('last_manifest_hash', %s, now())
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
        """,
        (new_hash,),
    )


def _record_pushed_manifest_hash(commit_resp, local_manifest: dict) -> None:
    """Cache last_manifest_hash to the sha256 of the manifest bytes that were
    ACTUALLY committed, which hf_write.hf_commit returns as `manifest_sha256`.

    Nothing rewrites the manifest anymore (version stamping went away with the
    central server), so this hash equals the hash of our local copy — but we
    still prefer the value from the commit response as the authoritative
    record of what landed on HF, so our own next freshness probe
    short-circuits and never flashes a phantom "update available" for a
    publish we already hold. Falls back to hashing the local manifest if the
    response carried none."""
    sha = commit_resp.get("manifest_sha256") if isinstance(commit_resp, dict) else None
    if sha:
        execute_query(
            """
            INSERT INTO sync_state (key, value, updated_at)
            VALUES ('last_manifest_hash', %s, now())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
            """,
            (sha,),
        )
        return
    _set_sync_state(local_manifest)


# --- Dedup checks ---


def _ce_name_already_published(name: str, exclude_id: int) -> Optional[dict]:
    """Return the existing published CE row if one exists with this name
    (and isn't ourselves), or None."""
    rows = execute_query_dict(
        """
        SELECT ce_id, name, public_id FROM cognitive_elements
        WHERE name = %s AND public_id IS NOT NULL AND ce_id != %s
        """,
        (name, exclude_id),
    )
    return rows[0] if rows else None


def _rule_name_already_published(name: str, exclude_id: int) -> Optional[dict]:
    rows = execute_query_dict(
        """
        SELECT rule_id, name, public_id FROM rules
        WHERE name = %s AND public_id IS NOT NULL AND rule_id != %s
        """,
        (name, exclude_id),
    )
    return rows[0] if rows else None


def _rule_set_name_already_published(name: str) -> Optional[dict]:
    """Return the existing published rule_sets row with this name, or None.
    No exclude_id: the publish flow mints a fresh rule_sets row, so any
    already-published row with the same name is a genuine conflict."""
    rows = execute_query_dict(
        """
        SELECT rule_set_id, name, public_id FROM rule_sets
        WHERE name = %s AND public_id IS NOT NULL
        """,
        (name,),
    )
    return rows[0] if rows else None


# --- Rule-set loaders / payload ---


def _load_classifier_row(classifier_id: int) -> Optional[dict]:
    """The private rule set being published (UI: a model-less guardrail).
    Only its name is exported — the model, training, and thresholds are
    never serialized."""
    rows = execute_query_dict(
        "SELECT classifier_id, name FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    return rows[0] if rows else None


def _rule_set_members(classifier_id: int) -> List[dict]:
    """The rules that make up this rule set, in author order (setup_id).

    Each entry: {setup_id, rule_id, display_name, public_id, is_local_draft,
    categories}. rule_id / public_id are NULL for a manual (rule_id NULL) or
    still-draft member — those are caught by the members-published-first gate.
    """
    return execute_query_dict(
        """
        SELECT rs.setup_id,
               rs.rule_id,
               COALESCE(rs.custom_name, r.name, 'Untitled rule') AS display_name,
               r.public_id,
               r.is_local_draft,
               r.categories
        FROM rule_setup rs
        LEFT JOIN rules r ON rs.rule_id = r.rule_id
        WHERE rs.classifier_id = %s
        ORDER BY rs.setup_id
        """,
        (classifier_id,),
    ) or []


def _build_rule_set_payload(
    name: str,
    description: str,
    category_ids: List[int],
    member_public_ids: List[str],
    public_id: str,
    published_at: str,
    creator: Optional[str],
) -> dict:
    """A thin pointer-collection: member_rules are the ORDERED member rule
    public_ids; categories are NAMES (the local DB stores int ids)."""
    payload = {
        "schema_version": 1,
        "public_id": public_id,
        "name": name,
        "description": description or "",
        "categories": _category_names_from_ids(category_ids),
        "member_rules": member_public_ids,
        "published_at": published_at,
    }
    if creator:
        payload["created_by_username"] = creator
    return payload


# --- publish_ce ---


def publish_ce(ce_id: int, publisher_user_id: Optional[int] = None) -> PublishResult:
    """Push a single CE + its excitation in one atomic HF commit.

    Pre: a local CE row with is_local_draft=true and a populated
    excitation_datasets row. Post: the CE row carries its new public_id,
    or it has been deleted (on hard error) / kept (on conflict / race).

    publisher_user_id is the user whose authenticated request triggered
    this publish. If supplied, their username is stamped onto the local
    row and into the HF payload so attribution flows through to /profile
    pages and the "by [user]" Browse link. None is allowed for legacy
    internal callers (bootstrap scripts, etc.) that don't have user
    context — in that case the row's existing created_by_username (if
    any) is preserved and used.
    """
    publisher_username = _resolve_username(publisher_user_id)

    # Step 1 — sync first, unless the local manifest was refreshed
    # very recently (background poller already covered us). The
    # race-checked push below catches any actual conflict regardless,
    # so skipping a redundant sync only costs a possible retry — and
    # saves a multi-second HF round-trip on the happy path.
    if not _sync_is_fresh():
        try:
            sync_library()
        except Exception as e:
            return PublishResult(PublishStatus.ERROR, error=f"Pre-publish sync failed: {e}")

    ce = _load_ce_row(ce_id)
    if not ce:
        return PublishResult(PublishStatus.ERROR, error=f"CE {ce_id} not found")
    if ce["public_id"]:
        return PublishResult(
            PublishStatus.ERROR,
            error=f"CE already published as {ce['public_id']}",
        )

    # Stamp creator on the local row before building the HF payload, so
    # the row in DB and the artifact JSON agree, and so heal-forward on
    # a mid-publish crash sees the right author. Only overwrite if we
    # have a fresh authenticated user; never blank out an existing value.
    if publisher_username and ce.get("created_by_username") != publisher_username:
        execute_query(
            "UPDATE cognitive_elements SET created_by_username = %s WHERE ce_id = %s",
            (publisher_username, ce_id),
        )
        ce["created_by_username"] = publisher_username

    samples = _load_excitation_samples(ce_id)
    if samples is None:
        return PublishResult(
            PublishStatus.ERROR,
            error="No training data for this CE; generate it before publishing",
        )

    # Step 2 — local dedup check.
    existing = _ce_name_already_published(ce["name"], ce_id)
    if existing:
        return PublishResult(
            PublishStatus.CONFLICT,
            name=ce["name"],
            conflict_with={
                "type": "ce", "name": existing["name"],
                "public_id": existing["public_id"],
            },
        )

    # Step 3 — fetch manifest first (we'll check the registry-side name
    # index before building anything).
    try:
        head_sha, manifest = _fetch_head_sha_and_manifest()
    except Exception as e:
        return PublishResult(PublishStatus.ERROR, error=f"Could not read registry HEAD: {e}")

    # Step 3b — registry name-index check. Catches the case where another
    # user just published a CE with the same name and our local sync
    # hasn't yet absorbed it (or skipped it because of our draft). This is
    # the second of the three layered defenses against same-name collisions.
    ce_name_index = manifest.get("ce_names", {}) or {}
    if ce["name"] in ce_name_index:
        existing_pid = ce_name_index[ce["name"]]
        return PublishResult(
            PublishStatus.CONFLICT,
            name=ce["name"],
            conflict_with={
                "type": "ce", "name": ce["name"],
                "public_id": existing_pid,
            },
            error="A CE with this name already exists in the public registry.",
        )

    # Step 4 — build payloads now that name is confirmed available.
    new_public_id = f"ce_{uuid.uuid4().hex}"
    published_at = _now_iso()

    ce_payload = _build_ce_payload(ce, new_public_id, published_at)
    excitation_payload = _build_excitation_payload(samples, new_public_id, published_at)

    # Calibration is optional — only publish if the local DB has one for
    # this CE. CEs created via the AI pipeline almost always have one
    # (the pipeline auto-generates calibration alongside excitation);
    # CEs imported / hand-built may not. Either case is fine.
    calibration_samples = _load_calibration_samples(ce_id)

    manifest.setdefault("ces", {})
    manifest["ces"][new_public_id] = published_at
    manifest.setdefault("ce_names", {})
    manifest["ce_names"][ce["name"]] = new_public_id
    if calibration_samples is not None:
        manifest.setdefault("ce_calibration", {})
        manifest["ce_calibration"][new_public_id] = published_at

    operations = [
        _op(f"public_ces/{new_public_id}.json", _to_bytes(ce_payload)),
        _op(f"public_excitation/excitation_{new_public_id}.json", _to_bytes(excitation_payload)),
        _op("manifest.json", _to_bytes(manifest)),
    ]
    if calibration_samples is not None:
        operations.append(_op(
            f"public_calibration/ce_{new_public_id}.json",
            _to_bytes(
                _build_ce_calibration_payload(calibration_samples, new_public_id, published_at)
            ),
        ))

    # Stamp the intent before the push so a process kill between push
    # success and local-row update can be recovered on next startup.
    # See services/hf_sync.recover_pending_publishes for the recovery
    # logic. The stamp commits immediately (autocommit on execute_query).
    execute_query(
        "UPDATE cognitive_elements SET pending_public_id = %s WHERE ce_id = %s",
        (new_public_id, ce_id),
    )

    # Step 4 — race-checked push via central server.
    try:
        commit_resp = _push_atomic(operations, head_sha, f"Publish CE: {ce['name']}")
    except Exception as e:
        # Push failed. Clear the stamp so the row goes back to a normal
        # draft state — the next session won't try to recover it.
        try:
            execute_query(
                "UPDATE cognitive_elements SET pending_public_id = NULL WHERE ce_id = %s",
                (ce_id,),
            )
        except Exception:
            pass

        if _is_race_error(e):
            try:
                sync_library()
            except Exception:
                pass
            existing = _ce_name_already_published(ce["name"], ce_id)
            if existing:
                return PublishResult(
                    PublishStatus.CONFLICT,
                    name=ce["name"],
                    conflict_with={
                        "type": "ce", "name": existing["name"],
                        "public_id": existing["public_id"],
                    },
                    error="Another user published this name during your push.",
                )
            return PublishResult(
                PublishStatus.RACE,
                name=ce["name"],
                error="Registry was updated during your push. Please retry.",
            )
        # Hard error — delete the local draft per the user's requirement.
        try:
            execute_query("DELETE FROM cognitive_elements WHERE ce_id = %s", (ce_id,))
        except Exception:
            logger.exception("[hf_publish] failed to delete CE %s after publish error", ce_id)
        return PublishResult(PublishStatus.ERROR, error=f"Publish failed (CE deleted): {e}")

    # Step 5 — finalize local row. Clears pending_public_id and stamps the
    # real public_id in a single statement so the row never carries both.
    execute_query(
        """
        UPDATE cognitive_elements
        SET public_id = %s, published_at = %s, is_local_draft = FALSE,
            pending_public_id = NULL
        WHERE ce_id = %s
        """,
        (new_public_id, published_at, ce_id),
    )
    _record_pushed_manifest_hash(commit_resp, manifest)


    return PublishResult(PublishStatus.SUCCESS, public_id=new_public_id, name=ce["name"])


# --- publish_rule ---


def publish_rule(rule_id: int, publisher_user_id: Optional[int] = None) -> PublishResult:
    """Push a rule and any of its draft CE dependencies in a single atomic
    commit. If the rule references CEs that are already published, those
    are reused (their public_ids land in ce_dependencies). Drafts get
    minted public_ids in this same commit, paired with their excitation
    files.

    Pre: rule row + rule_ce_link rows + every linked CE has either a
    public_id (already published) or a populated excitation_datasets row
    (will be published with the rule).

    Post: rule + every freshly published CE carry their new public_ids
    AND their created_by_username (the publishing user's name).
    On error: rule row is deleted, CE drafts are kept (still valid for
    a later publish).

    publisher_user_id is the authenticated user. Their username is
    stamped on the rule AND on every draft CE that gets published
    alongside it. None is allowed for legacy / bootstrap callers — the
    rows keep whatever creator they already had.
    """
    publisher_username = _resolve_username(publisher_user_id)

    # Skip the upfront sync when the local cache is fresh (see
    # _sync_is_fresh docstring for the safety argument — short version:
    # the race-checked push still catches actual conflicts).
    if not _sync_is_fresh():
        try:
            sync_library()
        except Exception as e:
            return PublishResult(PublishStatus.ERROR, error=f"Pre-publish sync failed: {e}")

    rule = _load_rule_row(rule_id)
    if not rule:
        return PublishResult(PublishStatus.ERROR, error=f"Rule {rule_id} not found")
    if rule["public_id"]:
        return PublishResult(
            PublishStatus.ERROR,
            error=f"Rule already published as {rule['public_id']}",
        )

    # Stamp creator on the rule row early so heal-forward sees the right
    # author if we crash between here and finalize.
    if publisher_username and rule.get("created_by_username") != publisher_username:
        execute_query(
            "UPDATE rules SET created_by_username = %s WHERE rule_id = %s",
            (publisher_username, rule_id),
        )
        rule["created_by_username"] = publisher_username

    necessary, fallback, sufficient = _ce_role_lists_for_rule(rule_id)
    all_linked_ces = list(necessary) + [c for g in fallback for c in g] + list(sufficient)

    if not all_linked_ces:
        return PublishResult(
            PublishStatus.ERROR,
            error="Rule has no linked CEs; cannot publish an empty rule",
        )

    # --- Holistic readiness gate -------------------------------------------
    # A rule may only publish when it was created 100% correctly. If anything
    # is still half-baked — the rule itself not finalized, a linked CE with no
    # training data, or the default test/calibration set missing/incomplete —
    # refuse with one clear, friendly message rather than publishing a broken
    # artifact. The user can regenerate and try again.
    _NOT_READY = (
        "This rule wasn't created properly — some of its parts didn't finish "
        "generating ({what}). Nothing was published. Please regenerate the rule "
        "and try again once everything is ready."
    )
    if rule.get("is_ready") is False:
        return PublishResult(PublishStatus.ERROR, name=rule["name"],
                             error=_NOT_READY.format(what="the rule is still finalizing"))

    # Every linked CE must be ready (finalized + has training data).
    _ce_ids = [c["ce_id"] for c in all_linked_ces]
    _unready_ce = execute_query_dict(
        """
        SELECT c.name
        FROM cognitive_elements c
        WHERE c.ce_id = ANY(%s)
          AND (
                c.is_ready = FALSE
             OR (c.public_id IS NULL
                 AND NOT EXISTS (SELECT 1 FROM excitation_datasets e WHERE e.ce_id = c.ce_id))
          )
        """,
        (_ce_ids,),
    ) or []
    if _unready_ce:
        names = ", ".join(r["name"] for r in _unready_ce)
        return PublishResult(PublishStatus.ERROR, name=rule["name"],
                             error=_NOT_READY.format(what=f"cognitive element(s) not ready: {names}"))

    # The default test/calibration set (positive / negative / positive_calibration).
    default_rows = _load_default_dataset_rows(rule_id)
    _by_type = {r["dataset_type"]: r for r in default_rows}
    _required_types = ("positive", "negative", "positive_calibration")
    _bad = [t for t in _required_types
            if t not in _by_type or _by_type[t].get("status") != "ready"]
    if _bad:
        return PublishResult(PublishStatus.ERROR, name=rule["name"],
                             error=_NOT_READY.format(what=f"default test/calibration set: {', '.join(_bad)}"))

    # Step 2 — local dedup check on the rule name.
    existing_rule = _rule_name_already_published(rule["name"], rule_id)
    if existing_rule:
        return PublishResult(
            PublishStatus.CONFLICT,
            name=rule["name"],
            conflict_with={
                "type": "rule", "name": existing_rule["name"],
                "public_id": existing_rule["public_id"],
            },
        )

    # Fetch HEAD's manifest once. Used for both the registry name-index
    # check below and for building the updated manifest at push time.
    try:
        head_sha, manifest = _fetch_head_sha_and_manifest()
    except Exception as e:
        return PublishResult(PublishStatus.ERROR, error=f"Could not read registry HEAD: {e}")

    # Registry-side name-index check on the rule. Catches collisions our
    # local sync may have skipped because of an existing local draft.
    rule_name_index = manifest.get("rule_names", {}) or {}
    if rule["name"] in rule_name_index:
        existing_pid = rule_name_index[rule["name"]]
        return PublishResult(
            PublishStatus.CONFLICT,
            name=rule["name"],
            conflict_with={
                "type": "rule", "name": rule["name"],
                "public_id": existing_pid,
            },
            error="A rule with this name already exists in the public registry.",
        )

    # Step 3 — partition linked CEs into "already published" vs "draft to publish here".
    # Dedup by ce_id since a single CE can show up in multiple role lists.
    seen_ce_ids = set()
    already_published_ces = []  # rows with public_id
    draft_ces = []               # rows we'll publish in this commit
    for c in all_linked_ces:
        if c["ce_id"] in seen_ce_ids:
            continue
        seen_ce_ids.add(c["ce_id"])
        if c["public_id"]:
            already_published_ces.append(c)
        else:
            draft_ces.append(c)

    # Drafts also get name-dedup'd (same race window as the rule). Two
    # checks: the local DB (catches an already-pulled published CE with
    # the same name) and the registry's ce_names index (catches a CE that
    # exists in the registry but was skipped during sync because of our
    # local draft).
    ce_name_index = manifest.get("ce_names", {}) or {}
    for d in draft_ces:
        existing_ce = _ce_name_already_published(d["name"], d["ce_id"])
        if existing_ce:
            return PublishResult(
                PublishStatus.CONFLICT,
                name=d["name"],
                conflict_with={
                    "type": "ce", "name": existing_ce["name"],
                    "public_id": existing_ce["public_id"],
                    "local_ce_id": d["ce_id"],
                },
                error=(
                    f"This rule's CE '{d['name']}' clashes with an already-published CE. "
                    "Rename or fork the CE before publishing the rule."
                ),
            )
        if d["name"] in ce_name_index:
            existing_pid = ce_name_index[d["name"]]
            return PublishResult(
                PublishStatus.CONFLICT,
                name=d["name"],
                conflict_with={
                    "type": "ce", "name": d["name"],
                    "public_id": existing_pid,
                    "local_ce_id": d["ce_id"],
                },
                error=(
                    f"This rule's CE '{d['name']}' clashes with a CE already in the public registry. "
                    "Rename or fork the CE before publishing the rule."
                ),
            )

    # For each draft CE, load its excitation. Required.
    draft_ce_excitations: Dict[int, list] = {}
    # Calibration samples are optional — load opportunistically so the
    # commit can carry them if the local DB has them, but a missing row
    # doesn't block the rule publish.
    draft_ce_calibrations: Dict[int, list] = {}
    for d in draft_ces:
        samples = _load_excitation_samples(d["ce_id"])
        if samples is None:
            return PublishResult(
                PublishStatus.ERROR,
                error=f"CE '{d['name']}' has no training data; generate before publishing",
            )
        draft_ce_excitations[d["ce_id"]] = samples
        cal_samples = _load_calibration_samples(d["ce_id"])
        if cal_samples is not None:
            draft_ce_calibrations[d["ce_id"]] = cal_samples

    # (Default-dataset readiness was already enforced by the holistic gate
    # near the top; `default_rows` / `_by_type` / `_required_types` from there
    # are reused below to build + finalize the dataset ops.)

    # Step 4 — generate public_ids for everything we're publishing.
    published_at = _now_iso()
    new_rule_public_id = f"rule_{uuid.uuid4().hex}"
    new_ce_public_ids: Dict[int, str] = {
        d["ce_id"]: f"ce_{uuid.uuid4().hex}" for d in draft_ces
    }

    # Build the name -> public_id map used by the rule payload.
    name_to_public_id: Dict[str, str] = {}
    for c in already_published_ces:
        name_to_public_id[c["name"]] = c["public_id"]
    for d in draft_ces:
        name_to_public_id[d["name"]] = new_ce_public_ids[d["ce_id"]]

    # Stamp creator on every draft CE that's getting published in this
    # commit. Only overwrites NULL or a different name — never blanks
    # out an existing creator. This must happen BEFORE _load_ce_row
    # below, so the row dict picks up the fresh value naturally.
    if publisher_username and draft_ces:
        draft_ce_ids = [d["ce_id"] for d in draft_ces]
        execute_query(
            """
            UPDATE cognitive_elements
            SET created_by_username = %s
            WHERE ce_id = ANY(%s)
              AND (created_by_username IS NULL OR created_by_username <> %s)
            """,
            (publisher_username, draft_ce_ids, publisher_username),
        )

    # Build payloads.
    operations: list = []

    for d in draft_ces:
        ce_row = _load_ce_row(d["ce_id"])
        ce_pid = new_ce_public_ids[d["ce_id"]]
        operations.append(_op(
            f"public_ces/{ce_pid}.json",
            _to_bytes(_build_ce_payload(ce_row, ce_pid, published_at)),
        ))
        operations.append(_op(
            f"public_excitation/excitation_{ce_pid}.json",
            _to_bytes(_build_excitation_payload(
                draft_ce_excitations[d["ce_id"]], ce_pid, published_at,
            )),
        ))
        if d["ce_id"] in draft_ce_calibrations:
            operations.append(_op(
                f"public_calibration/ce_{ce_pid}.json",
                _to_bytes(_build_ce_calibration_payload(
                    draft_ce_calibrations[d["ce_id"]], ce_pid, published_at,
                )),
            ))

    rule_payload = _build_rule_payload(
        rule, necessary, fallback, sufficient,
        new_rule_public_id, published_at, name_to_public_id,
    )
    operations.append(_op(
        f"public_rules/{new_rule_public_id}.json",
        _to_bytes(rule_payload),
    ))

    # The rule's DEFAULT test/calibration set (positive / negative /
    # positive_calibration) rides along in the same atomic commit, one file
    # per bucket. Dialogues + config only — never thresholds/metrics. This
    # is the shared, reproducible benchmark every adopter of the rule sees.
    # (CE-level calibration also still publishes alongside each CE.)
    for _t in _required_types:
        operations.append(_op(
            _rule_dataset_path(new_rule_public_id, _t),
            _to_bytes(_build_rule_dataset_payload(_by_type[_t], new_rule_public_id, published_at)),
        ))

    # Updated manifest. We already fetched head_sha + manifest at the top
    # of this function; just amend the in-memory dict here.
    manifest.setdefault("ces", {})
    manifest.setdefault("rules", {})
    manifest.setdefault("ce_names", {})
    manifest.setdefault("rule_names", {})
    manifest.setdefault("ce_calibration", {})
    manifest.setdefault("rule_datasets", {})
    for d in draft_ces:
        ce_pid = new_ce_public_ids[d["ce_id"]]
        manifest["ces"][ce_pid] = published_at
        manifest["ce_names"][d["name"]] = ce_pid
        if d["ce_id"] in draft_ce_calibrations:
            manifest["ce_calibration"][ce_pid] = published_at
    manifest["rules"][new_rule_public_id] = published_at
    manifest["rule_names"][rule["name"]] = new_rule_public_id
    # One entry per rule covers all three default dataset files.
    manifest["rule_datasets"][new_rule_public_id] = published_at

    operations.append(_op("manifest.json", _to_bytes(manifest)))

    # Stamp the intent on every row this commit will publish, so a crash
    # between push success and the local finalize step can be recovered on
    # next session. See services/hf_sync.recover_pending_publishes().
    for d in draft_ces:
        execute_query(
            "UPDATE cognitive_elements SET pending_public_id = %s WHERE ce_id = %s",
            (new_ce_public_ids[d["ce_id"]], d["ce_id"]),
        )
    execute_query(
        "UPDATE rules SET pending_public_id = %s WHERE rule_id = %s",
        (new_rule_public_id, rule_id),
    )
    # Stamp each default dataset row with its per-bucket composite public_id.
    for _t in _required_types:
        execute_query(
            "UPDATE test_datasets SET pending_public_id = %s WHERE dataset_id = %s",
            (f"{new_rule_public_id}_{_t}", _by_type[_t]["dataset_id"]),
        )

    # Step 5 — race-checked push.
    try:
        commit_resp = _push_atomic(
            operations, head_sha,
            f"Publish rule: {rule['name']} (+{len(draft_ces)} CE{'s' if len(draft_ces) != 1 else ''})",
        )
    except Exception as e:
        # Push failed. Clear all the stamps we just placed.
        try:
            execute_query("UPDATE rules SET pending_public_id = NULL WHERE rule_id = %s", (rule_id,))
            for d in draft_ces:
                execute_query(
                    "UPDATE cognitive_elements SET pending_public_id = NULL WHERE ce_id = %s",
                    (d["ce_id"],),
                )
            for r in default_rows:
                execute_query(
                    "UPDATE test_datasets SET pending_public_id = NULL WHERE dataset_id = %s",
                    (r["dataset_id"],),
                )
        except Exception:
            pass

        if _is_race_error(e):
            try:
                sync_library()
            except Exception:
                pass
            existing = _rule_name_already_published(rule["name"], rule_id)
            if existing:
                return PublishResult(
                    PublishStatus.CONFLICT,
                    name=rule["name"],
                    conflict_with={
                        "type": "rule", "name": existing["name"],
                        "public_id": existing["public_id"],
                    },
                    error="Another user published this rule name during your push.",
                )
            return PublishResult(
                PublishStatus.RACE,
                name=rule["name"],
                error="Registry was updated during your push. Please retry.",
            )
        # Hard error — delete only the rule (CE drafts stay reusable).
        try:
            execute_query("DELETE FROM rules WHERE rule_id = %s", (rule_id,))
        except Exception:
            logger.exception("[hf_publish] failed to delete rule %s after publish error", rule_id)
        return PublishResult(PublishStatus.ERROR, error=f"Publish failed (rule deleted): {e}")

    # Step 6 — finalize all local rows. Clears pending_public_id atomically
    # with the public_id stamp so the row never carries both.
    for d in draft_ces:
        execute_query(
            """
            UPDATE cognitive_elements
            SET public_id = %s, published_at = %s, is_local_draft = FALSE,
                pending_public_id = NULL
            WHERE ce_id = %s
            """,
            (new_ce_public_ids[d["ce_id"]], published_at, d["ce_id"]),
        )
    execute_query(
        """
        UPDATE rules
        SET public_id = %s, published_at = %s, is_local_draft = FALSE,
            pending_public_id = NULL
        WHERE rule_id = %s
        """,
        (new_rule_public_id, published_at, rule_id),
    )
    # Finalize the default dataset rows: stamp their composite public_id +
    # published_at, clear the pending stamp.
    for _t in _required_types:
        execute_query(
            """
            UPDATE test_datasets
            SET public_id = %s, published_at = %s, pending_public_id = NULL
            WHERE dataset_id = %s
            """,
            (f"{new_rule_public_id}_{_t}", published_at, _by_type[_t]["dataset_id"]),
        )
    _record_pushed_manifest_hash(commit_resp, manifest)


    return PublishResult(PublishStatus.SUCCESS, public_id=new_rule_public_id, name=rule["name"])


# --- publish_rule_set ---


def publish_rule_set(classifier_id: int, publisher_user_id: Optional[int] = None) -> PublishResult:
    """Publish a private rule set (a model-less guardrail / `classifiers` row)
    to the public registry as a model-agnostic, shareable rule collection.

    The published artifact is a SEPARATE `rule_sets` row + a thin
    public_rule_sets/<pid>.json record that references its member rules by
    their existing rule public_ids — it is NOT the private `classifiers` row,
    which stays fully editable / deletable. Only the rule selection is shared;
    the model, training data, thresholds, and metrics are never serialized.

    v1 contract — MEMBERS PUBLISHED FIRST: every member rule must already be
    public (have a public_id). If any member is still a draft (or a manual
    rule with no global backing row), the publish is REFUSED with a clear
    message and nothing is written. This keeps the commit a pure pointer
    collection with no dangling references and the smallest possible blast
    radius (no cascade of draft-rule publishes).

    Failure semantics mirror publish_rule, except the transient `rule_sets`
    row is DELETED on any non-success (the durable artifact is the private
    `classifiers` row, which we never touch): SUCCESS / CONFLICT (name taken) /
    RACE (HEAD moved) / ERROR.
    """
    publisher_username = _resolve_username(publisher_user_id)

    # Step 1 — sync first unless the local cache is fresh (the race-checked
    # push still catches any actual conflict).
    if not _sync_is_fresh():
        try:
            sync_library()
        except Exception as e:
            return PublishResult(PublishStatus.ERROR, error=f"Pre-publish sync failed: {e}")

    classifier = _load_classifier_row(classifier_id)
    if not classifier:
        return PublishResult(PublishStatus.ERROR, error=f"Rule set {classifier_id} not found")
    name = classifier["name"]

    members = _rule_set_members(classifier_id)
    if not members:
        return PublishResult(
            PublishStatus.ERROR, name=name,
            error="This rule set has no rules yet; add at least one rule before sharing.",
        )

    # --- Members-published-first gate -------------------------------------
    # Every member must already be a public rule (public_id set). List the
    # ones that aren't so the user knows exactly what to publish first.
    unpublished = [m["display_name"] for m in members if not m["public_id"]]
    if unpublished:
        listed = ", ".join(dict.fromkeys(unpublished))  # de-dup, keep order
        return PublishResult(
            PublishStatus.ERROR, name=name,
            error=(
                "Every rule in a shared rule set must be published first. "
                f"These rules aren't public yet: {listed}. "
                "Publish them from the rule editor, then share the set."
            ),
        )

    # Ordered, de-duplicated member public_ids + the union of their categories.
    member_public_ids: List[str] = []
    seen_pids = set()
    category_ids: List[int] = []
    seen_cats = set()
    for m in members:
        pid = m["public_id"]
        if pid not in seen_pids:
            seen_pids.add(pid)
            member_public_ids.append(pid)
        for cid in (m.get("categories") or []):
            if cid not in seen_cats:
                seen_cats.add(cid)
                category_ids.append(cid)

    # Step 2 — local dedup on the rule-set name.
    existing_set = _rule_set_name_already_published(name)
    if existing_set:
        return PublishResult(
            PublishStatus.CONFLICT,
            name=name,
            conflict_with={
                "type": "rule_set", "name": existing_set["name"],
                "public_id": existing_set["public_id"],
            },
        )

    # Step 3 — fetch HEAD + manifest; registry-side name-index check.
    try:
        head_sha, manifest = _fetch_head_sha_and_manifest()
    except Exception as e:
        return PublishResult(PublishStatus.ERROR, error=f"Could not read registry HEAD: {e}")

    rule_set_name_index = manifest.get("rule_set_names", {}) or {}
    if name in rule_set_name_index:
        return PublishResult(
            PublishStatus.CONFLICT,
            name=name,
            conflict_with={
                "type": "rule_set", "name": name,
                "public_id": rule_set_name_index[name],
            },
            error="A rule set with this name already exists in the public registry.",
        )

    # Step 4 — mint the public_id + build the record now that the name is free.
    new_public_id = f"ruleset_{uuid.uuid4().hex}"
    published_at = _now_iso()
    creator = publisher_username  # None for legacy/internal callers

    payload = _build_rule_set_payload(
        name, "", category_ids, member_public_ids, new_public_id, published_at, creator,
    )

    manifest.setdefault("rule_sets", {})
    manifest.setdefault("rule_set_names", {})
    manifest["rule_sets"][new_public_id] = published_at
    manifest["rule_set_names"][name] = new_public_id

    operations = [
        _op(f"public_rule_sets/{new_public_id}.json", _to_bytes(payload)),
        _op("manifest.json", _to_bytes(manifest)),
    ]

    # Create the TRANSIENT local rule_sets row + membership, stamped with
    # pending_public_id for crash recovery (recover_pending_publishes heals
    # forward / clears it on next sync). The private classifiers row is left
    # untouched; this rule_sets row is a derived publish artifact.
    rows = execute_query_dict(
        """
        INSERT INTO rule_sets
            (name, description, categories, is_local_draft, is_ready,
             created_by_username, pending_public_id)
        VALUES (%s, %s, %s, TRUE, TRUE, %s, %s)
        RETURNING rule_set_id
        """,
        (name, "", category_ids, creator, new_public_id),
    )
    rule_set_id = rows[0]["rule_set_id"]
    for pos, m in enumerate(members):
        # members are all published here; reference by local rule_id. Skip
        # exact duplicate (rule_set_id, rule_id) pairs — PK forbids them.
        execute_query(
            """
            INSERT INTO rule_set_member (rule_set_id, rule_id, position)
            VALUES (%s, %s, %s)
            ON CONFLICT (rule_set_id, rule_id) DO NOTHING
            """,
            (rule_set_id, m["rule_id"], pos),
        )

    def _drop_transient_row():
        try:
            execute_query("DELETE FROM rule_sets WHERE rule_set_id = %s", (rule_set_id,))
        except Exception:
            logger.exception("[hf_publish] failed to drop transient rule_set %s", rule_set_id)

    # Step 5 — race-checked push.
    try:
        commit_resp = _push_atomic(
            operations, head_sha,
            f"Publish rule set: {name} ({len(member_public_ids)} rule{'s' if len(member_public_ids) != 1 else ''})",
        )
    except Exception as e:
        if _is_race_error(e):
            # Drop our transient row, re-sync, and let the user retry.
            _drop_transient_row()
            try:
                sync_library()
            except Exception:
                pass
            existing = _rule_set_name_already_published(name)
            if existing:
                return PublishResult(
                    PublishStatus.CONFLICT, name=name,
                    conflict_with={
                        "type": "rule_set", "name": existing["name"],
                        "public_id": existing["public_id"],
                    },
                    error="Another user published this rule-set name during your push.",
                )
            return PublishResult(
                PublishStatus.RACE, name=name,
                error="Registry was updated during your push. Please retry.",
            )
        # Hard error — drop the transient artifact (the private rule set stays).
        _drop_transient_row()
        return PublishResult(PublishStatus.ERROR, name=name, error=f"Publish failed: {e}")

    # Step 6 — finalize: stamp public_id/published_at, flip out of draft,
    # clear the pending stamp.
    execute_query(
        """
        UPDATE rule_sets
        SET public_id = %s, published_at = %s, is_local_draft = FALSE,
            pending_public_id = NULL
        WHERE rule_set_id = %s
        """,
        (new_public_id, published_at, rule_set_id),
    )
    _record_pushed_manifest_hash(commit_resp, manifest)

    return PublishResult(PublishStatus.SUCCESS, public_id=new_public_id, name=name)
