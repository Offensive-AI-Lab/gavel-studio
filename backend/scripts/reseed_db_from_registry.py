"""Reseed the local database from the public HF registry.

Replaces the old file-based reseed (which read from D:\\gavel) with a fresh
pull from GavelPublicData/public-library. Drops every local table, recreates
the schema (including the registry-link columns added by DButils.init_database),
and inserts the seed library exactly as it lives on HuggingFace.

What this preserves:
  * Every local row from the registry carries its public_id, so the next
    library sync sees nothing new to pull (no duplicates).
  * CE rows include the matching excitation dataset.
  * Rule rows include the role-aware CE links (necessary / fallback /
    sufficient) wired up to local ce_ids.

Run with the project venv:
    .venv\\Scripts\\python.exe scripts\\reseed_db_from_registry.py

Requires:
    HF_TOKEN in backend/.env (read access is enough — the token only needs
    write access for the bootstrap scripts).
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow imports from backend/ when run as a script.
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from utils.PostgreSQL import execute_query, execute_query_dict
from utils.DButils import drop_all_tables, init_database, normalize_and_upsert_categories
from utils.embedding_utils import trigger_embedding

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    print("[!] huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


# --- Configuration ---
# Resolved in main() AFTER load_dotenv so an HF_REPO_ID/HF_REPO_TYPE override
# in backend/.env is honored — must match hf_sync.REPO_ID's resolution.

REPO_ID = "GavelPublicData/public-library"
REPO_TYPE = "dataset"


# --- Helpers ---


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pull_registry(token: str) -> Path:
    """Download the entire registry repo to the local HF cache and return
    the snapshot directory. snapshot_download is one HTTP call, much
    cheaper than per-file fetches."""
    print(f"[*] Pulling registry from {REPO_ID}...")
    snapshot_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=token,
    )
    return Path(snapshot_dir)


# --- Insert helpers ---


def upsert_ce_from_hf(record: dict) -> int:
    """Insert a CE from its HF JSON. Returns the local ce_id.

    The HF record carries publisher / published_at / public_id, all of which
    we propagate to the new row. is_local_draft = FALSE because the row
    originated from the registry, not from a local edit.
    """
    final_categories = normalize_and_upsert_categories(
        record.get("categories", []) or [], allow_new=True
    )
    row = execute_query_dict(
        """
        INSERT INTO cognitive_elements (
            name, definition, category, categories, examples,
            public_id, published_at, is_local_draft
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE)
        ON CONFLICT (name) DO UPDATE
        SET definition       = EXCLUDED.definition,
            category         = EXCLUDED.category,
            categories       = EXCLUDED.categories,
            examples         = EXCLUDED.examples,
            public_id        = EXCLUDED.public_id,
            published_at     = EXCLUDED.published_at,
            is_local_draft   = FALSE
        RETURNING ce_id
        """,
        (
            record["name"],
            record.get("definition", ""),
            record.get("category", "CONTEXT"),
            final_categories,
            json.dumps(record.get("examples", [])),
            record["public_id"],
            record.get("published_at"),
        ),
    )
    return row[0]["ce_id"]


def upsert_excitation_from_hf(ce_id: int, excitation_record: dict) -> None:
    """Store the samples list (the conversation array) under the local CE.
    The wrapper fields (publisher / published_at / schema_version) are
    redundant once the data is in the local DB — we keep only what the
    guardrail engine actually consumes."""
    samples = excitation_record.get("samples", [])
    payload = {"samples": samples, "sample_count": len(samples)}
    execute_query(
        """
        INSERT INTO excitation_datasets (ce_id, dataset)
        VALUES (%s, %s)
        ON CONFLICT (ce_id) DO UPDATE SET dataset = EXCLUDED.dataset
        """,
        (ce_id, json.dumps(payload)),
    )


def upsert_rule_from_hf(record: dict, ce_public_id_to_local_id: dict) -> int:
    """Insert a rule from its HF JSON, then create the role-aware ce_links.

    The HF record's ce_dependencies are public_ids; we resolve each to its
    local ce_id via the map built earlier. Any unknown public_id signals a
    bug (we should have pulled every dep first), so we fail loudly rather
    than silently dropping the link.
    """
    row = execute_query_dict(
        """
        INSERT INTO rules (
            name, predicate, categories, description,
            public_id, published_at, is_local_draft
        )
        VALUES (%s, %s, %s, %s, %s, %s, FALSE)
        ON CONFLICT (name) DO UPDATE
        SET predicate        = EXCLUDED.predicate,
            categories       = EXCLUDED.categories,
            description      = EXCLUDED.description,
            public_id        = EXCLUDED.public_id,
            published_at     = EXCLUDED.published_at,
            is_local_draft   = FALSE
        RETURNING rule_id
        """,
        (
            record["name"],
            record.get("predicate", ""),
            normalize_and_upsert_categories(
                record.get("categories", []) or [], allow_new=True
            ),
            record.get("definition", ""),
            record["public_id"],
            record.get("published_at"),
        ),
    )
    rule_id = row[0]["rule_id"]

    # Wipe any prior links (if reseed found an existing row by name) so we
    # don't end up with a mix of old and new links.
    execute_query("DELETE FROM rule_ce_link WHERE rule_id = %s", (rule_id,))

    necessary = record.get("necessary", []) or []
    fallback = record.get("fallback", []) or []
    sufficient = record.get("sufficient", []) or []

    def _resolve(name: str) -> int:
        # The role lists hold CE *names* (matching the local DB), while
        # ce_dependencies holds public_ids. Local CE rows have the names
        # we just inserted, so we can resolve by name.
        rows = execute_query_dict(
            "SELECT ce_id FROM cognitive_elements WHERE name = %s", (name,)
        )
        if not rows:
            raise RuntimeError(
                f"Rule '{record['name']}' references unknown CE '{name}'"
            )
        return rows[0]["ce_id"]

    # rule_ce_link.fallback_group is NOT NULL with the convention "0 for
    # non-fallback roles, group index for fallback rows". The PK is
    # (rule_id, ce_id, role, fallback_group), so the same CE can appear
    # under different roles without collision.
    for ce_name in necessary:
        execute_query(
            "INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group) "
            "VALUES (%s, %s, 'necessary', 0)",
            (rule_id, _resolve(ce_name)),
        )
    for group_idx, group in enumerate(fallback):
        for ce_name in group:
            execute_query(
                "INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group) "
                "VALUES (%s, %s, 'fallback', %s)",
                (rule_id, _resolve(ce_name), group_idx),
            )
    for ce_name in sufficient:
        execute_query(
            "INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group) "
            "VALUES (%s, %s, 'sufficient', 0)",
            (rule_id, _resolve(ce_name)),
        )

    return rule_id


# --- Main ---


def main():
    load_dotenv(dotenv_path=BACKEND_DIR / ".env")

    global REPO_ID, REPO_TYPE
    REPO_ID = os.getenv("HF_REPO_ID", REPO_ID)
    REPO_TYPE = os.getenv("HF_REPO_TYPE", REPO_TYPE)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[!] HF_TOKEN not set in backend/.env. Aborting before any DB modification.")
        sys.exit(1)

    print("=" * 60)
    print(f"RESEED FROM HF REGISTRY ({REPO_ID})")
    print("=" * 60)

    # 0. Pull the entire registry into the local HF cache. We do this BEFORE
    # any destructive DB action so a network failure can't leave the user
    # with a wiped DB and no way to repopulate it.
    snapshot = _pull_registry(token)
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        print(f"[!] manifest.json missing in the snapshot at {snapshot}")
        sys.exit(1)
    manifest = _load_json(manifest_path)
    ce_ids = list(manifest.get("ces", {}).keys())
    rule_ids = list(manifest.get("rules", {}).keys())
    print(f"[*] Manifest: {len(ce_ids)} CEs, {len(rule_ids)} rules")

    # Validate every record file is present locally before we drop the DB.
    missing = []
    for ce_pid in ce_ids:
        if not (snapshot / "public_ces" / f"{ce_pid}.json").is_file():
            missing.append(f"public_ces/{ce_pid}.json")
        if not (snapshot / "public_excitation" / f"excitation_{ce_pid}.json").is_file():
            missing.append(f"public_excitation/excitation_{ce_pid}.json")
    for rid in rule_ids:
        if not (snapshot / "public_rules" / f"{rid}.json").is_file():
            missing.append(f"public_rules/{rid}.json")
    if missing:
        print(f"[!] {len(missing)} record file(s) missing from the snapshot:")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        print("DB was NOT modified.")
        sys.exit(1)

    # 1. Drop & reinitialize. From here on, an interrupt leaves the local
    # DB in a partial state, but the registry is untouched, so re-running
    # this script always recovers.
    drop_all_tables()
    init_database()

    # 2. Insert CEs + their excitation datasets.
    seeded_ces = 0
    for ce_pid in ce_ids:
        ce_record = _load_json(snapshot / "public_ces" / f"{ce_pid}.json")
        excitation = _load_json(
            snapshot / "public_excitation" / f"excitation_{ce_pid}.json"
        )
        local_ce_id = upsert_ce_from_hf(ce_record)
        upsert_excitation_from_hf(local_ce_id, excitation)
        # Trigger embedding + tsvector update for hybrid search.
        try:
            trigger_embedding("ce", local_ce_id, ce_record["name"], ce_record.get("definition", ""))
        except Exception as e:
            print(f"  [!] Embedding for CE '{ce_record['name']}': {e}")
        seeded_ces += 1
    print(f"[OK] Seeded {seeded_ces} CEs with their excitation datasets")

    # 3. Build a public_id -> local_ce_id map for resolving rule deps.
    rows = execute_query_dict(
        "SELECT ce_id, public_id FROM cognitive_elements WHERE public_id IS NOT NULL"
    ) or []
    ce_public_to_local = {r["public_id"]: r["ce_id"] for r in rows}

    # 4. Insert rules.
    seeded_rules = 0
    for rid in rule_ids:
        rule_record = _load_json(snapshot / "public_rules" / f"{rid}.json")
        try:
            local_rule_id = upsert_rule_from_hf(rule_record, ce_public_to_local)
            seeded_rules += 1
            # Trigger embedding for the rule too.
            try:
                ce_definitions = " ".join(
                    rule_record.get("ce_dependencies", [])
                )
                trigger_embedding(
                    "rule",
                    local_rule_id,
                    rule_record["name"],
                    rule_record.get("predicate", ""),
                    ce_definitions,
                )
            except Exception as e:
                print(f"  [!] Embedding for rule '{rule_record['name']}': {e}")
        except Exception as e:
            print(f"  [!] Skipped rule '{rule_record.get('name', rid)}': {e}")
    print(f"[OK] Seeded {seeded_rules} rules")

    print()
    print(f"[OK] Reseed complete from {REPO_ID}.")
    print(f"     Snapshot at: {snapshot}")


if __name__ == "__main__":
    main()
