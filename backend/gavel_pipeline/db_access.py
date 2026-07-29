import collections
import json
from typing import Dict, List

from utils.sqlite_db import execute_query, execute_query_dict
from utils.embedding_utils import trigger_embedding
from utils.DButils import normalize_and_upsert_categories




def fetch_categories_dict() -> List[Dict[str, str]]:
    """Fetch active categories with id, name and description."""
    rows = execute_query_dict(
        "SELECT category_id, name, description FROM categories WHERE active = TRUE ORDER BY category_id"
    ) or []
    return [{"id": row["category_id"], "name": row["name"], "description": row["description"]} for row in rows]

def upsert_category(name: str, description: str):
    """Upsert a category with description."""
    execute_query(
        """
        INSERT INTO categories (name, description, active)
        VALUES (%s, %s, TRUE)
        ON CONFLICT (name)
        DO UPDATE SET description = EXCLUDED.description, active = TRUE
        """,
        (name, description)
    )

def fetch_reference_datasets() -> Dict[str, List]:
    """
    Fetches reference datasets from the 'excitation_datasets' table.
    Returns a dictionary mapping CE name to the raw dataset (list of conversation lists).
    """
    rows = execute_query_dict(
        """
        SELECT ce.name as ce_name, ed.dataset
        FROM excitation_datasets ed
        JOIN cognitive_elements ce ON ed.ce_id = ce.ce_id
        """
    )
    
    reference_datasets = {}
    for row in rows:
        ce_name = row['ce_name']
        dataset = row['dataset']
        
        # If dataset is returned as a string (JSON serialization), parse it first
        if isinstance(dataset, str):
            try:
                dataset = json.loads(dataset)
            except Exception:
                pass

        # Unpack if it follows the LongString/origin pattern (storage artifact)
        if isinstance(dataset, dict) and dataset.get("type") == "LongString" and "origin" in dataset:
            try:
                dataset_origin = dataset["origin"]
                if isinstance(dataset_origin, str):
                     dataset = json.loads(dataset_origin)
            except Exception:
                pass

        # Unwrap the {"samples": [...], "sample_count": N} envelope that
        # services/hf_sync._upsert_excitation writes for HF-pulled CEs.
        # load_reference_examples expects a flat list of conversations and
        # would otherwise raise KeyError(0) on every synced CE — visible as
        # the "Could not processing reference dataset for X: 0" warnings.
        if isinstance(dataset, dict) and isinstance(dataset.get("samples"), list):
            dataset = dataset["samples"]

        reference_datasets[ce_name] = dataset
        
    return reference_datasets

def fetch_ces_dict() -> Dict[str, dict]:
    """Return CE dict shaped like legacy JSON: {name: {definition, examples, note?}}.

    Skips is_ready=FALSE rows — those are in-flight AI-pipeline outputs
    that haven't finished generating yet, and showing them to the rule-
    generator prompt would let the AI propose CEs that are mid-creation.
    """
    rows = execute_query_dict(
        """
        SELECT ce_id, name, definition, note, examples
        FROM cognitive_elements
        WHERE is_ready = TRUE
        ORDER BY name
        """
    ) or []

    ces = {}
    for row in rows:
        examples = row.get("examples") or []
        ces[row["name"]] = {
            "definition": row.get("definition", ""),
            "examples": examples if isinstance(examples, list) else [],
        }
        if row.get("note"):
            ces[row["name"]]["note"] = row["note"]
    return ces


def fetch_rules_dict() -> Dict[str, dict]:
    """Return rules dict shaped like legacy JSON: {name: {necessary, fallback, sufficient, predicate}}.

    Skips is_ready=FALSE rows for the same reason fetch_ces_dict does.
    """
    rule_rows = execute_query_dict(
        """
        SELECT rule_id, name, predicate
        FROM rules
        WHERE is_ready = TRUE
        ORDER BY name
        """
    ) or []
    if not rule_rows:
        return {}

    link_rows = execute_query_dict(
        """
        SELECT r.rule_id,
               r.name AS rule_name,
               ce.name AS ce_name,
               link.role,
               link.fallback_group
        FROM rule_ce_link link
        JOIN rules r ON r.rule_id = link.rule_id
        JOIN cognitive_elements ce ON ce.ce_id = link.ce_id
        ORDER BY r.rule_id, link.role, link.fallback_group, ce.name
        """
    ) or []

    rules: Dict[int, dict] = {row["rule_id"]: {
        "name": row["name"], 
        "predicate": row.get("predicate", ""),
        "necessary": [], 
        "fallback": [], 
        "sufficient": []
    } for row in rule_rows}

    fallback_groups: Dict[int, Dict[int, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))

    for row in link_rows:
        rid = row["rule_id"]
        ce_name = row["ce_name"]
        role = row.get("role", "necessary")
        group = row.get("fallback_group", 0) or 0

        if rid not in rules:
            continue

        if role == "necessary":
            rules[rid]["necessary"].append(ce_name)
        elif role == "sufficient":
            rules[rid]["sufficient"].append(ce_name)
        elif role == "fallback":
            fallback_groups[rid][group].append(ce_name)

    # materialize fallback groups ordered by group id
    for rid, groups in fallback_groups.items():
        ordered = []
        for gid in sorted(groups.keys()):
            ordered.append(groups[gid])
        rules[rid]["fallback"] = ordered

    # convert to name-keyed dict
    named_rules: Dict[str, dict] = {}
    for rid, payload in rules.items():
        named_rules[payload["name"]] = {
            "necessary": payload.get("necessary", []),
            "fallback": payload.get("fallback", []),
            "sufficient": payload.get("sufficient", []),
            "predicate": payload.get("predicate", "")
        }
    return named_rules


def upsert_ces(new_ces: Dict[str, dict]) -> List[int]:
    """Insert/update CEs and return their ids."""
    if not new_ces:
        return []

    inserted_ids: List[int] = []
    for name, payload in new_ces.items():
        definition = payload.get("definition", "")
        category = payload.get("category", "CONTEXT")
        # Use provided categories list directly;
        raw_categories = payload.get("categories") or []
        categories = normalize_and_upsert_categories(raw_categories, allow_new=True) if raw_categories else []
        note = payload.get("note")
        examples = payload.get("examples") or []

        row = execute_query_dict(
            """
            INSERT INTO cognitive_elements (name, definition, category, categories, note, examples)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (name)
            DO UPDATE SET definition = EXCLUDED.definition,
                          category = EXCLUDED.category,
                          categories = EXCLUDED.categories,
                          note = COALESCE(EXCLUDED.note, cognitive_elements.note),
                          examples = COALESCE(EXCLUDED.examples, cognitive_elements.examples)
            RETURNING ce_id
            """,
            (name, definition, category, categories, note, examples),
        )
        if row:
            ce_id = row[0]["ce_id"] if isinstance(row[0], dict) else row[0][0]
            inserted_ids.append(ce_id)
            trigger_embedding("ce", ce_id, name, definition)
            
    return inserted_ids


def upsert_rule_with_links(rule_data: dict, mark_pending: bool = False) -> int:
    """Insert/update rule + CE links using role/fallback_group semantics. Returns rule_id.

    `mark_pending=True` is used by the AI pipeline to mark the rule as
    is_ready=FALSE so it doesn't appear in any user-facing list until
    /ai/embed-resources flips it to TRUE post-training. The boot-time
    IncompletePipelineRecovery wipes any is_ready=FALSE rows, so a crash
    cleanly looks like the rule was never created.

    On UPDATE (rule with this name already existed), we leave is_ready
    alone — we don't want to silently un-publish or un-finish a row that
    was already complete.
    """
    rule_name = rule_data["rule_name"]
    description = rule_data.get("description", "")
    # Predicate may not exist yet; store description as placeholder if missing.
    predicate = rule_data.get("predicate") or description or ""

    # Structural-dedup: a rule with the same role/fallback shape as an
    # existing rule under a DIFFERENT name is a functional duplicate
    # even though no name conflict fires. Catch it here so AI-generated
    # and manually-created rules both get checked at the single choke
    # point. `exclude_name=rule_name` lets the same rule be re-saved
    # (the AI pipeline updates predicate/description on retry runs).
    from sql_scripts.junction_scripts import (
        compute_rule_fingerprint_from_names,
        find_existing_rule_by_fingerprint,
    )
    new_fp = compute_rule_fingerprint_from_names(
        rule_data.get("necessary") or [],
        rule_data.get("fallback") or [],
        rule_data.get("sufficient") or [],
    )
    duplicate = find_existing_rule_by_fingerprint(new_fp, exclude_name=rule_name)
    if duplicate is not None:
        existing_name = duplicate.get("name") or "(unnamed)"
        raise ValueError(
            f"A rule with the same structure already exists as "
            f"'{existing_name}'. Two rules with identical CEs in "
            f"identical roles and fallback groups are functionally the "
            f"same — bookmark or fork the existing rule, or change the "
            f"new rule's logic."
        )

    raw_categories = rule_data.get("categories") or []
    categories = normalize_and_upsert_categories(raw_categories, allow_new=True)

    rule_row = execute_query_dict(
        """
        INSERT INTO rules (name, predicate, description, categories, is_ready)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (name)
        DO UPDATE SET predicate = EXCLUDED.predicate,
                      description = EXCLUDED.description,
                      categories = EXCLUDED.categories
        RETURNING rule_id
        """,
        (rule_name, predicate, description, categories, not mark_pending),
    )
    rule_id = rule_row[0]["rule_id"] if isinstance(rule_row[0], dict) else rule_row[0][0]

    # Clear existing links for this rule_id
    execute_query("DELETE FROM rule_ce_link WHERE rule_id = %s", (rule_id,))

    # Necessary
    for ce_name in rule_data.get("necessary", []) or []:
        execute_query(
            """
            INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group)
            SELECT %s, ce_id, 'necessary', 0 FROM cognitive_elements WHERE name = %s
            ON CONFLICT DO NOTHING
            """,
            (rule_id, ce_name),
        )

    # Sufficient
    for ce_name in rule_data.get("sufficient", []) or []:
        execute_query(
            """
            INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group)
            SELECT %s, ce_id, 'sufficient', 0 FROM cognitive_elements WHERE name = %s
            ON CONFLICT DO NOTHING
            """,
            (rule_id, ce_name),
        )

    # Fallback groups
    fallback = rule_data.get("fallback", []) or []
    for idx, group in enumerate(fallback, start=1):
        for ce_name in group:
            execute_query(
                """
                INSERT INTO rule_ce_link (rule_id, ce_id, role, fallback_group)
                SELECT %s, ce_id, 'fallback', %s FROM cognitive_elements WHERE name = %s
                ON CONFLICT DO NOTHING
                """,
                (rule_id, idx, ce_name),
            )

    # --- Auto-Embed Rule ---
    try:
        all_ce_names = set(rule_data.get("necessary", []) or [])
        all_ce_names.update(rule_data.get("sufficient", []) or [])
        for group in rule_data.get("fallback", []) or []:
            all_ce_names.update(group)
            
        ce_defs_str = ""
        if all_ce_names:
            # Postgres IN clause requires tuple
            names_tuple = tuple(all_ce_names)
            if len(names_tuple) == 1:
                # tuple('str') is ('s','t','r'), need ('str',)
                # execute_query_dict handles params better but let's build dynamic query carefully or just iterate?
                # Using ANY is cleaner
                rows = execute_query_dict(
                    "SELECT definition FROM cognitive_elements WHERE name = ANY(%s)",
                    (list(all_ce_names),)
                )
            else:
                rows = execute_query_dict(
                    "SELECT definition FROM cognitive_elements WHERE name = ANY(%s)",
                    (list(all_ce_names),)
                )
            
            if rows:
                ce_defs = [row['definition'] for row in rows if row.get('definition')]
                ce_defs_str = " ".join(ce_defs)

        trigger_embedding("rule", rule_id, rule_name, predicate, ce_defs_str)

    except Exception as e:
        print(f"[!] Embedding trigger failed for rule {rule_name}: {e}")

    return rule_id
