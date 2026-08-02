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
        # services/library_sync.ensure_excitation writes for registry-pulled CEs.
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
    """Return rules keyed by name, each with its groups+condition logic straight from the
    rules columns: {name: {ce_groups, condition, predicate, description}}.

    ce_groups is {group_name: [CE names]} and condition is the condition grammar
    string over those group names (utils/rule_condition.py); predicate is the
    derived display string. Skips is_ready=FALSE rows for the same reason
    fetch_ces_dict does.
    """
    rule_rows = execute_query_dict(
        """
        SELECT name, ce_groups, condition, predicate, description
        FROM rules
        WHERE is_ready = TRUE
        ORDER BY name
        """
    ) or []

    named_rules: Dict[str, dict] = {}
    for row in rule_rows:
        named_rules[row["name"]] = {
            "ce_groups": row.get("ce_groups") or {},
            "condition": row.get("condition") or "",
            "predicate": row.get("predicate") or "",
            "description": row.get("description") or "",
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
    """Insert/update a rule and its membership links from groups+condition logic. Returns rule_id.

    THE single writer for the `rules` table + `rule_ce_link` membership rows.
    `rule_data` = {
        "rule_name":   str,
        "description": str,
        "ce_groups":   {group_name: [CE NAMES]},
        "condition":   str (condition grammar over the group names),
        "categories":  optional [category names],
    }

    Steps:
      1. Validate (ce_groups, condition) via rule_condition.validate_condition
         — raises ValueError listing the problems.
      2. Structural dedup via compute_rule_fingerprint_v2: a rule whose firing
         logic matches an existing rule under a DIFFERENT name is a functional
         duplicate even though no name conflict fires. `exclude_name` lets the
         same rule be re-saved (the AI pipeline updates description on retry).
      3. Derive predicate = render_predicate(parse(condition), ce_groups) —
         the stored predicate is a DISPLAY string only, never evaluated.
      4. Upsert the rules row (ce_groups/condition are the logic source of
         truth) and rebuild rule_ce_link as pure membership rows from the
         union of group members. Unknown CE names raise ValueError.
      5. Trigger embedding over name + predicate + member CE definitions.

    `mark_pending=True` is used by the AI pipeline to mark the rule as
    is_ready=FALSE so it doesn't appear in any user-facing list until
    /ai/embed-resources flips it to TRUE post-training. The boot-time
    IncompletePipelineRecovery wipes any is_ready=FALSE rows, so a crash
    cleanly looks like the rule was never created.

    On UPDATE (rule with this name already existed), we leave is_ready
    alone — we don't want to silently un-publish or un-finish a row that
    was already complete.
    """
    from utils.rule_condition import parse, render_predicate, validate_condition
    from sql_scripts.junction_scripts import (
        compute_rule_fingerprint_v2,
        find_existing_rule_by_fingerprint_v2,
    )

    rule_name = rule_data["rule_name"]
    description = rule_data.get("description", "")
    ce_groups = {
        gname: list(members or [])
        for gname, members in (rule_data.get("ce_groups") or {}).items()
    }
    condition = (rule_data.get("condition") or "").strip()

    # 1. Validate the logic pair before anything touches the DB.
    problems = validate_condition(condition, ce_groups)
    if problems:
        raise ValueError(
            f"Invalid rule logic for '{rule_name}': " + "; ".join(problems)
        )

    # 2. Structural dedup at the single choke point (AI-generated and
    # manually-created rules both funnel through here).
    new_fp = compute_rule_fingerprint_v2(ce_groups, condition)
    duplicate = find_existing_rule_by_fingerprint_v2(new_fp, exclude_name=rule_name)
    if duplicate is not None:
        existing_name = duplicate.get("name") or "(unnamed)"
        raise ValueError(
            f"A rule with the same structure already exists as "
            f"'{existing_name}'. Two rules with identical CE groups and an "
            f"equivalent firing condition are functionally the same — "
            f"bookmark or fork the existing rule, or change the new rule's "
            f"logic."
        )

    # 3. Derived display predicate.
    predicate = render_predicate(parse(condition), ce_groups)

    # Resolve member CE names -> ids up front so a typo'd/unknown name fails
    # loudly instead of silently dropping a membership row.
    all_ce_names = sorted({m for members in ce_groups.values() for m in members})
    name_to_id: Dict[str, int] = {}
    ce_defs: List[str] = []
    if all_ce_names:
        rows = execute_query_dict(
            "SELECT ce_id, name, definition FROM cognitive_elements WHERE name = ANY(%s)",
            (all_ce_names,),
        ) or []
        name_to_id = {r["name"]: r["ce_id"] for r in rows}
        ce_defs = [r["definition"] for r in rows if r.get("definition")]
        missing = [n for n in all_ce_names if n not in name_to_id]
        if missing:
            raise ValueError(
                f"Rule '{rule_name}' references unknown CE(s): "
                + ", ".join(missing)
            )

    raw_categories = rule_data.get("categories") or []
    categories = normalize_and_upsert_categories(raw_categories, allow_new=True)

    # 4. Upsert the rules row + rebuild membership links.
    rule_row = execute_query_dict(
        """
        INSERT INTO rules (name, predicate, description, categories, ce_groups, condition, is_ready)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (name)
        DO UPDATE SET predicate = EXCLUDED.predicate,
                      description = EXCLUDED.description,
                      categories = EXCLUDED.categories,
                      ce_groups = EXCLUDED.ce_groups,
                      condition = EXCLUDED.condition
        RETURNING rule_id
        """,
        (rule_name, predicate, description, categories, ce_groups, condition,
         not mark_pending),
    )
    rule_id = rule_row[0]["rule_id"] if isinstance(rule_row[0], dict) else rule_row[0][0]

    # Membership = union of group members (pure membership rows; the logic
    # itself lives in ce_groups/condition on the rules row).
    execute_query("DELETE FROM rule_ce_link WHERE rule_id = %s", (rule_id,))
    for ce_name in all_ce_names:
        execute_query(
            """
            INSERT INTO rule_ce_link (rule_id, ce_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            """,
            (rule_id, name_to_id[ce_name]),
        )

    # 5. Auto-embed (best-effort; embedding failure must not fail the write).
    try:
        trigger_embedding("rule", rule_id, rule_name, predicate, " ".join(ce_defs))
    except Exception as e:
        print(f"[!] Embedding trigger failed for rule {rule_name}: {e}")

    return rule_id
