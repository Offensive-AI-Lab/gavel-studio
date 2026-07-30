from utils.sqlite_db import execute_query, execute_query_dict
from utils.embedding_utils import trigger_embedding
from utils.DButils import normalize_and_upsert_categories

# ---------------------------------------------------------
# COGNITIVE ELEMENTS (Direct Global Table)
# ---------------------------------------------------------

def create_ce(
    user_id: int,
    name: str,
    definition: str = "",
    category: str = "CONTEXT",
    categories: list = None,
    auto_embed: bool = True,
    mark_pending: bool = False,
):
    """
    Find an existing CE or create a new one in the public table.

    `mark_pending=True` is used by the AI pipeline to insert the row with
    is_ready=FALSE so it doesn't show up in any user-facing list until the
    pipeline flips it to TRUE post-training. If the pipeline never
    completes (crash / network drop / closed tab), the boot-time
    IncompletePipelineRecovery deletes the orphan.
    """
    # 1. Check if CE already exists globally
    query_find = "SELECT ce_id, name, definition, category, categories FROM cognitive_elements WHERE name = %s"
    existing = execute_query_dict(query_find, (name,))

    if existing:
        res = existing[0]
        res['is_new'] = False
        return res

    # 2. Insert into global table if it doesn't exist

    # Categories: Only use provided taxonomy categories; do NOT fall back to primary ACTION/CONTEXT
    cats_input = categories if categories else []
    normalized_categories = normalize_and_upsert_categories(cats_input, allow_new=True)

    query_insert = """
        INSERT INTO cognitive_elements (name, definition, category, categories, is_ready)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING ce_id, name, definition, category, categories, is_ready
    """
    new_ce = execute_query_dict(
        query_insert,
        (name, definition, category, normalized_categories, not mark_pending),
    )[0]
    new_ce['is_new'] = True

    # Auto-calculate embedding
    if auto_embed:
        trigger_embedding('ce', new_ce['ce_id'], new_ce['name'], new_ce['definition'])

    # Note: Excitation dataset generation should be triggered separately via API
    return new_ce


def get_user_ces():
    """
    Fetches all available Cognitive Elements so the user can
    search/select them for their rules. Filters out is_ready=FALSE rows
    (in-flight AI pipelines that haven't finished generating training data).

    Also hides a DRAFT CE whose training data (excitation) hasn't landed yet:
    with background generation, `embed-resources` flips a new CE is_ready=TRUE
    on Finish while its training set may still be generating. A draft CE only
    shows once its excitation row exists, so the user never sees / selects a
    half-generated CE. Public CEs always show (their data lives on HF and is
    pulled lazily, so a missing local excitation row is expected for them).
    """
    query = """
        SELECT ce.ce_id, ce.name, ce.definition, ce.category,
               (SELECT json_group_array(c.name) FROM categories c
                WHERE c.category_id IN (SELECT value FROM json_each(ce.categories))
               ) as "categories [JSONB]",
               ce.created_at,
               ce.is_local_draft,
               ce.created_by_username,
               ce.public_id,
               ce.examples,
               ed.dataset_id,
               CASE WHEN ed.dataset_id IS NOT NULL THEN true ELSE false END as "has_training_data [BOOLEAN]"
        FROM cognitive_elements ce
        LEFT JOIN excitation_datasets ed ON ce.ce_id = ed.ce_id
        WHERE ce.is_ready = TRUE
          AND (ce.is_local_draft = FALSE OR ed.dataset_id IS NOT NULL)
        ORDER BY ce.name ASC
    """
    return execute_query_dict(query)

def save_excitation_dataset(ce_id: int, dataset_json: dict):
    """
    Saves the excitation dataset (training data) for a CE to the database.
    """
    import json
    query = """
        INSERT INTO excitation_datasets (ce_id, dataset)
        VALUES (%s, %s)
        ON CONFLICT (ce_id) DO UPDATE SET
            dataset = EXCLUDED.dataset,
            created_at = now()
        RETURNING dataset_id
    """
    result = execute_query_dict(query, (ce_id, json.dumps(dataset_json)))
    return result[0] if result else None

def get_excitation_dataset(ce_id: int):
    """
    Retrieves the excitation dataset for a CE from the database.
    """
    import json
    query = """
        SELECT dataset_id, ce_id, dataset, created_at
        FROM excitation_datasets
        WHERE ce_id = %s
    """
    result = execute_query_dict(query, (ce_id,))
    if result:
        dataset_row = result[0]
        # Parse JSON string back to dict; tolerate malformed rows
        if isinstance(dataset_row.get('dataset'), str):
            try:
                dataset_row['dataset'] = json.loads(dataset_row['dataset'])
            except Exception as e:
                dataset_row['dataset_parse_error'] = str(e)
                dataset_row['dataset'] = {}
        return dataset_row
    return None


def save_calibration_dataset(ce_id: int, dataset_json: dict):
    """Saves the calibration dataset for a CE to the database."""
    import json
    query = """
        INSERT INTO calibration_datasets (ce_id, dataset)
        VALUES (%s, %s)
        ON CONFLICT (ce_id) DO UPDATE SET
            dataset = EXCLUDED.dataset,
            created_at = now()
        RETURNING dataset_id
    """
    result = execute_query_dict(query, (ce_id, json.dumps(dataset_json)))
    return result[0] if result else None


def get_calibration_dataset(ce_id: int):
    """Retrieves the calibration dataset for a CE from the database."""
    import json
    query = """
        SELECT dataset_id, ce_id, dataset, created_at
        FROM calibration_datasets
        WHERE ce_id = %s
    """
    result = execute_query_dict(query, (ce_id,))
    if result:
        row = result[0]
        if isinstance(row.get('dataset'), str):
            try:
                row['dataset'] = json.loads(row['dataset'])
            except Exception as e:
                row['dataset_parse_error'] = str(e)
                row['dataset'] = {}
        return row
    return None


# ---------------------------------------------------------
# RULE-LEVEL CALIBRATION — moved to `test_datasets` (dataset_type=
# 'positive_calibration'). The save_rule_calibration_dataset /
# get_rule_calibration_dataset helpers and the `rule_calibration_datasets`
# table itself are removed. Callers that previously used these helpers
# now go through the Test Sets flow (POST /ai/test-set/generate) and
# the calibration runner reads directly from test_datasets.
# ---------------------------------------------------------
# GLOBAL RULES (Public Library)
# ---------------------------------------------------------

def create_global_rule(name: str, predicate: str, ce_groups: dict = None,
                       condition: str = None):
    """Creates a community-standard rule template in the 'rules' table.

    `ce_groups` is {group_name: [CE NAMES]} and `condition` the v2 grammar
    string over those group names — the logic source of truth. `predicate`
    is the derived display string; when empty and a condition is given it
    is re-rendered here. Membership rule_ce_link rows are rebuilt from the
    union of group members (names resolved to ce_ids; unknown names are
    skipped — this legacy path creates CEs before calling in).
    """
    ce_groups = ce_groups or {}
    if condition and not predicate:
        from utils.rule_condition import parse, render_predicate
        predicate = render_predicate(parse(condition), ce_groups)

    # Insert/Update Rule
    query_rule = """
        INSERT INTO rules (name, predicate, ce_groups, condition)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (name) DO UPDATE SET predicate = EXCLUDED.predicate,
                                         ce_groups = EXCLUDED.ce_groups,
                                         condition = EXCLUDED.condition
        RETURNING rule_id
    """
    rule_id = execute_query_dict(
        query_rule, (name, predicate, ce_groups, condition)
    )[0]['rule_id']

    # Rebuild membership links from the union of group members.
    execute_query("DELETE FROM rule_ce_link WHERE rule_id = %s", (rule_id,))
    member_names = sorted({m for members in ce_groups.values() for m in (members or [])})
    ce_defs = []
    for ce_name in member_names:
        rows = execute_query_dict(
            "SELECT ce_id, definition FROM cognitive_elements WHERE name = %s",
            (ce_name,),
        ) or []
        if not rows:
            continue
        execute_query(
            "INSERT INTO rule_ce_link (rule_id, ce_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (rule_id, rows[0]['ce_id']),
        )
        if rows[0].get('definition'):
            ce_defs.append(rows[0]['definition'])

    # Auto-calculate embedding for rule
    trigger_embedding('rule', rule_id, name, predicate, " ".join(ce_defs))

    return rule_id


def _attach_rule_logic(rows: list) -> list:
    """Post-process rule rows that carry raw ce_groups/condition columns plus
    a flat active_ces membership list into the read contract: pops the raw
    columns and adds `logic` = {"groups": {group_name: [{ce_id, name}]},
    "condition", "predicate"}."""
    from sql_scripts.model_scripts import _build_logic_payload
    for row in rows or []:
        row["logic"] = _build_logic_payload(
            row.pop("ce_groups", None),
            row.pop("condition", None),
            row.get("predicate"),
            row.get("active_ces") or [],
        )
    return rows or []

def get_all_public_rules():
    """Fetches templates for the community browsing feature.

    Each row carries `logic` = {"groups": {group_name: [{ce_id, name}]},
    "condition", "predicate"} built from the rule's ce_groups/condition
    columns, plus `active_ces` as a flat [{ce_id, name}] membership list
    (CE-count displays / membership-only consumers).
    """
    query = """
        SELECT
            r.rule_id, r.name, r.predicate, r.description, r.type, r.created_at,
            r.is_local_draft,
            r.created_by_username,
            r.public_id,
            r.ce_groups,
            r.condition,
            (SELECT json_group_array(c.name) FROM categories c
             WHERE c.category_id IN (SELECT value FROM json_each(r.categories))
            ) as "categories [JSONB]",
            COALESCE(
                json_group_array(
                    json_object(
                        'ce_id', ce.ce_id,
                        'name', ce.name
                    )
                ) FILTER (WHERE ce.ce_id IS NOT NULL),
                '[]'
            ) as "active_ces [JSONB]",
            COALESCE(json_group_array(ce.name) FILTER (WHERE ce.ce_id IS NOT NULL), '[]') as "required_ces [JSONB]"
        FROM rules r
        LEFT JOIN rule_ce_link rl ON r.rule_id = rl.rule_id
        LEFT JOIN cognitive_elements ce ON rl.ce_id = ce.ce_id
        -- Published rules only. Local drafts (is_local_draft = TRUE) are
        -- surfaced separately via /library/drafts; including them here made a
        -- draft show twice in Browse (once from each list) and leaked drafts
        -- into the public library.
        WHERE r.is_ready = TRUE AND r.is_local_draft = FALSE
        GROUP BY r.rule_id
    """
    return _attach_rule_logic(execute_query_dict(query))


def get_all_public_rule_sets():
    """Public rule sets for the Community browsing feature. A rule set is a
    named, model-agnostic collection of already-published rules. Only published
    sets (is_local_draft = FALSE) are returned — a user's private rule sets live
    in the classifiers table and never appear here."""
    query = """
        SELECT
            rs.rule_set_id, rs.name, rs.description, rs.created_at,
            rs.is_local_draft, rs.created_by_username, rs.public_id,
            (SELECT json_group_array(c.name) FROM categories c
             WHERE c.category_id IN (SELECT value FROM json_each(rs.categories))
            ) AS "categories [JSONB]",
            COALESCE((
                -- Aggregate over a pre-ordered subquery: SQLite (< 3.44) has
                -- no ORDER BY inside aggregates, and it feeds rows to the
                -- aggregate in scan order, which the inner ORDER BY fixes.
                SELECT json_group_array(json_object(
                           'rule_id', m.rule_id,
                           'name', m.name,
                           'public_id', m.public_id,
                           'position', m.position
                       ))
                FROM (
                    SELECT r.rule_id, r.name, r.public_id, rsm.position
                    FROM rule_set_member rsm
                    JOIN rules r ON r.rule_id = rsm.rule_id
                    WHERE rsm.rule_set_id = rs.rule_set_id
                    ORDER BY rsm.position
                ) m
            ), '[]') AS "member_rules [JSONB]"
        FROM rule_sets rs
        WHERE rs.is_ready = TRUE AND rs.is_local_draft = FALSE
        ORDER BY rs.created_at DESC
    """
    return execute_query_dict(query)


def get_rule_set_detail(public_id: str):
    """One public rule set + its member rules for the rule-set detail page
    (each member carries `logic` + flat `active_ces` as in
    get_all_public_rules). Keyed by the library public_id. Returns None if no
    published set with that public_id exists locally."""
    set_rows = execute_query_dict(
        """
        SELECT rs.rule_set_id, rs.name, rs.description, rs.created_at,
               rs.is_local_draft, rs.created_by_username, rs.public_id,
               (SELECT json_group_array(c.name) FROM categories c
                WHERE c.category_id IN (SELECT value FROM json_each(rs.categories))
               ) AS "categories [JSONB]"
        FROM rule_sets rs
        WHERE rs.public_id = %s AND rs.is_local_draft = FALSE
        """,
        (public_id,),
    )
    if not set_rows:
        return None
    rs = set_rows[0]
    rs["member_rules"] = _attach_rule_logic(execute_query_dict(
        """
        SELECT rsm.position, r.rule_id, r.name, r.predicate, r.description,
               r.public_id, r.created_by_username,
               r.ce_groups, r.condition,
               COALESCE(
                   json_group_array(
                       json_object(
                           'ce_id', ce.ce_id,
                           'name', ce.name
                       )
                   ) FILTER (WHERE ce.ce_id IS NOT NULL),
                   '[]'
               ) AS "active_ces [JSONB]"
        FROM rule_set_member rsm
        JOIN rules r ON r.rule_id = rsm.rule_id
        LEFT JOIN rule_ce_link rl ON rl.rule_id = r.rule_id
        LEFT JOIN cognitive_elements ce ON ce.ce_id = rl.ce_id
        WHERE rsm.rule_set_id = %s
        GROUP BY rsm.position, r.rule_id, r.name, r.predicate, r.description,
                 r.public_id, r.created_by_username
        ORDER BY rsm.position
        """,
        (rs["rule_set_id"],),
    ) or [])
    return rs