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


# ---------------------------------------------------------
# BOOKMARK HELPERS (Rules & CEs)
# ---------------------------------------------------------
# Implementation lives in services/bookmarks.py — these functions are kept as
# thin compatibility shims so existing import paths stay valid. New code should
# call BookmarkService directly.
from services.bookmarks import BookmarkService  # noqa: E402


def add_rule_bookmark(user_id: int, rule_id: int):
    return BookmarkService.add("rule", user_id, rule_id)


def remove_rule_bookmark(user_id: int, rule_id: int):
    return BookmarkService.remove("rule", user_id, rule_id)


def list_rule_bookmarks(user_id: int):
    return BookmarkService.list("rule", user_id)


def add_ce_bookmark(user_id: int, ce_id: int):
    return BookmarkService.add("ce", user_id, ce_id)


def remove_ce_bookmark(user_id: int, ce_id: int):
    return BookmarkService.remove("ce", user_id, ce_id)


def list_ce_bookmarks(user_id: int):
    return BookmarkService.list("ce", user_id)

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

def create_global_rule(name: str, predicate: str, ce_ids: list):
    """Creates a community-standard rule template in the 'rules' table."""
    # Insert/Update Rule
    query_rule = """
        INSERT INTO rules (name, predicate) 
        VALUES (%s, %s) 
        ON CONFLICT (name) DO UPDATE SET predicate=EXCLUDED.predicate 
        RETURNING rule_id
    """
    rule_id = execute_query_dict(query_rule, (name, predicate))[0]['rule_id']
    
    # Link CEs to the public template in rule_ce_link
    ce_defs = []
    for ce_id in ce_ids:
        execute_query(
            "INSERT INTO rule_ce_link (rule_id, ce_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", 
            (rule_id, ce_id)
        )
        # Fetch definition for embedding
        rows = execute_query_dict("SELECT definition FROM cognitive_elements WHERE ce_id = %s", (ce_id,))
        if rows:
            ce_defs.append(rows[0]['definition'])
            
    # Auto-calculate embedding for rule
    ce_definitions_str = " ".join(ce_defs)
    trigger_embedding('rule', rule_id, name, predicate, ce_definitions_str)

    return rule_id

def get_all_public_rules():
    """Fetches templates for the community browsing feature."""
    query = """
        SELECT
            r.rule_id, r.name, r.predicate, r.description, r.type, r.created_at,
            r.is_local_draft,
            r.created_by_username,
            r.public_id,
            (SELECT json_group_array(c.name) FROM categories c
             WHERE c.category_id IN (SELECT value FROM json_each(r.categories))
            ) as "categories [JSONB]",
            COALESCE(
                json_group_array(
                    json_object(
                        'ce_id', ce.ce_id,
                        'name', ce.name,
                        'role', COALESCE(rl.role, 'necessary'),
                        'fallback_group', COALESCE(rl.fallback_group, 0)
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
    return execute_query_dict(query)


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
    """One public rule set + its member rules (each with their CEs/roles) for
    the rule-set detail page. Keyed by the HF public_id. Returns None if no
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
    rs["member_rules"] = execute_query_dict(
        """
        SELECT rsm.position, r.rule_id, r.name, r.predicate, r.description,
               r.public_id, r.created_by_username,
               COALESCE(
                   json_group_array(
                       json_object(
                           'ce_id', ce.ce_id,
                           'name', ce.name,
                           'role', COALESCE(rl.role, 'necessary'),
                           'fallback_group', COALESCE(rl.fallback_group, 0)
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
    ) or []
    return rs