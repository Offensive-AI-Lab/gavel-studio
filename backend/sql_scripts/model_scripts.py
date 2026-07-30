# backend/sql_scripts/model_scripts.py
import hashlib
from utils.sqlite_db import execute_query, execute_query_dict
# We import create_global_rule to re-use the logic for adding to the main table
from sql_scripts.definition_scripts import create_global_rule


# --- POLICY DRIFT / RETRAINING ---------------------------------------------

def compute_classifier_policy_fingerprint_v2(classifier_id: int) -> str:
    """Deterministic content fingerprint of a guardrail's CURRENT active policy.

    sha256 over the sorted per-setup (ce_groups, condition) fingerprints
    (compute_rule_fingerprint_v2) of the guardrail's ACTIVE setups.
    Independent of setup_id churn and rule order, so re-adding a removed rule
    (which mints a new setup_id) does NOT register as a change. Returns ''
    when no active setup carries logic."""
    from sql_scripts.junction_scripts import compute_rule_fingerprint_v2

    rows = execute_query_dict(
        """
        SELECT ce_groups, condition FROM rule_setup
        WHERE classifier_id = %s AND is_active = TRUE
        """,
        (classifier_id,),
    ) or []
    fps = sorted(
        compute_rule_fingerprint_v2(r["ce_groups"], r["condition"])
        for r in rows
        if r["ce_groups"]
    )
    canonical = ";".join(fps)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest() if canonical else ""


def reconcile_classifier_status(classifier_id: int) -> str:
    """Return the guardrail's TRUE status, self-healing the stored value.

    'Needs retraining' should reflect REAL drift — the current policy differing
    from the policy the model was trained on — not a sticky flag. We compare the
    live policy fingerprint against the snapshot captured at train time
    (`trained_policy_fingerprint`):

      * current == trained -> 'active'            (no drift)
      * current != trained -> 'needs_retraining'

    Only meaningful once a model exists (status active/needs_retraining) AND a
    fingerprint snapshot is present. Other states (untrained/training/error) and
    guardrails trained before fingerprinting existed pass through unchanged.
    The recomputed value is written back so gates reading the stored status
    (download/evaluate/monitor) stay correct.
    """
    rows = execute_query_dict(
        "SELECT status, trained_policy_fingerprint FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    ) or []
    if not rows:
        return "untrained"
    status = rows[0]["status"]
    trained_fp = rows[0].get("trained_policy_fingerprint")
    if status not in ("active", "needs_retraining") or not trained_fp:
        return status
    current_fp = compute_classifier_policy_fingerprint_v2(classifier_id)
    new_status = "active" if current_fp == trained_fp else "needs_retraining"
    if new_status != status:
        execute_query(
            "UPDATE classifiers SET status = %s WHERE classifier_id = %s",
            (new_status, classifier_id),
        )
    return new_status


def commit_trained_policy_snapshot(classifier_id: int) -> None:
    """Record the guardrail's CURRENT active policy as the trained snapshot.

    Writes trained_rule_setup_ids, trained_rule_names, trained_policy_fingerprint
    and trained_at together — the durable record of what the now-trained model
    was trained on. Drift detection (reconcile_classifier_status) and the Policy
    Logic Manager's "Up to Date / Retrain" button compare the live policy against
    this snapshot.

    Call ONLY on SUCCESSFUL training completion — local (trainer.py) AND cluster
    (the get-status completion path). Writing it any earlier (e.g. at training
    start) is the bug that made an INTERRUPTED run look 'Up to Date' on a model
    that was never produced; never recording it (the old cluster path) is the bug
    that left the drift banner stuck after a successful cluster retrain.

    Names use COALESCE(custom_name, rules.name) — the same identity the rule list
    exposes — so set-equality against the live selection is order/setup_id stable.
    """
    rows = execute_query_dict(
        """
        SELECT rs.setup_id, COALESCE(rs.custom_name, r.name) AS rule_name
        FROM rule_setup rs
        LEFT JOIN rules r ON rs.rule_id = r.rule_id
        WHERE rs.classifier_id = %s AND rs.is_active = TRUE
        ORDER BY rs.setup_id
        """,
        (classifier_id,),
    ) or []
    setup_ids = [r["setup_id"] for r in rows]
    names = [r["rule_name"] for r in rows]
    fingerprint = compute_classifier_policy_fingerprint_v2(classifier_id)
    execute_query(
        """
        UPDATE classifiers
        SET trained_rule_setup_ids     = %s,
            trained_rule_names         = %s,
            trained_policy_fingerprint = %s,
            trained_at                 = now()
        WHERE classifier_id = %s
        """,
        (setup_ids, names, fingerprint, classifier_id),
    )

    # A (re)train invalidates the previous model's calibration + evaluation.
    # The _POST_TRAIN_CLAUSE already HIDES them (created before the new
    # trained_at), but delete them outright so the Results page + history are
    # clean and the once-per-training lock starts fresh for the new model.
    # Leave any '*_running' markers alone — they may own a live cluster job and
    # are reaped by the boot-recovery path instead.
    execute_query(
        """
        DELETE FROM evaluation_results
        WHERE classifier_id = %s
          AND eval_type IN ('calibration', 'evaluation', 'calibration_error', 'evaluation_error')
        """,
        (classifier_id,),
    )


# --- MODELS & GUARDRAILS (Existing) ---
def register_model(user_id: int, name: str, storage_path: str, hf_token: str = None,
                   num_layers: int = None, selected_layers=None):
    query = ("INSERT INTO target_models (user_id, name, storage_path, hf_token, num_layers, selected_layers) "
             "VALUES (%s, %s, %s, %s, %s, %s) RETURNING model_id, name")
    result = execute_query_dict(query, (user_id, name, storage_path, hf_token, num_layers, selected_layers))
    return result[0] if result else None

def get_user_models(user_id: int):
    # Never select hf_token — it must not leave the backend. Expose a
    # boolean `has_hf_token` so the UI can show "token saved" if useful.
    return execute_query_dict(
        """SELECT model_id, user_id, name, storage_path, created_at,
                  num_layers, selected_layers,
                  (hf_token IS NOT NULL) AS has_hf_token
           FROM target_models WHERE user_id = %s ORDER BY created_at DESC""",
        (user_id,),
    )

def update_model_layers(model_id: int, user_id: int, selected_layers):
    """Persist a model's chosen LLM layer range ([start, end))."""
    rows = execute_query_dict(
        "UPDATE target_models SET selected_layers = %s WHERE model_id = %s AND user_id = %s "
        "RETURNING model_id, selected_layers",
        (selected_layers, model_id, user_id),
    )
    return rows[0] if rows else None

def create_classifier(user_id: int, name: str, model_id: int | None = None):
    """Create a guardrail owned by `user_id`. model_id may be NULL — an
    unattached guardrail holds a rule set until a model is picked at train time.
    """
    query = (
        "INSERT INTO classifiers (user_id, model_id, name) VALUES (%s, %s, %s) "
        "RETURNING classifier_id, model_id, name, status"
    )
    result = execute_query_dict(query, (user_id, model_id, name))
    return result[0] if result else None

def get_model_classifiers(model_id: int):
    query = "SELECT classifier_id, model_id, name, status, model_path, training_log, created_at FROM classifiers WHERE model_id = %s ORDER BY created_at DESC"
    return execute_query_dict(query, (model_id,))

def get_user_classifiers(user_id: int):
    """Every guardrail (classifier) a user owns, across all models AND the
    unattached ones (model_id IS NULL → model_name NULL). Mirrors the columns
    get_model_classifiers returns, plus model_name and a live rule_count for the
    card, and the training-banner fields the Guardrails page polls."""
    query = """
        SELECT c.classifier_id, c.model_id, c.name, c.status, c.model_path,
               c.training_log, c.training_phase_detail, c.created_at, c.trained_at,
               c.folder_id,
               tm.name AS model_name,
               (SELECT COUNT(*) FROM rule_setup rs WHERE rs.classifier_id = c.classifier_id) AS rule_count
        FROM classifiers c
        LEFT JOIN target_models tm ON c.model_id = tm.model_id
        WHERE c.user_id = %s
        ORDER BY c.created_at DESC
    """
    return execute_query_dict(query, (user_id,))

def attach_model_to_classifier(classifier_id: int, model_id: int):
    """Bind an unattached guardrail to a model. Returns the updated row."""
    result = execute_query_dict(
        "UPDATE classifiers SET model_id = %s WHERE classifier_id = %s "
        "RETURNING classifier_id, model_id, name, status",
        (model_id, classifier_id),
    )
    return result[0] if result else None

def clone_classifier_policy(source_classifier_id: int, target_model_id: int,
                            user_id: int, name: str | None = None):
    """Deep-copy a guardrail's rule set into a NEW, UNTRAINED guardrail attached
    to `target_model_id`. Copies the per-guardrail policy layer only —
    rule_setup rows and their setup_ce_link rows (rules/CEs are global and stay
    referenced by id). Does NOT copy status / trained_* snapshot / model_path /
    training_config, so the copy starts 'untrained' and must be retrained for
    its model. The name is auto-deduped within the target model so the one-click
    action never collides. Returns {classifier_id, model_id, name, status}.
    """
    src = execute_query_dict(
        "SELECT name FROM classifiers WHERE classifier_id = %s", (source_classifier_id,))
    if not src:
        raise ValueError(f"Guardrail {source_classifier_id} not found")
    base = (name or src[0]["name"]).strip()

    # Dedupe within the target model (case-insensitive, like create_new_classifier).
    candidate, n = base, 1
    while execute_query_dict(
            "SELECT 1 FROM classifiers WHERE model_id = %s AND LOWER(name) = LOWER(%s)",
            (target_model_id, candidate)):
        n += 1
        candidate = f"{base} (copy)" if n == 2 else f"{base} (copy {n - 1})"

    new = execute_query_dict(
        "INSERT INTO classifiers (user_id, model_id, name) VALUES (%s, %s, %s) "
        "RETURNING classifier_id, model_id, name, status",
        (user_id, target_model_id, candidate))[0]
    new_classifier_id = new["classifier_id"]

    # Copy each rule_setup (incl. its per-guardrail ce_groups/condition logic
    # copy), then remap its setup_ce_link membership rows to the new setup_id
    # (per-row, because each insert mints a fresh setup_id).
    setups = execute_query_dict(
        "SELECT setup_id, rule_id, custom_name, predicate, ce_groups, condition, is_active "
        "FROM rule_setup WHERE classifier_id = %s ORDER BY setup_id",
        (source_classifier_id,)) or []
    for s in setups:
        new_setup_id = execute_query_dict(
            "INSERT INTO rule_setup (classifier_id, rule_id, custom_name, predicate, ce_groups, condition, is_active) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING setup_id",
            (new_classifier_id, s["rule_id"], s["custom_name"], s["predicate"],
             s["ce_groups"], s["condition"], s["is_active"]),
        )[0]["setup_id"]
        execute_query(
            "INSERT INTO setup_ce_link (setup_id, ce_id) "
            "SELECT %s, ce_id FROM setup_ce_link WHERE setup_id = %s",
            (new_setup_id, s["setup_id"]),
        )
    return new

# --- RULE LINKING LOGIC ---

def resolve_editor_logic(groups: dict, condition: str):
    """Translate the rule editor's logic shape ({group_name: [ce_id]},
    condition) into the stored v2 shape.

    The frontend holds ce_ids; the DB stores CE NAMES inside ce_groups
    (names are the identity the condition grammar and trained label keys
    speak). Resolves every referenced ce_id to its name, validates the
    (ce_groups, condition) pair via rule_condition.validate_condition, and
    renders the display predicate.

    Returns (ce_groups, condition, predicate, ce_ids) where ce_groups is
    {group_name: [CE names]} and ce_ids is the sorted membership union.
    Raises ValueError on unknown ce_ids or invalid logic.
    """
    from utils.rule_condition import parse, render_predicate, validate_condition

    groups = groups or {}
    condition = (condition or "").strip()

    ce_ids = sorted({int(cid) for members in groups.values() for cid in (members or [])})
    name_map: dict = {}
    if ce_ids:
        rows = execute_query_dict(
            "SELECT ce_id, name FROM cognitive_elements WHERE ce_id = ANY(%s)",
            (ce_ids,),
        ) or []
        name_map = {r["ce_id"]: r["name"] for r in rows}
        missing = [str(cid) for cid in ce_ids if cid not in name_map]
        if missing:
            raise ValueError(f"Unknown CE id(s): {', '.join(missing)}")

    ce_groups = {
        gname: [name_map[int(cid)] for cid in (members or [])]
        for gname, members in groups.items()
    }

    problems = validate_condition(condition, ce_groups)
    if problems:
        raise ValueError("Invalid rule logic: " + "; ".join(problems))

    predicate = render_predicate(parse(condition), ce_groups)
    return ce_groups, condition, predicate, ce_ids


def _replace_setup_membership(setup_id: int, ce_ids: list):
    """Rebuild a setup's membership links (pure membership — the logic lives
    in rule_setup.ce_groups/condition)."""
    execute_query("DELETE FROM setup_ce_link WHERE setup_id = %s", (setup_id,))
    for ce_id in ce_ids:
        execute_query(
            "INSERT INTO setup_ce_link (setup_id, ce_id) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (setup_id, ce_id),
        )


# 1. LINK EXISTING PUBLIC RULE (User selects from list)
def add_rule_to_classifier(classifier_id: int, public_rule_id: int):
    # Fetch original rule details
    result = execute_query_dict("SELECT * FROM rules WHERE rule_id = %s", (public_rule_id,))
    if not result:
        raise ValueError(f"Rule {public_rule_id} not found")
    pub_rule = result[0]

    # Create a local copy in rule_setup linked to the public rule. The
    # per-guardrail ce_groups/condition copy starts as a verbatim snapshot of
    # the rule's logic; private edits later diverge it without touching the
    # library rule.
    query_setup = """
        INSERT INTO rule_setup (classifier_id, rule_id, custom_name, predicate, ce_groups, condition)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING setup_id
    """
    setup_id = execute_query_dict(
        query_setup,
        (classifier_id, public_rule_id, pub_rule['name'], pub_rule['predicate'],
         pub_rule.get('ce_groups') or {}, pub_rule.get('condition')),
    )[0]['setup_id']

    # Copy the CE membership rows from public to private.
    execute_query(
        """
        INSERT INTO setup_ce_link (setup_id, ce_id)
        SELECT %s, ce_id FROM rule_ce_link WHERE rule_id = %s
        """,
        (setup_id, public_rule_id),
    )
    return setup_id

def fork_public_rule_set_to_classifier(rule_set_public_id: str, user_id: int,
                                       name: str | None = None):
    """Fork a PUBLIC rule set into a NEW private, model-less rule set owned by
    `user_id`. ADD-BY-REFERENCE: every member rule is the existing PUBLIC rule
    (referenced by id via add_rule_to_classifier), so NO new rules/CEs are
    minted. This deliberately sidesteps the structural-fingerprint dedup that
    rejects an unchanged rule fork — letting "fork a rule set as-is" actually
    work. The copy starts UNTRAINED with no model; the user attaches a model and
    trains later (the normal model-last flow). Returns {classifier_id, model_id,
    name, status}.
    """
    rs = execute_query_dict(
        "SELECT rule_set_id, name FROM rule_sets "
        "WHERE public_id = %s AND is_local_draft = FALSE",
        (rule_set_public_id,),
    )
    if not rs:
        raise ValueError(f"Public rule set {rule_set_public_id} not found")
    rule_set_id = rs[0]["rule_set_id"]
    base = (name or rs[0]["name"]).strip()

    # Dedupe among the user's OTHER model-less rule sets (case-insensitive),
    # mirroring clone_classifier_policy's '(copy)'/'(copy N)' scheme.
    candidate, n = base, 1
    while execute_query_dict(
            "SELECT 1 FROM classifiers "
            "WHERE user_id = %s AND model_id IS NULL AND LOWER(name) = LOWER(%s)",
            (user_id, candidate)):
        n += 1
        candidate = f"{base} (copy)" if n == 2 else f"{base} (copy {n - 1})"

    new = create_classifier(user_id, candidate, model_id=None)
    new_classifier_id = new["classifier_id"]

    members = execute_query_dict(
        "SELECT rule_id FROM rule_set_member WHERE rule_set_id = %s ORDER BY position",
        (rule_set_id,),
    ) or []
    for m in members:
        add_rule_to_classifier(new_classifier_id, m["rule_id"])

    return new


# 2. CREATE MANUALLY (User types by hand -> Local only)
def create_custom_rule_setup(classifier_id: int, name: str):
    """Creates a private rule. rule_id is NULL; logic starts empty
    (ce_groups = {}, condition = NULL) until the editor saves it."""
    query = """
        INSERT INTO rule_setup (classifier_id, custom_name, predicate, rule_id, ce_groups, condition)
        VALUES (%s, %s, '', NULL, %s, NULL)
        RETURNING setup_id
    """
    result = execute_query_dict(query, (classifier_id, name, {}))
    return result[0]['setup_id'] if result else None

# 3. CREATE WITH AI (Global Table -> Then Link)
def create_and_link_global_rule(classifier_id: int, name: str, predicate: str, ce_ids: list):
    """
    1. Creates rule in 'rules' table (Global).
    2. Links it to this guardrail using add_rule_to_classifier logic.

    Legacy path: the caller speaks a flat ce_id list, which maps to a single
    'required' group with condition 'all of required' (every CE necessary —
    the pre-v24 default for role-less links).
    """
    names = []
    if ce_ids:
        rows = execute_query_dict(
            "SELECT ce_id, name FROM cognitive_elements WHERE ce_id = ANY(%s)",
            (list(ce_ids),),
        ) or []
        names = sorted(r["name"] for r in rows)

    ce_groups = {"required": names} if names else {}
    condition = "all of required" if names else None

    # Step A: Create in Global Table
    # Note: create_global_rule comes from definition_scripts.py
    global_rule_id = create_global_rule(name, predicate, ce_groups=ce_groups, condition=condition)

    # Step B: Link to Guardrail
    setup_id = add_rule_to_classifier(classifier_id, global_rule_id)
    return setup_id

# --- FETCHING & UPDATING ---

def get_classifier_rules(classifier_id: int):
    """Rule setups for a guardrail's Policy Logic page.

    Each row carries:
      * `logic` — {"groups": {group_name: [{ce_id, name}]}, "condition",
        "predicate"} built from the setup's ce_groups/condition columns
        (the v2 logic contract the rule editor speaks).
      * `active_ces` — flat [{ce_id, name}] membership list (kept so CE-count
        displays and membership-only consumers keep working).
    """
    query = """
        SELECT
            rs.setup_id,
            rs.classifier_id,
            rs.rule_id as source_rule_id,
            rs.custom_name,
            rs.predicate,
            rs.ce_groups,
            rs.condition,
            rs.is_active,
            -- Surface the source rule's draft flag so the UI can show a
            -- "Publish to library" button on local drafts. Setups whose
            -- rule_id is NULL (manual bookmark rules) collapse to NULL here;
            -- the frontend treats NULL as "not yet a publishable record".
            r.is_local_draft AS is_local_draft,
            -- Surface the source rule's library identity too, so the card on
            -- this page can show the rating widget (needs public_id + author)
            -- and the "What this rule detects" explanation (description), the
            -- same as Browse / the Rule page.
            r.public_id AS public_id,
            r.description AS description,
            r.created_by_username AS created_by_username,
            -- Correlated scalar subqueries replace the Postgres LATERAL join:
            -- ordered name list + id list of the source rule's categories.
            COALESCE(
                (SELECT json_group_array(name) FROM (
                    SELECT c.name FROM categories c
                    WHERE c.category_id IN (SELECT value FROM json_each(r.categories))
                    ORDER BY c.name
                )),
                '[]'
            ) AS "categories [JSONB]",
            COALESCE(
                (SELECT json_group_array(c.category_id) FROM categories c
                 WHERE c.category_id IN (SELECT value FROM json_each(r.categories))),
                r.categories,
                '[]'
            ) AS "category_ids [JSONB]",
            COALESCE(
                json_group_array(
                    json_object(
                        'ce_id', ce.ce_id,
                        'name', ce.name
                    )
                ) FILTER (WHERE ce.ce_id IS NOT NULL),
                '[]'
            ) as "active_ces [JSONB]"
        FROM rule_setup rs
        LEFT JOIN rules r ON rs.rule_id = r.rule_id
        LEFT JOIN setup_ce_link link ON rs.setup_id = link.setup_id
        LEFT JOIN cognitive_elements ce ON link.ce_id = ce.ce_id
        WHERE rs.classifier_id = %s
          -- Hide rule setups whose underlying rule is still being created.
          -- A NULL rule_id (manual setups with no backing rule yet) is fine
          -- to show; only filter when there IS a backing rule and it's
          -- not yet ready.
          AND (rs.rule_id IS NULL OR r.is_ready = TRUE)
        GROUP BY rs.setup_id, r.categories, r.is_local_draft, r.public_id,
                 r.description, r.created_by_username
        ORDER BY rs.setup_id ASC
    """
    rows = execute_query_dict(query, (classifier_id,)) or []
    for row in rows:
        row["logic"] = _build_logic_payload(
            row.pop("ce_groups", None),
            row.pop("condition", None),
            row.get("predicate"),
            row.get("active_ces") or [],
        )
    return rows


def _build_logic_payload(ce_groups, condition, predicate, membership: list) -> dict:
    """Shape a stored (ce_groups, condition, predicate) triple into the
    editor/read contract: {"groups": {group_name: [{ce_id, name}]},
    "condition", "predicate"}. `membership` is the flat [{ce_id, name}]
    list used to resolve member names back to ce_ids (a name missing from
    membership — which shouldn't happen — degrades to ce_id None instead of
    dropping the member)."""
    name_to_id = {m["name"]: m["ce_id"] for m in membership if m.get("name")}
    groups = {
        gname: [{"ce_id": name_to_id.get(n), "name": n} for n in (members or [])]
        for gname, members in (ce_groups or {}).items()
    }
    return {
        "groups": groups,
        "condition": condition or "",
        "predicate": predicate or "",
    }


def update_private_setup(setup_id: int, groups: dict, condition: str):
    """Update a setup's logic from the editor shape ({group_name: [ce_id]},
    condition string).

    Resolves ids -> names, validates the (ce_groups, condition) pair, then
    rewrites the setup's membership links + ce_groups/condition/predicate.
    If the setup is backed by the user's own LOCAL DRAFT rule, the draft's
    rules row + rule_ce_link membership are patched in place too (the
    in-place edit path — no new rule entity, no draft clutter). A setup
    backed by a PUBLIC rule only gets its private copy updated; forking is
    the caller's job (fork_setup_to_draft).

    Returns the derived display predicate. Raises ValueError on invalid
    logic or unknown ce_ids.
    """
    ce_groups, condition, predicate, ce_ids = resolve_editor_logic(groups, condition)

    _replace_setup_membership(setup_id, ce_ids)
    execute_query(
        "UPDATE rule_setup SET ce_groups = %s, condition = %s, predicate = %s WHERE setup_id = %s",
        (ce_groups, condition, predicate, setup_id),
    )

    # In-place path: keep the backing draft rule in lockstep so My Drafts
    # always shows the logic the user last saved.
    backing = execute_query_dict(
        """
        SELECT r.rule_id
        FROM rule_setup rs
        JOIN rules r ON rs.rule_id = r.rule_id
        WHERE rs.setup_id = %s AND r.is_local_draft = TRUE
        """,
        (setup_id,),
    ) or []
    if backing:
        rule_id = backing[0]["rule_id"]
        execute_query(
            "UPDATE rules SET ce_groups = %s, condition = %s, predicate = %s WHERE rule_id = %s",
            (ce_groups, condition, predicate, rule_id),
        )
        execute_query("DELETE FROM rule_ce_link WHERE rule_id = %s", (rule_id,))
        for ce_id in ce_ids:
            execute_query(
                "INSERT INTO rule_ce_link (rule_id, ce_id) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                (rule_id, ce_id),
            )

    return predicate


def create_draft_rule_from_bookmarked(name: str, groups: dict, condition: str,
                                      categories: list | None = None, description: str = ""):
    """Create a GUARDRAIL-AGNOSTIC draft rule from bookmarked CEs.

    `groups` is the editor shape {group_name: [ce_id]}; `condition` is the v2
    grammar string over those group names. Ids are translated to CE names for
    storage.

    Does NOT attach the rule to any guardrail — it only writes the canonical
    `rules` row + `rule_ce_link` membership entries (via
    `upsert_rule_with_links`), exactly like the AI rule pipeline. The result
    is a local draft (is_ready=FALSE until finalized) that lands in the
    user's Drafts; the user adds it to a guardrail later.

    `mark_pending=True` keeps it hidden until the wizard finalizes it, and lets
    the boot-time IncompletePipelineRecovery wipe it cleanly if the user
    abandons the wizard. Structural dedup is enforced globally inside
    `upsert_rule_with_links`. Returns (rule_id, predicate).
    """
    if not groups or not any(members for members in groups.values()):
        raise ValueError("groups cannot be empty")

    ce_groups, condition, predicate, _ = resolve_editor_logic(groups, condition)

    rule_data = {
        "rule_name": name,
        "ce_groups": ce_groups,
        "condition": condition,
        "description": description or "",
        "categories": categories or [],
    }
    from gavel_pipeline.db_access import upsert_rule_with_links
    rule_id = upsert_rule_with_links(rule_data, mark_pending=True)
    return rule_id, predicate


def fork_setup_to_draft(
    setup_id: int,
    user_id: int,
    new_name: str,
    groups: dict,
    condition: str,
    add_bookmark: bool = False,
):
    """
    Promote an in-progress setup edit into a NEW private draft rule.

    `groups` is the editor shape {group_name: [ce_id]}; `condition` is the
    v2 grammar string over those group names.

    Used by the rule-editor Save flow when the source rule is either
    (a) a public library rule the user is forking, or (b) a manual setup
    that hasn't been backed by a rules-table row yet. The user's own
    drafts use the simpler in-place `update_private_setup` path instead
    so editing-then-editing-again doesn't clutter "My Drafts" with stale
    versions.

    Steps:
      1. Validate `new_name` is non-empty and free of name conflicts in
         the global rules table (the column has a UNIQUE constraint, so
         we probe up-front for a clean error message).
      2. Resolve/validate the logic, compute the v2 structural fingerprint
         and reject duplicates of any rule the user could observe — same
         logic the editor's pre-save check applies, but re-run server-side
         because the client could be stale. The per-guardrail scan reads
         sibling rule_setup.ce_groups/condition copies.
      3. Build the new rules row + membership links via
         `upsert_rule_with_links` (gives us is_local_draft=TRUE for free).
      4. Replace the setup's membership rows + ce_groups/condition/
         predicate, and repoint setup.rule_id + setup.custom_name to the
         new rule.
      5. Optionally insert a bookmark for the user so the rule shows up
         in My Bookmarks for reuse on other guardrails.
      6. Return the new rule_id + the derived predicate.
    """
    name = (new_name or "").strip()
    if not name:
        raise ValueError("Rule name is required when saving an edited rule as a draft.")

    # Find the guardrail owning this setup — we need it to scope the
    # within-guardrail dedup probe and to mark needs_retraining later.
    setup_row = execute_query_dict(
        "SELECT classifier_id FROM rule_setup WHERE setup_id = %s",
        (setup_id,),
    )
    if not setup_row:
        raise ValueError(f"setup_id {setup_id} not found")
    classifier_id = setup_row[0]["classifier_id"]

    # 1. Name uniqueness probe. The rules.name column is UNIQUE — a
    # collision will fail the eventual INSERT with an opaque integrity
    # error; probing up-front lets the route surface a clean
    # "already taken" message instead.
    existing = execute_query_dict(
        "SELECT rule_id FROM rules WHERE LOWER(name) = LOWER(%s)",
        (name,),
    )
    if existing:
        raise ValueError(
            f"A rule named '{name}' already exists. Pick a different name "
            f"to save your edit as a new draft."
        )

    # 2. Resolve + validate the logic, then structural dedup. Mirrors the
    # route-level check_rule_duplicate endpoint but re-runs here so a stale
    # client can't smuggle a duplicate past us.
    ce_groups, condition, predicate, ce_ids = resolve_editor_logic(groups, condition)

    from sql_scripts.junction_scripts import (
        compute_rule_fingerprint_v2,
        find_existing_rule_by_fingerprint_v2,
        find_existing_rule_setup_by_fingerprint_v2,
    )
    fingerprint = compute_rule_fingerprint_v2(ce_groups, condition)

    # Within the guardrail's other setups (excluding the one we're editing).
    sibling = find_existing_rule_setup_by_fingerprint_v2(
        classifier_id, fingerprint, exclude_setup_id=setup_id
    )
    if sibling is not None:
        raise ValueError(
            f"Same logic as another rule in this guardrail "
            f"('{sibling.get('custom_name') or 'unnamed'}'). Differentiate the "
            f"logic before forking."
        )
    # And against the global rules table (excluding by NAME — none of
    # our existing rule names match `name` since we just probed for that).
    global_dup = find_existing_rule_by_fingerprint_v2(fingerprint)
    if global_dup is not None:
        raise ValueError(
            f"Same logic as existing rule '{global_dup.get('name')}'. "
            f"Modify the rule structure before forking."
        )

    # 3. Build the new rules row + membership links.
    rule_data = {
        "rule_name": name,
        "ce_groups": ce_groups,
        "condition": condition,
        "description": "",
        "categories": [],
    }
    from gavel_pipeline.db_access import upsert_rule_with_links
    new_rule_id = upsert_rule_with_links(rule_data)

    # 4. Replace the setup's membership + logic and repoint at the new rule.
    _replace_setup_membership(setup_id, ce_ids)
    execute_query(
        "UPDATE rule_setup SET rule_id = %s, custom_name = %s, predicate = %s, "
        "ce_groups = %s, condition = %s WHERE setup_id = %s",
        (new_rule_id, name, predicate, ce_groups, condition, setup_id),
    )

    # Editing the policy means the trained guardrail no longer matches
    # what's selected — flag for retraining.
    execute_query(
        "UPDATE classifiers SET status = 'needs_retraining' WHERE classifier_id = %s AND status = 'active'",
        (classifier_id,),
    )

    # 5. Optional bookmark for cross-guardrail reuse.
    if add_bookmark:
        try:
            from services.bookmarks import BookmarkService
            BookmarkService.add("rule", new_rule_id)
        except Exception:
            # Bookmark is opt-in convenience; a duplicate-bookmark race
            # or transient failure shouldn't roll back the fork itself.
            pass

    return {"rule_id": new_rule_id, "predicate": predicate}


def delete_rule_setup(setup_id: int):
    execute_query("DELETE FROM rule_setup WHERE setup_id = %s", (setup_id,))
    return True


def delete_model(model_id: int):
    """
    Removes a target model. 
    Cascade: Deletes all linked guardrails -> rule_setups -> setup_ce_links.
    """
    query = "DELETE FROM target_models WHERE model_id = %s"
    execute_query(query, (model_id,))
    return True

def delete_classifier(classifier_id: int):
    """
    Removes a guardrail.
    Cascade: Deletes all linked rule_setups -> setup_ce_links.
    """
    query = "DELETE FROM classifiers WHERE classifier_id = %s"
    execute_query(query, (classifier_id,))
    return True
