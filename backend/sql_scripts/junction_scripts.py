from utils.sqlite_db import execute_query, execute_query_dict

# --- JUNCTION TABLE LOGIC (Rule Instance <-> CE) ---

def link_ce_to_setup(setup_id: int, ce_id: int):
    """
    Saves the connection in setup_ce_link so the CE stays on the 
    specific rule card even if the rule is private.
    """
    query = """
        INSERT INTO setup_ce_link (setup_id, ce_id) 
        VALUES (%s, %s) 
        ON CONFLICT DO NOTHING
    """
    try:
        execute_query(query, (setup_id, ce_id))
        return True
    except Exception as e:
        print(f"Error linking CE to setup: {e}")
        return False

def unlink_ce_from_setup(setup_id: int, ce_id: int):
    """
    Removes the specific link when the user deletes a tag from a rule card.
    Maintains a clean design space for the guardrail.
    """
    query = "DELETE FROM setup_ce_link WHERE setup_id = %s AND ce_id = %s"
    try:
        execute_query(query, (setup_id, ce_id))
        return True
    except Exception as e:
        print(f"Error unlinking CE from setup: {e}")
        return False

def get_ces_for_setup(setup_id: int):
    """
    Fetches all Cognitive Elements specifically linked to this setup instance.
    """
    query = """
        SELECT ce.ce_id, ce.name
        FROM cognitive_elements ce
        JOIN setup_ce_link link ON ce.ce_id = link.ce_id
        WHERE link.setup_id = %s
    """
    return execute_query_dict(query, (setup_id,))


# ---------------------------------------------------------------------------
# Rule structural fingerprint
# ---------------------------------------------------------------------------
#
# A rule's "logic" is fully described by which CEs play which role and how
# fallback groups are partitioned. Two rules with the same role/fallback
# structure but different names are functionally identical at scoring time —
# the user gets confused if both show up in their library, and the trained
# guardrail wastes capacity learning duplicate detectors. The fingerprint
# below normalizes that structure into a stable string so we can dedup on
# create.
#
# Format (chosen for human-readability when surfaced in error messages):
#   N:[ce_id_1,ce_id_2,...]|F:[[a,b],[c]]|S:[ce_id_x,...]
# where each list is sorted, fallback groups are individually sorted then
# the list-of-groups is itself sorted. fallback_group integer values are
# discarded — only the partition matters, not the user's group numbering.

def compute_rule_fingerprint_from_links(ce_links: list) -> str:
    """ce_links is a list of {ce_id, role, fallback_group}. Used by the
    bookmarked-CE path where the request body already speaks ce_ids."""
    necessary, sufficient = [], []
    fallback_groups: dict = {}
    for link in ce_links or []:
        ce_id = link.get("ce_id")
        if ce_id is None:
            continue
        role = (link.get("role") or "necessary").lower()
        if role == "necessary":
            necessary.append(ce_id)
        elif role == "sufficient":
            sufficient.append(ce_id)
        elif role == "fallback":
            grp = int(link.get("fallback_group", 0) or 0)
            fallback_groups.setdefault(grp, []).append(ce_id)

    necessary_sorted = sorted(necessary)
    sufficient_sorted = sorted(sufficient)
    fallback_normalized = sorted(
        tuple(sorted(group)) for group in fallback_groups.values()
    )
    return f"N:{tuple(necessary_sorted)}|F:{fallback_normalized}|S:{tuple(sufficient_sorted)}"


def compute_rule_fingerprint_from_names(necessary, fallback, sufficient) -> str:
    """Same fingerprint, but the inputs are CE NAMES grouped by role.
    Used by the AI pipeline / public-rule path (`upsert_rule_with_links`),
    which speaks names because rule_ce_link is hydrated by name lookup.

    Names are translated to ce_ids before fingerprinting so the result
    is comparable to fingerprints computed from ce_id-shaped inputs."""
    all_names = set(necessary or [])
    all_names.update(sufficient or [])
    for group in (fallback or []):
        all_names.update(group)
    if not all_names:
        return compute_rule_fingerprint_from_links([])

    rows = execute_query_dict(
        "SELECT ce_id, name FROM cognitive_elements WHERE name = ANY(%s)",
        (list(all_names),),
    ) or []
    name_to_id = {r["name"]: r["ce_id"] for r in rows}

    links: list = []
    for ce_name in (necessary or []):
        cid = name_to_id.get(ce_name)
        if cid is not None:
            links.append({"ce_id": cid, "role": "necessary", "fallback_group": 0})
    for ce_name in (sufficient or []):
        cid = name_to_id.get(ce_name)
        if cid is not None:
            links.append({"ce_id": cid, "role": "sufficient", "fallback_group": 0})
    for idx, group in enumerate((fallback or []), start=1):
        for ce_name in group:
            cid = name_to_id.get(ce_name)
            if cid is not None:
                links.append({"ce_id": cid, "role": "fallback", "fallback_group": idx})
    return compute_rule_fingerprint_from_links(links)


def find_existing_rule_setup_by_fingerprint(classifier_id: int, fingerprint: str):
    """Scan rule_setup rows in this guardrail and return the first that
    matches the structural fingerprint, or None. Used by manual rule
    creation (bookmarked-CE flow) to refuse duplicates with clearer
    messaging than a name conflict.

    The fingerprint comparison happens in Python rather than SQL because
    the storage shape (rule_setup + setup_ce_link) doesn't have a stable
    serialized fingerprint column — adding one would tie us to recompute
    on every CE-link edit, which is more invasive than just walking the
    guardrail's setups (typically <100 per guardrail in practice).
    """
    rows = execute_query_dict(
        """
        SELECT rs.setup_id, rs.custom_name,
               COALESCE(
                   json_group_array(
                       json_object(
                           'ce_id', scl.ce_id,
                           'role', scl.role,
                           'fallback_group', scl.fallback_group
                       )
                   ) FILTER (WHERE scl.ce_id IS NOT NULL),
                   '[]'
               ) AS "links [JSONB]"
        FROM rule_setup rs
        LEFT JOIN setup_ce_link scl ON scl.setup_id = rs.setup_id
        WHERE rs.classifier_id = %s
        GROUP BY rs.setup_id, rs.custom_name
        """,
        (classifier_id,),
    ) or []
    for row in rows:
        if compute_rule_fingerprint_from_links(row["links"]) == fingerprint:
            return row
    return None


def find_existing_rule_by_fingerprint(fingerprint: str, exclude_name: str = None):
    """Scan the GLOBAL `rules` table for a rule with the same structural
    fingerprint. Returns the first match (with name + rule_id) or None.

    Used by `upsert_rule_with_links` to flag a same-structure-different-
    name collision before writing. `exclude_name` lets the caller
    re-save a rule under its OWN name without tripping the dedup."""
    rows = execute_query_dict(
        """
        SELECT r.rule_id, r.name,
               COALESCE(
                   json_group_array(
                       json_object(
                           'ce_id', rcl.ce_id,
                           'role', rcl.role,
                           'fallback_group', rcl.fallback_group
                       )
                   ) FILTER (WHERE rcl.ce_id IS NOT NULL),
                   '[]'
               ) AS "links [JSONB]"
        FROM rules r
        LEFT JOIN rule_ce_link rcl ON rcl.rule_id = r.rule_id
        WHERE (%s IS NULL OR r.name <> %s)
        GROUP BY r.rule_id, r.name
        """,
        (exclude_name, exclude_name),
    ) or []
    for row in rows:
        if compute_rule_fingerprint_from_links(row["links"]) == fingerprint:
            return row
    return None