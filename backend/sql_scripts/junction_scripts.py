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
# Rule structural fingerprint — v2 (groups + condition model)
# ---------------------------------------------------------------------------
#
# Under the gavel-rules data model a rule's logic is (ce_groups, condition):
# named groups of CE names plus a condition string in the v2 grammar
# (utils/rule_condition.py). Two rules are structurally identical when their
# firing logic is the same REGARDLESS of what the author called the groups —
# so the fingerprint canonicalizes: referenced groups are keyed by their
# sorted member tuple (names, not ids — ce_groups stores names and that is
# the identity conditions speak), renamed g0..gN in that canonical order,
# and the condition AST is serialized with commutative and/or children
# sorted. Groups never referenced by the condition (e.g. the migrated
# 'supporting' bucket) do not affect firing and are excluded.

def compute_rule_fingerprint_v2(ce_groups: dict | None, condition: str | None) -> str:
    """Canonical fingerprint of a (ce_groups, condition) pair.

    ce_groups: {group_name: [ce_name, ...]}. condition: v2 grammar string or
    None/empty (legacy rules with membership but no logic). Falls back to a
    membership-only fingerprint when the condition is absent or unparseable
    (unparseable should never happen for validated rows, but a fingerprint
    must never raise — dedup is best-effort)."""
    from utils.rule_condition import ConditionError, parse, referenced_groups

    groups = ce_groups or {}
    members_of = {
        name: tuple(sorted(set(members or [])))
        for name, members in groups.items()
    }

    all_members = tuple(sorted({m for t in members_of.values() for m in t}))
    if not (condition or "").strip():
        return f"M:{all_members}"

    try:
        ast = parse(condition)
    except ConditionError:
        return f"M:{all_members}"

    refs = sorted(
        (name for name in referenced_groups(ast) if name in members_of),
        key=lambda n: (members_of[n], n),
    )
    canonical_name = {name: f"g{i}" for i, name in enumerate(refs)}

    def serialize(node) -> str:
        kind = node[0]
        if kind in ("and", "or"):
            inner = ",".join(sorted(serialize(c) for c in node[1]))
            return f"{'a' if kind == 'and' else 'o'}({inner})"
        if kind == "not":
            return f"n({serialize(node[1])})"
        _, quant, name = node
        cname = canonical_name.get(name, name)
        return f"s({quant},{cname})"

    canon_groups = "|".join(
        f"{canonical_name[n]}:{members_of[n]}" for n in refs
    )
    return f"G:{canon_groups}|C:{serialize(ast)}"


def find_existing_rule_by_fingerprint_v2(fingerprint: str, exclude_name: str = None):
    """Scan the GLOBAL `rules` table for a rule whose (ce_groups, condition)
    logic has the same canonical fingerprint. Returns the first match
    ({rule_id, name, ...}) or None.

    Used by `upsert_rule_with_links` (and the editor's check-duplicate probe)
    to flag a same-structure-different-name collision before writing.
    `exclude_name` lets a rule be re-saved under its OWN name without
    tripping the dedup. Rules with no group members carry no logic and are
    never reported as duplicates of each other.
    """
    rows = execute_query_dict(
        """
        SELECT rule_id, name, ce_groups, condition
        FROM rules
        WHERE (%s IS NULL OR name <> %s)
        """,
        (exclude_name, exclude_name),
    ) or []
    for row in rows:
        groups = row.get("ce_groups") or {}
        if not any(members for members in groups.values()):
            continue  # no logic — nothing to duplicate
        if compute_rule_fingerprint_v2(groups, row.get("condition")) == fingerprint:
            return row
    return None


def find_existing_rule_setup_by_fingerprint_v2(
    classifier_id: int, fingerprint: str, exclude_setup_id: int = None
):
    """Scan this guardrail's rule_setup rows for one whose per-guardrail
    (ce_groups, condition) copy matches the canonical fingerprint. Returns
    the first match ({setup_id, custom_name, ...}) or None.

    Used by the rule editor's Save flow to refuse a fork that duplicates a
    sibling rule in the same guardrail. `exclude_setup_id` skips the setup
    currently being edited so an unchanged save doesn't match itself.
    Setups with no group members (fresh custom setups) are never matched.
    """
    rows = execute_query_dict(
        """
        SELECT setup_id, custom_name, ce_groups, condition
        FROM rule_setup
        WHERE classifier_id = %s
          AND (%s IS NULL OR setup_id <> %s)
        """,
        (classifier_id, exclude_setup_id, exclude_setup_id),
    ) or []
    for row in rows:
        groups = row.get("ce_groups") or {}
        if not any(members for members in groups.values()):
            continue
        if compute_rule_fingerprint_v2(groups, row.get("condition")) == fingerprint:
            return row
    return None
