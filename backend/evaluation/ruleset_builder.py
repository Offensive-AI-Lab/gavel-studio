# evaluation/ruleset_builder.py
# Builds the unified ruleset dict from the DB (rule_setup + setup_ce_link tables).
# Maps the platform's role names onto the reference evaluation format:
#   necessary  → all_required
#   fallback   → any_of (grouped by fallback_group)
#   sufficient → supporting
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _sanitize_ce_name(name: str) -> str:
    """Normalize a CE name the SAME way the trainer does when building the
    guardrail's ``labels`` dict (classifier_engine.trainer._sanitize_label).

    The trained guardrail only knows CEs by their sanitized names (e.g.
    "provide or give" -> "provide_or_give"). The ruleset, however, is built
    straight from ``cognitive_elements.name`` (raw). If we don't sanitize here,
    a rule's required CE name ("provide or give") won't match the labels-dict
    key ("provide_or_give"), so convert_labels_to_tensors / load_any_of_conditions
    silently drop it — making rules look like they're "missing required CEs"
    even when every CE has triggered. Keep this regex in lockstep with
    trainer._sanitize_label.
    """
    if not name:
        return name
    return re.sub(r'[^\w\-]', '_', name).strip('_') or "label"


def build_unified_ruleset(classifier_id: int) -> Dict[str, dict]:
    """Query DB and build a unified ruleset dict for evaluation.

    Returns:
        {
            "use_case_name": {
                "all_required": ["CE_Name_1"],
                "any_of": [["CE_A", "CE_B"], ["CE_C"]],
                "supporting": ["CE_D"],
                "enabled": True
            },
            ...
        }

    Selection logic:
      * If the guardrail has a frozen training snapshot (trained_rule_setup_ids
        is not NULL/empty), the ruleset is built ONLY from those setup_ids —
        regardless of how the user has since edited the live rule_setup.
        That's what evaluation, calibration, and the realtime guardrail
        need: the trained weights only know the CEs that were active at
        training time, so scoring against newly-added rules would be
        meaningless.
      * Otherwise (never trained, or snapshot was cleared), fall back to
        every rule_setup row currently attached to the guardrail.
    """
    from utils.sqlite_db import execute_query_dict

    snapshot = execute_query_dict(
        "SELECT trained_rule_setup_ids FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    ) or []
    trained_ids = (snapshot[0].get("trained_rule_setup_ids") if snapshot else None) or []

    # Single SELECT shared between both branches — only the WHERE clause
    # differs. Inlined as a format-string with a placeholder predicate to
    # avoid duplicating 7 lines of column projections.
    base_query = """
        SELECT
            rs.setup_id,
            COALESCE(rs.custom_name, r.name) AS rule_name,
            rs.is_active,
            rs.predicate,
            ce.name AS ce_name,
            scl.role,
            scl.fallback_group
        FROM rule_setup rs
        LEFT JOIN rules r ON rs.rule_id = r.rule_id
        JOIN setup_ce_link scl ON rs.setup_id = scl.setup_id
        JOIN cognitive_elements ce ON scl.ce_id = ce.ce_id
        WHERE {where}
        ORDER BY rs.setup_id, scl.role, scl.fallback_group
    """

    rows: list = []
    used_snapshot = False

    if trained_ids:
        rows = execute_query_dict(
            base_query.format(where="rs.classifier_id = %s AND rs.setup_id = ANY(%s)"),
            (classifier_id, trained_ids),
        ) or []
        if rows:
            used_snapshot = True
            logger.info(
                f"build_unified_ruleset(classifier {classifier_id}): using trained "
                f"snapshot of {len(trained_ids)} setup_id(s)"
            )
        else:
            # Snapshot setup_ids point at rows that no longer exist in
            # rule_setup — usually because the user deleted a rule and
            # re-added it, which mints a new setup_id even if the rule
            # name and content are identical. Falling back to the live
            # rule_setup is the least-bad option: calibration / evaluation
            # still run, and the frontend's drift banner already tells
            # the user to retrain (since current setup_ids != snapshot).
            logger.warning(
                f"build_unified_ruleset(classifier {classifier_id}): trained "
                f"snapshot {trained_ids} is orphaned (no matching rule_setup "
                f"rows). Falling back to live rule_setup so calibration / "
                f"evaluation can still run; user should retrain."
            )

    if not used_snapshot:
        rows = execute_query_dict(
            base_query.format(where="rs.classifier_id = %s"),
            (classifier_id,),
        ) or []

    # Group by rule (setup_id)
    rules: Dict[int, dict] = {}
    for row in rows:
        sid = row["setup_id"]
        if sid not in rules:
            rules[sid] = {
                "name": row["rule_name"],
                "enabled": bool(row["is_active"]),
                "all_required": [],
                "any_of_groups": {},  # fallback_group -> [ce_names]
                "supporting": [],
            }

        # Sanitize to match the trained guardrail's labels-dict keys (see
        # _sanitize_ce_name). No-op for names that are already underscore/word
        # safe, so existing rulesets are unaffected.
        ce_name = _sanitize_ce_name(row["ce_name"])
        role = (row["role"] or "necessary").lower()
        fb_group = row["fallback_group"] or 0

        if role == "necessary":
            rules[sid]["all_required"].append(ce_name)
        elif role == "fallback":
            rules[sid]["any_of_groups"].setdefault(fb_group, []).append(ce_name)
        elif role == "sufficient":
            rules[sid]["supporting"].append(ce_name)
        else:
            # Unknown role — treat as necessary
            rules[sid]["all_required"].append(ce_name)

    # Build final unified dict keyed by rule name
    unified = {}
    for sid, rule in rules.items():
        name = rule["name"] or f"rule_{sid}"
        # Convert any_of_groups dict to sorted list of lists
        any_of = [
            group for _, group in sorted(rule["any_of_groups"].items())
        ] if rule["any_of_groups"] else []

        unified[name] = {
            "all_required": rule["all_required"],
            "any_of": any_of,
            "supporting": rule["supporting"],
            "enabled": rule["enabled"],
        }

    logger.info(f"Built unified ruleset for classifier {classifier_id}: {len(unified)} rules")
    return unified


def get_classifier_labels(classifier_id: int) -> Dict[str, int]:
    """Load the labels dict from the trained guardrail metadata.

    Returns:
        Dict mapping sanitized CE name -> label index (e.g. {"Tax_Evasion": 0, "Bribery": 1}).
        Returns empty dict if guardrail is not trained.
    """
    import json
    import os
    from classifier_engine.trainer import classifier_workdir

    try:
        meta_path = os.path.join(classifier_workdir(classifier_id), "classifier_meta.json")
    except ValueError:
        # guardrail row vanished — caller's already in error territory.
        return {}
    if not os.path.exists(meta_path):
        logger.warning(f"No classifier_meta.json for classifier {classifier_id}")
        return {}

    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("labels", {})


def get_classifier_metadata(classifier_id: int) -> Optional[dict]:
    """Load full guardrail metadata (labels, dims, layers, etc.)."""
    import json
    import os
    from classifier_engine.trainer import classifier_workdir

    try:
        meta_path = os.path.join(classifier_workdir(classifier_id), "classifier_meta.json")
    except ValueError:
        return None
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        return json.load(f)
