# evaluation/ruleset_builder.py
# Builds the unified ruleset dict from the DB (rule_setup.ce_groups/condition).
# A rule's logic is named CE groups + a v2-grammar condition over the group
# names ("all of hook and 1 of action" — utils/rule_condition.py). Member CE
# names are sanitized to match the trained guardrail's labels-dict keys.
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
                "groups": {"hook": ["CE_A"], "action": ["CE_B", "CE_C"]},
                "condition": "all of hook and 1 of action",
                "enabled": True
            },
            ...
        }

    Member names inside `groups` are sanitized to the trained labels-dict
    vocabulary; group names pass through untouched (they only ever meet the
    condition parser, never the model). Setups without logic (empty groups or
    no condition — e.g. a migrated supporting-only rule) are omitted: they
    have no firing semantics.

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
    # differs. The logic lives on the rule_setup row itself now; the
    # setup_ce_link junction is membership-only and not needed here.
    base_query = """
        SELECT
            rs.setup_id,
            COALESCE(rs.custom_name, r.name) AS rule_name,
            rs.is_active,
            rs.ce_groups,
            rs.condition
        FROM rule_setup rs
        LEFT JOIN rules r ON rs.rule_id = r.rule_id
        WHERE {where}
        ORDER BY rs.setup_id
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

    # Build the unified dict keyed by rule name. Sanitize member names to
    # the trained labels-dict vocabulary (see _sanitize_ce_name); no-op for
    # names that are already underscore/word safe.
    unified = {}
    for row in rows:
        groups = row.get("ce_groups") or {}
        condition = (row.get("condition") or "").strip()
        if not groups or not condition:
            logger.debug(
                "build_unified_ruleset: setup %s has no logic (groups=%s, "
                "condition=%r) — omitted", row["setup_id"], bool(groups), condition,
            )
            continue
        name = row["rule_name"] or f"rule_{row['setup_id']}"
        unified[name] = {
            "groups": {
                gname: [_sanitize_ce_name(m) for m in (members or [])]
                for gname, members in groups.items()
            },
            "condition": condition,
            "enabled": bool(row["is_active"]),
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
