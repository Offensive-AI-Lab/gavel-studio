import logging
from fastapi import APIRouter, HTTPException, Depends
from sql_scripts.user_scripts import get_user_by_id
from pydantic import BaseModel
from typing import List, Dict
from utils.sqlite_db import execute_query_dict
from utils.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Response Models ---
class DashboardStats(BaseModel):
    total_models: int
    total_classifiers: int
    active_classifiers: int
    total_rules: int
    total_ces: int
    total_evaluations: int
    total_test_datasets: int

class DashboardResponse(BaseModel):
    user_info: Dict[str, str]
    stats: DashboardStats
    recent_activity: List[dict]
    classifier_summary: List[dict]


# --- Endpoint ---
@router.get("/{user_id}")
def get_dashboard_data(user_id: int, current_user: int = Depends(get_current_user)):
    # Ownership: a user may only read their OWN dashboard (no cross-user peeking).
    if current_user != user_id:
        raise HTTPException(status_code=403, detail="Not authorized for this dashboard")
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        # Count models
        models = execute_query_dict(
            "SELECT COUNT(*) as count FROM target_models WHERE user_id = %s", (user_id,)
        )
        total_models = models[0]["count"] if models else 0

        # Count guardrails and active ones. Owned directly via
        # classifiers.user_id so unattached guardrails (no model yet) count too.
        classifiers = execute_query_dict("""
            SELECT c.status, COUNT(*) as count
            FROM classifiers c
            WHERE c.user_id = %s
            GROUP BY c.status
        """, (user_id,))
        status_counts = {r["status"]: r["count"] for r in (classifiers or [])}
        total_classifiers = sum(status_counts.values())
        active_classifiers = status_counts.get("active", 0) + status_counts.get("needs_retraining", 0)

        # Count THIS user's rules / CEs — everything they created, whether it's
        # still a local draft or already published (i.e. what they see in
        # Library "Browse + Drafts"). Counting the global tables here was wrong:
        # it returned every user's items and drifted from manual deletions on HF.
        username = user.get("username")
        rules = execute_query_dict(
            "SELECT COUNT(*) as count FROM rules WHERE LOWER(created_by_username) = LOWER(%s)",
            (username,),
        )
        total_rules = rules[0]["count"] if rules else 0

        ces = execute_query_dict(
            "SELECT COUNT(*) as count FROM cognitive_elements WHERE LOWER(created_by_username) = LOWER(%s)",
            (username,),
        )
        total_ces = ces[0]["count"] if ces else 0

        # Count evaluations
        evals = execute_query_dict("""
            SELECT COUNT(*) as count FROM evaluation_results er
            JOIN classifiers c ON er.classifier_id = c.classifier_id
            WHERE c.user_id = %s
        """, (user_id,))
        total_evaluations = evals[0]["count"] if evals else 0

        # Count the user's own (private custom) test datasets. Test sets are
        # rule-scoped now (v10); defaults aren't owned by anyone, so we count
        # only what this user created.
        tests = execute_query_dict("""
            SELECT COUNT(*) as count FROM test_datasets td
            WHERE td.user_id = %s
        """, (user_id,))
        total_test_datasets = tests[0]["count"] if tests else 0

        # Per-guardrail summary (name, status, #rules, #CEs, last eval)
        classifier_summary = execute_query_dict("""
            SELECT
                c.classifier_id,
                c.name AS classifier_name,
                c.status,
                tm.name AS model_name,
                (SELECT COUNT(*) FROM rule_setup rs WHERE rs.classifier_id = c.classifier_id) AS rule_count,
                (SELECT COUNT(DISTINCT scl.ce_id) FROM rule_setup rs2
                 JOIN setup_ce_link scl ON rs2.setup_id = scl.setup_id
                 WHERE rs2.classifier_id = c.classifier_id) AS ce_count,
                (SELECT MAX(er2.created_at) FROM evaluation_results er2
                 WHERE er2.classifier_id = c.classifier_id) AS "last_evaluation [TIMESTAMPTZ]"
            FROM classifiers c
            LEFT JOIN target_models tm ON c.model_id = tm.model_id
            WHERE c.user_id = %s
            ORDER BY c.classifier_id DESC
            LIMIT 10
        """, (user_id,))

        # Recent activity: latest evaluations and training events
        recent = execute_query_dict("""
            SELECT
                'evaluation' AS event_type,
                er.eval_type AS detail,
                c.name AS classifier_name,
                er.created_at
            FROM evaluation_results er
            JOIN classifiers c ON er.classifier_id = c.classifier_id
            WHERE c.user_id = %s
            ORDER BY er.created_at DESC
            LIMIT 10
        """, (user_id,))

    except Exception as e:
        logger.error(f"Dashboard stats query failed: {e}")
        # Fallback to zeros if tables don't exist yet
        total_models = total_classifiers = active_classifiers = 0
        total_rules = total_ces = total_evaluations = total_test_datasets = 0
        classifier_summary = []
        recent = []

    stats = {
        "total_models": total_models,
        "total_classifiers": total_classifiers,
        "active_classifiers": active_classifiers,
        "total_rules": total_rules,
        "total_ces": total_ces,
        "total_evaluations": total_evaluations,
        "total_test_datasets": total_test_datasets,
    }

    # Serialize datetimes in recent activity
    activity = []
    for r in (recent or []):
        entry = dict(r)
        if entry.get("created_at"):
            entry["created_at"] = str(entry["created_at"])
        activity.append(entry)

    summary = []
    for s in (classifier_summary or []):
        entry = dict(s)
        if entry.get("last_evaluation"):
            entry["last_evaluation"] = str(entry["last_evaluation"])
        summary.append(entry)

    return {
        "user_info": {
            "username": user.get("username"),
        },
        "stats": stats,
        "recent_activity": activity,
        "classifier_summary": summary,
    }