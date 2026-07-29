"""Generate a rule's plain-English explanation (rules.description).

Used by the build-from-CEs flow: that path has no AI to write a description, so
the user can type one — and if they leave it blank, we derive it from the misuse
scenario they confirm on the Test & Calibration step. "Only if empty" means an
AI-pipeline rule (which already has a description) is never overwritten.
"""
import logging

from utils.sqlite_db import execute_query, execute_query_dict

logger = logging.getLogger(__name__)


def _generate_from_scenario(scenario: str) -> str:
    """One-to-two sentence, user-facing explanation distilled from the misuse
    scenario. Returns '' on any LLM failure (caller treats it as 'skip')."""
    from gavel_pipeline.rule_generator import call_llm

    prompt = (
        "A content-safety detection rule is being created. Below is the misuse "
        "scenario it is meant to catch. Write a concise 1-2 sentence, plain-English "
        "explanation of what the rule detects — the real-world behavior or content "
        "that makes it fire. Write for a product user, not a developer. Do NOT add a "
        "preamble like 'This rule'; output ONLY the explanation sentence(s).\n\n"
        f"Misuse scenario:\n{scenario.strip()}\n"
    )
    text, err = call_llm([{"role": "user", "content": prompt}], temperature=0.3)
    if err or not text:
        logger.warning("[rule_descriptions] LLM failed: %s", err)
        return ""
    return text.strip().strip('"').strip()


def fill_description_from_scenario_if_empty(rule_id: int, scenario: str) -> bool:
    """If the rule has no explanation yet, derive one from the misuse scenario
    and store it. Returns True iff a description was written.

    Best-effort and idempotent: a rule that already has a description (the user
    typed one, or it came from the AI pipeline) is left untouched.
    """
    if not scenario or not scenario.strip():
        return False
    rows = execute_query_dict(
        "SELECT description FROM rules WHERE rule_id = %s", (rule_id,)
    ) or []
    if not rows or (rows[0].get("description") or "").strip():
        return False
    desc = _generate_from_scenario(scenario)
    if not desc:
        return False
    execute_query("UPDATE rules SET description = %s WHERE rule_id = %s", (desc, rule_id))
    logger.info("[rule_descriptions] derived description for rule %s from its scenario", rule_id)
    return True
