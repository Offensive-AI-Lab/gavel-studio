"""
Single-shot CE generator. Mirrors the rule pipeline's pattern: takes a user
description, builds a comprehensive prompt with existing CEs and taxonomy,
calls the LLM ONCE, and returns a fully-structured CE (with definition, type,
examples, categories) ready to feed into /ce-training/generate.

This replaces the multi-turn CE chat. The advantages:
  - No conversational drift — categories never get "forgotten" because the
    JSON schema requires them.
  - Built-in orthogonality check — the prompt forces the model to refuse
    if the concept is already covered by existing CEs.
  - Structured output (response_format=json_object) — no parser fragility.
"""
import os
import json
import re
import warnings
from typing import Dict, Optional, Tuple

import litellm

from gavel_pipeline.db_access import fetch_categories_dict
from utils.sqlite_db import execute_query_dict


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "ce_generator_prompt.md")
CE_GENERATOR_MODEL = "gpt-4.1"  # gpt-4.1 with structured output is reliable here


def _load_prompt_template() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _format_existing_ces() -> str:
    """Format every published or local-draft CE as a markdown bullet list
    for the orthogonality-check section of the prompt. Skips is_ready=FALSE
    rows (in-flight pipeline outputs that shouldn't be visible yet).
    """
    rows = execute_query_dict(
        """
        SELECT name, definition, category, is_local_draft
        FROM cognitive_elements
        WHERE is_ready = TRUE
        ORDER BY name
        """
    ) or []
    if not rows:
        return "(no existing CEs yet — anything you create will be the first.)"
    lines = []
    for r in rows:
        name = r.get("name") or ""
        definition = (r.get("definition") or "").strip().replace("\n", " ")
        kind = (r.get("category") or "").strip().upper() or "CE"
        draft_tag = " *(draft)*" if r.get("is_local_draft") else ""
        lines.append(f"- **{name}** [{kind}]{draft_tag} — {definition}")
    return "\n".join(lines)


def _format_categories(cats: list) -> str:
    return "\n".join(
        f"- **{c['name']}**: {c.get('description', '')}" for c in cats
    )


def categorize_ce_with_llm(ce_name: str, definition: str) -> Dict:
    """
    Single-shot LLM call to assign 1-2 categories to a CE based on its
    name + definition. Used as a safety-net fallback in /ce-training/generate
    when categories somehow arrive empty.

    Returns dict with:
      - "assigned_categories": list of existing category names (canonical case)
      - "new_category_name": str (only set if no existing fits)
      - "new_category_description": str
    """
    result = {"assigned_categories": [], "new_category_name": "", "new_category_description": ""}
    try:
        cats = fetch_categories_dict()
        if not cats:
            return result
        taxonomy_text = "\n".join(
            f"- **{c['name']}**: {c.get('description', '')}" for c in cats
        )
        existing_names = [c["name"] for c in cats]

        system_prompt = (
            "You are a Senior AI Taxonomy Architect. Your job: assign a Cognitive Element (CE) "
            "to 1-2 categories from an existing taxonomy. Only propose a new category if NO existing "
            "category genuinely fits. Strongly prefer existing categories — multi-label is allowed."
        )
        user_prompt = (
            f"**CE Name:** {ce_name}\n"
            f"**CE Definition:** {definition}\n\n"
            f"**Existing Taxonomy:**\n{taxonomy_text}\n\n"
            f"Pick 1-2 existing category NAMES that best match this CE. Use the exact names from "
            f"the taxonomy above (case-sensitive). Only propose `new_category` if NONE of the existing "
            f"categories fit at all — and you must justify why each existing category is unsuitable.\n\n"
            f"Respond with JSON only:\n"
            f"{{\n"
            f'  "assigned_categories": ["<exact existing category name>", ...],\n'
            f'  "new_category": null OR {{"name": "<broad new category>", "description": "<~20 words>"}}\n'
            f"}}"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Pydantic serializer warnings", UserWarning)
            resp = litellm.completion(
                model=CE_GENERATOR_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
        data = json.loads(resp.choices[0].message.content)

        assigned = data.get("assigned_categories") or []
        if isinstance(assigned, str):
            assigned = [assigned]
        existing_lower = {n.lower(): n for n in existing_names}
        for item in assigned:
            if not isinstance(item, str):
                continue
            key = item.strip().lower()
            if key in existing_lower and existing_lower[key] not in result["assigned_categories"]:
                result["assigned_categories"].append(existing_lower[key])

        if not result["assigned_categories"]:
            new_cat = data.get("new_category")
            if isinstance(new_cat, dict):
                nc_name = (new_cat.get("name") or "").strip()
                nc_desc = (new_cat.get("description") or "").strip()
                if nc_name and nc_name.lower() not in existing_lower:
                    result["new_category_name"] = nc_name
                    result["new_category_description"] = nc_desc
                elif nc_name.lower() in existing_lower:
                    result["assigned_categories"].append(existing_lower[nc_name.lower()])

        print(f"[ce_generator] categorize_ce_with_llm: {ce_name} -> {result}")
    except Exception as e:
        print(f"[!] categorize_ce_with_llm failed for {ce_name!r}: {e}")
    return result


def _format_history(history: list) -> str:
    """Format prior clarification Q&A as text for the prompt."""
    if not history:
        return "(no prior clarifications — this is the first turn)"
    lines = []
    for i, entry in enumerate(history, start=1):
        q = (entry.get("question") or "").strip()
        a = (entry.get("answer") or "").strip()
        if q or a:
            lines.append(f"Round {i}:")
            if q:
                lines.append(f"  AI asked: {q}")
            if a:
                lines.append(f"  User answered: {a}")
    return "\n".join(lines) if lines else "(no prior clarifications)"


def generate_ce(
    user_description: str,
    prefer_type: Optional[str] = None,
    history: Optional[list] = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    CE generation with optional clarification flow.

    The LLM decides per call whether to:
      - Generate the full CE (if the description is clear enough), OR
      - Ask ONE clarifying question (if vague/ambiguous).

    The frontend accumulates clarification Q&A in `history` and re-calls
    until the LLM returns a generated CE (or refuses).

    Args:
        user_description: latest free-text input from the user
        prefer_type: "ACTION" or "CONTEXT" if the user specified, else None
        history: list of {"question": str, "answer": str} for prior clarifications

    Returns:
        (ce_data, error_message).
          - On clarification needed: ({"needs_clarification": True, "clarification_question": "..."}, None)
          - On refusal: ({"refuse": True, "refuse_reason": "..."}, None)
          - On success: (full ce_data dict, None)
          - On error: (None, error_message)
    """
    if not user_description or not user_description.strip():
        return None, "Empty description"

    try:
        template = _load_prompt_template()
    except Exception as e:
        return None, f"Could not load prompt template: {e}"

    try:
        cats = fetch_categories_dict()
    except Exception as e:
        return None, f"Could not fetch categories: {e}"

    if not cats:
        return None, "No categories in DB — cannot assign one"

    existing_ces_text = _format_existing_ces()
    categories_text = _format_categories(cats)
    history_text = _format_history(history or [])

    prompt = (
        template
        .replace("{user_description}", user_description.strip())
        .replace("{prefer_type}", (prefer_type or "").strip())
        .replace("{existing_ces}", existing_ces_text)
        .replace("{current_categories}", categories_text)
        .replace("{history}", history_text)
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Pydantic serializer warnings", UserWarning)
            resp = litellm.completion(
                model=CE_GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
        raw = resp.choices[0].message.content
    except Exception as e:
        return None, f"LLM call failed: {e}"

    print(f"[ce_generator] raw LLM output (first 2000 chars):\n{raw[:2000]}")

    try:
        ce_data = json.loads(raw)
    except Exception:
        # Try to extract JSON from any wrapping the model might add
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None, "LLM output was not valid JSON"
        try:
            ce_data = json.loads(m.group(0))
        except Exception as e:
            return None, f"Could not parse JSON: {e}"

    # Clarification path: short-circuit before validating CE fields
    if ce_data.get("needs_clarification"):
        question = (ce_data.get("clarification_question") or "").strip()
        if not question:
            return None, "LLM signaled needs_clarification but no question provided"
        return {
            "needs_clarification": True,
            "clarification_question": question,
        }, None

    # Honor refusal
    if ce_data.get("refuse"):
        return ce_data, None

    # Validate required fields (only when generating)
    required = ["name", "type", "definition", "in_scope_examples"]
    missing = [f for f in required if not ce_data.get(f)]
    if missing:
        return None, f"LLM output missing required fields: {missing}"

    # Validate categories
    assigned = ce_data.get("assigned_categories") or []
    new_cat = ce_data.get("new_category")
    has_new_cat = isinstance(new_cat, dict) and new_cat.get("name") and new_cat.get("description")
    if not assigned and not has_new_cat:
        return None, "LLM output has no categories — both assigned_categories and new_category are empty"

    # Resolve assigned_categories against the existing taxonomy (case-insensitive
    # match → canonical case). Drops anything that doesn't exist.
    existing_lookup = {c["name"].lower(): c["name"] for c in cats}
    canonical_assigned = []
    for item in assigned:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if key in existing_lookup:
            canonical_assigned.append(existing_lookup[key])
    ce_data["assigned_categories"] = canonical_assigned

    # If the LLM proposed a new_category but its name actually matches an
    # existing one, treat it as existing instead (avoid duplicate categories).
    if has_new_cat:
        nc_key = new_cat["name"].strip().lower()
        if nc_key in existing_lookup:
            existing_name = existing_lookup[nc_key]
            if existing_name not in canonical_assigned:
                canonical_assigned.append(existing_name)
            ce_data["assigned_categories"] = canonical_assigned
            ce_data["new_category"] = None

    # Final check: if after canonicalization both are empty, fail
    if not ce_data["assigned_categories"] and not (
        isinstance(ce_data.get("new_category"), dict) and ce_data["new_category"].get("name")
    ):
        return None, "After canonicalization, no valid categories remain"

    # Normalize type to uppercase
    ce_data["type"] = (ce_data.get("type") or "").upper()
    if ce_data["type"] not in ("ACTION", "CONTEXT"):
        return None, f"Invalid type: {ce_data['type']!r} (must be ACTION or CONTEXT)"

    return ce_data, None
