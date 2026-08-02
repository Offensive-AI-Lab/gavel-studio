"""Loader for the neutral evaluation corpus.

The corpus is a pool of short, domain-agnostic conversations the guardrail
must NOT fire on. It's the third evaluation split (alongside the positive +
negative test sets) and is what gives a meaningful false-positive rate against
universal everyday content.

Split into the reference two pseudo use-cases:
  - conversational : small-talk / opinions / chit-chat
  - instructive    : how-to / factual / informational Q&A

ALL-OR-NOTHING: the corpus lives ONLY in the gavel-rules library
(neutral/<category>/conversations.json), synced into the `neutral_corpus` DB
table. There is no bundled fallback — if the data isn't present locally,
evaluation refuses to run (see routes/evaluation.py) rather than silently
scoring against a partial set. It is generated and pushed to the registry by
local admin tooling under backend/scripts/ (not part of the running app).
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

CATEGORIES = ("conversational", "instructive")


# --- conversation identity ---------------------------------------------------

def _canonical(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Normalize a conversation to a stable shape for hashing/storage:
    a list of {role, content} with trimmed content, lowercased roles."""
    out: List[Dict[str, str]] = []
    for m in conversation or []:
        if not isinstance(m, dict):
            continue
        out.append({
            "role": str(m.get("role", "")).strip().lower(),
            "content": str(m.get("content", "")).strip(),
        })
    return out


def content_hash(conversation: List[Dict[str, str]]) -> str:
    """Stable sha256 of a conversation's canonical form. Used as the dedup key
    so the same dialogue is never stored (or published) twice."""
    payload = json.dumps(_canonical(conversation), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- runtime loaders (DB only) ----------------------------------------------

def load_neutral_corpus_by_category() -> Dict[str, List[List[Dict[str, str]]]]:
    """Return neutral conversations grouped by category, from the synced
    `neutral_corpus` DB table. Returns empty lists when the table is empty or
    unavailable — there is NO bundled fallback (all-or-nothing); the caller
    treats an empty corpus as "cannot evaluate".
    """
    grouped: Dict[str, List[List[Dict[str, str]]]] = {c: [] for c in CATEGORIES}
    try:
        from utils.sqlite_db import execute_query_dict
        rows = execute_query_dict("SELECT category, conversation FROM neutral_corpus") or []
    except Exception as e:
        logger.warning("neutral_corpus DB read failed: %s", e)
        return grouped

    for r in rows:
        cat = r["category"] if r["category"] in CATEGORIES else "conversational"
        conv = r["conversation"]
        if isinstance(conv, str):
            conv = json.loads(conv)
        if isinstance(conv, list) and len(conv) >= 2:
            grouped[cat].append(conv)
    return grouped


def load_neutral_corpus() -> List[List[Dict[str, str]]]:
    """Flat list of all neutral conversations (both categories)."""
    grouped = load_neutral_corpus_by_category()
    return [conv for cat in CATEGORIES for conv in grouped.get(cat, [])]
