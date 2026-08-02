# evaluation/realtime.py
# Backend-facing realtime CE monitoring. The PURE classification + generation core
# now lives in classifier_engine.realtime_core (the package shipped to remote
# workers), so the SAME code runs locally AND in the warm remote realtime job.
# This module re-exports the core under the historical names and adds the
# single-turn convenience used by the local path.
from __future__ import annotations
import logging
from typing import Dict, List, Tuple

import torch

from classifier_engine.realtime_core import (  # noqa: F401
    extract_value_cache,
    extract_assistant_reps as _extract_assistant_reps,
    classify_spans as _classify_spans,
    classify_conversation_turns,
    generate_and_classify,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def classify_conversation(
    messages: List[Dict[str, str]],
    model,
    tokenizer,
    classifier,
    meta: dict,
) -> Tuple[List[Dict], List[Dict]]:
    """Classify the LAST assistant turn of an EXISTING conversation (no
    generation). `messages` is a normalized List[{"role","content"}]."""
    window_size = meta.get("rnn_sequence_length", 5)

    asst_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if (messages[i].get("role") or "").lower() == "assistant":
            asst_idx = i
            break
    if asst_idx is None:
        return [], []
    prompt_messages = messages[:asst_idx]
    full_messages = messages[:asst_idx + 1]
    if not prompt_messages:
        return [], []

    assistant_reps, assistant_tokens = _extract_assistant_reps(
        model, tokenizer, prompt_messages, full_messages, meta,
    )
    if assistant_reps.shape[1] == 0:
        return [], []
    return _classify_spans(
        classifier, assistant_reps, assistant_tokens, tokenizer, window_size,
    )
