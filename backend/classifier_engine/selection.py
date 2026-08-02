# classifier_engine/selection.py
#
# Candidate-model selection on calibration data. Several RNN fits on the same
# extracted features are indistinguishable by validation metrics (val comes
# from the same synthetic excitation distribution as training and saturates at
# ~1.0), but they differ a lot in how well each CE head transfers to REAL
# dialogues — which is what calibration and evaluation measure. This module
# scores every candidate on the calibration dialogues and the trainer keeps
# the one whose WEAKEST CE transfers best (rules are AND-chains, so one bad
# head sinks a whole use case).
#
# Hard rule (same as inference_core): NO database / FastAPI / network imports.
# This must import cleanly on remote workers.
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from classifier_engine.dialogue_dataset import DialogueDataset
from classifier_engine.inference_core import (
    extract_attention_readouts,
    build_assistant_windows,
    rnn_logits_over_windows,
    _normalize_conversation,
)

logger = logging.getLogger(__name__)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U with average ranks for ties).
    Dependency-free so the remote job needs nothing beyond numpy."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = y_score[order]
    i, next_rank = 0, 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (2 * next_rank + (j - i)) / 2.0
        next_rank += j - i + 1
        i = j + 1
    sum_pos = ranks[y_true.astype(bool)].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def score_candidates_on_calibration(
    candidates: List[torch.nn.Module],
    llm,
    tokenizer,
    entries: List[dict],          # [{"conversation": <conv>, "ce": "<sanitized name>"}]
    labels: Dict[str, int],       # sanitized CE name -> output index
    *,
    window_size: int,
    selected_layers,
    device: torch.device,
    max_length: Optional[int] = 1024,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
    """Score every candidate RNN on the calibration dialogues in ONE LLM pass.

    Per dialogue the attention readouts are extracted once; each candidate then
    consumes them with a cheap GRU forward. A dialogue's score for a CE is its
    max window probability — matching trigger semantics (patience=1: a CE
    fires if ANY window crosses its threshold). Truncation (`max_length`) is
    applied identically to every candidate, so the ranking is fair.

    Returns one dict per candidate:
        {"per_ce_auc": {ce: auc}, "min_auc": float, "mean_auc": float}
    AUC is one-vs-rest over dialogues: positives are the dialogues tagged with
    that CE, negatives are every other calibration dialogue.
    """
    for m in candidates:
        m.eval()

    num_classes = len(labels)
    dialogue_scores: List[List[np.ndarray]] = [[] for _ in candidates]
    dialogue_ce_idx: List[int] = []
    total = len(entries)

    for i, entry in enumerate(entries):
        ce_name = entry.get("ce")
        if ce_name not in labels:
            continue
        if progress_callback and i % 10 == 0:
            try:
                progress_callback(i + 1, total)
            except Exception:
                pass

        conv = _normalize_conversation(entry.get("conversation", ""))

        def _windows_for(eff_max_length):
            dataset = DialogueDataset([conv], tokenizer, max_length=eff_max_length)
            item = dataset[0]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=device)
            asst_mask_1d = item["assistant_mask"][: input_ids.shape[1]].bool()
            if int(asst_mask_1d.sum().item()) == 0:
                return None, None
            readouts = extract_attention_readouts(llm, input_ids, attention_mask, selected_layers)
            S_len = readouts.shape[1]
            token_vecs = readouts[0, :S_len].reshape(S_len, -1).float()
            windows = build_assistant_windows(asst_mask_1d, window_size, window_size)
            return (token_vecs, windows) if windows else (None, None)

        token_vecs = windows = None
        attempts = [max_length] + [c for c in (512,) if c < (max_length or 1 << 30)]
        for eff in attempts:
            try:
                token_vecs, windows = _windows_for(eff)
                break
            except RuntimeError as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if "out of memory" in str(e).lower() and eff is not attempts[-1]:
                    continue
                logger.warning(f"[selection] dialogue {i} failed: {e}")
                break
            except Exception as e:
                logger.warning(f"[selection] dialogue {i} failed: {e}")
                break
        if token_vecs is None or not windows:
            continue

        per_candidate = []
        for rnn in candidates:
            logits = rnn_logits_over_windows(rnn, token_vecs, windows, device=device)
            if logits.size == 0:
                break
            probs = 1.0 / (1.0 + np.exp(-logits))           # sigmoid
            per_candidate.append(probs.max(axis=0))          # max window prob per CE
        if len(per_candidate) != len(candidates):
            continue  # commit all-or-nothing so scores stay aligned with ground truth
        for c_idx, vec in enumerate(per_candidate):
            dialogue_scores[c_idx].append(vec)
        dialogue_ce_idx.append(labels[ce_name])

    y = np.array(dialogue_ce_idx)
    results = []
    for c_idx in range(len(candidates)):
        scores = np.stack(dialogue_scores[c_idx]) if dialogue_scores[c_idx] else np.zeros((0, num_classes))
        per_ce = {}
        for ce_name, idx in labels.items():
            if len(y) == 0:
                continue
            auc = _roc_auc((y == idx).astype(np.int8), scores[:, idx])
            if not np.isnan(auc):
                per_ce[ce_name] = round(auc, 4)
        if per_ce:
            vals = list(per_ce.values())
            results.append({
                "per_ce_auc": per_ce,
                "min_auc": min(vals),
                "mean_auc": round(sum(vals) / len(vals), 4),
            })
        else:
            results.append({"per_ce_auc": {}, "min_auc": float("nan"), "mean_auc": float("nan")})
    return results


def pick_best_candidate(scores: List[dict]) -> int:
    """Index of the candidate whose weakest CE transfers best (max min-AUC,
    mean-AUC as tie-break). Falls back to 0 if scoring produced nothing."""
    best_idx, best_key = 0, (-1.0, -1.0)
    for i, s in enumerate(scores):
        mn, mean = s.get("min_auc"), s.get("mean_auc")
        if mn is None or (isinstance(mn, float) and np.isnan(mn)):
            continue
        key = (mn, mean if mean is not None else -1.0)
        if key > best_key:
            best_key, best_idx = key, i
    return best_idx
