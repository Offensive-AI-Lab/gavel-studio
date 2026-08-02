# evaluation/inference.py
# Runs inference on dialogues using a trained guardrail.
# Loads the target LLM + trained RNN, extracts per-window logits.
# Uses DialogueDataset for proper multi-turn chat template handling.
from __future__ import annotations
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from classifier_engine.RNN import TopicRNN
# The pure windowing inference lives in classifier_engine.inference_core so the
# LOCAL path here and the REMOTE path (compute_jobs/infer_job.py) share ONE
# implementation — identical per-window logits regardless of where it ran.
# Re-exported here for backward compatibility with existing imports.
from classifier_engine.inference_core import (  # noqa: F401
    _extract_value_cache,
    extract_attention_readouts,
    rnn_logits_over_windows,
    _normalize_conversation,
    run_inference_core,
)
# NOTE: classifier_engine.trainer (which pulls in DB code) is imported LAZILY
# inside load_trained_classifier, so this module stays importable on remote workers.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_trained_classifier(classifier_id: int, device: torch.device = None):
    """Load a trained RNN guardrail and its metadata.

    Returns:
        Tuple of (rnn_model, metadata_dict) or raises ValueError.
    """
    if device is None:
        from utils.device import get_torch_device
        device = get_torch_device()

    # Lazy import: classifier_engine.trainer pulls in DB code, which a remote worker
    # has no config for. Keeping it here lets this module import remotely.
    from classifier_engine.trainer import classifier_workdir
    work_dir = classifier_workdir(classifier_id)
    meta_path = os.path.join(work_dir, "classifier_meta.json")
    rnn_path = os.path.join(work_dir, "trained_rnn.pth")

    if not os.path.exists(meta_path):
        raise ValueError(f"Guardrail {classifier_id} has no trained model (no metadata)")
    if not os.path.exists(rnn_path):
        raise ValueError(f"Guardrail {classifier_id} has no trained model (no .pth)")

    with open(meta_path) as f:
        meta = json.load(f)

    rnn = TopicRNN(
        num_layers=meta["n_layers"],
        input_dim=meta["readout_dim"],
        hidden_dim=meta["hidden_dim"],
        num_rnn_layers=meta["num_rnn_layers"],
        num_topics=meta["num_classes"],
        rnn_type="GRU",
    ).to(device)
    rnn.load_state_dict(torch.load(rnn_path, map_location=device))
    rnn.eval()

    return rnn, meta


# extract_attention_readouts / rnn_logits_over_windows / _normalize_conversation
# now live in classifier_engine.inference_core (imported above), shared with the
# remote inference job so per-window logits are identical local vs remote.

def run_inference_on_dialogues(
    classifier_id: int,
    dialogues: List[dict],
    batch_size: int = 2,
    max_length: Optional[int] = None,   # None = keep the WHOLE dialogue (reference-parity); the core's per-dialogue OOM fallback truncates only what won't fit the GPU.
    window_stride: int = 0,   # 0 => non-overlapping (stride = window_size); matches training + reference
    progress_callback=None,
) -> List[dict]:
    """Run full inference pipeline on a list of dialogues.

    Args:
        classifier_id: ID of the trained guardrail.
        dialogues: List of dicts, each containing:
            - conversation: list of {role, content} messages OR raw text
            - metadata: dict with split, usecase_path, dialogue_id (for evaluation)
        batch_size: LLM batch size (keep small for memory).
        max_length: Max tokenization length.
        window_stride: Stride for sliding window.
        progress_callback: Optional callable(stage, detail).

    Returns:
        List of dicts with:
            - logits: numpy array (num_windows, num_topics)
            - metadata: original metadata dict
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from utils.device import get_torch_device, empty_device_cache, get_llm_device_map
    device = get_torch_device()

    def _progress(stage, detail=""):
        if progress_callback:
            progress_callback(stage, detail)
        logger.info(f"[Inference] [{stage}] {detail}")

    # 1. Load trained guardrail
    _progress("load", "Loading trained guardrail")
    rnn_model, meta = load_trained_classifier(classifier_id, device)

    # 2. Load LLM
    model_path = meta["model_path"]
    _progress("load", f"Loading LLM from: {model_path}")
    from classifier_engine.utils_train import load_model_and_tokenizer
    llm, tokenizer = load_model_and_tokenizer(model_path, device_map=get_llm_device_map())

    # Cooperative cancellation: abort the (GPU-heavy) loop if the guardrail was
    # deleted mid-run. Mirrors training's TrainingCancelled — the loop checks
    # every 10 dialogues and raises InferenceCancelled (a BaseException) so the
    # calibration/eval worker stops instead of finishing on a deleted guardrail.
    from classifier_engine.inference_core import InferenceCancelled
    from classifier_engine.trainer import _classifier_deleted

    def _cancel_check():
        if _classifier_deleted(classifier_id):
            raise InferenceCancelled(classifier_id)

    # 3. Process dialogues via the SHARED core (identical to the remote path).
    try:
        results = run_inference_core(
            rnn_model, meta, llm, tokenizer, dialogues, device,
            max_length=max_length, window_stride=window_stride,
            progress_callback=progress_callback,
            cancel_check=_cancel_check,
        )
    finally:
        # Free LLM memory even if inference raised.
        del llm
        empty_device_cache()

    return results
