# evaluation/model_cache.py
# LRU cache for loaded LLM + guardrail pairs.
# Replaces Streamlit's @st.cache_resource pattern.
import logging
import threading
from collections import OrderedDict
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: OrderedDict = OrderedDict()  # key -> (llm, tokenizer, rnn, meta)
_MAX_ENTRIES = 2  # Keep at most 2 loaded models (LLMs are large)


def _cache_key(classifier_id: int) -> str:
    return f"classifier_{classifier_id}"


_mtime: dict = {}  # key -> meta-file mtime when cached


def get_cached_models(classifier_id: int) -> Optional[Tuple]:
    """Return (llm, tokenizer, rnn_model, meta) if cached, else None.

    Invalidates the cache if classifier_meta.json has been modified since
    the entry was cached — guards against stale schemas after retraining
    or manual patches."""
    key = _cache_key(classifier_id)
    with _lock:
        if key not in _cache:
            return None
        try:
            import os
            from classifier_engine.trainer import classifier_workdir
            meta_path = os.path.join(classifier_workdir(classifier_id), "classifier_meta.json")
            current_mtime = os.path.getmtime(meta_path)
            if _mtime.get(key) != current_mtime:
                logger.info(f"Meta file changed for {key}, invalidating cache")
                evicted = _cache.pop(key)
                _mtime.pop(key, None)
                _free_model(evicted)
                return None
        except OSError:
            evicted = _cache.pop(key)
            _mtime.pop(key, None)
            _free_model(evicted)
            return None
        _cache.move_to_end(key)
        return _cache[key]


def cache_models(classifier_id: int, llm, tokenizer, rnn_model, meta: dict):
    """Store models in cache, evicting LRU if over limit."""
    key = _cache_key(classifier_id)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            _cache[key] = (llm, tokenizer, rnn_model, meta)
        else:
            # Evict LRU if at capacity
            while len(_cache) >= _MAX_ENTRIES:
                evicted_key, evicted = _cache.popitem(last=False)
                _mtime.pop(evicted_key, None)
                logger.info(f"Evicting cached model: {evicted_key}")
                _free_model(evicted)
            _cache[key] = (llm, tokenizer, rnn_model, meta)
            logger.info(f"Cached models for {key} ({len(_cache)}/{_MAX_ENTRIES})")

        try:
            import os
            from classifier_engine.trainer import classifier_workdir
            meta_path = os.path.join(classifier_workdir(classifier_id), "classifier_meta.json")
            _mtime[key] = os.path.getmtime(meta_path)
        except OSError:
            _mtime.pop(key, None)


def evict(classifier_id: int):
    """Remove a specific guardrail from cache."""
    key = _cache_key(classifier_id)
    with _lock:
        if key in _cache:
            evicted = _cache.pop(key)
            _mtime.pop(key, None)
            _free_model(evicted)
            logger.info(f"Evicted {key} from cache")


def clear_cache():
    """Remove all cached models."""
    with _lock:
        for key in list(_cache.keys()):
            _free_model(_cache.pop(key))
        _mtime.clear()
        logger.info("Model cache cleared")


def _free_model(entry):
    """Free GPU memory for evicted models."""
    try:
        del entry
        from utils.device import empty_device_cache
        empty_device_cache()
    except Exception:
        pass


# Per-guardrail locks that serialize the EXPENSIVE load (download + read the
# ~15 GB LLM into memory). Distinct from `_lock` (which only guards the cache
# dict) so cache reads / model-status checks for OTHER guardrails never block
# during a multi-minute load.
_load_locks: dict = {}
_load_locks_guard = threading.Lock()


def _get_load_lock(classifier_id: int) -> threading.Lock:
    with _load_locks_guard:
        lk = _load_locks.get(classifier_id)
        if lk is None:
            lk = threading.Lock()
            _load_locks[classifier_id] = lk
        return lk


def load_or_get(classifier_id: int, device: torch.device = None):
    """Load guardrail models from cache or disk.

    Returns:
        Tuple of (llm, tokenizer, rnn_model, meta).
    """
    cached = get_cached_models(classifier_id)
    if cached:
        logger.info(f"Using cached models for classifier {classifier_id}")
        return cached

    # Serialize the load. Without this, two requests that both miss the cache
    # (e.g. click a sample, navigate away, click another before the first 15 GB
    # download finished) start TWO concurrent downloads of the same model and
    # collide on HuggingFace's per-file locks ("Fetching 3 files: 0/3" hangs) —
    # or load two copies into RAM. With it, the first request loads; the rest
    # wait on this lock, then hit the now-warm cache below.
    with _get_load_lock(classifier_id):
        # Double-checked: a concurrent request may have finished loading while we
        # were waiting for the lock.
        cached = get_cached_models(classifier_id)
        if cached:
            logger.info(f"Using cached models for classifier {classifier_id} (loaded by a concurrent request)")
            return cached

        if device is None:
            from utils.device import get_torch_device
            device = get_torch_device()

        from evaluation.inference import load_trained_classifier
        from classifier_engine.utils_train import load_model_and_tokenizer

        rnn_model, meta = load_trained_classifier(classifier_id, device)
        llm, tokenizer = load_model_and_tokenizer(meta["model_path"])

        # The LLM may have fallen back from MPS to CPU (too large for this Mac's
        # GPU). The realtime guardrail runs on the LLM's per-token readouts, so
        # the RNN must live on the SAME device as the LLM — otherwise its forward
        # gets a CPU tensor while its weights are on MPS (device mismatch).
        try:
            llm_device = next(llm.parameters()).device
            if next(rnn_model.parameters()).device != llm_device:
                rnn_model = rnn_model.to(llm_device)
        except Exception:
            pass

        cache_models(classifier_id, llm, tokenizer, rnn_model, meta)
        return llm, tokenizer, rnn_model, meta
