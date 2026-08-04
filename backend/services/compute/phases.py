"""The training-phase vocabulary, shared by everything that produces or renders it.

The stage keys are emitted by BOTH training implementations — the in-process
trainer (`classifier_engine/trainer.py` `_progress()` calls) and the standalone
`compute_jobs/train_job.py` child — so they must agree. Keeping the map here
(instead of in the route that renders it) is what stops the two from drifting.

Pure data + one function: no torch, no DB, no FastAPI. Safe to import anywhere,
including from boot-time crash recovery.
"""

TRAINING_PHASE_LABELS = {
    "init":      "Preparing",
    "data":      "Loading datasets",
    "load_llm":  "Loading language model",
    "split":     "Splitting train/validation",
    "extract":   "Extracting embeddings",
    "train_rnn": "Training RNN",
    "save":      "Saving model",
    "done":      "complete",
}


def phase_label(stage: str) -> str:
    """User-facing label for a raw stage key. Unknown keys degrade to a
    Title-cased version rather than showing the user a bare identifier."""
    stage = (stage or "").strip()
    if not stage:
        return ""
    return TRAINING_PHASE_LABELS.get(stage, stage.replace("_", " ").title())
