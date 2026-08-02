# backend/classifier_engine/cancellation.py
# Lightweight (torch-free) cancellation signals shared across the ML pipeline.
#
# Kept dependency-free ON PURPOSE so route modules (e.g. routes/evaluation.py)
# can `except InferenceCancelled` without importing torch at app startup — the
# heavy inference_core / inference modules import it from here.


class InferenceCancelled(BaseException):
    """Raised inside the inference loop when the guardrail is deleted mid-run, so
    a local calibration/evaluation background task stops promptly instead of
    burning GPU on a guardrail that no longer exists.

    Subclasses BaseException (not Exception) ON PURPOSE — mirrors
    trainer.TrainingCancelled: best-effort hooks wrapped in `except Exception:
    pass` won't swallow the cancel, and the calibration/evaluation workers'
    generic `except Exception` won't mistake it for a real failure (which would
    try to INSERT an error row for the now-deleted guardrail)."""
