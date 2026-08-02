"""Adapter between our DB-backed system and the reference code.

The reference functions (in classifier_engine/reference/) expect:
  * `labels: Dict[str, int]`            — CE name → index
  * `unified_ruleset: dict`             — rule name → {all_required, supporting, any_of, enabled}
  * `dialogue_data: List[{logits, metadata}]`
                                        — already produced by evaluation/inference.run_inference_on_dialogues()
  * `output_dir: str`                   — writes thresholds.json + plots + CSVs

This module:
  1. Loads `labels` from the guardrail's on-disk meta.json
  2. Builds the unified_ruleset from rule_setup (delegates to ruleset_builder.build_unified_ruleset)
  3. Runs LLM+RNN inference on calibration/test conversations (delegates to inference.run_inference_on_dialogues)
  4. Runs reference `calibrate()` / `evaluate()` against a tempdir, then parses the
     artifacts back into our DB JSONB shape

The goal is algorithmic identity with the reference pipeline — we copy their
code rather than reimplement, so any future re-syncing is the only change needed
to stay aligned.
"""
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Importing classifier_engine.reference sets up the gavel.* sys.modules alias.
# Must come BEFORE any `from gavel...` import below.
import classifier_engine.reference  # noqa: F401

from gavel.evaluation.calibration import calibrate as reference_calibrate
from gavel.evaluation.metrics import evaluate as reference_evaluate

from evaluation.ruleset_builder import build_unified_ruleset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. labels (CE name → index)
# ---------------------------------------------------------------------------

def load_classifier_labels(classifier_id: int, user_id: int) -> Dict[str, int]:
    """Read `labels` dict from the trained guardrail's on-disk metadata.

    Falls back to looking under any user_id directory if user_id is not
    known (e.g. background job context without the original request).
    """
    base = Path(__file__).resolve().parent.parent / "trained_classifiers"
    candidates: List[Path] = []
    if user_id is not None:
        candidates.append(base / str(user_id) / f"classifier_{classifier_id}" / "classifier_meta.json")
    # Fallback: scan every user dir until we find this guardrail
    for user_dir in base.glob("*/"):
        candidates.append(user_dir / f"classifier_{classifier_id}" / "classifier_meta.json")

    for p in candidates:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            labels = meta.get("labels")
            if labels:
                return labels
    raise FileNotFoundError(
        f"classifier_meta.json not found for classifier_id={classifier_id} "
        f"(searched under {base})"
    )


# ---------------------------------------------------------------------------
# 2. calibration: reference calibrate() + artifact capture
# ---------------------------------------------------------------------------

def run_calibration(
    classifier_id: int,
    labels: Dict[str, int],
    dialogue_data: List[Dict],
    patience_values: Optional[List[int]] = None,
) -> Dict[str, dict]:
    """Run the reference calibration pipeline.

    Returns the parsed thresholds.json content: a dict mapping topic
    name → {threshold, patience, youden_j, tpr_at_optimal, fpr_at_optimal}.

    The reference function writes plots to disk; we let them go to a
    tempdir so they're cleaned up automatically. If you want to keep the
    plots, use `run_calibration_with_plots()`.
    """
    unified_ruleset = build_unified_ruleset(classifier_id)

    with tempfile.TemporaryDirectory(prefix=f"gavel-calib-{classifier_id}-") as tmpdir:
        reference_calibrate(
            output_dir=tmpdir,
            labels=labels,
            unified_ruleset=unified_ruleset,
            dialogue_data=dialogue_data,
            patience_values=patience_values,
            show_progress=False,
            generate_plots=False,  # skip plot rendering for speed
            logger=logger,
        )

        thresholds_path = Path(tmpdir) / "thresholds.json"
        if not thresholds_path.is_file():
            raise RuntimeError(
                "Reference calibrate() did not produce thresholds.json — "
                "check that dialogue_data has 'usecase_level'/'CE_level' splits in metadata"
            )
        with open(thresholds_path, "r", encoding="utf-8") as f:
            return json.load(f)


def run_calibration_with_plots(
    classifier_id: int,
    labels: Dict[str, int],
    dialogue_data: List[Dict],
    plot_dir: str,
    patience_values: Optional[List[int]] = None,
) -> Dict[str, dict]:
    """Same as run_calibration() but persists per-topic plots under `plot_dir`.

    The caller is responsible for deleting plot_dir when done."""
    unified_ruleset = build_unified_ruleset(classifier_id)
    os.makedirs(plot_dir, exist_ok=True)
    reference_calibrate(
        output_dir=plot_dir,
        labels=labels,
        unified_ruleset=unified_ruleset,
        dialogue_data=dialogue_data,
        patience_values=patience_values,
        show_progress=False,
        generate_plots=True,
        logger=logger,
    )
    with open(Path(plot_dir) / "thresholds.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 3. evaluation: reference evaluate() + artifact capture
# ---------------------------------------------------------------------------

def run_evaluation(
    classifier_id: int,
    labels: Dict[str, int],
    dialogue_data: List[Dict],
    thresholds: Dict[str, dict],
    compute_auc: bool = True,
) -> Dict[str, object]:
    """Run the reference evaluation pipeline.

    Returns a dict containing every metric the reference evaluate() computes,
    plus copies of the produced CSVs (read back from the tempdir and
    re-emitted as in-memory lists so the caller can persist them in
    evaluation_results JSONB).
    """
    unified_ruleset = build_unified_ruleset(classifier_id)

    with tempfile.TemporaryDirectory(prefix=f"gavel-eval-{classifier_id}-") as tmpdir:
        # Reference evaluate() reads thresholds + ruleset from JSON on disk.
        # Write them out to the tempdir.
        thr_path = Path(tmpdir) / "thresholds.json"
        with open(thr_path, "w", encoding="utf-8") as f:
            json.dump(thresholds, f)

        ruleset_path = Path(tmpdir) / "unified_ruleset.json"
        with open(ruleset_path, "w", encoding="utf-8") as f:
            json.dump(unified_ruleset, f)

        result = reference_evaluate(
            output_dir=tmpdir,
            labels=labels,
            thresholds_path=str(thr_path),
            unified_ruleset_path=str(ruleset_path),
            dialogue_data=dialogue_data,
            compute_auc=compute_auc,
            show_progress=False,
            logger=logger,
        )

        # Read CSVs back so the caller can store them in JSONB
        csvs: Dict[str, list] = {}
        for csv_file in Path(tmpdir).glob("*.csv"):
            try:
                import csv as _csv
                with open(csv_file, "r", encoding="utf-8") as f:
                    csvs[csv_file.stem] = list(_csv.DictReader(f))
            except Exception as e:
                logger.warning(f"Could not parse {csv_file.name}: {e}")

    # The result dict from reference_evaluate() may contain pandas objects;
    # serialise them so the caller can write to JSONB.
    serialisable = _serialise_eval_result(result)
    serialisable["csvs"] = csvs
    return serialisable


def _serialise_eval_result(result: dict) -> dict:
    """Convert pandas DataFrames inside the reference eval result to plain
    lists of dicts so the whole thing fits into the JSONB result columns."""
    import pandas as pd

    out = {}
    for key, value in (result or {}).items():
        if isinstance(value, pd.DataFrame):
            out[key] = value.to_dict(orient="records")
        elif isinstance(value, pd.Series):
            out[key] = value.to_dict()
        elif isinstance(value, dict):
            out[key] = _serialise_eval_result(value)
        elif isinstance(value, list):
            out[key] = [
                _serialise_eval_result(v) if isinstance(v, dict) else v
                for v in value
            ]
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# 4. utilities exposed for tests
# ---------------------------------------------------------------------------

def reference_compute_triggers(logits, thresholds, patience: int = 1):
    """Pass-through to the reference compute_triggers() so callers (tests,
    debugging tools) don't need to know about the sys.modules alias."""
    from gavel.evaluation.metrics import compute_triggers
    return compute_triggers(logits, thresholds, patience)
