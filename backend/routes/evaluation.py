# backend/routes/evaluation.py
# Endpoints for guardrail evaluation: calibration, evaluation, and results retrieval.
import json
import logging
import math
import os
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from utils.auth import get_current_user
from utils.ownership import require_classifier_owner
from utils.sqlite_db import execute_query, execute_query_dict
from classifier_engine.cancellation import InferenceCancelled  # torch-free

logger = logging.getLogger(__name__)
# Every endpoint here is /{classifier_id}/… — guard the whole router so the
# caller must own that guardrail (auth + ownership) before any handler runs.
router = APIRouter(dependencies=[Depends(require_classifier_owner)])


def _jsonb(obj) -> str:
    """Serialize for a Postgres jsonb column, NaN/Infinity-safe.

    json.dumps emits the bare literals `NaN` / `Infinity` / `-Infinity` for
    non-finite floats, which are valid Python output but INVALID JSON — Postgres
    rejects them ("invalid input syntax for type json: Token NaN is invalid").
    Evaluation metrics legitimately produce NaN (e.g. ROC_AUC for a use-case
    that only has one class), so replace every non-finite float with null first.
    """
    def clean(o):
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    return json.dumps(clean(obj))


# Snapshot-aware filter clause appended to every read of evaluation_results.
#
# When a guardrail is retrained, classifiers.trained_at is bumped to now()
# and the model on disk is replaced — but the evaluation_results rows from
# the previous training (calibration thresholds, evaluation metrics, etc.)
# stay in the table. Reading those after a retrain would silently apply
# stale thresholds / show stale metrics that no longer correspond to the
# current model.
#
# This clause filters every result lookup to only rows created AFTER the
# guardrail was last trained, hiding pre-retrain rows without deleting
# them — old runs are still available via DB archeology, just not surfaced.
#
# COALESCE: a never-trained guardrail has trained_at = NULL, in which case
# we want the filter to be a no-op (year-1 sentinel: every real timestamp is
# after it, so `created_at >= '0001-01-01'` is always true — the SQLite
# stand-in for Postgres' '-infinity').
# An untrained guardrail shouldn't have any evaluation_results rows anyway,
# but being defensive here avoids hiding rows that might exist (e.g. left
# over from a crash mid-train).
_POST_TRAIN_CLAUSE = (
    "AND created_at >= COALESCE("
    "(SELECT trained_at FROM classifiers WHERE classifier_id = %s),"
    " '0001-01-01 00:00:00')"
)


# --- Live progress: a short, human-readable phase on the *_running row --------
#
# The '*_running' marker row's `metrics` column is otherwise unused, and
# get_evaluation_results() already returns it, so we use it to publish a live
# phase string ("Fetching data…", "Running on the remote GPU…", etc.). The
# frontend polls results and shows this line — the calibrate/evaluate analogue
# of training's phase indicator. Best-effort: a failed UPDATE never breaks the
# run.

def _set_phase(running_eval_id: Optional[int], phase: str):
    if running_eval_id is None:
        return
    try:
        execute_query(
            "UPDATE evaluation_results SET metrics = %s::jsonb WHERE eval_id = %s",
            (json.dumps({"phase": phase}), running_eval_id),
        )
    except Exception:
        pass


# --- Inference dispatch: remote worker -> local GPU/MPS/CPU --------------------

def _run_inference(classifier_id: int, inference_input: list, running_eval_id: int = None,
                   set_phase=None):
    """Run the GPU-heavy inference for calibration/evaluation.

    Priority mirrors training: 1) remote GPU worker, 2) local GPU/MPS, 3) local
    CPU. When a worker is configured we offload the inference there (the remote
    job uses the SAME shared core, so logits are identical), block-polling until
    the logits come back; on ANY remote failure we fall back to local inference
    (run_inference_on_dialogues itself picks CUDA/MPS/CPU). The light
    metric/threshold math always runs locally on the returned logits.

    Crash-safety: when a remote job is submitted, its pointer is stashed in the
    caller's '*_running' row (`running_eval_id`) so boot recovery can
    cancel/clean the orphaned job — and because the final result row is only
    written on full success, a crash leaves the run as if it never happened.
    """
    from services import compute

    def _phase(text):
        if set_phase:
            set_phase(text)

    # Resolve the trained artifacts up front: a remote provider ships these to
    # the GPU; the local provider reloads them itself from the workdir.
    from classifier_engine.trainer import classifier_workdir
    work_dir = classifier_workdir(classifier_id)
    meta = {}
    try:
        with open(os.path.join(work_dir, "classifier_meta.json")) as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    rnn_path = os.path.join(work_dir, "trained_rnn.pth")

    def _stash(info):
        # A remote provider calls this once its GPU job is submitted, so boot
        # recovery can cancel/clean an orphaned job if the backend dies mid-run.
        if running_eval_id is None:
            return
        try:
            execute_query(
                "UPDATE evaluation_results SET plots = %s::jsonb WHERE eval_id = %s",
                (json.dumps({"remote_job": dict(info or {})}), running_eval_id),
            )
        except Exception:
            pass

    spec = compute.InferenceSpec(
        classifier_id=classifier_id,
        model_hf_path=meta.get("model_path"),
        classifier_meta=meta,
        dialogues=inference_input,
        rnn_path=rnn_path,
    )

    # Failover ladder: remote_worker -> local GPU -> local CPU. We start from
    # the availability-aware pick (get_provider probes/caches reachability, so
    # a configured-but-down worker doesn't cost a connect timeout) and descend
    # the rest of the chain only on failure. Each tier that fails for a
    # recoverable reason hands off to the next; the CPU tier is the always-works
    # floor.
    from utils.device import force_cpu
    full_chain = compute.failover_providers(compute.Workload.INFERENCE, include_cpu_tier=True)
    top = compute.get_provider(compute.Workload.INFERENCE)
    chain = full_chain[full_chain.index(top.name):] if top.name in full_chain else full_chain

    last_err = None
    for idx, pname in enumerate(chain):
        prov = compute.provider_by_name(pname)
        if prov is None:
            continue
        try:
            if idx > 0:
                where_n = "local CPU" if pname == "local_cpu" else pname
                _phase(f"Compute tier failed — retrying inference on {where_n} "
                       f"(this can take longer)…")
            # on_submit (crash-recovery stash) only matters for the off-box tier.
            on_sub = _stash if pname == "remote_worker" else None
            if pname == "local_cpu":
                with force_cpu():
                    return prov.run_inference(spec, on_phase=_phase)
            return prov.run_inference(spec, on_phase=_phase, on_submit=on_sub)
        except InferenceCancelled:
            raise  # guardrail deleted mid-run — abort, never fail over
        except compute.ComputeError as e:
            last_err = e
            if not e.retryable_local:
                raise  # deterministic failure the lower tiers would hit too
            logger.warning(f"[evaluation] {pname} inference failed for classifier "
                           f"{classifier_id}; trying next tier: {e.message}")
            continue
        except Exception as e:
            # Non-ComputeError (e.g. a CUDA OOM/driver error on the local GPU tier).
            # Only worth retrying when a CPU tier remains below us.
            last_err = e
            if pname == "local" and "local_cpu" in chain[idx + 1:]:
                logger.warning(f"[evaluation] local GPU inference failed for classifier "
                               f"{classifier_id}; retrying on CPU: {e}")
                continue
            raise
    raise last_err or RuntimeError("All compute tiers failed for inference.")


# --- Request Schemas ---

class CalibrateRequest(BaseModel):
    """Request body for starting calibration."""
    test_dataset_id: Optional[int] = None  # Use stored test dataset (single)
    test_dataset_ids: Optional[List[int]] = None  # Multiple datasets combined
    dialogue_data: Optional[List[dict]] = None  # Or pass inline dialogue data
    patience_values: Optional[List[int]] = None
    # Recalibrate an already-calibrated rule set without retraining it. Needed
    # because calibration now runs automatically after training: the user's only
    # way to redo it when the calibration DATA changes (new CE calibration sets,
    # regenerated dialogues) would otherwise be a pointless full retrain.
    force: bool = False


class EvaluateRequest(BaseModel):
    """Request body for starting evaluation."""
    test_dataset_id: Optional[int] = None
    test_dataset_ids: Optional[List[int]] = None  # Multiple datasets combined
    dialogue_data: Optional[List[dict]] = None
    # Whether to also include the bundled neutral corpus as a third split
    # (reference-parity FPR estimation against universal everyday content).
    # Default ON: every evaluation gets it for free.
    include_neutral: bool = True


# --- Background task runners ---

def _run_calibration(classifier_id: int, patience_values: list = None):
    """Background task: runs calibration using the reference
    algorithm (threshold × patience sweep, optimizing per-topic Youden-J).

    Loads CE-level calibration sets AND rule-level (usecase-level)
    calibration sets, feeds both into the reference calibrate(), and
    writes the resulting thresholds + plots to the DB.

    The "reference" pipeline is the reference implementation
    copied verbatim into backend/classifier_engine/reference/. The adapter
    at evaluation/adapter.py orchestrates the call.
    """
    import os
    from evaluation import adapter as eval_adapter
    from classifier_engine.trainer import classifier_workdir

    # Mark calibration as running so the frontend can detect it after a page
    # refresh. Capture the row id so a remote job's pointer can be stashed on it
    # (for crash recovery), and so this exact marker can be cleaned up.
    _running = execute_query_dict(
        """INSERT INTO evaluation_results (classifier_id, eval_type, thresholds, metrics, plots)
           VALUES (%s, 'calibration_running', NULL, NULL, NULL) RETURNING eval_id""",
        (classifier_id,),
    )
    running_eval_id = _running[0]["eval_id"] if _running else None

    def set_phase(p):
        _set_phase(running_eval_id, p)

    # Lazy-pull any CE calibration sets that exist in the registry but aren't
    # yet cached locally. Same for rule-level calibration. The seed library
    # ships both, so for a fresh install + guardrail built from public
    # rules, this lets the user click "Run Calibration" without first
    # generating data manually.
    set_phase("Fetching calibration data…")
    try:
        from services.library_sync import ensure_ce_calibrations_for_classifier, ensure_rule_aux_for_classifier
        ensure_ce_calibrations_for_classifier(classifier_id)
        try:
            ensure_rule_aux_for_classifier(classifier_id)
        except Exception as rule_err:
            logger.warning(f"[evaluation] rule-aux lazy fetch failed: {rule_err}")
    except Exception as fetch_err:
        logger.warning(f"[evaluation] CE-calibration lazy fetch failed: {fetch_err}")

    try:
        # Resolve the guardrail's user_id (needed to locate
        # trained_classifiers/<user_id>/classifier_<id>/classifier_meta.json).
        owner_rows = execute_query_dict(
            """SELECT m.user_id FROM classifiers c
               JOIN target_models m ON c.model_id = m.model_id
               WHERE c.classifier_id = %s""",
            (classifier_id,),
        ) or []
        user_id = owner_rows[0]["user_id"] if owner_rows else None

        labels = eval_adapter.load_classifier_labels(classifier_id, user_id)
        if not labels:
            raise ValueError("Guardrail has no trained model or labels")

        set_phase("Loading calibration datasets…")
        # CE-level calibration data: one conversation per CE per dialogue
        ce_rows = execute_query_dict("""
            SELECT ce.ce_id, ce.name, cd.dataset
            FROM cognitive_elements ce
            JOIN calibration_datasets cd ON ce.ce_id = cd.ce_id
            WHERE ce.name = ANY(%s)
        """, (list(labels.keys()),)) or []

        inference_input: list = []
        for row in ce_rows:
            ce_name = row["name"]
            dataset = row["dataset"]
            if isinstance(dataset, str):
                dataset = json.loads(dataset)
            conversations = (
                dataset.get("conversations")
                or dataset.get("samples")
                or []
            )
            for i, conv in enumerate(conversations):
                inference_input.append({
                    "conversation": conv,
                    "metadata": {
                        "split": "CE_level",
                        "usecase_path": ce_name,
                        "dialogue_id": f"calib_ce_{ce_name}_{i}",
                    },
                })

        # Rule-level (usecase-level) calibration data: end-to-end conversations
        # that exercise the rule's full predicate. Each rule has exactly ONE
        # calibration set — the auto-generated default (is_default=TRUE) — so we
        # select it directly (custom user-defined sets were removed).
        active_rules = execute_query_dict("""
            SELECT DISTINCT rs.rule_id, COALESCE(rs.custom_name, r.name) AS rule_name
            FROM rule_setup rs
            LEFT JOIN rules r ON rs.rule_id = r.rule_id
            WHERE rs.classifier_id = %s AND rs.is_active = TRUE
        """, (classifier_id,)) or []
        active_rule_ids = [r["rule_id"] for r in active_rules if r.get("rule_id")]
        rule_id_to_name = {r["rule_id"]: r["rule_name"] for r in active_rules if r.get("rule_id")}

        rule_scoped_rows = []
        if active_rule_ids:
            rule_scoped_rows = execute_query_dict("""
                SELECT dataset_id, rule_id, conversations FROM (
                    SELECT dataset_id, rule_id, conversations,
                           ROW_NUMBER() OVER (
                               PARTITION BY rule_id ORDER BY created_at DESC
                           ) AS rn
                    FROM test_datasets
                    WHERE rule_id = ANY(%s)
                      AND dataset_type = 'positive_calibration'
                      AND is_default = TRUE
                      AND status = 'ready'
                ) ranked WHERE rn = 1
            """, (active_rule_ids,)) or []

        # Each calibration row is tagged with its own rule's name.
        for ds_row in rule_scoped_rows:
            rname = rule_id_to_name.get(ds_row["rule_id"])
            if not rname:
                continue
            convs = ds_row.get("conversations") or []
            if isinstance(convs, str):
                convs = json.loads(convs)
            for i, conv in enumerate(convs):
                inference_input.append({
                    "conversation": conv,
                    "metadata": {
                        "split": "usecase_level",
                        "usecase_path": rname,
                        "dialogue_id": f"calib_uc_{rname}_{ds_row['dataset_id']}_{i}",
                    },
                })

        if not inference_input:
            raise ValueError(
                "No calibration datasets found. Generate CE calibration "
                "(per-CE) AND rule calibration (per rule) before calibrating."
            )

        logger.info(
            f"Running inference on {len(inference_input)} calibration conversations "
            f"({len(ce_rows)} CEs, {len(rule_scoped_rows)} rule sets)..."
        )
        processed_data = _run_inference(classifier_id, inference_input,
                                        running_eval_id=running_eval_id, set_phase=set_phase)
        if not processed_data:
            raise ValueError("Inference produced no results — check conversations and model")

        set_phase("Computing per-CE thresholds (Youden-J)…")
        # Persist per-topic plots alongside the trained guardrail so the
        # user can download them later if they want. Plots are NOT
        # written into JSONB to avoid bloating evaluation_results.
        plot_dir = os.path.join(classifier_workdir(classifier_id), "calibration")
        optimal_thresholds = eval_adapter.run_calibration_with_plots(
            classifier_id=classifier_id,
            labels=labels,
            dialogue_data=processed_data,
            plot_dir=plot_dir,
            patience_values=patience_values or [1],
        )

        set_phase("Saving calibration thresholds…")
        # Save to DB. `plots` records the on-disk plot directory; the
        # adapter writes the actual PNGs there. JSONB stays small.
        execute_query(
            """INSERT INTO evaluation_results (classifier_id, eval_type, thresholds, metrics, plots)
               VALUES (%s, 'calibration', %s::jsonb, NULL, %s::jsonb)""",
            (
                classifier_id,
                _jsonb(optimal_thresholds),
                json.dumps({"plot_dir": plot_dir}),
            ),
        )

        logger.info(f"Calibration complete for classifier {classifier_id}")

    except InferenceCancelled:
        # Guardrail was deleted mid-calibration. The cascade delete already
        # removed the '*_running' row; do NOT write an error row (it would FK-fail
        # on the gone guardrail anyway). Just stop quietly.
        logger.info(f"Calibration cancelled — classifier {classifier_id} was deleted mid-run")
        return
    except Exception as e:
        logger.exception(f"Calibration failed for classifier {classifier_id}: {e}")
        execute_query(
            """INSERT INTO evaluation_results (classifier_id, eval_type, thresholds, metrics, plots)
               VALUES (%s, 'calibration_error', NULL, %s::jsonb, NULL)""",
            (classifier_id, json.dumps({"error": str(e)})),
        )


def _run_evaluation(classifier_id: int, dataset_pairs: list, include_neutral: bool = True):
    """Background task: runs evaluation against the reference
    algorithm (Youden-J-calibrated thresholds applied, per-usecase
    TPR/FPR/Accuracy/F1 computed, AUC reported).

    Args:
        classifier_id: Guardrail ID.
        dataset_pairs: List of (conversations, dataset_type) tuples, or
            (conversations, dataset_type, rule_name) triples. When the third
            element is present, that batch is attributed to that rule's
            use-case (so multi-rule guardrails get a true per-rule breakdown).
    """
    from evaluation import adapter as eval_adapter
    from evaluation.ruleset_builder import build_unified_ruleset

    _running = execute_query_dict(
        """INSERT INTO evaluation_results (classifier_id, eval_type, thresholds, metrics, plots)
           VALUES (%s, 'evaluation_running', NULL, NULL, NULL) RETURNING eval_id""",
        (classifier_id,),
    )
    running_eval_id = _running[0]["eval_id"] if _running else None

    def set_phase(p):
        _set_phase(running_eval_id, p)

    set_phase("Fetching evaluation data…")
    try:
        from services.library_sync import ensure_rule_aux_for_classifier
        ensure_rule_aux_for_classifier(classifier_id)
    except Exception as fetch_err:
        logger.warning(f"[evaluation] rule aux lazy fetch failed: {fetch_err}")

    # Neutral corpus: the gavel-rules registry no longer ships one (removed
    # upstream — arbitrary role assignment made it questionable for
    # false-positive measurement), so there is nothing to fetch. Whatever rows
    # a previously-synced DB still holds keep working; an empty table now
    # SKIPS the neutral split with a warning instead of hard-failing.

    try:
        owner_rows = execute_query_dict(
            """SELECT m.user_id FROM classifiers c
               JOIN target_models m ON c.model_id = m.model_id
               WHERE c.classifier_id = %s""",
            (classifier_id,),
        ) or []
        user_id = owner_rows[0]["user_id"] if owner_rows else None

        labels = eval_adapter.load_classifier_labels(classifier_id, user_id)
        if not labels:
            raise ValueError("Guardrail has no trained model or labels")

        set_phase("Loading evaluation datasets…")
        # Ruleset still needed locally for the inference batcher.
        ruleset = build_unified_ruleset(classifier_id)
        if not ruleset:
            raise ValueError("No rules configured for this guardrail")

        # Append the neutral split when local data exists. The registry no
        # longer provides this corpus, so absence is expected on fresh
        # installs — evaluate WITHOUT the split (the benign-chat FPR rows
        # simply don't appear) rather than refusing to run.
        if include_neutral:
            from evaluation.neutral_corpus import load_neutral_corpus_by_category, CATEGORIES
            grouped = load_neutral_corpus_by_category()
            if sum(len(grouped.get(c, [])) for c in CATEGORIES) == 0:
                logger.warning(
                    "[evaluation] no neutral corpus in the local DB — running "
                    "without the neutral (benign-chat false-positive) split"
                )
            for cat in CATEGORIES:
                convs = grouped.get(cat) or []
                if convs:
                    dataset_pairs = [*dataset_pairs, (convs, "neutral", cat)]

        # Run inference on all datasets and combine. The reference algorithm
        # recognizes three splits: "positive" (rule should fire), "negative"
        # (rule should not fire — domain-shared hard negatives), and
        # "neutral" (universal everyday content). Unknown ds_types fall
        # back to "positive" for backwards compatibility with legacy rows.
        inference_input = []
        for item in dataset_pairs:
            convos, ds_type = item[0], item[1]
            # 3-tuples (default eval) carry the owning rule's name; 2-tuples
            # (legacy explicit-dataset / inline / neutral) don't.
            usecase_path = item[2] if len(item) > 2 else None
            split_label = ds_type if ds_type in ("positive", "negative", "neutral") else "positive"
            inference_input.extend(_build_eval_inference_input(
                convos, ds_type, ruleset, split_label=split_label, usecase_path=usecase_path,
            ))

        # ONE dispatched inference for the whole evaluation (remote -> local
        # GPU/CPU) so a multi-rule eval is a single remote job, not N.
        logger.info(f"Running inference on {len(inference_input)} evaluation conversations...")
        processed_data = _run_inference(classifier_id, inference_input,
                                        running_eval_id=running_eval_id, set_phase=set_phase)

        if not processed_data:
            raise ValueError("Inference produced no results — check conversations and model")

        set_phase("Loading calibrated thresholds…")

        # Load calibrated thresholds — only ones produced AFTER the
        # guardrail's most recent training. Stale thresholds from a
        # previous model would silently apply incorrect cutoffs.
        thresholds_result = execute_query_dict(
            f"""SELECT thresholds FROM evaluation_results
               WHERE classifier_id = %s AND eval_type = 'calibration'
                 AND thresholds IS NOT NULL
                 {_POST_TRAIN_CLAUSE}
               ORDER BY created_at DESC, eval_id DESC LIMIT 1""",
            (classifier_id, classifier_id),
        )
        thresholds = thresholds_result[0]["thresholds"] if thresholds_result else None
        if not thresholds:
            raise ValueError(
                "No calibrated thresholds found for this guardrail. "
                "Run calibration before evaluating."
            )

        set_phase("Scoring & computing per-use-case metrics (TPR / FPR / F1)…")
        # Hand the work to the reference algorithm via the adapter.
        eval_result = eval_adapter.run_evaluation(
            classifier_id=classifier_id,
            labels=labels,
            dialogue_data=processed_data,
            thresholds=thresholds,
            compute_auc=True,
        )

        # The adapter already serialized DataFrames to lists/dicts.
        # eval_result includes the CSVs (per the reference evaluate() writing
        # usecase_metrics_fprtpr.csv + usecase_weighted_averages.csv +
        # weighted_averages_ces_*.csv + label_statistics.csv). Persist
        # everything under metrics so downstream UI can render them.
        metrics = dict(eval_result)
        # Record how many of which split made it into the run, so the UI
        # can label sample sizes correctly (e.g. "FPR computed against
        # 32 neutral conversations + 50 negatives"). Counts use the
        # dataset_pairs we constructed at request time — the same
        # batches the reference algorithm just consumed.
        split_counts: dict = {}
        for item in dataset_pairs:
            convos, ds_type = item[0], item[1]
            split_counts[ds_type] = split_counts.get(ds_type, 0) + len(convos)
        metrics["split_counts"] = split_counts
        plots = {}

        execute_query(
            """INSERT INTO evaluation_results (classifier_id, eval_type, thresholds, metrics, plots)
               VALUES (%s, 'evaluation', %s::jsonb, %s::jsonb, %s::jsonb)""",
            (
                classifier_id,
                _jsonb(thresholds),
                _jsonb(metrics),
                _jsonb(plots) if plots else None,
            ),
        )

        logger.info(f"Evaluation complete for classifier {classifier_id}")

    except InferenceCancelled:
        # Guardrail was deleted mid-evaluation — cascade already removed the
        # '*_running' row. Stop quietly (no error row; it would FK-fail anyway).
        logger.info(f"Evaluation cancelled — classifier {classifier_id} was deleted mid-run")
        return
    except Exception as e:
        logger.exception(f"Evaluation failed for classifier {classifier_id}: {e}")
        execute_query(
            """INSERT INTO evaluation_results (classifier_id, eval_type, thresholds, metrics, plots)
               VALUES (%s, 'evaluation_error', NULL, %s::jsonb, NULL)""",
            (classifier_id, json.dumps({"error": str(e)})),
        )


def _load_test_dataset_dialogues(dataset_id: int) -> tuple:
    """Load dialogue data from a stored test dataset.

    Returns:
        Tuple of (conversations_list, dataset_type_str).
    """
    result = execute_query_dict(
        "SELECT conversations, dataset_type FROM test_datasets WHERE dataset_id = %s AND status = 'ready'",
        (dataset_id,),
    )
    if not result:
        raise ValueError(f"Test dataset {dataset_id} not found or not ready")

    convos = result[0]["conversations"]
    dataset_type = result[0].get("dataset_type", "positive")
    if isinstance(convos, str):
        convos = json.loads(convos)
    return convos, dataset_type


def _load_multiple_test_datasets(dataset_ids: List[int]) -> list:
    """Load and combine multiple test datasets into a single list of (conversations, dataset_type) pairs."""
    combined = []
    for ds_id in dataset_ids:
        convos, ds_type = _load_test_dataset_dialogues(ds_id)
        combined.append((convos, ds_type))
    return combined


def _load_default_eval_pairs(classifier_id: int) -> list:
    """(convos, dataset_type, rule_name) triples used when /evaluate is called
    with no explicit dataset ids: each active rule's positive + negative set.

    Each set is tagged with ITS OWN rule's name so the reference evaluator
    attributes the conversations to the correct use-case and produces a real
    PER-RULE breakdown. The name is COALESCE(custom_name, rules.name) — exactly
    the key build_unified_ruleset() uses, so the evaluator's ground-truth lookup
    (`malicious_use_cases_ruleset[gt_uc_name]`) matches. Lazily pulls the defaults
    from the registry first so a freshly-adopted public rule works without the
    user generating anything."""
    try:
        from services.library_sync import ensure_rule_aux_for_classifier
        ensure_rule_aux_for_classifier(classifier_id)
    except Exception as e:
        logger.warning(f"[evaluation] default-eval lazy fetch failed: {e}")

    active = execute_query_dict(
        "SELECT DISTINCT rs.rule_id, COALESCE(rs.custom_name, r.name) AS rule_name "
        "FROM rule_setup rs LEFT JOIN rules r ON rs.rule_id = r.rule_id "
        "WHERE rs.classifier_id = %s AND rs.is_active = TRUE AND rs.rule_id IS NOT NULL",
        (classifier_id,),
    ) or []
    rule_id_to_name = {r["rule_id"]: r["rule_name"] for r in active if r.get("rule_id")}
    active_rule_ids = list(rule_id_to_name.keys())
    if not active_rule_ids:
        return []

    rows = execute_query_dict(
        """
        SELECT rule_id, dataset_type, conversations FROM (
            SELECT rule_id, dataset_type, conversations,
                   ROW_NUMBER() OVER (
                       PARTITION BY rule_id, dataset_type
                       ORDER BY is_default ASC, created_at DESC
                   ) AS rn
            FROM test_datasets
            WHERE rule_id = ANY(%s)
              AND dataset_type IN ('positive', 'negative')
              AND status = 'ready'
        ) ranked WHERE rn = 1
        """,
        (active_rule_ids,),
    ) or []

    pairs = []
    for r in rows:
        rname = rule_id_to_name.get(r["rule_id"])
        if not rname:
            continue
        convs = r.get("conversations") or []
        if isinstance(convs, str):
            convs = json.loads(convs)
        pairs.append((convs, r["dataset_type"], rname))
    return pairs


def _build_eval_inference_input(
    raw_conversations: list,
    dataset_type: str,
    ruleset: dict,
    split_label: str,
    usecase_path: str = None,
) -> list:
    """Package one batch of raw conversations into inference_input dialogues
    (conversation + metadata) — NO inference. The caller collects these across
    all batches and runs a single dispatched inference (one remote job).

    `usecase_path` is the rule this batch belongs to (a ruleset key); the
    evaluator uses it to pick each dialogue's ground truth, so per-rule sets MUST
    carry their own rule's name for a correct per-rule breakdown. Falls back to
    the first rule only when unknown (legacy explicit-dataset / inline paths).
    """
    rule_names = list(ruleset.keys())
    if not rule_names:
        raise ValueError("No rules configured for this guardrail")
    uc = usecase_path or rule_names[0]

    # dialogue_id is namespaced by use-case so positives/negatives from different
    # rules can't collide.
    return [
        {
            "conversation": conv,
            "metadata": {
                "split": split_label,
                "usecase_path": uc,
                "dialogue_id": f"{uc}_{dataset_type}_{i}",
            },
        }
        for i, conv in enumerate(raw_conversations)
    ]


# --- Endpoints ---

def _has_post_train_success(classifier_id: int, success_type: str) -> bool:
    """True if a SUCCESSFUL run of the given kind ('calibration' / 'evaluation')
    already exists for the guardrail's CURRENT training.

    Calibration and evaluation are once-per-training: a failed run leaves only an
    '*_error' / '*_running' row (so the user can retry), but a success writes the
    plain 'calibration' / 'evaluation' row and locks further runs. Retraining
    bumps classifiers.trained_at, which the _POST_TRAIN_CLAUSE keys off — so the
    old success no longer counts and the operation unlocks again.
    """
    rows = execute_query_dict(
        f"""SELECT 1 FROM evaluation_results
           WHERE classifier_id = %s AND eval_type = %s
             {_POST_TRAIN_CLAUSE}
           LIMIT 1""",
        (classifier_id, success_type, classifier_id),
    )
    return bool(rows)


@router.post("/{classifier_id}/calibrate")
def start_calibration(
    classifier_id: int,
    req: CalibrateRequest,
    background_tasks: BackgroundTasks,
    _: int = Depends(get_current_user),
):
    """Start threshold calibration using per-CE calibration datasets from DB."""
    _verify_classifier_trained(classifier_id)

    from services.auto_calibration import calibration_state
    state = calibration_state(classifier_id)
    if state == "running":
        raise HTTPException(
            status_code=409,
            detail="Calibration is already running for this rule set.",
        )
    if state == "calibrated" and not req.force:
        raise HTTPException(
            status_code=409,
            detail="This rule set is already calibrated. Use Recalibrate to run it again.",
        )

    background_tasks.add_task(
        _run_calibration, classifier_id, req.patience_values
    )
    return {"success": True, "message": "Calibration started — using per-CE calibration datasets"}


@router.post("/{classifier_id}/evaluate")
def start_evaluation(
    classifier_id: int,
    req: EvaluateRequest,
    background_tasks: BackgroundTasks,
    _: int = Depends(get_current_user),
):
    """Start evaluation in background. Uses test dataset or inline data."""
    _verify_classifier_trained(classifier_id)

    # Evaluation needs the calibrated thresholds — you can't evaluate before
    # calibrating. Block it up front with a clear message instead of letting the
    # run start and fail with "no thresholds".
    if not _has_post_train_success(classifier_id, "calibration"):
        raise HTTPException(
            status_code=409,
            detail="Calibrate this rule set before evaluating it.",
        )

    if _has_post_train_success(classifier_id, "evaluation"):
        raise HTTPException(
            status_code=409,
            detail="This rule set is already evaluated. Retrain it to re-evaluate.",
        )

    if req.test_dataset_ids:
        dataset_pairs = _load_multiple_test_datasets(req.test_dataset_ids)
    elif req.test_dataset_id:
        convos, ds_type = _load_test_dataset_dialogues(req.test_dataset_id)
        dataset_pairs = [(convos, ds_type)]
    elif req.dialogue_data:
        dataset_pairs = [(req.dialogue_data, "positive")]
    else:
        # No explicit datasets — fall back to the active rules' default
        # (or custom-preferred) positive + negative sets.
        dataset_pairs = _load_default_eval_pairs(classifier_id)
        if not dataset_pairs:
            raise HTTPException(
                status_code=400,
                detail="No test datasets provided and no default sets available for this rule set's rules",
            )

    # Neutral split is fetched + appended INSIDE the background task (after the
    # registry pull), so it sees the full corpus — and the run hard-fails there
    # if the neutral data is unavailable (all-or-nothing).
    background_tasks.add_task(_run_evaluation, classifier_id, dataset_pairs, req.include_neutral)
    return {"success": True, "message": "Evaluation started in background"}


@router.get("/{classifier_id}/results")
def get_evaluation_results(classifier_id: int, _: int = Depends(get_current_user)):
    """Get latest calibration and evaluation results for a guardrail.

    Filters out results produced before the guardrail's most recent
    training so a retrain visibly clears the page until new runs land —
    instead of confusingly displaying stale numbers from the previous model.
    """
    # eval_id DESC tiebreak: created_at defaults have whole-second precision,
    # so a '*_error' row written in the same second as its '*_running' marker
    # would otherwise lose the DESC LIMIT 1 pick — leaving the UI showing a
    # dead run as "running" until the next restart's crash recovery.
    calibration = execute_query_dict(
        f"""SELECT eval_id, eval_type, thresholds, metrics, plots, created_at
           FROM evaluation_results
           WHERE classifier_id = %s
             AND eval_type IN ('calibration', 'calibration_error', 'calibration_running')
             {_POST_TRAIN_CLAUSE}
           ORDER BY created_at DESC, eval_id DESC LIMIT 1""",
        (classifier_id, classifier_id),
    )
    evaluation = execute_query_dict(
        f"""SELECT eval_id, eval_type, thresholds, metrics, plots, created_at
           FROM evaluation_results
           WHERE classifier_id = %s
             AND eval_type IN ('evaluation', 'evaluation_error', 'evaluation_running')
             {_POST_TRAIN_CLAUSE}
           ORDER BY created_at DESC, eval_id DESC LIMIT 1""",
        (classifier_id, classifier_id),
    )
    return {
        "calibration": calibration[0] if calibration else None,
        "evaluation": evaluation[0] if evaluation else None,
    }


@router.get("/{classifier_id}/thresholds")
def get_calibrated_thresholds(classifier_id: int, _: int = Depends(get_current_user)):
    """Get the latest calibrated thresholds for a guardrail.

    Only returns thresholds produced AFTER the guardrail's most recent
    training; any pre-retrain calibration is hidden behind a 404 so the
    user is forced to recalibrate against the current model.
    """
    result = execute_query_dict(
        f"""SELECT thresholds FROM evaluation_results
           WHERE classifier_id = %s AND eval_type = 'calibration'
             AND thresholds IS NOT NULL
             {_POST_TRAIN_CLAUSE}
           ORDER BY created_at DESC, eval_id DESC LIMIT 1""",
        (classifier_id, classifier_id),
    )
    if not result or not result[0]["thresholds"]:
        raise HTTPException(status_code=404, detail="No calibrated thresholds found")
    return {"thresholds": result[0]["thresholds"]}


@router.get("/{classifier_id}/calibration-status")
def get_calibration_data_status(classifier_id: int, _: int = Depends(get_current_user)):
    """Check which CEs the trained guardrail weights actually know about,
    and whether each one has a calibration dataset locally.

    The authoritative CE list comes from `classifier_meta.json["labels"]`
    — that file is written at training time and contains exactly the
    output heads the trained model has. We do NOT read from rule_setup
    here: setup_ids are volatile (a deleted-and-recreated rule mints a
    new id) and the live rule_setup may have been edited since training,
    so it's a misleading source for "what does the model know".

    For untrained guardrails (no meta.json yet), fall back to the live
    rule_setup so the page shows something actionable — the user can see
    which CEs they're about to train on.

    Lazy-pulls any CE calibration sets that exist in the registry but aren't
    cached locally yet, so public-library CEs appear as Ready without
    forcing the user to click Calibrate first to trigger the fetch.
    """
    try:
        from services.library_sync import (
            ensure_ce_calibrations_for_classifier, ensure_rule_aux_for_classifier,
        )
        ensure_ce_calibrations_for_classifier(classifier_id)
        # Also pull each rule's test sets (positive/negative) so entering the
        # Evaluation page lazily downloads them from the registry — the frontend
        # then refreshes and the Evaluate button unlocks without a manual sync.
        try:
            ensure_rule_aux_for_classifier(classifier_id)
        except Exception as aux_err:
            logger.warning(f"[evaluation] calibration-status rule-aux fetch failed: {aux_err}")
    except Exception as fetch_err:
        logger.warning(f"[evaluation] calibration-status lazy fetch failed: {fetch_err}")

    from evaluation.ruleset_builder import get_classifier_labels
    trained_ce_names = list(get_classifier_labels(classifier_id).keys())

    if trained_ce_names:
        # Trained guardrail — exact list from the on-disk model metadata.
        rows = execute_query_dict("""
            SELECT ce.ce_id, ce.name,
                   (cd.dataset_id IS NOT NULL) AS has_calibration
            FROM cognitive_elements ce
            LEFT JOIN calibration_datasets cd ON ce.ce_id = cd.ce_id
            WHERE ce.name = ANY(%s)
            GROUP BY ce.ce_id, ce.name, cd.dataset_id
        """, (trained_ce_names,)) or []

        # If a name in meta.json no longer exists in cognitive_elements
        # (rare — only if the user deleted the CE row entirely after
        # training), still surface it so the user knows the model
        # references a CE that's gone.
        found = {r["name"] for r in rows}
        for name in trained_ce_names:
            if name not in found:
                rows.append({
                    "ce_id": None,
                    "name": name,
                    "has_calibration": False,
                })
    else:
        # Untrained — show CEs from the live rule_setup so the user can
        # see what they're about to train on.
        rows = execute_query_dict("""
            SELECT ce.ce_id, ce.name,
                   (cd.dataset_id IS NOT NULL) AS has_calibration
            FROM setup_ce_link scl
            JOIN rule_setup rs ON scl.setup_id = rs.setup_id
            JOIN cognitive_elements ce ON scl.ce_id = ce.ce_id
            LEFT JOIN calibration_datasets cd ON ce.ce_id = cd.ce_id
            WHERE rs.classifier_id = %s
            GROUP BY ce.ce_id, ce.name, cd.dataset_id
        """, (classifier_id,)) or []

    ces = rows
    all_ready = all(c.get("has_calibration") for c in ces)
    return {"ces": ces, "all_ready": all_ready, "total": len(ces)}


@router.get("/{classifier_id}/results/history")
def get_results_history(classifier_id: int, limit: int = 10, _: int = Depends(get_current_user)):
    """Get evaluation results history for a guardrail — current model only.

    Pre-retrain rows still live in the table (no destructive write on
    retrain) but are hidden here so the user only sees results that
    correspond to the current trained weights.
    """
    results = execute_query_dict(
        f"""SELECT eval_id, eval_type, created_at
           FROM evaluation_results
           WHERE classifier_id = %s
             {_POST_TRAIN_CLAUSE}
           ORDER BY created_at DESC LIMIT %s""",
        (classifier_id, classifier_id, limit),
    )
    return {"results": results or []}


def _verify_classifier_trained(classifier_id: int):
    """Verify guardrail exists and is trained."""
    result = execute_query_dict(
        "SELECT classifier_id, status FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    if not result:
        raise HTTPException(status_code=404, detail="Rule Set not found")
    if result[0]["status"] not in ("active", "needs_retraining"):
        raise HTTPException(
            status_code=400,
            detail=f"Rule Set must be trained first (current status: {result[0]['status']})"
        )
