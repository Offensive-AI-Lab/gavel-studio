# classifier_engine/trainer.py
# Orchestrates the full training pipeline for a user's guardrail.
# Takes a classifier_id, loads CE datasets from DB, extracts LLM embeddings,
# trains the RNN guardrail, and saves the result.
import os
import json
import re
import shutil
import logging
import torch
from typing import Dict

from classifier_engine.RNN import TopicRNN, train_rnn_model, train_rnn_candidates
from classifier_engine.utils_train import (
    load_model_and_tokenizer,
    create_dataloaders_from_directory,
    split_dataset_into_train_val,
    create_dataloaders_for_sequences,
    extract_per_sequence_reps,
    _head_geometry,
)

logger = logging.getLogger(__name__)

# Root for every trained guardrail on disk. Layout is:
#
#   trained_classifiers/
#     <user_id>/
#       classifier_<classifier_id>/
#         trained_rnn.pth
#         classifier_meta.json
#         dataset/...
#         calibration/...
#         evaluation/...
#
# Per-user namespacing keeps two users' classifier_5 directories from
# colliding and makes "delete everything for a user" trivial. All path
# resolution goes through `classifier_workdir(...)` below — never join
# TRAINED_MODELS_DIR with classifier_<id> directly anymore.
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_classifiers")


def _resolve_user_id(classifier_id: int) -> int:
    """Look up the owning user via guardrail → model → user.

    Cached at the call-site by passing user_id explicitly in tight loops.
    Raises ValueError on a missing guardrail so callers fail loudly
    instead of silently writing to a junk path.
    """
    from utils.sqlite_db import execute_query_dict
    rows = execute_query_dict(
        """
        SELECT m.user_id
        FROM classifiers c
        JOIN target_models m ON c.model_id = m.model_id
        WHERE c.classifier_id = %s
        """,
        (classifier_id,),
    ) or []
    if not rows:
        raise ValueError(f"guardrail {classifier_id} not found (or has no owning model)")
    return int(rows[0]["user_id"])


def classifier_workdir(classifier_id: int, user_id: int = None) -> str:
    """Absolute filesystem path for a guardrail's working directory.

    Pass `user_id` if you already have it (avoids a DB roundtrip in the
    hot path); otherwise it's resolved from the guardrail row.
    """
    if user_id is None:
        user_id = _resolve_user_id(classifier_id)
    return os.path.join(TRAINED_MODELS_DIR, str(user_id), f"classifier_{classifier_id}")


def delete_classifier_workdir(classifier_id: int, user_id: int) -> None:
    """Remove the on-disk artifacts for a deleted guardrail.

    user_id MUST be passed explicitly: this is invoked AFTER the
    guardrail row (and possibly its owning model) has been deleted, so
    the JOIN-through-target_models lookup `classifier_workdir()` does
    by default would return nothing. Caller's responsibility to capture
    the user_id before the DB delete.

    No-op if the dir doesn't exist — that's the normal state for an
    untrained guardrail. If rmtree fails (file lock, permission), we
    log and return; the boot-time OrphanedClassifierDirRecovery sweep
    is the safety net.

    Also prunes the empty `<user_id>/` parent dir if the deleted
    guardrail was the user's last one. Mirrors the same housekeeping
    the boot-time recovery does, so disk state is consistent regardless
    of which path cleaned it up.
    """
    work_dir = os.path.join(TRAINED_MODELS_DIR, str(user_id), f"classifier_{classifier_id}")
    if os.path.isdir(work_dir):
        try:
            shutil.rmtree(work_dir)
            logger.info(f"[delete] removed classifier workdir: {work_dir}")
        except Exception as e:
            logger.warning(f"[delete] failed to remove {work_dir}: {e}")
            return

    user_dir = os.path.join(TRAINED_MODELS_DIR, str(user_id))
    try:
        if os.path.isdir(user_dir) and not os.listdir(user_dir):
            os.rmdir(user_dir)
    except OSError:
        # Racy with another concurrent delete or a freshly-created
        # workdir from a parallel training run — harmless either way.
        pass

# Default training hyperparameters (cloud-aware: conservative defaults)
DEFAULT_TRAINING_CONFIG = {
    "batch_size_text": 4,       # matches reference config.json (feature-extraction batch; affects throughput/memory only, not the model)
    "max_length": 256,
    "rnn_sequence_length": 5,
    "batch_size": 64,           # RNN mini-batch — matches reference config.json (was 16)
    "epochs": 10,
    # Which LLM hidden layers to read attention-value features from. The MIDDLE
    # band carries the best features for this probe: the reference
    # reads layers 13..26 of Mistral-7B's 32 (selected_layers_range = [13, 27]).
    # An explicit [start, stop) is used when valid for the model; otherwise we
    # fall back to the same middle band scaled to the model's depth (see below).
    # This REPLACES the old "last N layers" heuristic (which read layers 24..31
    # and gave markedly worse per-use-case detection — the eval-parity bug).
    "selected_layers_range": [13, 27],
    "num_layers_to_use": 8,     # legacy fallback width only; ignored when selected_layers_range is set
    "hidden_dim": 256,
    "num_rnn_layers": 3,
    "learning_rate": 3e-4,
    # Candidate refinement: several independent reference-parity fits are run on
    # the same cached features and the one whose WEAKEST CE generalizes best to
    # the calibration dialogues is kept (min per-CE ROC-AUC, mean as tie-break).
    # Validation metrics can't rank candidates — they saturate at ~1.0 for every
    # fit — while transfer to real dialogues varies a lot between fits and is
    # exactly what calibration/evaluation measure. Set to 1 for a single fit.
    "refinement_rounds": 5,
    # Per-CE cap on calibration dialogues used for candidate scoring (bounds the
    # one-off scoring pass; the dialogues are scored once for ALL candidates).
    "selection_calib_per_ce": 25,
}


def get_training_config(classifier_id: int) -> dict:
    """Load training config from DB, merged with defaults. The guardrail's MODEL
    can pin which LLM layers to use (target_models.selected_layers) — when set it
    overrides the default range, so the per-model layer choice drives training."""
    from utils.sqlite_db import execute_query_dict
    result = execute_query_dict(
        "SELECT training_config FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    stored = (result[0].get("training_config") or {}) if result else {}
    cfg = {**DEFAULT_TRAINING_CONFIG, **stored}
    rng = _model_selected_layers(classifier_id)
    if rng:
        cfg["selected_layers_range"] = rng
    return cfg


def _model_selected_layers(classifier_id: int):
    """The [start, end) LLM-layer range pinned on this guardrail's model, or None."""
    from utils.sqlite_db import execute_query_dict
    rows = execute_query_dict(
        "SELECT tm.selected_layers FROM classifiers c "
        "JOIN target_models tm ON c.model_id = tm.model_id WHERE c.classifier_id = %s",
        (classifier_id,),
    )
    sel = rows[0].get("selected_layers") if rows else None
    return list(sel) if sel and len(sel) == 2 else None


def _sanitize_label(name: str) -> str:
    """Convert CE name to filesystem-safe label (matches directory name rules)."""
    return re.sub(r'[^\w\-]', '_', name).strip('_') or "label"


def get_classifier_info(classifier_id: int):
    from utils.sqlite_db import execute_query_dict
    query = """
        SELECT c.classifier_id, c.model_id, c.name, c.status,
               tm.storage_path, tm.name as model_name
        FROM classifiers c
        JOIN target_models tm ON c.model_id = tm.model_id
        WHERE c.classifier_id = %s
    """
    result = execute_query_dict(query, (classifier_id,))
    return result[0] if result else None


def get_classifier_ces_with_datasets(classifier_id: int):
    """
    Return all CEs linked to this guardrail's rules that have excitation datasets.
    Returns list of {ce_id, name, dataset_json}.
    """
    from utils.sqlite_db import execute_query_dict
    query = """
        SELECT DISTINCT
            ce.ce_id,
            ce.name,
            ed.dataset
        FROM rule_setup rs
        JOIN setup_ce_link scl ON rs.setup_id = scl.setup_id
        JOIN cognitive_elements ce ON scl.ce_id = ce.ce_id
        LEFT JOIN excitation_datasets ed ON ce.ce_id = ed.ce_id
        WHERE rs.classifier_id = %s
          AND ed.dataset IS NOT NULL
        ORDER BY ce.ce_id
    """
    return execute_query_dict(query, (classifier_id,)) or []


def update_classifier_status(classifier_id: int, status: str, model_path: str = None, training_log: str = None):
    from utils.sqlite_db import execute_query
    if model_path is not None and training_log is not None:
        execute_query(
            "UPDATE classifiers SET status = %s, model_path = %s, training_log = %s WHERE classifier_id = %s",
            (status, model_path, training_log, classifier_id),
        )
    elif model_path is not None:
        execute_query(
            "UPDATE classifiers SET status = %s, model_path = %s WHERE classifier_id = %s",
            (status, model_path, classifier_id),
        )
    elif training_log is not None:
        execute_query(
            "UPDATE classifiers SET status = %s, training_log = %s WHERE classifier_id = %s",
            (status, training_log, classifier_id),
        )
    else:
        execute_query(
            "UPDATE classifiers SET status = %s WHERE classifier_id = %s",
            (status, classifier_id),
        )


def _extract_training_data(dataset_raw) -> list:
    """
    Parse a CE excitation dataset (stored as JSON text or dict) into a list of conversations.
    Handles both the full dataset dict format and raw list formats.
    """
    if isinstance(dataset_raw, str):
        try:
            data = json.loads(dataset_raw)
        except (json.JSONDecodeError, TypeError):
            return []
    else:
        data = dataset_raw

    # Full dataset dict format: {"samples": [...]} or legacy {"training_data": [...]}
    if isinstance(data, dict):
        conversations = data.get("samples", data.get("training_data", []))
        if isinstance(conversations, list):
            return conversations
        return []

    # Raw list of conversations
    if isinstance(data, list):
        return data

    return []


def prepare_training_data(ces_with_datasets: list, dataset_dir: str) -> Dict[str, int]:
    """
    Write CE datasets to the training directory as JSON files.
    Returns LABELS dict mapping sanitized CE name -> index.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    labels = {}
    label_idx = 0

    for ce in ces_with_datasets:
        ce_name = ce['name']
        safe_name = _sanitize_label(ce_name)
        conversations = _extract_training_data(ce.get('dataset'))

        if not conversations:
            logger.warning(f"CE '{ce_name}' has no valid conversations, skipping")
            continue

        # Write JSON file
        out_path = os.path.join(dataset_dir, f"{safe_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2)

        labels[safe_name] = label_idx
        label_idx += 1
        logger.info(f"  CE '{ce_name}' -> label {label_idx - 1} ({len(conversations)} conversations)")

    return labels


def fetch_calibration_entries(classifier_id: int, ces_with_datasets: list, per_ce: int = 25) -> list:
    """CE-level calibration dialogues for candidate selection, capped per CE
    (deterministic: the first `per_ce` of each set; the sets are static).

    Returns [{"conversation": <conv>, "ce": "<sanitized label>"}]. Best-effort:
    any failure returns [] and the caller falls back to a single fit."""
    try:
        from services.library_sync import ensure_ce_calibrations_for_classifier
        ensure_ce_calibrations_for_classifier(classifier_id)
    except Exception as fetch_err:
        logger.warning(f"[Trainer] CE-calibration lazy fetch failed: {fetch_err}")
    try:
        from utils.sqlite_db import execute_query_dict
        names = [ce["name"] for ce in ces_with_datasets]
        rows = execute_query_dict("""
            SELECT ce.name, cd.dataset
            FROM cognitive_elements ce
            JOIN calibration_datasets cd ON ce.ce_id = cd.ce_id
            WHERE ce.name = ANY(%s)
        """, (names,)) or []
        entries = []
        for row in rows:
            dataset = row["dataset"]
            if isinstance(dataset, str):
                dataset = json.loads(dataset)
            conversations = (dataset.get("conversations") or dataset.get("samples") or [])[:per_ce]
            safe = _sanitize_label(row["name"])
            entries.extend({"conversation": c, "ce": safe} for c in conversations)
        return entries
    except Exception as e:
        logger.warning(f"[Trainer] Calibration fetch for candidate selection failed: {e}")
        return []


class TrainingCancelled(BaseException):
    """Raised inside run_training when the guardrail row is deleted mid-run, so a
    local background-training thread stops promptly instead of training (and then
    trying to save) a guardrail that no longer exists.

    Subclasses BaseException (not Exception) ON PURPOSE: the per-epoch RNN
    progress callback is wrapped in `except Exception: pass` (RNN.py), as are
    other best-effort hooks. Deriving from BaseException means those generic
    handlers don't swallow the cancel — it propagates straight to run_training's
    explicit `except TrainingCancelled`, so a delete during the RNN epoch loop
    (not just at a stage boundary) still aborts promptly."""


def _classifier_deleted(classifier_id: int) -> bool:
    """True only when we can CONFIRM the guardrail row is gone. A transient DB
    error returns False so a momentary blip never aborts a healthy run."""
    try:
        rows = execute_query_dict(
            "SELECT 1 AS ok FROM classifiers WHERE classifier_id = %s", (classifier_id,)
        )
    except Exception:
        return False
    return not rows


def run_training(classifier_id: int, progress_callback=None):
    """
    Full training pipeline for a guardrail.

    Steps:
    1. Fetch guardrail + model info from DB
    2. Fetch CE datasets from DB
    3. Write training data to working directory
    4. Load the LLM
    5. Split data into train/val
    6. Extract per-sequence attention representations
    7. Train the RNN guardrail
    8. Save the trained model
    9. Update DB status to 'active'

    Args:
        classifier_id: The guardrail to train.
        progress_callback: Optional callable(stage: str, detail: str) for status updates.

    Cloud readiness:
        - All files written to TRAINED_MODELS_DIR/{classifier_id}/
        - No absolute paths in saved artifacts
        - Designed to run in a background worker (Celery, etc.) in production
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from utils.device import get_torch_device, empty_device_cache, get_llm_device_map
    device = get_torch_device()
    print(f"\n{'='*60}", flush=True)
    print(f"[Trainer] Starting training for classifier {classifier_id} on {device}", flush=True)
    print(f"{'='*60}\n", flush=True)
    logger.info(f"[Trainer] Starting training for classifier {classifier_id} on {device}")

    def _progress(stage, detail=""):
        msg = f"[Trainer] [{stage}] {detail}" if detail else f"[Trainer] [{stage}]"
        print(msg, flush=True)
        logger.info(msg)
        if progress_callback:
            try:
                progress_callback(stage, detail)
            except Exception:
                pass
        # Cooperative cancellation: if the guardrail was deleted while this run
        # is in flight (e.g. the user hit "remove" mid-training), stop at this
        # stage boundary instead of burning compute on a guardrail that's gone.
        if _classifier_deleted(classifier_id):
            raise TrainingCancelled(classifier_id)

    try:
        # 1. Guardrail info + training config
        _progress("init", "Loading guardrail info")
        classifier = get_classifier_info(classifier_id)
        if not classifier:
            raise ValueError(f"Guardrail {classifier_id} not found")

        cfg = get_training_config(classifier_id)
        _progress("init", "Preparing training configuration")
        logger.info(f"[Trainer] Config: epochs={cfg['epochs']}, lr={cfg['learning_rate']}, "
                    f"hidden={cfg['hidden_dim']}, rnn_layers={cfg['num_rnn_layers']}, "
                    f"seq_len={cfg['rnn_sequence_length']}, llm_layers={cfg.get('selected_layers_range') or cfg['num_layers_to_use']}")

        update_classifier_status(classifier_id, 'training')

        # 2. CE datasets
        _progress("data", "Fetching CE datasets from database")
        ces_with_datasets = get_classifier_ces_with_datasets(classifier_id)
        if not ces_with_datasets:
            raise ValueError("No CEs with training datasets found. Generate excitation datasets for your CEs first.")

        # NOTE: the trained-policy snapshot (trained_rule_names /
        # trained_policy_fingerprint / trained_at) is committed ONLY on
        # successful completion, at the bottom of this try block, via
        # commit_trained_policy_snapshot(). It must NOT be written here at the
        # start: an interrupted run (server killed mid-training) would otherwise
        # leave the snapshot pointing at a policy whose model was never produced,
        # making the guardrail falsely look "Up to Date".

        # 3. Working directory — namespaced under the owning user so two
        # users' classifier_5 directories can't collide.
        work_dir = classifier_workdir(classifier_id)
        dataset_dir = os.path.join(work_dir, "dataset")

        # Retrain wipes the previous artifacts in place (same folder
        # name, fresh contents) so the user's expectation of "the
        # classifier folder name doesn't move on retrain" holds.
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir, exist_ok=True)

        # Drop the old (LLM, RNN, meta) trio from the in-memory model
        # cache so the realtime route's next call has to reload from
        # disk. Without this, the cache keeps serving the previous model
        # across retrains — the user retrains on rule Y but the realtime
        # monitor stays bound to rule X. Routes that need the model
        # (realtime / evaluation / calibration) all reject during
        # training via the status='training' check, so evicting now is
        # safe; load_or_get repopulates lazily once status flips back.
        try:
            from evaluation.model_cache import evict as _evict_cache
            _evict_cache(classifier_id)
        except Exception as evict_err:
            logger.warning(f"[Trainer] Cache evict failed for classifier {classifier_id}: {evict_err}")

        # 4. Write training data
        _progress("data", f"Writing training data for {len(ces_with_datasets)} CEs")
        labels = prepare_training_data(ces_with_datasets, dataset_dir)
        if not labels:
            raise ValueError("No valid training data. Ensure CE excitation datasets contain conversations.")

        logger.info(f"[Trainer] Labels: {labels}")

        # 4b. Calibration dialogues for candidate selection. Fetched up front so
        # a missing calibration library degrades to a single fit BEFORE we spend
        # GPU time on extra candidates.
        calib_entries = fetch_calibration_entries(
            classifier_id, ces_with_datasets,
            per_ce=int(cfg.get("selection_calib_per_ce", 25)),
        )
        rounds = max(1, int(cfg.get("refinement_rounds", 5)))
        if not calib_entries and rounds > 1:
            logger.warning("[Trainer] No calibration data available — training a single candidate")
            rounds = 1
        logger.info(f"[Trainer] Candidate refinement: rounds={rounds}, "
                    f"selection dialogues={len(calib_entries)}")

        # 5. Load LLM
        model_path = classifier['storage_path']
        _progress("load_llm", f"Loading LLM from: {model_path}")
        llm, tokenizer = load_model_and_tokenizer(model_path, device_map=get_llm_device_map())

        # Pick which LLM hidden layers to read features from. Use the MIDDLE band
        # (the reference reads layers 13..26 of Mistral-7B's 32 via
        # selected_layers_range = [13, 27]), NOT the last N layers — the middle
        # layers carry far better features for this probe; the old last-N heuristic
        # (layers 24..31) gave much worse per-use-case detection.
        total_layers = llm.config.num_hidden_layers
        _rng = cfg.get("selected_layers_range")
        if _rng and len(_rng) == 2 and _rng[1] > _rng[0] >= 0:
            # CLAMP the chosen range to the model's real depth — any layer beyond
            # num_hidden_layers is ignored (the UI lets the user pick up to 100
            # when the model's layer count isn't known up front).
            _start = min(_rng[0], total_layers - 1)
            _end = min(_rng[1], total_layers)
            if _end <= _start:
                _end = _start + 1
            selected_layers = list(range(_start, _end))
        else:
            # Same middle band, scaled to this model's depth (reproduces [13, 27)
            # for a 32-layer LLM; degrades gracefully for smaller models).
            _start = round(total_layers * 13 / 32)
            _end = max(_start + 1, round(total_layers * 27 / 32))
            selected_layers = list(range(_start, _end))
        n_layers = len(selected_layers)
        _progress("load_llm", f"Using layers {selected_layers[0]}-{selected_layers[-1]} ({n_layers}) of {total_layers}")

        # 6. Split dataset into train/val
        _progress("split", "Splitting dataset into train/val")
        split_dataset_into_train_val(dataset_root_path=dataset_dir, train_ratio=0.8, random_seed=42)

        # 7. Create text dataloaders
        _progress("extract", "Creating text dataloaders")
        text_dataloaders = create_dataloaders_from_directory(
            base_directory=dataset_dir,
            tokenizer=tokenizer,
            batch_size=cfg["batch_size_text"],
            max_length=cfg["max_length"],
        )

        # 8. Extract per-sequence representations
        seq_train = os.path.join(work_dir, "sequences", "train")
        seq_val = os.path.join(work_dir, "sequences", "val")

        _progress("extract", "Extracting LLM representations for train set")
        extract_per_sequence_reps(
            dataloaders=text_dataloaders["train_dataloaders"],
            model=llm,
            tokenizer=tokenizer,
            selected_layers=selected_layers,
            save_root=seq_train,
            dtype=torch.float16,
        )

        _progress("extract", "Extracting LLM representations for val set")
        extract_per_sequence_reps(
            dataloaders=text_dataloaders["val_dataloaders"],
            model=llm,
            tokenizer=tokenizer,
            selected_layers=selected_layers,
            save_root=seq_val,
            dtype=torch.float16,
        )

        # Free LLM memory before RNN training
        del llm
        empty_device_cache()

        # 9. Create sequence dataloaders
        _progress("train_rnn", "Creating sequence dataloaders")
        rnn_seq_config = {"RNN_sequence_length": cfg["rnn_sequence_length"]}
        dataloaders_new, class_counts, used_min = create_dataloaders_for_sequences(
            base_directory=work_dir,
            labels=labels,
            batch_size=cfg["batch_size"],
            config=rnn_seq_config,
            seed=42,
            num_workers=4,   # matches reference (gavel scripts/train.py)
        )

        # 10. Compute RNN input dimension from LLM geometry
        # We need the model config without loading weights again
        from transformers import AutoConfig
        llm_config = AutoConfig.from_pretrained(model_path)
        if "gemma" in llm_config.model_type:
            n_v_heads = llm_config.text_config.num_key_value_heads
            head_dim = llm_config.text_config.head_dim
        else:
            n_q_heads = llm_config.num_attention_heads
            n_v_heads = llm_config.num_key_value_heads
            head_dim = getattr(llm_config, "head_dim", None) or (llm_config.hidden_size // n_q_heads)
        readout_dim = n_v_heads * head_dim
        num_classes = len(labels)

        _progress("train_rnn", f"Training guardrail: {num_classes} classes, input_dim={readout_dim}, layers={n_layers}")

        # Fresh TopicRNN per candidate fit; each fit is an exact-parity
        # reference run (gavel/scripts/train.py semantics).
        def _build_rnn():
            return TopicRNN(
                num_layers=n_layers,
                input_dim=readout_dim,
                hidden_dim=cfg["hidden_dim"],
                num_rnn_layers=cfg["num_rnn_layers"],
                num_topics=num_classes,
                rnn_type="GRU",
            ).to(device)

        checkpoint_dir = os.path.join(work_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        training_log_entries = []

        # User-facing progress is ONE continuous figure across the whole
        # optimization — per-fit/per-epoch structure stays in server logs only.
        def _rnn_progress(step, total_steps, metrics):
            pct = min(99, int(round(100.0 * step / max(1, total_steps))))
            _progress("train_rnn", f"Optimizing guardrail — {pct}%")
            entry = {
                "progress": pct,
                "train_loss": round(metrics.get("train_loss", 0), 6),
                "val_loss": round(metrics.get("val_loss", 0), 6),
                "val_accuracy": round(float(metrics.get("val_accuracy", 0)), 4),
                "learning_rate": metrics.get("learning_rate"),
            }
            training_log_entries.append(entry)
            # Stream log to DB for live monitoring
            try:
                log_json = json.dumps(training_log_entries)
                update_classifier_status(classifier_id, status='training', training_log=log_json)
            except Exception:
                pass

        _progress("train_rnn", "Optimizing guardrail — 0%")
        candidates = train_rnn_candidates(
            _build_rnn,
            rounds=rounds,
            base_seed=42,
            progress_callback=_rnn_progress,
            labels_dict=labels,
            train_loader=dataloaders_new["train"],
            val_loader=dataloaders_new["val"],
            epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            train_class_counts=class_counts["train"],
            val_class_counts=class_counts["val"],
            checkpoint_dir=checkpoint_dir,
            use_wandb=False,
        )

        if len(candidates) > 1 and calib_entries:
            # Score all candidates in ONE pass over the calibration dialogues
            # (LLM extraction once per dialogue; each candidate adds only a GRU
            # forward) and keep the one whose weakest CE transfers best.
            _progress("train_rnn", "Validating guardrail…")
            from classifier_engine.selection import score_candidates_on_calibration, pick_best_candidate
            llm, _sel_tokenizer = load_model_and_tokenizer(model_path, device_map=get_llm_device_map())

            def _sel_progress(done, total):
                _progress("train_rnn", f"Validating guardrail — {min(99, int(100.0 * done / max(1, total)))}%")

            scores = score_candidates_on_calibration(
                candidates, llm, _sel_tokenizer, calib_entries, labels,
                window_size=cfg["rnn_sequence_length"],
                selected_layers=selected_layers,
                device=device,
                progress_callback=_sel_progress,
            )
            best_idx = pick_best_candidate(scores)
            for c_idx, s in enumerate(scores):
                logger.info(f"[Trainer] candidate {c_idx + 1}/{len(candidates)}: "
                            f"min_auc={s.get('min_auc')} mean_auc={s.get('mean_auc')}")
            logger.info(f"[Trainer] selected candidate {best_idx + 1}/{len(candidates)}")
            trained_rnn = candidates[best_idx]
            del llm
            empty_device_cache()
        else:
            trained_rnn = candidates[0]

        # 11. Save trained RNN
        _progress("save", "Saving trained model")
        rnn_path = os.path.join(work_dir, "trained_rnn.pth")
        torch.save(trained_rnn.state_dict(), rnn_path)

        # Save metadata needed for inference
        meta = {
            "labels": labels,
            "num_classes": num_classes,
            "readout_dim": readout_dim,
            "n_layers": n_layers,
            "hidden_dim": cfg["hidden_dim"],
            "num_rnn_layers": cfg["num_rnn_layers"],
            "rnn_sequence_length": cfg["rnn_sequence_length"],
            "learning_rate": cfg["learning_rate"],
            "model_path": model_path,
            "selected_layers": [list(selected_layers)[0], list(selected_layers)[-1] + 1],
            "training_config": cfg,
        }
        meta_path = os.path.join(work_dir, "classifier_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # 12. Update DB with final structured log
        final_log = json.dumps(training_log_entries) if training_log_entries else f"Trained on {num_classes} classes: {list(labels.keys())}"
        update_classifier_status(
            classifier_id,
            status='active',
            model_path=rnn_path,
            training_log=final_log,
        )

        # 13. Commit the trained-policy snapshot NOW that training succeeded:
        # trained_rule_setup_ids / trained_rule_names / trained_policy_fingerprint
        # / trained_at, all in one write. Doing it here (not at the start) means
        # an interrupted run never updates the snapshot, so the guardrail keeps
        # reflecting its last SUCCESSFUL training — and a failed retrain doesn't
        # lose history (the evaluation_results filter still keys off trained_at).
        try:
            from sql_scripts.model_scripts import commit_trained_policy_snapshot
            commit_trained_policy_snapshot(classifier_id)
        except Exception as snap_err:
            logger.error(f"[Trainer] Failed to commit trained-policy snapshot for classifier {classifier_id}: {snap_err}")

        # 14. Chain straight into calibration. Monitoring an uncalibrated rule
        # set is meaningless (every CE falls back to a 0.5 threshold), and the
        # step takes no user input — so don't make the user remember it.
        try:
            from services.auto_calibration import schedule_post_training_calibration
            schedule_post_training_calibration(classifier_id)
        except Exception as cal_err:
            logger.error(f"[Trainer] Auto-calibration not scheduled for classifier {classifier_id}: {cal_err}")

        _progress("done", f"Training complete. Model saved to {rnn_path}")

    except TrainingCancelled:
        # The guardrail was deleted mid-run. The delete path already removed the
        # DB row + workdir, so there's nothing to mark 'error' — just free GPU
        # memory and return quietly (do NOT re-raise; the background task wrapper
        # would otherwise log a spurious failure traceback).
        print(f"[Trainer] Training cancelled — classifier {classifier_id} was removed mid-run; stopping.", flush=True)
        logger.info(f"[Trainer] Training cancelled for deleted classifier {classifier_id}")
        try:
            empty_device_cache()
        except Exception:
            pass
        return

    except Exception as e:
        error_msg = str(e)
        print(f"\n[Trainer] *** TRAINING FAILED *** classifier {classifier_id}: {error_msg}", flush=True)
        logger.error(f"[Trainer] Training failed for classifier {classifier_id}: {error_msg}")
        update_classifier_status(classifier_id, status='error', training_log=error_msg)
        raise
