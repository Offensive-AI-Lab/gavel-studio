#!/usr/bin/env python3
"""Standalone guardrail training script for remote GPU workers.

Runs entirely without the GAVEL backend — no DB access, no FastAPI. All
inputs come from files in the job directory; all outputs go to the same
directory. The backend is responsible for staging inputs before the job
starts and retrieving outputs after it finishes.

Usage (called by the worker harness, not directly by a user):
    python train_job.py \
        --job-dir /home/<user>/gavel_jobs/<job_id> \
        [--device cuda]

The job directory must contain:
    job_payload.json      prepared by the backend, contains:
        {
            "model_hf_path": "meta-llama/Llama-3.2-3B-Instruct",
            "labels": {"ce_name_a": 0, "ce_name_b": 1, ...},
            "config": { ...training hyperparams... },
            "classifier_id": 42,
            "user_id": 7
        }
    dataset/
        <ce_name_a>.json  list of conversations (excitation data)
        <ce_name_b>.json
        ...

The script writes to the same job directory:
    trained_rnn.pth           the trained guardrail weights
    classifier_meta.json      labels, config, metrics, model geometry
    training_log.json         epoch-by-epoch train/val metrics
    status.json               {"status": "success"|"failed", "error": "...", "elapsed_s": N}

Exit codes:
    0   success
    1   bad arguments / missing input files
    2   training failed (OOM, NaN loss, etc.)
    3   model load failed
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback

# Add the gavel_code directory to sys.path so `classifier_engine` is
# importable. On the worker, classifier_engine/ lives at
# ~/gavel_code/classifier_engine/ — the parent dir needs to be on path.
_code_dir = os.path.expanduser("~/gavel_code")
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

import torch


def _status(job_dir, status, error=None, elapsed=None):
    """Write status.json so the backend can read the outcome."""
    payload = {"status": status}
    if error:
        payload["error"] = str(error)[:2000]
    if elapsed is not None:
        payload["elapsed_s"] = round(elapsed, 1)
    path = os.path.join(job_dir, "status.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[train_job] status={status}" + (f" error={error}" if error else ""))


def _sanitize_label(name):
    return re.sub(r'[^\w\-]', '_', name).strip('_') or "label"


def main():
    parser = argparse.ArgumentParser(description="GAVEL remote training job")
    parser.add_argument("--job-dir", required=True, help="Path to the job directory")
    parser.add_argument("--device", default="auto", help="'cuda', 'cpu', or 'auto' (default)")
    args = parser.parse_args()

    job_dir = args.job_dir
    t0 = time.time()

    # ---- Validate inputs ----
    payload_path = os.path.join(job_dir, "job_payload.json")
    dataset_dir = os.path.join(job_dir, "dataset")

    if not os.path.isfile(payload_path):
        _status(job_dir, "failed", f"Missing {payload_path}")
        sys.exit(1)
    if not os.path.isdir(dataset_dir):
        _status(job_dir, "failed", f"Missing {dataset_dir}")
        sys.exit(1)

    with open(payload_path) as f:
        payload = json.load(f)

    # Gated-model auth: the backend (which has DB access the worker lacks)
    # resolved target_models.hf_token and shipped it in the payload. Export it so
    # transformers/huggingface_hub authenticate the base-model download — the
    # from_pretrained calls below pass no token and fall back to this env var.
    _hf_token = payload.get("hf_token")
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

    model_hf_path = payload.get("model_hf_path")
    labels = payload.get("labels", {})
    config = payload.get("config", {})

    if not model_hf_path:
        _status(job_dir, "failed", "model_hf_path missing from payload")
        sys.exit(1)
    if not labels:
        _status(job_dir, "failed", "labels dict empty in payload")
        sys.exit(1)

    # ---- Device setup ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[train_job] device={device}, model={model_hf_path}, labels={len(labels)}")

    # ---- Training config (merge defaults) ----
    defaults = {
        "batch_size_text": 4,   # matches reference config.json (feature-extraction batch; throughput/memory only)
        "max_length": 256,
        "rnn_sequence_length": 5,
        "batch_size": 64,   # matches reference config.json (was 16)
        "epochs": 10,
        # Middle-band LLM layers (reference: layers 13..26 of Mistral-7B's
        # 32). Explicit [start, stop) wins when valid; else a scaled middle band.
        # Replaces the old last-N heuristic (layers 24..31), which scored much worse.
        "selected_layers_range": [13, 27],
        "num_layers_to_use": 8,   # legacy fallback width only; ignored when range is set
        "hidden_dim": 256,
        "num_rnn_layers": 3,
        "learning_rate": 3e-4,
    }
    cfg = {**defaults, **config}

    try:
        # ---- Step 1: Load LLM ----
        print(f"[train_job] Loading LLM: {model_hf_path}", flush=True)
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

        # Left-pad + legacy=False to match the local loader (utils_train.load_model_and_tokenizer)
        # and the reference (training/utils.py). Decoder-only models must be left-padded.
        tokenizer = AutoTokenizer.from_pretrained(model_hf_path, padding_side="left", legacy=False, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm_config = AutoConfig.from_pretrained(model_hf_path, trust_remote_code=True)
        total_layers = llm_config.num_hidden_layers
        # MIDDLE band, matching the reference (layers 13..26 of
        # Mistral-7B's 32), NOT the last N — middle layers give much better
        # features for this probe.
        _rng = cfg.get("selected_layers_range")
        if _rng and 0 <= _rng[0] < _rng[1] <= total_layers:
            selected_layers = list(range(_rng[0], _rng[1]))
        else:
            _start = round(total_layers * 13 / 32)
            _end = max(_start + 1, round(total_layers * 27 / 32))
            selected_layers = list(range(_start, _end))
        n_layers = len(selected_layers)
        print(f"[train_job] LLM has {total_layers} layers, using middle layers "
              f"{selected_layers[0]}-{selected_layers[-1]} ({n_layers})", flush=True)

        # Determine device_map for the LLM
        if device.type == "cuda":
            device_map = "auto"
        else:
            device_map = {"": "cpu"}

        llm = AutoModelForCausalLM.from_pretrained(
            model_hf_path,
            device_map=device_map,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            output_hidden_states=True,
            attn_implementation="eager",
        )
        llm.eval()
        print(f"[train_job] LLM loaded successfully", flush=True)

    except Exception as e:
        _status(job_dir, "failed", f"Model load failed: {e}", time.time() - t0)
        traceback.print_exc()
        sys.exit(3)

    try:
        # ---- Step 2: Split dataset into train/val ----
        print(f"[train_job] Splitting dataset into train/val", flush=True)

        # The classifier_engine utilities expect a specific directory structure.
        # We add the project to sys.path so imports work on the worker.
        # The worker staging script copies the classifier_engine/ directory.
        from classifier_engine.utils_train import (
            split_dataset_into_train_val,
            create_dataloaders_from_directory,
            create_dataloaders_for_sequences,
            extract_per_sequence_reps,
        )
        from classifier_engine.RNN import TopicRNN, train_rnn_model, train_rnn_candidates

        split_dataset_into_train_val(dataset_root_path=dataset_dir, train_ratio=0.8, random_seed=42)

        # ---- Step 3: Create text dataloaders ----
        print(f"[train_job] Creating text dataloaders", flush=True)
        text_dataloaders = create_dataloaders_from_directory(
            base_directory=dataset_dir,
            tokenizer=tokenizer,
            batch_size=cfg["batch_size_text"],
            max_length=cfg["max_length"],
        )

        # ---- Step 4: Extract per-sequence representations ----
        seq_train = os.path.join(job_dir, "sequences", "train")
        seq_val = os.path.join(job_dir, "sequences", "val")

        print(f"[train_job] Extracting LLM representations (train set)...", flush=True)
        extract_per_sequence_reps(
            dataloaders=text_dataloaders["train_dataloaders"],
            model=llm,
            tokenizer=tokenizer,
            selected_layers=selected_layers,
            save_root=seq_train,
            dtype=torch.float16,
        )

        print(f"[train_job] Extracting LLM representations (val set)...", flush=True)
        extract_per_sequence_reps(
            dataloaders=text_dataloaders["val_dataloaders"],
            model=llm,
            tokenizer=tokenizer,
            selected_layers=selected_layers,
            save_root=seq_val,
            dtype=torch.float16,
        )

        # Calibration dialogues for candidate selection (staged by the backend;
        # absent file -> single fit, fully backward compatible).
        calib_path = os.path.join(job_dir, "calibration_input.json")
        calib_entries = []
        if os.path.isfile(calib_path):
            try:
                with open(calib_path) as f:
                    calib_entries = json.load(f) or []
            except Exception as calib_err:
                print(f"[train_job] Could not read calibration_input.json: {calib_err}", flush=True)
        rounds = max(1, int(cfg.get("refinement_rounds", 5)))
        if not calib_entries and rounds > 1:
            print("[train_job] No selection dialogues shipped — training a single candidate", flush=True)
            rounds = 1
        print(f"[train_job] Candidate refinement: rounds={rounds}, selection dialogues={len(calib_entries)}", flush=True)

        # Free LLM memory before RNN training (reloaded afterwards for the
        # candidate-selection pass; the job-local HF cache makes that cheap).
        del llm
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[train_job] LLM freed, starting RNN training", flush=True)

        # ---- Step 5: Create sequence dataloaders ----
        rnn_seq_config = {"RNN_sequence_length": cfg["rnn_sequence_length"]}
        dataloaders_new, class_counts, used_min = create_dataloaders_for_sequences(
            base_directory=job_dir,
            labels=labels,
            batch_size=cfg["batch_size"],
            config=rnn_seq_config,
            seed=42,
            num_workers=4,   # matches reference (gavel scripts/train.py)
        )

        # ---- Step 6: Compute RNN input dimension ----
        if "gemma" in llm_config.model_type:
            n_v_heads = llm_config.text_config.num_key_value_heads
            head_dim = llm_config.text_config.head_dim
        else:
            n_q_heads = llm_config.num_attention_heads
            n_v_heads = llm_config.num_key_value_heads
            head_dim = getattr(llm_config, "head_dim", None) or (llm_config.hidden_size // n_q_heads)
        readout_dim = n_v_heads * head_dim
        num_classes = len(labels)

        print(f"[train_job] RNN: {num_classes} classes, input_dim={readout_dim}, "
              f"hidden={cfg['hidden_dim']}, rnn_layers={cfg['num_rnn_layers']}", flush=True)

        # ---- Step 7: Train candidate RNNs, keep the best calibration-transfer one ----
        def _build_rnn():
            return TopicRNN(
                num_layers=n_layers,
                input_dim=readout_dim,
                hidden_dim=cfg["hidden_dim"],
                num_rnn_layers=cfg["num_rnn_layers"],
                num_topics=num_classes,
                rnn_type="GRU",
            ).to(device)

        checkpoint_dir = os.path.join(job_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        training_log = []

        # training_log.json is synced into the platform UI — entries carry one
        # continuous progress figure, no per-fit/per-epoch structure. The remote
        # stdout keeps the detailed view for debugging.
        def _step_callback(step, total_steps, metrics):
            pct = min(99, int(round(100.0 * step / max(1, total_steps))))
            entry = {
                "progress": pct,
                "train_loss": round(metrics.get("train_loss", 0) or 0, 6),
                "val_loss": round(metrics.get("val_loss", 0) or 0, 6),
                "val_accuracy": round(float(metrics.get("val_accuracy", 0) or 0), 4),
                "learning_rate": metrics.get("learning_rate"),
            }
            training_log.append(entry)
            print(f"[train_job] step {step}/{total_steps} "
                  f"val_loss={metrics.get('val_loss', float('nan')):.4f} "
                  f"val_acc={metrics.get('val_accuracy', float('nan')):.4f}", flush=True)
            with open(os.path.join(job_dir, "training_log.json"), "w") as f:
                json.dump(training_log, f, indent=2)

        print(f"[train_job] Training {rounds} candidate(s) (exact-parity fits)", flush=True)
        candidates = train_rnn_candidates(
            _build_rnn,
            rounds=rounds,
            base_seed=42,
            progress_callback=_step_callback,
            labels_dict=labels,
            train_loader=dataloaders_new["train"],
            val_loader=dataloaders_new["val"],
            epochs=cfg["epochs"],
            train_class_counts=class_counts.get("train", {}) if class_counts else {},
            val_class_counts=class_counts.get("val", {}) if class_counts else {},
            checkpoint_dir=checkpoint_dir,
            learning_rate=cfg["learning_rate"],
            use_wandb=False,
        )

        if len(candidates) > 1 and calib_entries:
            # Reload the LLM (job-local HF cache → fast) and score all candidates
            # in ONE pass over the selection dialogues; keep the model whose
            # weakest CE transfers best (min per-CE ROC-AUC, mean tie-break).
            from classifier_engine.selection import score_candidates_on_calibration, pick_best_candidate
            print(f"[train_job] Reloading LLM for candidate selection", flush=True)
            llm = AutoModelForCausalLM.from_pretrained(
                model_hf_path,
                device_map=device_map,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                trust_remote_code=True,
                output_hidden_states=True,
                attn_implementation="eager",
            )
            llm.eval()

            def _sel_progress(done, total):
                print(f"[train_job] selection scoring {done}/{total}", flush=True)

            scores = score_candidates_on_calibration(
                candidates, llm, tokenizer, calib_entries, labels,
                window_size=cfg["rnn_sequence_length"],
                selected_layers=selected_layers,
                device=device,
                progress_callback=_sel_progress,
            )
            best_idx = pick_best_candidate(scores)
            for c_idx, s in enumerate(scores):
                print(f"[train_job] candidate {c_idx + 1}/{len(candidates)}: "
                      f"min_auc={s.get('min_auc')} mean_auc={s.get('mean_auc')}", flush=True)
            print(f"[train_job] selected candidate {best_idx + 1}/{len(candidates)}", flush=True)
            trained_rnn = candidates[best_idx]
            del llm
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            trained_rnn = candidates[0]
        final_metrics = training_log[-1] if training_log else {}

        # ---- Step 8: Save results ----
        model_path = os.path.join(job_dir, "trained_rnn.pth")
        torch.save(trained_rnn.state_dict(), model_path)
        print(f"[train_job] Saved trained model to {model_path}", flush=True)

        # Match the local trainer's meta schema exactly — the
        # calibration/inference code reads top-level rnn_sequence_length,
        # learning_rate, model_path, and expects selected_layers as a
        # [first, last+1] pair, not the full list.
        sel_list = list(selected_layers)
        meta = {
            "labels": labels,
            "num_classes": num_classes,
            "readout_dim": readout_dim,
            "n_layers": n_layers,
            "hidden_dim": cfg["hidden_dim"],
            "num_rnn_layers": cfg["num_rnn_layers"],
            "rnn_sequence_length": cfg["rnn_sequence_length"],
            "learning_rate": cfg["learning_rate"],
            "model_path": model_hf_path,
            "selected_layers": [sel_list[0], sel_list[-1] + 1],
            "training_config": cfg,
            "rnn_type": "GRU",
            "total_layers": total_layers,
            "final_metrics": final_metrics if final_metrics else {},
            "class_counts": class_counts if class_counts else {},
        }
        with open(os.path.join(job_dir, "classifier_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Final training log
        with open(os.path.join(job_dir, "training_log.json"), "w") as f:
            json.dump(training_log, f, indent=2)

        elapsed = time.time() - t0
        _status(job_dir, "success", elapsed=elapsed)
        print(f"[train_job] Training complete in {elapsed:.1f}s", flush=True)

    except torch.cuda.OutOfMemoryError:
        elapsed = time.time() - t0
        _status(job_dir, "failed", "CUDA out of memory. Try a smaller model or reduce batch_size_text.", elapsed)
        traceback.print_exc()
        sys.exit(2)
    except Exception as e:
        elapsed = time.time() - t0
        _status(job_dir, "failed", str(e), elapsed)
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
