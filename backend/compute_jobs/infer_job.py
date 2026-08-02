#!/usr/bin/env python3
"""Standalone INFERENCE job for remote GPU workers.

Runs the GPU-heavy part of calibration/evaluation — the target-LLM + RNN
windowed inference — on the worker, and returns the per-window logits. The
light metric/threshold math then runs back on the GAVEL backend.

Crucially it calls the SAME shared core as the local path
(classifier_engine.inference_core.run_inference_core), so the per-window logits
are byte-for-byte identical whether the compute ran remotely or locally.

Like train_job.py: no DB, no FastAPI — everything is files in the job
directory.

Usage (called by the worker harness):
    python infer_job.py --job-dir <jobs_root>/<job_id> --device auto

job_payload.json (uploaded by the backend):
    {
      "model_hf_path": "...",                 # HF model id (== meta.model_path)
      "classifier_meta": { ...meta... },       # classifier_meta.json contents
      "dialogues": [ {"conversation": [...], "metadata": {...}}, ... ],
      "max_length": 256,
      "window_stride": 0,
      "classifier_id": 42
    }
    trained_rnn.pth                            # the trained RNN weights

Outputs (same job_dir):
    logits.json   -> {"results": [ {"logits": [[...]], "metadata": {...}}, ... ]}
    status.json   -> {"status": "success"|"failed", "error": "...", "elapsed_s": N}
"""
import argparse
import json
import os
import sys
import time
import traceback

_code_dir = os.path.expanduser("~/gavel_code")
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

import torch


def _status(job_dir, status, error=None, elapsed=None):
    payload = {"status": status}
    if error:
        payload["error"] = str(error)[:2000]
    if elapsed is not None:
        payload["elapsed_s"] = round(elapsed, 1)
    with open(os.path.join(job_dir, "status.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[infer_job] status={status}" + (f" error={error}" if error else ""), flush=True)


def main():
    parser = argparse.ArgumentParser(description="GAVEL remote inference job")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    job_dir = args.job_dir
    t0 = time.time()

    payload_path = os.path.join(job_dir, "job_payload.json")
    if not os.path.isfile(payload_path):
        _status(job_dir, "failed", f"Missing {payload_path}")
        sys.exit(1)

    with open(payload_path) as f:
        payload = json.load(f)

    # Gated-model auth: the backend resolved target_models.hf_token (DB access the
    # worker lacks) and shipped it here. Export it so the base-model download
    # authenticates — load_model_and_tokenizer passes token=None and falls back to
    # this env var.
    _hf_token = payload.get("hf_token")
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token

    meta = payload.get("classifier_meta") or {}
    dialogues = payload.get("dialogues") or []
    model_hf_path = payload.get("model_hf_path") or meta.get("model_path")
    _ml = payload.get("max_length", None)   # None => NO truncation (matches reference eval)
    max_length = int(_ml) if _ml is not None else None
    window_stride = int(payload.get("window_stride", 0))   # 0 => non-overlapping (resolved to window_size in run_inference_core)
    rnn_path = os.path.join(job_dir, "trained_rnn.pth")

    if not model_hf_path:
        _status(job_dir, "failed", "Missing model_hf_path / meta.model_path")
        sys.exit(1)
    if not os.path.isfile(rnn_path):
        _status(job_dir, "failed", f"Trained RNN not found: {rnn_path}")
        sys.exit(1)
    if not dialogues:
        # Nothing to do — write an empty (but valid) result so the backend can
        # proceed instead of hanging.
        with open(os.path.join(job_dir, "logits.json"), "w") as f:
            json.dump({"results": []}, f)
        _status(job_dir, "success", elapsed=time.time() - t0)
        return

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[infer_job] device={device}, model={model_hf_path}, dialogues={len(dialogues)}", flush=True)

    try:
        from classifier_engine.RNN import TopicRNN
        from classifier_engine.utils_train import load_model_and_tokenizer
        from classifier_engine.inference_core import run_inference_core

        # Rebuild the trained RNN from the geometry stored in meta.
        print("[infer_job] Loading trained RNN...", flush=True)
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

        # Load the target LLM (downloaded from HF on the worker).
        print(f"[infer_job] Loading LLM {model_hf_path}...", flush=True)
        device_map = "auto" if device.type == "cuda" else "cpu"
        llm, tokenizer = load_model_and_tokenizer(model_hf_path, device_map=device_map)

        # The SHARED core — identical to the local path.
        print(f"[infer_job] Running inference on {len(dialogues)} dialogues...", flush=True)
        results = run_inference_core(
            rnn, meta, llm, tokenizer, dialogues, device,
            max_length=max_length, window_stride=window_stride,
        )

        # Serialize: numpy logits -> nested lists.
        out = {
            "results": [
                {"logits": r["logits"].tolist(), "metadata": r.get("metadata", {})}
                for r in results
            ]
        }
        with open(os.path.join(job_dir, "logits.json"), "w") as f:
            json.dump(out, f)

        elapsed = time.time() - t0
        print(f"[infer_job] Done: {len(results)}/{len(dialogues)} dialogues in {elapsed:.0f}s", flush=True)
        _status(job_dir, "success", elapsed=elapsed)

    except Exception as e:
        traceback.print_exc()
        _status(job_dir, "failed", error=str(e), elapsed=time.time() - t0)
        sys.exit(1)


if __name__ == "__main__":
    main()
