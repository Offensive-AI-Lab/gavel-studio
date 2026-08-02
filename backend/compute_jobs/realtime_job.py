#!/usr/bin/env python3
"""Long-lived WARM REALTIME inference server for remote GPU workers.

Loads the target LLM + RNN ONCE and then serves realtime classification requests
from the GAVEL backend via files in the session dir — so realtime works on ANY
client PC (Windows / Mac / weak laptops): the heavy LLM forward runs here on the
remote GPU, the client just sends a conversation and renders the result.

No socket needed: the worker harness drops request files into the session dir;
this job polls for them and writes responses back to the same dir.

Session dir layout:
    job_payload.json          # {model_hf_path, classifier_meta}  (uploaded at submit)
    trained_rnn.pth           # the trained RNN weights            (uploaded at submit)
    requests/<id>.json        # backend writes a request
    responses/<id>.json       # this job writes the result (atomic via .tmp+rename)
    keepalive                 # backend `touch`es it on each client ping (idle reset)
    stop                      # backend writes it to ask a clean exit
    ready                     # written once the model is loaded (backend waits on it)
    heartbeat                 # rewritten every loop with a unix ts (liveness probe)
    status.json               # {status: loading|ready|stopped|failed, error?}

Request:  {"id": "...", "mode": "stored"|"live", ...payload}
  stored: {"messages": [{"role","content"}, ...]}
          -> {"per_turn": {"<turn_idx>": {"windows":[...], "tokens":[...]}}}
  live:   {"user_message","system_prompt","history","max_new_tokens"}
          -> {"generated_text": "...", "windows":[...], "tokens":[...]}
Response: {"id","ok":true,"result":{...}} | {"id","ok":false,"error":"..."}
(The job returns RAW per-window/per-token logits; the backend applies the
calibrated thresholds + rule predicates, so recalibration needs no restart.)

Exits on: stop sentinel or idle timeout (no request/keepalive for IDLE_TIMEOUT
s). Each request failure is isolated — one bad dialogue or a CUDA OOM never
kills the server.
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

IDLE_TIMEOUT = int(os.environ.get("GAVEL_RT_IDLE", "900"))   # 15 min with no activity → self-exit
POLL_INTERVAL = float(os.environ.get("GAVEL_RT_POLL", "0.4"))


def main():
    parser = argparse.ArgumentParser(description="GAVEL warm realtime job")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    job_dir = args.job_dir
    req_dir = os.path.join(job_dir, "requests")
    resp_dir = os.path.join(job_dir, "responses")
    os.makedirs(req_dir, exist_ok=True)
    os.makedirs(resp_dir, exist_ok=True)

    hb_path = os.path.join(job_dir, "heartbeat")
    ka_path = os.path.join(job_dir, "keepalive")
    stop_path = os.path.join(job_dir, "stop")

    def _heartbeat():
        try:
            with open(hb_path, "w") as f:
                f.write(str(time.time()))
        except OSError:
            pass

    def _status(status, error=None):
        d = {"status": status}
        if error:
            d["error"] = str(error)[:2000]
        try:
            with open(os.path.join(job_dir, "status.json"), "w") as f:
                json.dump(d, f)
        except OSError:
            pass
        print(f"[realtime_job] status={status}" + (f" error={error}" if error else ""), flush=True)

    def _last_keepalive_ts():
        # The backend `touch`es keepalive on every client ping; its mtime is the
        # client-liveness signal that keeps this job alive while the user is in
        # realtime but idle (reading results).
        try:
            return os.path.getmtime(ka_path)
        except OSError:
            return 0.0

    _status("loading")
    _heartbeat()

    try:
        payload = json.load(open(os.path.join(job_dir, "job_payload.json")))
        # Gated-model auth: the backend resolved target_models.hf_token (DB access
        # the worker lacks) and shipped it here. Export it so the base-model
        # download authenticates — the loader passes no token and falls back to it.
        _hf_token = payload.get("hf_token")
        if _hf_token:
            os.environ["HF_TOKEN"] = _hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
        meta = payload.get("classifier_meta") or {}
        model_ref = payload.get("model_hf_path") or meta.get("model_path")
        rnn_path = os.path.join(job_dir, "trained_rnn.pth")
        if not model_ref or not os.path.isfile(rnn_path):
            _status("failed", "missing model_hf_path or trained_rnn.pth")
            sys.exit(1)

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"[realtime_job] device={device}, model={model_ref}", flush=True)

        from classifier_engine.RNN import TopicRNN
        from classifier_engine.utils_train import load_model_and_tokenizer
        from classifier_engine.realtime_core import (
            classify_conversation_turns, generate_and_classify,
        )

        print("[realtime_job] Loading trained RNN...", flush=True)
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

        print(f"[realtime_job] Loading LLM {model_ref}...", flush=True)
        device_map = "auto" if device.type == "cuda" else "cpu"
        llm, tokenizer = load_model_and_tokenizer(model_ref, device_map=device_map)

        # Align the RNN to the LLM's device (the LLM may be sharded by accelerate).
        try:
            llm_device = next(llm.parameters()).device
            if next(rnn.parameters()).device != llm_device:
                rnn = rnn.to(llm_device)
        except Exception:
            pass

        # Ready — the backend was waiting on this marker.
        with open(os.path.join(job_dir, "ready"), "w") as f:
            f.write("1")
        _status("ready")
        _heartbeat()
        print("[realtime_job] READY — serving requests", flush=True)

        last_activity = time.time()

        def _handle(req):
            mode = req.get("mode", "stored")
            if mode == "live":
                text, windows, tokens = generate_and_classify(
                    user_input=req.get("user_message", ""),
                    system_prompt=req.get("system_prompt", "You are a helpful assistant."),
                    model=llm, tokenizer=tokenizer, classifier=rnn, meta=meta,
                    max_new_tokens=int(req.get("max_new_tokens", 128)),
                    history=req.get("history"),
                )
                return {"generated_text": text, "windows": windows, "tokens": tokens}
            per_turn = classify_conversation_turns(
                messages=req.get("messages") or [], model=llm, tokenizer=tokenizer,
                classifier=rnn, meta=meta,
            )
            return {"per_turn": {str(k): {"windows": w, "tokens": t} for k, (w, t) in per_turn.items()}}

        while True:
            _heartbeat()

            # Clean exit on request.
            if os.path.exists(stop_path):
                print("[realtime_job] stop sentinel — exiting", flush=True)
                break

            # Idle self-termination: no request AND no client keepalive for too long
            # (covers a client that left realtime ungracefully / a dead backend).
            idle = time.time() - max(last_activity, _last_keepalive_ts())
            if idle > IDLE_TIMEOUT:
                print(f"[realtime_job] idle {idle:.0f}s > {IDLE_TIMEOUT}s — exiting", flush=True)
                break

            try:
                pending = sorted(f for f in os.listdir(req_dir) if f.endswith(".json"))
            except OSError:
                pending = []
            if not pending:
                time.sleep(POLL_INTERVAL)
                continue

            for rf in pending:
                rpath = os.path.join(req_dir, rf)
                try:
                    req = json.load(open(rpath))
                except Exception:
                    # Malformed request — write an error response (id from the
                    # filename) so the backend fails fast instead of polling to
                    # the full timeout, then drop the file.
                    rid = rf[:-5]
                    try:
                        _tmp = os.path.join(resp_dir, rid + ".json.tmp")
                        with open(_tmp, "w") as f:
                            json.dump({"id": rid, "ok": False, "error": "malformed request"}, f)
                        os.replace(_tmp, os.path.join(resp_dir, rid + ".json"))
                    except OSError:
                        pass
                    try:
                        os.remove(rpath)
                    except OSError:
                        pass
                    continue
                # Claim the request (process once). NOTE: the backend does NOT
                # retry — if the job dies between here and writing the response the
                # request is lost and the backend times out (its 12s liveness probe
                # usually catches a dead job sooner).
                try:
                    os.remove(rpath)
                except OSError:
                    pass

                rid = req.get("id") or rf[:-5]
                try:
                    result = _handle(req)
                    out = {"id": rid, "ok": True, "result": result}
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    out = {"id": rid, "ok": False, "error": "cuda_oom"}
                    print(f"[realtime_job] OOM on request {rid}", flush=True)
                except Exception as e:
                    traceback.print_exc()
                    out = {"id": rid, "ok": False, "error": str(e)[:500]}

                # Atomic write so the backend never reads a half-written response.
                tmp = os.path.join(resp_dir, rid + ".json.tmp")
                final = os.path.join(resp_dir, rid + ".json")
                try:
                    with open(tmp, "w") as f:
                        json.dump(out, f)
                    os.replace(tmp, final)
                except OSError as e:
                    print(f"[realtime_job] failed to write response {rid}: {e}", flush=True)

                last_activity = time.time()
                _heartbeat()

        _status("stopped")

    except Exception as e:
        traceback.print_exc()
        _status("failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
