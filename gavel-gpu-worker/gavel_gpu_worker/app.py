"""gavel-gpu-worker HTTP API.

Endpoints (all bearer-authed except /health):
    GET  /health                      liveness + version (UNAUTH)
    GET  /capabilities                accelerator, engine version, what it supports
    POST /infer        (multipart)    spec json + rnn file -> {job_id}
    GET  /infer/{id}                  job status
    GET  /infer/{id}/result           logits.json (when done)
    POST /train        (json)         payload (config + dataset) -> {job_id}
    GET  /train/{id}                  job status (+ progress)
    GET  /train/{id}/model            trained_rnn.pth + classifier_meta.json (zip)
    POST /train/{id}/cancel
    POST /session/start (multipart)   spec json + rnn file -> {session_id}
    GET  /session/{id}/status         queued|loading|ready|dead|stopped
    POST /session/{id}/analyze (json) one realtime request -> result
    POST /session/{id}/keepalive
    POST /session/{id}/end
"""
import gzip
import json

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from . import __version__, config
from .auth import require_token
from .orchestrator import ORCH

app = FastAPI(title="gavel-gpu-worker", version=__version__)


def _accelerator() -> str:
    try:
        import torch
        if config.DEVICE == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _parse_spec(spec: str) -> dict:
    try:
        return json.loads(spec)
    except Exception:
        raise HTTPException(status_code=400, detail="`spec` must be valid JSON.")


async def _read_spec(spec, spec_gz) -> dict:
    """Read the job spec from either a plain `spec` form field (back-compat) or a
    gzipped `spec_gz` file part. The eval spec inlines the whole neutral corpus
    (several MB); gzipping it keeps the multipart request under hosted-proxy body
    limits (e.g. RunPod's ~40 MB), which otherwise truncate it -> a 400/disconnect."""
    if spec_gz is not None:
        try:
            return json.loads(gzip.decompress(await spec_gz.read()))
        except Exception:
            raise HTTPException(status_code=400, detail="`spec_gz` must be gzipped JSON.")
    if spec:
        return _parse_spec(spec)
    raise HTTPException(status_code=400, detail="Missing spec (provide `spec` or `spec_gz`).")


# ---------------------------------------------------------------------------
# health / capabilities
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": __version__,
            "engine_version": config.engine_version(),
            "accelerator": _accelerator()}


@app.get("/capabilities", dependencies=[Depends(require_token)])
def capabilities():
    acc = _accelerator()
    return {
        "version": __version__,
        "engine_version": config.engine_version(),
        "accelerator": acc,
        "supports": ["training", "inference", "realtime"],
        "max_realtime_sessions": 1,
        "active_sessions": ORCH.active_session_count(),
        "detail": f"GPU worker ({acc})",
    }


# ---------------------------------------------------------------------------
# inference (calibration + evaluation)
# ---------------------------------------------------------------------------

@app.post("/infer", dependencies=[Depends(require_token)])
async def infer(rnn: UploadFile = File(...),
                spec: str = Form(None),
                spec_gz: UploadFile = File(None)):
    payload = await _read_spec(spec, spec_gz)
    rnn_bytes = await rnn.read()
    if not rnn_bytes:
        raise HTTPException(status_code=400, detail="Empty rnn file.")
    job_id = ORCH.submit_batch("infer", payload, rnn_bytes)
    return {"job_id": job_id}


@app.get("/infer/{job_id}", dependencies=[Depends(require_token)])
def infer_status(job_id: str):
    st = ORCH.get_batch(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="No such job.")
    return st


@app.get("/infer/{job_id}/result", dependencies=[Depends(require_token)])
def infer_result(job_id: str):
    st = ORCH.get_batch(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="No such job.")
    if st["state"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not done (state={st['state']}).")
    path = ORCH.batch_result_path(job_id)
    if not path:
        raise HTTPException(status_code=500, detail="Result missing.")
    return FileResponse(path, media_type="application/json", filename="logits.json")


@app.delete("/infer/{job_id}", dependencies=[Depends(require_token)])
def infer_cleanup(job_id: str):
    """Delete a finished infer job's scratch dir (the uploaded rnn + the inlined
    eval corpus + logits). The backend calls this right after fetching the result
    so per-run data doesn't pile up on a limited-disk box. Idempotent."""
    ORCH.cleanup_batch(job_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

@app.post("/train", dependencies=[Depends(require_token)])
def train(payload: dict):
    dataset_files = payload.pop("dataset_files", None) or {}
    job_id = ORCH.submit_batch("train", payload, rnn_bytes=None, dataset_files=dataset_files)
    return {"job_id": job_id}


@app.get("/train/{job_id}", dependencies=[Depends(require_token)])
def train_status(job_id: str):
    st = ORCH.get_batch(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="No such job.")
    return st


@app.get("/train/{job_id}/model", dependencies=[Depends(require_token)])
def train_model(job_id: str):
    st = ORCH.get_batch(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="No such job.")
    if st["state"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not done (state={st['state']}).")
    data = ORCH.batch_model_zip(job_id)
    if not data:
        raise HTTPException(status_code=500, detail="Trained model artifacts missing.")
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{job_id}_model.zip"'})


@app.post("/train/{job_id}/cancel", dependencies=[Depends(require_token)])
def train_cancel(job_id: str):
    if not ORCH.cancel_batch(job_id):
        raise HTTPException(status_code=404, detail="No such job.")
    return {"cancelled": True}


@app.delete("/train/{job_id}", dependencies=[Depends(require_token)])
def train_cleanup(job_id: str):
    """Delete a finished train job's scratch dir (dataset + trained artifacts) once
    the backend has pulled the model. Idempotent."""
    ORCH.cleanup_batch(job_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# realtime (warm session)
# ---------------------------------------------------------------------------

@app.post("/session/start", dependencies=[Depends(require_token)])
async def session_start(spec: str = Form(...), rnn: UploadFile = File(...)):
    payload = _parse_spec(spec)
    rnn_bytes = await rnn.read()
    if not rnn_bytes:
        raise HTTPException(status_code=400, detail="Empty rnn file.")
    sid = ORCH.start_session(payload, rnn_bytes)
    if sid is None:
        raise HTTPException(status_code=409, detail="GPU busy — a session or job is already running.")
    return {"session_id": sid}


@app.get("/session/{session_id}/status", dependencies=[Depends(require_token)])
def session_status(session_id: str):
    return {"status": ORCH.session_status(session_id)}


@app.post("/session/{session_id}/analyze", dependencies=[Depends(require_token)])
def session_analyze(session_id: str, req: dict):
    try:
        return {"result": ORCH.analyze(session_id, req)}
    except KeyError:
        raise HTTPException(status_code=404, detail="No such session.")
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Realtime analyze timed out.")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/session/{session_id}/keepalive", dependencies=[Depends(require_token)])
def session_keepalive(session_id: str):
    if not ORCH.keepalive(session_id):
        raise HTTPException(status_code=404, detail="No such session.")
    return {"ok": True}


@app.post("/session/{session_id}/end", dependencies=[Depends(require_token)])
def session_end(session_id: str):
    ORCH.end_session(session_id)
    return {"ended": True}


def main():
    """Console entrypoint (`gavel-gpu-worker`)."""
    import uvicorn
    if not config.WORKER_TOKEN:
        print("[gavel-gpu-worker] WARNING: WORKER_TOKEN is not set — every request "
              "will be rejected with 503 until you set it.", flush=True)
    print(f"[gavel-gpu-worker] v{__version__} | accelerator={_accelerator()} | "
          f"engine={config.engine_version()} | code_dir={config.code_dir()}", flush=True)
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
