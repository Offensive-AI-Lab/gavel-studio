# gavel-gpu-worker

A small server you run **on a GPU machine** (a rented cloud GPU, a lab box, anything with an NVIDIA card) so your GAVEL backend can run its GPU work there — **training, calibration/evaluation, and the realtime session** — over HTTPS.

It's the second way to get a GPU, alongside *your own machine*. Use it when you don't have a local GPU. No SSH — just an HTTPS URL and a token.

```
GAVEL backend ──HTTPS──► gavel-gpu-worker ──runs──► compute_jobs/*_job.py ──► the GPU
```

Under the hood it runs the **exact same job scripts** as the local path, so results are identical no matter where the GPU is.

---

## Setup

### RunPod (quickstart)

1. **Deploy a Pod** with a GPU and a PyTorch template. In the deploy screen, under **Expose HTTP Ports**, add **`8000`** (this is the port the worker listens on — RunPod gives it a public HTTPS URL through its proxy).
2. **Find your Pod's URL.** RunPod proxies port 8000 at:

   ```
   https://<POD_ID>-8000.proxy.runpod.net
   ```

   Replace `<POD_ID>` with your Pod's ID (the **Connect** button on the Pod shows the exact URL for port 8000). This is the only piece that's unique to you — everything else below is copy-paste.

3. **Open the Pod's web terminal** and run, in this order:

   ```bash
   cd /workspace
   git clone https://github.com/Offensive-AI-Lab/gavel-studio.git
   cd gavel-studio/gavel-gpu-worker
   bash setup_worker.sh --url https://<POD_ID>-8000.proxy.runpod.net
   ```

   The first run installs the ML stack (torch etc. — a few minutes) and downloads nothing else until you actually train. When it finishes it prints two lines:

   ```ini
   GPU_WORKER_URL=https://<POD_ID>-8000.proxy.runpod.net
   GPU_WORKER_TOKEN=<generated token>
   ```

4. **Paste those two lines into the backend's `backend/.env`** and restart the backend. Done — training, calibration, evaluation, and realtime now run on the Pod.

> **Re-running is safe and won't re-download.** `bash setup_worker.sh` restarts the worker cleanly; `pip` skips packages that are already installed, and the model cache is reused — so a second run does **not** re-download torch or the LLM. To keep the **same token** across re-runs (so your `.env` stays valid), pass it back: `bash setup_worker.sh --token "$GPU_WORKER_TOKEN" --url https://<POD_ID>-8000.proxy.runpod.net`.

> Before the quickstart, the repo must be present on the box (the worker copies the backend's ML code from it). `bash setup_worker.sh --help` lists all options.

### Easiest — one command (any provider)

From inside `gavel-studio/gavel-gpu-worker`:

```bash
bash setup_worker.sh
```

It does everything: stages the ML code, installs the dependencies, generates a `WORKER_TOKEN`, starts the worker in the background, and prints the two lines to paste into your backend's `backend/.env`. Safe to re-run (it restarts the worker cleanly, no re-download). Pass your provider's HTTPS URL with `--url https://…` to have it printed for you.

### Or do it manually

First **stage the engine code** (copies the backend's ML code next to the worker so it produces identical results):

```bash
python scripts/stage_engine.py
```

**Option A — no Docker** (e.g. a cloud GPU pod, Colab):

```bash
pip install -r gavel_code/engine-requirements.txt   # the ML stack (torch, transformers…)
pip install .                                        # the worker itself
export WORKER_TOKEN=<a long secret you choose>       # e.g. openssl rand -hex 24
gavel-gpu-worker                                      # serves on :8000
```

**Option B — Docker:**

```bash
docker build -t gavel-gpu-worker .
docker run --gpus all -e WORKER_TOKEN=<a long secret you choose> -p 8000:8000 gavel-gpu-worker
```

> **HTTPS is required.** The worker serves plain HTTP on `:8000` — put HTTPS in front of it (most cloud GPU providers give you a public HTTPS URL automatically; otherwise use a tunnel like `cloudflared`, an ALB, or Caddy). The backend **refuses** a worker URL that isn't `https://`, because the token travels in the request.

---

## Point your backend at it

This is all the configuration there is. In the **backend's** `backend/.env`:

```ini
GPU_WORKER_URL=https://<your-worker-host>
GPU_WORKER_TOKEN=<the WORKER_TOKEN you set above>
```

Restart the backend. That's it — training, calibration, evaluation, and realtime now run on this GPU. The backend auto-detects the worker from `GPU_WORKER_URL`; you don't set anything else. (If the worker is ever unreachable, the backend automatically falls back to running locally.)

---

## Worker settings (env vars)

Only `WORKER_TOKEN` is required. The rest have sensible defaults.

| Variable | Default | What it does |
|---|---|---|
| `WORKER_TOKEN` | — (**required**) | The secret the backend must send on every request. Pick anything long; put the same value in the backend's `GPU_WORKER_TOKEN`. |
| `GAVEL_HF_CACHE` | unset | Folder to cache downloaded models in. The model downloads **once** and is reused by every job for the Pod's lifetime (see *Data & disk* below). |
| `WORKER_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu`. |
| `WORKER_PORT` | `8000` | Port to listen on. |
| `GAVEL_JOBS_DIR` | `~/gavel_worker_jobs` | Scratch space for each job. **Deleted after each run** (the backend deletes it once it has the result). |
| `GAVEL_JOB_RETENTION` | `1800` | Safety-net seconds before the sweeper purges a leftover job dir (only matters if a client died mid-run or the worker restarted). |
| `GAVEL_CODE_DIR` | auto | Where the staged engine code lives. Leave it alone unless you staged it somewhere custom. |

---

## Data & disk (what's stored, what's deleted)

The worker keeps the box clean:

- **Per-job data is deleted after each run.** Everything a train/infer/realtime job writes to `~/gavel_worker_jobs/<job>` — the uploaded classifier weights, the inlined training/eval datasets, the trained model, the logits — is removed the moment the backend has fetched the result (and a sweeper purges anything left behind by a crash or restart). Nothing accumulates run-to-run.
- **The downloaded LLM stays cached for the Pod's lifetime.** A model from Hugging Face is downloaded **once** and reused by every job after that. It is *not* re-downloaded per run, and it is *not* deleted between runs — because re-downloading multi-GB weights on every run would waste the GPU time you're paying for by the hour. It lives only on the Pod's ephemeral disk.
- **User-uploaded (local-file) models never reach the worker.** If a classifier's model is a file on your own machine rather than an HF repo, that job runs **locally** instead — the worker only handles HF models it can download itself.

**To free the disk and stop paying: terminate the Pod.** The Pod's disk is ephemeral, so destroying it removes the model cache and all scratch in one go (and stops billing). That's the intended "save money" path — keep the model cached while you're actively using the Pod, then tear the Pod down when you're done. If you're on a *very* small volume and want the model gone between runs too, point `GAVEL_HF_CACHE` at a disposable path and clear it yourself (accepting a re-download on the next run).

---

## Keep it in sync with the backend

The worker and backend must run the **same** engine code, or the backend refuses to use the worker (so you can never silently get different results). `GET /health` reports an `engine_version` hash; the backend compares it to its own.

**Whenever the backend's `classifier_engine` is updated, re-run `python scripts/stage_engine.py`** (and rebuild the Docker image if you use Option B).

## Good to know

- **One job at a time.** A single GPU runs one heavy thing at once — jobs queue, and an active realtime session holds the GPU until it ends. A second concurrent request gets `409 GPU busy`.
- **Quick check it's alive:** `curl http://localhost:8000/health` → returns status, version, and `accelerator` (`cuda` or `cpu`).
