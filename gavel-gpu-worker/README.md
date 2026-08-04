# gavel-gpu-worker

A small server you run on a machine with an NVIDIA GPU — a rented cloud GPU, a lab box, anything with a card in it. Your GAVEL Studio backend sends its heavy work to that machine over HTTPS: training, calibration, evaluation, and live monitoring. The results come back to Studio.

Use it when the machine running Studio has no GPU of its own. There is no SSH involved. You run one command on the GPU machine, then paste two lines into Studio's configuration.

See the [repository README](../README.md) for what GAVEL Studio is and how to run it.

## Run it

On the GPU machine you need Python 3.10 or newer, pip, and a public HTTPS address for port 8000. Then:

```sh
git clone https://github.com/Offensive-AI-Lab/gavel-studio.git
cd gavel-studio/gavel-gpu-worker
bash setup_worker.sh --url https://<your-public-https-address>
```

The first run installs PyTorch and the rest of the machine-learning packages, which takes a few minutes. When it finishes it prints two lines:

```ini
GPU_WORKER_URL=https://<your-public-https-address>
GPU_WORKER_TOKEN=<a generated secret>
```

Put both lines in the backend's `backend/.env` on the machine running Studio, and restart the backend. That is the whole configuration. Training, calibration, evaluation, and live monitoring now run on the GPU machine.

Re-running `bash setup_worker.sh` is safe: it restarts the worker and skips anything already installed. Add `--token "$GPU_WORKER_TOKEN"` to keep the same secret so your `backend/.env` stays valid. `bash setup_worker.sh --help` lists the other options (`--port`, `--foreground`, `--no-start`).

### On RunPod

Deploy a pod with a GPU and a PyTorch template. In the deploy screen, under **Expose HTTP Ports**, add `8000`. RunPod then serves that port at `https://<POD_ID>-8000.proxy.runpod.net` — the pod's **Connect** button shows the exact address. Use it as `--url` in the command above.

### HTTPS is required

The worker itself serves plain HTTP on port 8000. Something must put HTTPS in front of it. Most cloud GPU providers do this for you; otherwise use a tunnel such as `cloudflared`, or a reverse proxy. The backend refuses an address that is not `https://` (localhost aside), because the secret travels with every request.

### Without the setup script

Stage the engine code first, then install and start the worker:

```sh
python scripts/stage_engine.py
pip install -r gavel_code/engine-requirements.txt
pip install .
WORKER_TOKEN=<a long secret you choose> gavel-gpu-worker
```

Or with Docker:

```sh
python scripts/stage_engine.py
docker build -t gavel-gpu-worker .
docker run --gpus all -e WORKER_TOKEN=<a long secret you choose> -p 8000:8000 gavel-gpu-worker
```

Either way, put that same secret in the backend's `GPU_WORKER_TOKEN`.

## Settings

Only `WORKER_TOKEN` is required. All of these are environment variables on the GPU machine.

| Variable | Default | What it does |
|---|---|---|
| `WORKER_TOKEN` | — (**required**) | The secret the backend must send with every request. Use the same value as the backend's `GPU_WORKER_TOKEN`. |
| `WORKER_PORT` | `8000` | Port the worker listens on. |
| `WORKER_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. |
| `GAVEL_HF_CACHE` | unset | Folder for downloaded models. Set it to a mounted volume under Docker so the cache survives restarts. |
| `GAVEL_JOBS_DIR` | `~/gavel_worker_jobs` | Scratch space for each job. Cleared after every run. |

## Check it is working

```sh
curl http://localhost:8000/health
```

It reports the version and whether it found a GPU (`accelerator` is `cuda` or `cpu`).

If you started the worker with `setup_worker.sh`, its output is in `~/gavel_worker.log` and you can stop it with `kill $(cat ~/.gavel_worker.pid)`.

## Test it

From the `gavel-gpu-worker` folder, with the worker installed and `pytest` available:

```sh
pip install pytest
python -m pytest tests/
```

## Good to know

* **Keep it in step with Studio.** After the backend is updated, re-run `python scripts/stage_engine.py` on the GPU machine (and rebuild the Docker image if you use Docker). The backend checks the versions match and will not use a worker that is out of step.
* **One job at a time.** A single GPU runs one heavy task at once. Jobs queue, and a live monitoring session holds the GPU until it ends.
* **Scratch is deleted, models are kept.** Everything a job writes is removed once Studio has the result. A model downloaded from Hugging Face stays cached and is reused by later jobs. To free the disk and stop paying, shut the machine down.
* **Models loaded from a file on your own machine stay there.** Those jobs run on the Studio machine instead. The worker only handles models it can download itself.
* **If the worker is unreachable, Studio runs the work locally.**
