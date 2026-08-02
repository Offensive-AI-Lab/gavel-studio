# gavel-gpu-worker — runs GAVEL's GPU workloads over HTTPS on any GPU host.
#
# Build (from the gavel-gpu-worker/ dir, AFTER staging the engine code):
#     python scripts/stage_engine.py
#     docker build -t gavel-gpu-worker .
#
# Run (on a CUDA host):
#     docker run --gpus all -e WORKER_TOKEN=<secret> -p 8000:8000 gavel-gpu-worker
#
# The PyTorch CUDA base image already ships torch + CUDA runtime; we add the
# rest of the engine deps from the backend's pinned requirements so logits match.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    GAVEL_CODE_DIR=/app/gavel_code \
    GAVEL_JOBS_DIR=/app/jobs \
    WORKER_HOST=0.0.0.0 \
    WORKER_PORT=8000

WORKDIR /app

# 1) Engine code (compute_jobs/ + classifier_engine/ + utils/ + evaluation/) + its pins.
COPY gavel_code/ /app/gavel_code/

# 2) Engine ML deps (torch already present in the base; pip skips/satisfies it).
RUN if [ -f /app/gavel_code/engine-requirements.txt ]; then \
        pip install --no-cache-dir -r /app/gavel_code/engine-requirements.txt ; \
    fi

# 3) The worker package + its HTTP deps.
COPY pyproject.toml requirements.txt /app/
COPY gavel_gpu_worker/ /app/gavel_gpu_worker/
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# NB: this serves plain HTTP on :8000 — terminate TLS in front (RunPod proxy, a
# cloudflared/ngrok tunnel, an ALB, or Caddy). The backend rejects non-HTTPS URLs.
CMD ["gavel-gpu-worker"]
