# Python 3.12. The slim variant skips a few hundred MB of build tools we
# don't need at runtime.
FROM python:3.12-slim

# The database is SQLite via Python's stdlib sqlite3 — no client libraries
# needed. build-essential is here because some optional deps
# (sentence-transformers pulls in scientific packages) occasionally fall back
# to source builds.
# curl is here so the healthcheck below can hit /health without extra layers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch wheel source. Default = the CPU-only index: on linux/amd64 the plain
# PyPI `torch` is the CUDA build (~5 GB with its nvidia-* deps), which the
# default CPU-training compose never uses. docker-compose.gpu.yml overrides
# this back to PyPI so local-GPU builds get the CUDA wheels.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# Copy requirements first so docker caches the (slow) pip install layer.
# As long as requirements.txt doesn't change, rebuilds skip pip entirely.
# torch is installed first from TORCH_INDEX so the requirements.txt line
# (torch>=2.0.0) is already satisfied and pip doesn't pull it from PyPI.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend. In dev, this gets shadowed by the volume
# mount in docker-compose.yml, but the COPY here keeps the image self-
# contained so it can run without compose (e.g. for prod deploys later).
COPY . .

EXPOSE 8000

# Probe /health so `depends_on: condition: service_healthy` works for
# anything wanting to wait until the backend is actually serving.
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
  CMD curl --fail http://localhost:8000/health || exit 1

# The default command is overridden in docker-compose.yml (with --reload).
# This CMD is the production-style fallback for `docker run` outside compose.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
