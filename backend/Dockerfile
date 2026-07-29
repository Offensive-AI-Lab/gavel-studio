# Python 3.12 — same version CI runs against. The slim variant skips a few
# hundred MB of build tools we don't need at runtime.
FROM python:3.12-slim

# The database is SQLite via Python's stdlib sqlite3 — no client libraries
# needed. build-essential is here because some optional deps
# (sentence-transformers pulls in scientific packages) occasionally fall back
# to source builds.
# curl is here so the healthcheck below can hit /health without extra layers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    openssh-client \
 && rm -rf /var/lib/apt/lists/*
# openssh-client provides ssh/scp — the SLURM "direct" path
# (services/cluster_direct.py) shells out to them to submit training jobs to
# the cluster. With it, GPU training works from inside the container (compute
# runs remotely on the cluster; no local GPU passthrough needed).

WORKDIR /app

# Copy requirements first so docker caches the (slow) pip install layer.
# As long as requirements.txt doesn't change, rebuilds skip pip entirely.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 # Compat pins discovered while wiring up CI — see README §Tests.
 && pip install --no-cache-dir "httpx<0.28" "argon2-cffi==25.1.0"

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
