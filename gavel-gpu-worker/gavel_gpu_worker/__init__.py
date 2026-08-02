"""gavel-gpu-worker — a self-contained HTTP service that runs GAVEL's GPU
workloads (training, calibration/evaluation inference, and the warm realtime
session) on whatever GPU it's running on (RunPod, AWS, Colab, a rented box…).

It is a thin HTTP + local-subprocess adapter around the EXACT same job scripts
the backend runs locally (compute_jobs/infer_job.py / train_job.py / realtime_job.py),
so the logits are byte-for-byte identical to the local path.

Run it (Docker):  docker run --gpus all -e WORKER_TOKEN=... -p 8000:8000 gavel-gpu-worker
Run it (pip):     pip install gavel-gpu-worker && gavel-gpu-worker
"""
__version__ = "0.1.0"
