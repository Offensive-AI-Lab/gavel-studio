"""Stage the GAVEL engine code into ./gavel_code so the worker (and the Docker
build) can find it. Copies the shared compute job scripts (backend/compute_jobs/)
+ backend/classifier_engine/ + backend/utils/ + backend/evaluation/ — plus the
backend requirements so the image installs the SAME ML stack as the main backend
(identical logits).

Run from the gavel-gpu-worker/ dir:
    python scripts/stage_engine.py          # finds the repo automatically
"""
import argparse
import shutil
import sys
from pathlib import Path

# gavel-gpu-worker/ lives INSIDE the repo, so the repo root is two levels up from
# this script (scripts/ -> gavel-gpu-worker/ -> <repo>). Computed from __file__ so
# it works no matter what directory you run from.
_DEFAULT_REPO = str(Path(__file__).resolve().parents[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=_DEFAULT_REPO,
                    help="Path to the gavel-studio repo (default: the repo this worker lives in).")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(__file__).resolve().parent.parent / "gavel_code"
    srcs = {
        repo / "backend" / "compute_jobs": out / "compute_jobs",
        repo / "backend" / "classifier_engine": out / "classifier_engine",
        repo / "backend" / "utils": out / "utils",
        repo / "backend" / "evaluation": out / "evaluation",
    }
    for src in srcs:
        if not src.is_dir():
            sys.exit(f"[abort] not found: {src} — is --repo correct?")

    out.mkdir(parents=True, exist_ok=True)
    for src, dst in srcs.items():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache", "reference/**/__pycache__"))
        print(f"[ok] {src}  ->  {dst}")

    # Carry the backend's ML requirements so the image pins identical versions.
    req = repo / "backend" / "requirements.txt"
    if req.is_file():
        shutil.copy2(req, out / "engine-requirements.txt")
        print(f"[ok] {req}  ->  {out / 'engine-requirements.txt'}")

    print(f"\nStaged engine code at: {out}")
    print(f"engine version: see `python -c \"import gavel_gpu_worker.config as c; print(c.engine_version())\"`")


if __name__ == "__main__":
    main()
