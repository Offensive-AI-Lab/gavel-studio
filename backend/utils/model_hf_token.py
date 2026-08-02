"""Resolve the stored HF access token for a model ref.

Shared by every off-box compute backend (e.g. the remote GPU worker):
those machines have no DB of their own, so the backend resolves a gated/private
base model's token HERE (with DB access) at submit time and ships it inside the
private per-job payload. Returns None for public models / no match / any error,
in which case the loader proceeds anonymously (correct for public models).
"""
from typing import Optional


def resolve_model_hf_token(model_ref: str) -> Optional[str]:
    if not model_ref:
        return None
    try:
        from utils.sqlite_db import execute_query_dict
        rows = execute_query_dict(
            "SELECT hf_token FROM target_models "
            "WHERE storage_path = %s AND hf_token IS NOT NULL LIMIT 1",
            (model_ref,),
        )
        return rows[0]["hf_token"] if rows else None
    except Exception:
        return None
