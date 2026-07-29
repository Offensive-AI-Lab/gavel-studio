import threading
from functools import lru_cache
from typing import List, Tuple

from utils.sqlite_db import execute_query

# Configuration mimicking process_library.py
FUNCTIONAL_KEYWORDS = ["detect", "classify", "score", "probability", "risk", "evaluate", "flag", "monitor"]
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Embeddings are stored as raw float32 bytes in a BLOB column (384 dims for
# MiniLM). services/library_search.py reads them back with numpy.frombuffer
# and does brute-force cosine — at library scale (thousands of rows) that is
# a sub-millisecond matrix product, so no vector index is needed.
EMBEDDING_DTYPE = "float32"


def embedding_to_blob(embedding) -> bytes:
    import numpy as np
    return np.asarray(embedding, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes):
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32)


class EmbeddingManager:
    _instance = None
    # Lock guards singleton construction against the library-sync
    # ThreadPoolExecutor (8 parallel workers). Without it, multiple
    # threads concurrently see _instance is None and each fire their
    # own SentenceTransformer init — and SentenceTransformer's load
    # path is NOT reentrant on MPS, so racers fail with the meta-tensor
    # error.
    _lock = threading.Lock()

    def __init__(self):
        # Lazy import — sentence_transformers + transformers + torch is ~3-5s of import time.
        # Keeping this inside __init__ means startup of unrelated routes stays fast.
        print("[*] Loading Embedding Models... (this runs once)")
        from sentence_transformers import SentenceTransformer
        from utils.device import get_torch_device

        # Pass device explicitly. Without this, current transformers releases
        # init the model on the `meta` device and SentenceTransformer's
        # internal `.to(device)` blows up with
        #   "Cannot copy out of meta tensor; no data!"
        # — which is what was breaking CI on /library/search. Passing device
        # to the constructor takes the safe loading path.
        # Auto-detect through the centralized helper: CUDA > MPS (Apple
        # Silicon) > CPU. So a Mac dev box uses the Metal backend, a Linux
        # GPU box uses CUDA, and a CI / CPU-only box silently falls back.
        device = str(get_torch_device())
        # transformers >= 4.50 defaults to low_cpu_mem_usage=True, which puts
        # weights on the `meta` device. SentenceTransformer's internal
        # `.to(device)` then fails with "Cannot copy out of meta tensor; no
        # data!" — most visibly when device is MPS (Apple Silicon).
        # Force the eager loading path so weights have real data from the start.
        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=device,
            model_kwargs={"low_cpu_mem_usage": False},
        )
        print(f"[*] Embedding Models Loaded on {device}.")

    @classmethod
    def get_instance(cls):
        # Double-checked locking: fast path returns immediately once the
        # singleton exists, slow path acquires the lock only on first
        # call. After a winning thread completes __init__, every other
        # thread that was waiting on the lock sees a non-None _instance
        # and skips the constructor entirely.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _infer_type(self, asset_kind: str, text: str) -> str:
        lowered = text.lower()
        for token in FUNCTIONAL_KEYWORDS:
            if token in lowered:
                return "detector"
        return "rule" if asset_kind == "rule" else "detector"

    def embed_and_update(self, asset_kind: str, asset_id: int, name: str, body: str, ce_definitions: str = ""):
        """
        Calculates embedding, infers type, and updates the database record.
        asset_kind: 'ce' or 'rule'
        body: definition for CE, predicate for Rule (or description)

        Full-text indexing needs no work here anymore: the FTS5 side-tables
        (rules_fts / ces_fts) are maintained by triggers on the base tables,
        so they were already updated when the row itself was written.
        """
        try:
            if asset_kind == "ce":
                text = body.strip() or name.strip()
            else:
                # For Rules: Name + Predicate + Semantic Definitions
                text = f"{name}. {body}. Concepts: {ce_definitions}".strip()

            # Infer type
            functional_type = self._infer_type(asset_kind, text)

            # Calculate embedding (normalized, so search can use a dot product)
            embedding = self.embedder.encode(text, normalize_embeddings=True).tolist()

            # Update Database
            target_table = "rules" if asset_kind == "rule" else "cognitive_elements"
            id_field = "rule_id" if asset_kind == "rule" else "ce_id"

            query = f"""
                UPDATE {target_table}
                SET embedding = %s,
                    type = %s
                WHERE {id_field} = %s
            """

            execute_query(query, (embedding_to_blob(embedding), functional_type, asset_id))
            print(f"[✓] Auto-embedded & Indexed {asset_kind} '{name}' (ID: {asset_id})")

        except Exception as e:
            print(f"[!] Failed to auto-embed {asset_kind} '{name}': {e}")
            import traceback
            traceback.print_exc()

    def embed_text(self, text: str) -> List[float]:
        """Return normalized embedding for arbitrary text (uncached — used at write time)."""
        return self.embedder.encode(text.strip(), normalize_embeddings=True).tolist()


@lru_cache(maxsize=1024)
def _embed_query_cached(text: str) -> Tuple[float, ...]:
    """LRU-cached embedding for query-time use. Cache key is the trimmed lowercased text;
    SentenceTransformer encode is ~30-100ms, repeated queries (pagination, re-runs, common
    autocomplete prefixes) hit cache and return in microseconds."""
    manager = EmbeddingManager.get_instance()
    return tuple(manager.embedder.encode(text, normalize_embeddings=True).tolist())


def embed_query(text: str) -> List[float]:
    """Cached entry-point for query-side embeddings. Use this from search routes."""
    if not text:
        return []
    return list(_embed_query_cached(text.strip().lower()))


def trigger_embedding(asset_kind: str, asset_id: int, name: str, body: str, ce_definitions: str = ""):
    """Helper function to trigger embedding update safely."""
    try:
        manager = EmbeddingManager.get_instance()
        manager.embed_and_update(asset_kind, asset_id, name, body, ce_definitions)
    except Exception as e:
        # Surface the full traceback so genuine failures (network, OOM,
        # disk full, etc.) aren't reduced to a one-line shrug. The
        # singleton race that produced the noisy "meta tensor" errors
        # used to live here is now fixed in EmbeddingManager.get_instance().
        import traceback
        print(f"[!] Error triggering embedding for {asset_kind} '{name}' (id={asset_id}): {e}")
        traceback.print_exc()
