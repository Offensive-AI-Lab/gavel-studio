"""Hybrid library search service (SQLite edition).

A single entry point — `HybridSearchService.search()` — handles both Rule and CE
queries (and any future asset type) through the same pipeline. Routes call this
service; they don't write SQL.

Three signals per asset type, fused with Reciprocal Rank Fusion exactly like
the Postgres version did:
  1. Semantic — brute-force cosine over float32-BLOB embeddings (numpy). The
     library is thousands of rows, so a full matrix product is sub-millisecond
     and needs no vector index.
  2. Keyword  — FTS5 MATCH on rules_fts / ces_fts, ranked by bm25 with a 10x
     weight on the name column (replaces setweight A/B + ts_rank_cd).
  3. Name     — pg_trgm-style trigram similarity computed in Python on rows
     whose name contains the query as a substring.

SOLID intent (unchanged):
  * Single responsibility — this module owns the retrieval math. Routes own
    HTTP I/O and validation. Hydration (joining categories/CEs) lives elsewhere.
  * Open/closed — adding a new searchable asset type means adding one entry to
    ASSET_REGISTRY. The service code, fusion math, and routes stay unchanged.
  * Dependency inversion — the service receives a callable embedder, so it can
    be unit-tested without loading SentenceTransformer.
"""
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from utils.sqlite_db import execute_query_dict


# Reciprocal Rank Fusion constant. 60 from Cormack et al. 2009 — robust default.
_RRF_K = 60
# Weight on the trigram name-match bonus. Empirically calibrated so a verbatim
# name hit ranks above the strongest pure-semantic top-1 (1/(60+1) ≈ 0.0164)
# without drowning out keyword fusion.
_NAME_BOOST = 0.5
# bm25 column weights (name, body): name hits outrank body hits, mirroring the
# old setweight('A') / setweight('B') ratio.
_BM25_NAME_WEIGHT = 10.0
_BM25_BODY_WEIGHT = 1.0


@dataclass(frozen=True)
class AssetSpec:
    """Tells `HybridSearchService` how to query one searchable asset type.

    Anything with (id, name, body, embedding, categories), an FTS5 side-table
    and an optional bookmark join table can be plugged in by adding an entry to
    `ASSET_REGISTRY` — no service or route changes required.
    """
    asset_type: str               # tag emitted in result rows ("rule", "ce", ...)
    table: str                    # main table name
    id_col: str                   # primary key column name
    content_col: str              # body column surfaced as `content` in results
    fts_table: str                # FTS5 side-table (rowid == id_col)
    bookmark_table: Optional[str] = None  # for "search my bookmarks" — same id_col


# Open/closed extension point. To add a new searchable asset type, append here.
ASSET_REGISTRY: dict = {
    "rule": AssetSpec(
        asset_type="rule",
        table="rules",
        id_col="rule_id",
        content_col="predicate",
        fts_table="rules_fts",
        bookmark_table="rule_bookmarks",
    ),
    "ce": AssetSpec(
        asset_type="ce",
        table="cognitive_elements",
        id_col="ce_id",
        content_col="definition",
        fts_table="ces_fts",
        bookmark_table="ce_bookmarks",
    ),
}


# --------------------------------------------------------------------- helpers

def _trigrams(text: str) -> set:
    """pg_trgm-style trigram set: lowercase, split into words, pad each word
    with two leading and one trailing space, take all 3-grams."""
    grams = set()
    for word in re.findall(r"\w+", text.lower()):
        padded = f"  {word} "
        for i in range(len(padded) - 2):
            grams.add(padded[i:i + 3])
    return grams


def trigram_similarity(a: str, b: str) -> float:
    """|A ∩ B| / |A ∪ B| over trigram sets — same formula as pg_trgm's
    similarity(). Not bit-identical to Postgres, but the same shape and range."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _fts_match_expr(query_text: str) -> str:
    """Build a safe FTS5 MATCH expression: each token double-quoted (so FTS5
    operator characters in user input can't break the query), implicit AND
    between tokens — the same semantics websearch_to_tsquery gave for plain
    word queries."""
    tokens = re.findall(r"\w+", query_text)
    return " ".join(f'"{t}"' for t in tokens)


class HybridSearchService:
    """RRF-fused hybrid search across one or more asset types.

    Each signal contributes its top-N candidates, ranked. RRF combines them via
    1 / (k + rank). The name signal is added as an additive bonus to catch
    short / acronym queries that keyword search under-ranks.

    The service is stateless; it can be a long-lived singleton or constructed
    per request — both are fine.
    """

    def __init__(self, embedder: Callable[[str], List[float]]):
        # Inject the embedding function so tests can substitute a fake.
        self._embed = embedder

    # ------------------------------------------------------------------ public API

    def search(
        self,
        *,
        query_text: str,
        asset_types: Sequence[str],
        category_ids: Optional[Sequence[int]] = None,
        bookmark_user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Return ranked candidate rows. Each row carries id, asset_type, name,
        content, type, categories, final_score. Empty list on no hits or empty
        query."""
        query_text = (query_text or "").strip()
        if not query_text:
            return []

        specs = [ASSET_REGISTRY[t] for t in asset_types if t in ASSET_REGISTRY]
        if not specs:
            return []

        query_vector = self._embed(query_text)
        if not query_vector:
            return []

        wanted_categories = set(int(c) for c in (category_ids or []))

        results: List[dict] = []
        for spec in specs:
            results.extend(
                self._search_one(
                    spec=spec,
                    query_text=query_text,
                    query_vector=query_vector,
                    wanted_categories=wanted_categories,
                    bookmark_user_id=bookmark_user_id,
                    limit=limit,
                )
            )

        results.sort(key=lambda r: r["final_score"], reverse=True)
        return results[:limit]

    # ------------------------------------------------------------ per-asset search

    def _search_one(
        self,
        *,
        spec: AssetSpec,
        query_text: str,
        query_vector: List[float],
        wanted_categories: set,
        bookmark_user_id: Optional[int],
        limit: int,
    ) -> List[dict]:
        candidates = self._fetch_candidates(spec, bookmark_user_id)
        if wanted_categories:
            candidates = [
                c for c in candidates
                if wanted_categories & set(c.get("categories") or [])
            ]
        if not candidates:
            return []

        by_id = {c["id"]: c for c in candidates}
        scores = {cid: 0.0 for cid in by_id}

        # 1) Semantic — normalized embeddings, so cosine == dot product.
        for rank, cid in enumerate(
            self._semantic_ranking(candidates, query_vector, limit), start=1
        ):
            scores[cid] += 1.0 / (_RRF_K + rank)

        # 2) Keyword — FTS5 bm25 (lower is better; ORDER BY score ASC).
        for rank, cid in enumerate(
            self._keyword_ranking(spec, query_text, by_id, limit), start=1
        ):
            scores[cid] += 1.0 / (_RRF_K + rank)

        # 3) Name — trigram-similarity bonus on substring name matches.
        q_lower = query_text.lower()
        for c in candidates:
            name = c.get("name") or ""
            if q_lower in name.lower():
                scores[c["id"]] += trigram_similarity(name, query_text) * _NAME_BOOST

        out = []
        for cid, score in scores.items():
            if score <= 0.0:
                continue
            row = dict(by_id[cid])
            row.pop("embedding", None)
            row["asset_type"] = spec.asset_type
            row["final_score"] = score
            out.append(row)
        out.sort(key=lambda r: r["final_score"], reverse=True)
        return out[:limit]

    def _fetch_candidates(self, spec: AssetSpec, bookmark_user_id: Optional[int]) -> List[dict]:
        join_clause = ""
        params: tuple = ()
        if bookmark_user_id is not None and spec.bookmark_table:
            join_clause = (
                f"JOIN {spec.bookmark_table} bk "
                f"ON x.{spec.id_col} = bk.{spec.id_col} AND bk.user_id = %s"
            )
            params = (int(bookmark_user_id),)
        sql = f"""
            SELECT x.{spec.id_col} AS id, x.name, x.{spec.content_col} AS content,
                   x.type, x.categories, x.is_local_draft, x.created_by_username,
                   x.public_id, x.embedding
            FROM {spec.table} x
            {join_clause}
        """
        return execute_query_dict(sql, params or None) or []

    @staticmethod
    def _semantic_ranking(candidates: List[dict], query_vector: List[float], limit: int) -> List[int]:
        import numpy as np
        from utils.embedding_utils import blob_to_embedding

        ids, vectors = [], []
        dim = len(query_vector)
        for c in candidates:
            blob = c.get("embedding")
            if not blob:
                continue
            vec = blob_to_embedding(blob)
            if vec.shape[0] != dim:
                continue  # stale row embedded with a different model
            ids.append(c["id"])
            vectors.append(vec)
        if not ids:
            return []
        matrix = np.stack(vectors)
        sims = matrix @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-sims)[:limit]
        return [ids[i] for i in order]

    @staticmethod
    def _keyword_ranking(spec: AssetSpec, query_text: str, by_id: dict, limit: int) -> List[int]:
        match_expr = _fts_match_expr(query_text)
        if not match_expr:
            return []
        try:
            # No SQL LIMIT here: the candidate scoping (bookmarks / categories)
            # was applied to `by_id`, and ranks must be computed WITHIN that
            # scoped set — capping the global FTS result first would silently
            # drop scoped rows that rank below the global top-N (the Postgres
            # version filtered inside the keyword CTE before its LIMIT). The
            # match set is bounded by the library size, so this stays cheap.
            rows = execute_query_dict(
                f"""
                SELECT rowid AS id,
                       bm25({spec.fts_table}, {_BM25_NAME_WEIGHT}, {_BM25_BODY_WEIGHT}) AS score
                FROM {spec.fts_table}
                WHERE {spec.fts_table} MATCH %s
                ORDER BY score ASC
                """,
                (match_expr,),
            ) or []
        except Exception:
            # An exotic query FTS5 refuses to parse should degrade to "no
            # keyword signal", not a 500 — semantic + name still rank.
            return []
        scoped = [r["id"] for r in rows if r["id"] in by_id]
        return scoped[:limit]
