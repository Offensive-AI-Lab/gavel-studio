import asyncio
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.sqlite_db import execute_query, execute_query_dict
from utils.auth import get_current_user
from utils.embedding_utils import embed_query
from services.library_search import HybridSearchService

router = APIRouter()

# One service instance for the whole route — the service is stateless and the
# embedder it wraps is itself an LRU-cached function, so this is safe and saves
# us from re-wiring dependencies on every request.
_search_service = HybridSearchService(embedder=embed_query)



def _normalize_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]



class SearchResult(BaseModel):
    id: int
    asset_type: str
    name: str
    content: Optional[str]
    ces: List[str] = []
    # Group-aware CE list for rule results. Each entry is
    # {ce_id, name, groups: [group_name, ...]} — the logic group(s) the CE
    # belongs to in the rule's ce_groups. Empty for CE results.
    # The bare `ces` list above stays for back-compat — RuleCard
    # prefers active_ces when present and falls back to ces otherwise.
    active_ces: List[dict] = []
    # The rule's v2 firing condition (grammar over its group names);
    # None for CE results.
    condition: Optional[str] = None
    categories: List[str] = []
    type: Optional[str] = None
    hybrid_score: float
    rerank_score: Optional[float] = None
    is_local_draft: Optional[bool] = None
    examples: List[dict] = []
    # Phase 2 attribution. Populated for every artifact whose creator is
    # known; absent on legacy rows that predate the artist feature and
    # haven't been backfilled. The "by [user]" link on RuleCard /
    # CognitiveElementCard checks for truthiness before rendering.
    created_by_username: Optional[str] = None
    # Phase 3 ratings: cards use public_id to call the /ratings/* API.
    # NULL on drafts; the StarRating widget no-ops when missing.
    public_id: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    candidates_examined: int
    total_results: int = 0
    page: int = 1
    page_size: int = 10



def _hydrate_results(paged_candidates: List[dict]) -> List[SearchResult]:
    """Helper to fetch category names, CE links, and format results."""
    # 1. Collect IDs
    rule_ids = [c["id"] for c in paged_candidates if c["asset_type"] == "rule"]
    all_cat_ids = set()
    for c in paged_candidates:
        if c.get("categories"):
            all_cat_ids.update(c["categories"])
    
    # 2. Batch Fetch Categories
    cat_map = {}
    if all_cat_ids:
        c_rows = execute_query_dict("SELECT category_id, name FROM categories WHERE category_id = ANY(%s)", (list(all_cat_ids),)) or []
        cat_map = {r["category_id"]: r["name"] for r in c_rows}
        
    # 3. Batch Fetch CEs + logic for Rules. Each active_ces entry carries
    # the logic group name(s) the CE belongs to (from rules.ce_groups) so
    # the frontend RuleCard can badge members by group; the rule's firing
    # condition rides along per row. We still expose the flat name list as
    # `ces` for any caller that only needs names.
    rule_ce_map = {}          # rule_id -> [name, ...]
    rule_active_ces_map = {}  # rule_id -> [{ce_id, name, groups: [gname]}, ...]
    rule_condition_map = {}   # rule_id -> condition string
    if rule_ids:
        logic_rows = execute_query_dict(
            "SELECT rule_id, ce_groups, condition FROM rules WHERE rule_id = ANY(%s)",
            (rule_ids,),
        ) or []
        groups_of_by_rule = {}  # rule_id -> {ce_name: [gname, ...]}
        for r in logic_rows:
            rule_condition_map[r["rule_id"]] = r.get("condition")
            groups_of: dict = {}
            for gname, members in (r.get("ce_groups") or {}).items():
                for member in members or []:
                    groups_of.setdefault(member, []).append(gname)
            groups_of_by_rule[r["rule_id"]] = groups_of

        ce_sql = """
        SELECT rcl.rule_id, rcl.ce_id, ce.name
        FROM rule_ce_link rcl
        JOIN cognitive_elements ce ON rcl.ce_id = ce.ce_id
        WHERE rcl.rule_id = ANY(%s)
        ORDER BY rcl.rule_id, ce.name
        """
        ce_rows = execute_query_dict(ce_sql, (rule_ids,)) or []
        from collections import defaultdict
        name_map = defaultdict(list)
        active_map = defaultdict(list)
        for r in ce_rows:
            name_map[r["rule_id"]].append(r["name"])
            active_map[r["rule_id"]].append({
                "ce_id": r["ce_id"],
                "name": r["name"],
                "groups": groups_of_by_rule.get(r["rule_id"], {}).get(r["name"], []),
            })
        rule_ce_map = name_map
        rule_active_ces_map = active_map

    # 4. Batch fetch examples for CEs
    ce_ids = [c["id"] for c in paged_candidates if c["asset_type"] == "ce"]
    ce_examples_map = {}
    if ce_ids:
        ex_rows = execute_query_dict(
            "SELECT ce_id, examples FROM cognitive_elements WHERE ce_id = ANY(%s)",
            (ce_ids,),
        ) or []
        for r in ex_rows:
            raw = r.get("examples") or []
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = []
            ce_examples_map[r["ce_id"]] = raw if isinstance(raw, list) else []

    # 5. Assemble Results
    results = []
    for cand in paged_candidates:
        # Map Categories
        c_ids = cand.get("categories") or []
        c_names = [cat_map.get(cid, str(cid)) for cid in c_ids if cid in cat_map]

        # Map CEs
        c_ces = []
        c_active_ces: list = []
        if cand["asset_type"] == "rule":
            c_ces = rule_ce_map.get(cand["id"], [])
            c_active_ces = rule_active_ces_map.get(cand["id"], [])

        results.append(SearchResult(
            id=cand["id"],
            asset_type=cand["asset_type"],
            name=cand["name"],
            content=cand["content"],
            ces=c_ces,
            active_ces=c_active_ces,
            condition=rule_condition_map.get(cand["id"]) if cand["asset_type"] == "rule" else None,
            categories=c_names,
            type=cand["type"],
            hybrid_score=cand.get("final_score", 0),
            rerank_score=None,
            is_local_draft=cand.get("is_local_draft"),
            examples=ce_examples_map.get(cand["id"], []) if cand["asset_type"] == "ce" else [],
            created_by_username=cand.get("created_by_username"),
            public_id=cand.get("public_id"),
        ))
    return results

@router.get("/categories", response_model=List[str])
async def get_all_categories():
    """Fetch all active category names from the database."""
    try:
        rows = execute_query_dict("SELECT name FROM categories ORDER BY name ASC") or []
        return [row["name"] for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def _run_hybrid_search(
    *,
    query_text: str,
    requested_assets: set,
    category_ids: List[int],
    limit: int,
    bookmark_user_id: Optional[int] = None,
) -> List[dict]:
    """Thin route-level wrapper around HybridSearchService that maps database
    schema errors back into helpful HTTP responses. The service itself stays
    HTTP-agnostic so it can be reused (and unit-tested) outside FastAPI."""
    try:
        return _search_service.search(
            query_text=query_text,
            asset_types=list(requested_assets),
            category_ids=category_ids,
            bookmark_user_id=bookmark_user_id,
            limit=limit,
        )
    except Exception as e:
        msg = str(e)
        print(f"Hybrid Search Error: {msg}")
        if "no such table" in msg:
            raise HTTPException(status_code=500, detail="Database schema outdated. Restart the backend (init_database rebuilds the schema), or delete backend/db/gavel.sqlite3 for a clean re-sync from the registry.")
        raise HTTPException(status_code=500, detail=msg)



@router.get("/search", response_model=SearchResponse)
async def search_library(
    q: str = Query(..., description="User search query"),
    categories: Optional[str] = Query(None, description="Comma-separated category names"),
    asset_types: Optional[str] = Query(None, description="Comma-separated asset types: rule, ce"),
    author: Optional[str] = Query(None, description="Filter results to a single author username (case-insensitive)"),
    page: int = Query(1, ge=1, description="Page number for pagination"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page"),
    candidate_limit: int = Query(100, ge=10, le=200, description="Fast retrieval pool size"),
):
    try:
        q = (q or "").strip()
        if len(q) > 512:
            raise HTTPException(status_code=400, detail="Search query must be at most 512 characters")
        # Author filter — Phase 4 "browse by author" facet. Lowercased
        # so we can match case-insensitively against the CITEXT column
        # in the post-filter below.
        author_norm = (author or "").strip().lower() or None
        requested_assets = {atype.lower() for atype in _normalize_list(asset_types)} or {"rule", "ce"}
        if not requested_assets.issubset({"rule", "ce"}):
            raise HTTPException(status_code=400, detail="asset_types must be a comma-separated list of 'rule' and/or 'ce'")

        # 1. Resolve Categories to IDs
        category_ids = []
        if categories:
            cat_names = _normalize_list(categories)
            if cat_names:
                q_cats = "SELECT category_id FROM categories WHERE name = ANY(%s)"
                rows = execute_query_dict(q_cats, (cat_names,)) or []
                category_ids = [r["category_id"] for r in rows]
                if not category_ids:
                    return SearchResponse(query=q, results=[], candidates_examined=0)

        # 2. Check Input (Empty q, categories, AND author all missing?)
        if not q and not category_ids and not author_norm:
            return SearchResponse(query=q, results=[], candidates_examined=0)

        search_query = q if q and q != "*" else ""
        perform_search = bool(search_query)

        # 3. Execute Search (Hybrid or Browse)
        candidates = []
        total_hits_estimate = 0
        
        if perform_search:
            # A. Hybrid Search — delegate to the shared service.
            candidates = _run_hybrid_search(
                query_text=search_query,
                requested_assets=requested_assets,
                category_ids=category_ids,
                limit=candidate_limit,
            )
            total_hits_estimate = len(candidates)
        else:
            # B. Browse / Filter (No Query)
            # Simple SQL to list items by category/recency
            browse_sqls = []

            # Helper for category filter
            cat_filter = ""
            if category_ids:
                cat_ids_str = ",".join(map(str, category_ids))
                cat_filter = (
                    "AND EXISTS (SELECT 1 FROM json_each(categories) "
                    f"WHERE json_each.value IN ({cat_ids_str}))"
                )
            # Author filter (browse path). CITEXT column makes the
            # equality case-insensitive automatically, but we use a
            # parameterized placeholder rather than inlining to keep the
            # SQL injection surface zero.
            author_filter = ""
            author_params: list = []
            if author_norm:
                author_filter = "AND created_by_username = %s"
                author_params.append(author_norm)

            if "rule" in requested_assets:
                browse_sqls.append(f"SELECT rule_id as id, 'rule' as asset_type, name, predicate as content, type, categories, is_local_draft, created_by_username, public_id, 1.0 as final_score FROM rules WHERE 1=1 {cat_filter} {author_filter}")
            if "ce" in requested_assets:
                browse_sqls.append(f"SELECT ce_id as id, 'ce' as asset_type, name, definition as content, type, categories, is_local_draft, created_by_username, public_id, 1.0 as final_score FROM cognitive_elements WHERE 1=1 {cat_filter} {author_filter}")

            browse_sql = f"SELECT * FROM ({' UNION ALL '.join(browse_sqls)}) as unified ORDER BY name ASC LIMIT {candidate_limit}"
            # Author param appears once per UNION arm.
            params_tuple = tuple(author_params * len(browse_sqls))
            candidates = execute_query_dict(browse_sql, params_tuple) or []
            total_hits_estimate = len(candidates)

        # Author post-filter for the hybrid path (cheaper than threading
        # the parameter into the search service). The hybrid path returns
        # at most `candidate_limit` rows so the post-filter is bounded.
        if author_norm and perform_search:
            candidates = [
                c for c in candidates
                if (c.get("created_by_username") or "").lower() == author_norm
            ]
            total_hits_estimate = len(candidates)

        if not candidates:
            return SearchResponse(query=q, results=[], candidates_examined=0)

        # 4. Pagination (in Python for simplified MVP, since RRF limit was small)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_candidates = candidates[start_idx:end_idx]
        
        if not paged_candidates:
             return SearchResponse(query=q, results=[], candidates_examined=len(candidates), total_results=len(candidates), page=page, page_size=page_size)

        # 5. Hydration (Fetch CEs and Category Names)
        results = _hydrate_results(paged_candidates)

        return SearchResponse(
            query=q,
            results=results,
            candidates_examined=len(candidates),
            total_results=total_hits_estimate,
            page=page,
            page_size=page_size
        )

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))



# -----------------------------------------------------------------------------
# BOOKMARK SEARCH LOGIC (Merged)
# -----------------------------------------------------------------------------

@router.get("/bookmarks/search", response_model=SearchResponse)
async def search_bookmarks(
    user_id: int = Query(..., description="User ID to filter bookmarks"),
    q: str = Query("", description="User search query (optional when filtering by category only)"),
    categories: Optional[str] = Query(None, description="Comma-separated category names"),
    asset_types: Optional[str] = Query(None, description="Comma-separated asset types: rule, ce"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page"),
    candidate_limit: int = Query(100, ge=10, le=200, description="Fast retrieval pool size"),
):
    """Search the user's bookmarks. Mirrors /library/search: hybrid path
    when `q` is provided, category-only browse path when only categories
    are selected. Both paths scope to rows the user actually bookmarked.

    Pre-fix, this endpoint silently dropped the `categories` parameter and
    refused to return anything for an empty query, so the bookmarks page's
    category chips filtered down to zero results even when the user had
    matching bookmarks.
    """
    try:
        q = (q or "").strip()
        if len(q) > 512:
            raise HTTPException(status_code=400, detail="Search query must be at most 512 characters")
        requested_assets = {atype.lower() for atype in _normalize_list(asset_types)} or {"rule", "ce"}
        if not requested_assets.issubset({"rule", "ce"}):
            raise HTTPException(status_code=400, detail="asset_types must be a comma-separated list of 'rule' and/or 'ce'")

        # 1. Resolve category NAMES (what the frontend sends) into IDs.
        #    Identical pattern to /library/search above.
        category_ids: List[int] = []
        if categories:
            cat_names = _normalize_list(categories)
            if cat_names:
                rows = execute_query_dict(
                    "SELECT category_id FROM categories WHERE name = ANY(%s)",
                    (cat_names,),
                ) or []
                category_ids = [r["category_id"] for r in rows]
                if not category_ids:
                    # User passed category names that don't exist — no hits possible.
                    return SearchResponse(query=q, results=[], candidates_examined=0)

        # 2. No q and no categories → nothing to do.
        if not q and not category_ids:
            return SearchResponse(query=q, results=[], candidates_examined=0)

        # 3. Execute search. Two paths, same as /library/search:
        #    - Hybrid (semantic + keyword + name) when `q` is present.
        #    - Direct SQL browse when only categories are selected; the
        #      hybrid service short-circuits on empty query, so we can't
        #      reuse it for the category-only case.
        candidates: List[dict] = []
        if q:
            candidates = _run_hybrid_search(
                query_text=q,
                requested_assets=requested_assets,
                category_ids=category_ids,
                limit=candidate_limit,
                bookmark_user_id=user_id,
            )
        else:
            # Category-only browse, scoped to this user's bookmarks via JOIN.
            cat_filter = ""
            if category_ids:
                cat_ids_str = ",".join(map(str, category_ids))
                cat_filter = (
                    "AND EXISTS (SELECT 1 FROM json_each(r.categories) "
                    f"WHERE json_each.value IN ({cat_ids_str}))"
                )

            browse_sqls: List[str] = []
            if "rule" in requested_assets:
                browse_sqls.append(
                    f"""
                    SELECT r.rule_id AS id, 'rule' AS asset_type, r.name,
                           r.predicate AS content, r.type, r.categories,
                           r.is_local_draft, r.created_by_username,
                           r.public_id, 1.0 AS final_score
                    FROM rules r
                    JOIN rule_bookmarks rb ON rb.rule_id = r.rule_id
                    WHERE rb.user_id = %(user_id)s
                    {cat_filter}
                    """
                )
            if "ce" in requested_assets:
                # CE table aliased as `r` so the cat_filter prefix matches both branches.
                browse_sqls.append(
                    f"""
                    SELECT r.ce_id AS id, 'ce' AS asset_type, r.name,
                           r.definition AS content, r.type, r.categories,
                           r.is_local_draft, r.created_by_username,
                           r.public_id, 1.0 AS final_score
                    FROM cognitive_elements r
                    JOIN ce_bookmarks cb ON cb.ce_id = r.ce_id
                    WHERE cb.user_id = %(user_id)s
                    {cat_filter}
                    """
                )

            if browse_sqls:
                browse_sql = (
                    f"SELECT * FROM ({' UNION ALL '.join(browse_sqls)}) as unified "
                    f"ORDER BY name ASC LIMIT {candidate_limit}"
                )
                candidates = execute_query_dict(browse_sql, {"user_id": user_id}) or []

        if not candidates:
            return SearchResponse(query=q, results=[], candidates_examined=0)

        # 4. Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_candidates = candidates[start_idx:end_idx]

        if not paged_candidates:
             return SearchResponse(query=q, results=[], candidates_examined=len(candidates), total_results=len(candidates), page=page, page_size=page_size)

        # 5. Hydration
        results = _hydrate_results(paged_candidates)

        return SearchResponse(
            query=q,
            results=results,
            candidates_examined=len(candidates),
            total_results=len(candidates),
            page=page,
            page_size=page_size
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------------
# REGISTRY SYNC
# -----------------------------------------------------------------------------


class SyncResponse(BaseModel):
    changed: bool
    ces_added: int = 0
    rules_added: int = 0
    rule_sets_added: int = 0
    # Records already present locally but re-pulled because the registry's
    # content_hash changed (an upstream edit kept the same public_id).
    ces_refreshed: int = 0
    rules_refreshed: int = 0
    rule_sets_refreshed: int = 0
    categories_synced: int = 0
    skipped_records: List[str] = []
    errors: List[str] = []


@router.get("/sync", response_model=SyncResponse)
async def sync_with_registry(
    force: bool = Query(False, description="If true, ignore the cached HEAD SHA and re-check every record."),
    _: int = Depends(get_current_user),
):
    """Pull new records from the gavel-rules registry into the local DB.

    Idempotent and safe to call at any time. Cheap when the registry has
    not changed since the last sync (a single HEAD-SHA probe); fetches
    only the deltas otherwise. See services/library_sync.py for the
    algorithm.
    """
    from services.library_sync import sync_library

    try:
        result = sync_library(force=force)
        return SyncResponse(**result.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/check-updates")
def check_updates(_: int = Depends(get_current_user)):
    """Cheap "is the local cache out of date?" probe.

    Compares the cached `last_synced_sha` against the gavel-rules repo's
    current HEAD commit SHA without pulling any records. Retained as a
    manual fallback; the live UI now learns about updates via the push
    stream below (`/library/events`) instead of a timer.
    """
    from services.library_sync import check_for_updates
    return check_for_updates()


# --- Live update push: backend -> frontend (SSE) ---

_SSE_KEEPALIVE_S = 25  # idle comment frame; keeps proxies from closing the stream


def _sse_frame(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@router.get("/events")
async def library_events_stream(request: Request):
    """Server-Sent-Events stream that pushes an `update_available` / `synced`
    signal the instant the registry poller notices the gavel-rules repo moved
    ahead of the local cache — so the sidebar can surface a "click to sync"
    badge with zero polling by the frontend. It never applies the update; the
    user clicks to sync.

    Public + non-sensitive by design: the stream carries only a
    "you are / aren't behind the registry" signal, never library content, and
    the backend is a localhost single-user process. EventSource also cannot
    attach an Authorization header, so gating this on a token would force a
    token-in-querystring dance for no real security gain.
    """
    from services import library_events as bus

    async def _initial_available() -> bool:
        # Greet with the freshest state so a tab that opened AFTER an update
        # still shows the badge. Cheap (one HEAD-SHA compare); off-loop
        # because it does blocking network I/O. Falls back to the last pushed
        # state.
        try:
            from services.library_sync import check_for_updates
            st = await asyncio.to_thread(check_for_updates)
            if st.get("checked"):
                return bool(st.get("available"))
        except Exception:
            pass
        return bool(bus.current_state().get("available"))

    async def event_gen():
        q = bus.subscribe()
        try:
            yield _sse_frame({"event": "connected", "available": await _initial_available()})
            while True:
                if await request.is_disconnected():
                    break
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=_SSE_KEEPALIVE_S)
                    yield _sse_frame(evt)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # comment frame — no event, just keep-alive
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so frames flush live
        },
    )


# -----------------------------------------------------------------------------
# NAME-CONFLICT LOOKUPS (used by AI pipeline early-detection + UI rename inputs)
#
# There is no in-studio publish anymore — contributions to the public library
# go by pull request to the gavel-rules GitHub repository. These probes exist
# so a draft doesn't silently shadow (or get shadowed by) a library record of
# the same name.
# -----------------------------------------------------------------------------


class CheckNameResponse(BaseModel):
    """Result of a name-conflict probe against the local library cache."""
    exists: bool
    public_id: Optional[str] = None
    # Lightweight summary of the existing record. Empty if exists=false or
    # the record couldn't be loaded from local cache.
    summary: Optional[dict] = None


def _lookup_conflict_summary(kind: str, public_id: str) -> Optional[dict]:
    """Return a small, UI-friendly summary of an existing public record so
    the conflict modal can show it without a separate fetch. Pulled from
    the local cache (post-sync). Returns None if the record isn't in the
    local DB — the UI can fall back to /library/record/{kind}/{public_id}
    for a fresh fetch in that case."""
    if kind == "rule":
        rows = execute_query_dict(
            """
            SELECT r.rule_id AS local_id, r.name, r.predicate, r.description,
                   r.public_id, r.published_at, r.categories
            FROM rules r WHERE r.public_id = %s
            """,
            (public_id,),
        )
    elif kind == "ce":
        rows = execute_query_dict(
            """
            SELECT ce.ce_id AS local_id, ce.name, ce.definition,
                   ce.category, ce.public_id, ce.published_at, ce.categories
            FROM cognitive_elements ce WHERE ce.public_id = %s
            """,
            (public_id,),
        )
    else:
        return None
    if not rows:
        return None
    row = rows[0]
    # Resolve category int IDs to names if present.
    cat_names: list = []
    if row.get("categories"):
        cat_rows = execute_query_dict(
            "SELECT name FROM categories WHERE category_id = ANY(%s)",
            (list(row["categories"]),),
        ) or []
        cat_names = [c["name"] for c in cat_rows]
    summary = {
        "kind": kind,
        "name": row["name"],
        "public_id": row["public_id"],
        "local_id": row["local_id"],
        "published_at": row.get("published_at").isoformat() if row.get("published_at") else None,
        "categories": cat_names,
    }
    if kind == "rule":
        summary["predicate"] = row.get("predicate") or ""
        summary["description"] = row.get("description") or ""
    else:
        summary["definition"] = row.get("definition") or ""
        summary["category"] = row.get("category") or ""
    return summary


_CHECK_NAME_TABLES = {
    "rule": "rules",
    "ce": "cognitive_elements",
    "rule_set": "rule_sets",
}


@router.get("/check-name", response_model=CheckNameResponse)
async def check_name(
    kind: str = Query(..., description="'rule', 'ce' or 'rule_set'"),
    name: str = Query(..., description="Name to probe against the synced library."),
    _: int = Depends(get_current_user),
):
    """Lightweight name-conflict probe. Used by the AI-pipeline early-check
    and the UI rename input.

    Purely LOCAL: looks the name up among the library records already synced
    into the local DB (is_local_draft = FALSE rows mirror the gavel-rules
    registry after a sync). No network, no token. If found, also returns a
    summary of the existing record so the UI can preview it inline.
    """
    table = _CHECK_NAME_TABLES.get(kind)
    if not table:
        raise HTTPException(status_code=400, detail="kind must be 'rule', 'ce' or 'rule_set'")
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    try:
        rows = execute_query_dict(
            f"""
            SELECT public_id FROM {table}
            WHERE name = %s AND is_local_draft = FALSE AND public_id IS NOT NULL
            """,
            (name.strip(),),
        ) or []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read the local library: {exc}")

    if not rows:
        return CheckNameResponse(exists=False)

    pid = rows[0]["public_id"]
    summary = _lookup_conflict_summary(kind, pid)  # rule/ce only; None otherwise
    return CheckNameResponse(exists=True, public_id=pid, summary=summary)


class CleanupResponse(BaseModel):
    """Result of POST /library/cleanup-local-drafts."""
    rules_deleted: int = 0
    ces_deleted: int = 0
    kept_for_conflict: int = 0


@router.get("/drafts")
async def list_local_drafts(_: int = Depends(get_current_user)):
    """Return every rule and CE in the local DB that's still
    is_local_draft = TRUE — i.e., user-created content that isn't part of
    the public gavel-rules library (contributions go by pull request).

    Powers the "My Drafts" page where the user reviews their local-only
    work. Read-only; no DB writes happen here.
    """
    try:
        rule_rows = execute_query_dict(
            """
            SELECT
                r.rule_id, r.name, r.predicate, r.description,
                r.ce_groups, r.condition,
                r.created_at,
                (SELECT json_group_array(c.name) FROM categories c
                 WHERE c.category_id IN (SELECT value FROM json_each(r.categories))
                ) AS "categories [JSONB]",
                COALESCE(
                    json_group_array(
                        json_object(
                            'ce_id', ce.ce_id,
                            'name', ce.name
                        )
                    ) FILTER (WHERE ce.ce_id IS NOT NULL),
                    '[]'
                ) AS "active_ces [JSONB]"
            FROM rules r
            LEFT JOIN rule_ce_link rl ON r.rule_id = rl.rule_id
            LEFT JOIN cognitive_elements ce ON rl.ce_id = ce.ce_id
            WHERE r.is_local_draft = TRUE AND r.is_ready = TRUE
              -- Hide a draft while its default test/calibration set is still
              -- being generated (in the background). It pops into Drafts/Browse
              -- only once the sets are ready, so the user never sees a
              -- half-generated rule.
              AND NOT EXISTS (
                  SELECT 1 FROM test_datasets td
                  WHERE td.rule_id = r.rule_id
                    AND td.is_default = TRUE
                    AND td.status IN ('generating', 'pending')
              )
            GROUP BY r.rule_id
            ORDER BY r.created_at DESC NULLS LAST, r.rule_id DESC
            """
        ) or []

        # Attach the logic group name(s) each member CE belongs to (from
        # ce_groups); `condition` stays a top-level field on the row. The
        # raw ce_groups column is dropped in favor of the annotated list.
        for row in rule_rows:
            groups_of: dict = {}
            for gname, members in (row.pop("ce_groups", None) or {}).items():
                for member in members or []:
                    groups_of.setdefault(member, []).append(gname)
            for entry in row.get("active_ces") or []:
                entry["groups"] = groups_of.get(entry.get("name"), [])

        ce_rows = execute_query_dict(
            """
            SELECT
                ce.ce_id, ce.name, ce.definition, ce.category, ce.created_at,
                (SELECT json_group_array(c.name) FROM categories c
                 WHERE c.category_id IN (SELECT value FROM json_each(ce.categories))
                ) AS "categories [JSONB]",
                ce.is_local_draft,
                ce.examples,
                CASE WHEN ed.dataset_id IS NOT NULL THEN TRUE ELSE FALSE END AS "has_training_data [BOOLEAN]"
            FROM cognitive_elements ce
            LEFT JOIN excitation_datasets ed ON ce.ce_id = ed.ce_id
            WHERE ce.is_local_draft = TRUE AND ce.is_ready = TRUE
              -- Hide a draft CE whose training set is still generating in the
              -- background (no excitation row yet), so it only appears once
              -- ready — same reasoning as draft rules above.
              AND ed.dataset_id IS NOT NULL
            ORDER BY ce.created_at DESC NULLS LAST, ce.ce_id DESC
            """
        ) or []

        return {"rules": rule_rows, "ces": ce_rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/drafts/rule/{rule_id}")
async def delete_draft_rule(rule_id: int, _: int = Depends(get_current_user)):
    """Delete a single local-draft rule. Refuses if the row is already
    published (is_local_draft = FALSE) — published rows mirror the
    gavel-rules registry and would just reappear on the next sync.

    Cascade rules in the schema take care of rule_ce_link rows.
    """
    try:
        row = execute_query_dict(
            "SELECT rule_id, name, is_local_draft FROM rules WHERE rule_id = %s",
            (rule_id,),
        ) or []
        if not row:
            raise HTTPException(status_code=404, detail="Rule not found")
        if not row[0]["is_local_draft"]:
            raise HTTPException(
                status_code=400,
                detail="Rule is published; cannot delete from local DB",
            )
        execute_query("DELETE FROM rules WHERE rule_id = %s", (rule_id,))
        return {"status": "deleted", "rule_id": rule_id, "name": row[0]["name"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drafts/ce/{ce_id}/dependent-rules")
async def get_ce_dependent_draft_rules(ce_id: int, _: int = Depends(get_current_user)):
    """List the draft rules that reference this CE. The UI fetches this
    before showing the delete-CE confirm dialog so the user can see (and
    is warned about) which rules will be cascade-deleted along with the
    CE. Published rules are intentionally not returned — they shouldn't
    reference a draft CE in the first place, and we don't touch published
    state from this view.
    """
    try:
        rows = execute_query_dict(
            """
            SELECT DISTINCT r.rule_id, r.name
            FROM rules r
            JOIN rule_ce_link rl ON r.rule_id = rl.rule_id
            WHERE rl.ce_id = %s AND r.is_local_draft = TRUE
            ORDER BY r.name
            """,
            (ce_id,),
        ) or []
        return {"rules": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/drafts/ce/{ce_id}")
async def delete_draft_ce(ce_id: int, _: int = Depends(get_current_user)):
    """Delete a single local-draft CE plus every draft rule that depends
    on it (cascade). Refuses if the CE is published.

    Why cascade: a draft rule that loses one of its CEs becomes structurally
    invalid (the predicate references a name no longer in the library), so
    the user has to either drop the rule or detach the CE before deleting.
    The UI warns about this via /drafts/ce/{ce_id}/dependent-rules and
    surfaces the rule list in the confirm dialog.

    Returns the deleted rule list so the UI can update its state without a
    second round-trip.
    """
    try:
        row = execute_query_dict(
            "SELECT ce_id, name, is_local_draft FROM cognitive_elements WHERE ce_id = %s",
            (ce_id,),
        ) or []
        if not row:
            raise HTTPException(status_code=404, detail="CE not found")
        if not row[0]["is_local_draft"]:
            raise HTTPException(
                status_code=400,
                detail="CE is published; cannot delete from local DB",
            )

        # Snapshot the dependent draft rules before we delete the CE — once
        # the CE row is gone the rule_ce_link rows cascade out and we can't
        # enumerate them anymore.
        dep_rules = execute_query_dict(
            """
            SELECT DISTINCT r.rule_id, r.name
            FROM rules r
            JOIN rule_ce_link rl ON r.rule_id = rl.rule_id
            WHERE rl.ce_id = %s AND r.is_local_draft = TRUE
            """,
            (ce_id,),
        ) or []

        deleted_rules = []
        for r in dep_rules:
            try:
                execute_query("DELETE FROM rules WHERE rule_id = %s", (r["rule_id"],))
                deleted_rules.append({"rule_id": r["rule_id"], "name": r["name"]})
            except Exception as rule_err:
                print(
                    f"[drafts] failed to cascade-delete rule {r['rule_id']} ({r['name']}): {rule_err}"
                )

        execute_query("DELETE FROM cognitive_elements WHERE ce_id = %s", (ce_id,))
        return {
            "status": "deleted",
            "ce_id": ce_id,
            "name": row[0]["name"],
            "deleted_rules": deleted_rules,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-local-drafts", response_model=CleanupResponse)
async def cleanup_local_drafts(_: int = Depends(get_current_user)):
    """Sweep stranded local drafts left behind by interrupted AI pipelines,
    cancelled flows, or backend crashes.

    Purely LOCAL — no network, no token. A draft is "stranded" when its name
    does NOT collide with a synced library record of the same kind (the
    is_local_draft = FALSE rows mirror the gavel-rules registry after a
    sync), so deleting it loses nothing that exists upstream. Drafts whose
    name IS taken by a library record are kept: that collision is something
    the user must resolve (rename the draft, or drop it deliberately).

    Best called AFTER /library/sync so the local mirror is fresh.
    """
    def _published_names(table: str) -> set:
        rows = execute_query_dict(
            f"SELECT name FROM {table} WHERE is_local_draft = FALSE"
        ) or []
        return {r["name"] for r in rows}

    rules_deleted = 0
    ces_deleted = 0
    kept = 0

    library_rule_names = _published_names("rules")
    drafts = execute_query_dict(
        "SELECT rule_id, name FROM rules WHERE is_local_draft = TRUE"
    ) or []
    for d in drafts:
        if d["name"] in library_rule_names:
            kept += 1
            continue
        try:
            execute_query("DELETE FROM rules WHERE rule_id = %s", (d["rule_id"],))
            rules_deleted += 1
        except Exception:
            pass

    library_ce_names = _published_names("cognitive_elements")
    drafts = execute_query_dict(
        "SELECT ce_id, name FROM cognitive_elements WHERE is_local_draft = TRUE"
    ) or []
    for d in drafts:
        if d["name"] in library_ce_names:
            kept += 1
            continue
        try:
            execute_query("DELETE FROM cognitive_elements WHERE ce_id = %s", (d["ce_id"],))
            ces_deleted += 1
        except Exception:
            pass

    return CleanupResponse(
        rules_deleted=rules_deleted,
        ces_deleted=ces_deleted,
        kept_for_conflict=kept,
    )


class PublicRecordResponse(BaseModel):
    """A single public record by public_id, returned as a UI-friendly summary.

    For now this only reads from the local cache; if the user wants a
    record that isn't local yet, they should sync first.
    """
    found: bool
    summary: Optional[dict] = None


@router.get("/record/{kind}/{public_id}", response_model=PublicRecordResponse)
async def get_public_record(
    kind: str,
    public_id: str,
    _: int = Depends(get_current_user),
):
    """Fetch a single public record's summary by its public_id, from the
    local cache (post-sync). Used by the conflict modal's preview pane.
    """
    if kind not in ("rule", "ce"):
        raise HTTPException(status_code=400, detail="kind must be 'rule' or 'ce'")
    summary = _lookup_conflict_summary(kind, public_id)
    return PublicRecordResponse(found=bool(summary), summary=summary)