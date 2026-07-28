import asyncio
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.PostgreSQL import execute_query, execute_query_dict
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
    # Role-aware CE list for rule results. Each entry is
    # {ce_id, name, role, fallback_group}. Empty for CE results.
    # The bare `ces` list above stays for back-compat — RuleCard
    # prefers active_ces when present and falls back to ces otherwise.
    active_ces: List[dict] = []
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
        
    # 3. Batch Fetch CEs for Rules (Linked entities). Pulls role +
    # fallback_group so the frontend RuleCard can show NECESSARY /
    # SUFFICIENT / FALLBACK badges instead of defaulting everything to
    # NECESSARY. We still expose the flat name list as `ces` for any
    # caller that only needs names; new callers should use active_ces.
    rule_ce_map = {}          # rule_id -> [name, ...]
    rule_active_ces_map = {}  # rule_id -> [{ce_id, name, role, fallback_group}, ...]
    if rule_ids:
        ce_sql = """
        SELECT rcl.rule_id, rcl.ce_id, ce.name,
               COALESCE(rcl.role, 'necessary')      AS role,
               COALESCE(rcl.fallback_group, 0)      AS fallback_group
        FROM rule_ce_link rcl
        JOIN cognitive_elements ce ON rcl.ce_id = ce.ce_id
        WHERE rcl.rule_id = ANY(%s)
        ORDER BY rcl.rule_id, rcl.role, rcl.fallback_group, ce.name
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
                "role": r["role"],
                "fallback_group": r["fallback_group"],
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
    """Thin route-level wrapper around HybridSearchService that maps Postgres
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
        if "relation" in msg and "does not exist" in msg:
            raise HTTPException(status_code=500, detail="Database schema outdated. Please run 'reseed_db_from_registry.py' to update tables.")
        if "vector" in msg:
            raise HTTPException(status_code=500, detail="PostgreSQL extension 'pgvector' is missing. Please install it.")
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
                cat_filter = f"AND categories && ARRAY[{cat_ids_str}]"
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
                cat_filter = f"AND r.categories && ARRAY[{cat_ids_str}]"

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
    # Records already present locally but re-pulled because the registry's
    # published_at was newer (an upstream edit kept the same public_id).
    ces_refreshed: int = 0
    rules_refreshed: int = 0
    categories_synced: int = 0
    neutral_synced: int = 0
    skipped_records: List[str] = []
    errors: List[str] = []


@router.get("/sync", response_model=SyncResponse)
async def sync_with_registry(
    force: bool = Query(False, description="If true, ignore the cached manifest hash and re-fetch every missing record."),
    _: int = Depends(get_current_user),
):
    """Pull new records from the public HF registry into the local DB.

    Idempotent and safe to call on every login. Cheap when the registry
    has not changed since the last sync (a single small fetch + a hash
    compare); fetches only the deltas otherwise. See
    services/hf_sync.py for the algorithm.
    """
    from services.hf_sync import sync_library

    try:
        result = sync_library(force=force)
        return SyncResponse(**result.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/check-updates")
def check_updates(_: int = Depends(get_current_user)):
    """Cheap "is the local cache out of date?" probe.

    Compares the cached `last_manifest_hash` against HF's current
    manifest hash without pulling any records. Retained as a manual
    fallback; the live UI now learns about updates via the push stream
    below (`/library/events`) instead of a timer.
    """
    from services.hf_sync import check_for_updates
    return check_for_updates()


# --- Live update push: backend -> frontend (SSE) ---

_SSE_KEEPALIVE_S = 25  # idle comment frame; keeps proxies from closing the stream


def _sse_frame(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@router.get("/events")
async def library_events_stream(request: Request):
    """Server-Sent-Events stream that pushes an `update_available` / `synced`
    signal the instant the central server's version_update lands — so the
    sidebar can surface a "click to sync" badge with zero polling (this mirrors
    the central -> backend control plane one layer down). It never applies the
    update; the user clicks to sync.

    Public + non-sensitive by design: the stream carries only a
    "you are / aren't behind the registry" signal, never library content, and
    the backend is a localhost single-user process (same rationale as the
    central server's public /hf/head-sha). EventSource also cannot attach an
    Authorization header, so gating this on the JWT would force a
    token-in-querystring dance for no real security gain.
    """
    from services import library_events as bus

    async def _initial_available() -> bool:
        # Greet with the freshest state so a tab that opened AFTER an update
        # still shows the badge. Cheap (one manifest-hash compare); off-loop
        # because it does blocking HF I/O. Falls back to the last pushed state.
        try:
            from services.hf_sync import check_for_updates
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
# REGISTRY PUBLISH (push local drafts to HF)
# -----------------------------------------------------------------------------


class PublishConflictInfo(BaseModel):
    type: str
    name: str
    public_id: str
    # For a CE-name clash inside a rule publish: the local draft CE's id, so the
    # UI can offer a one-click "adopt the existing public CE" (replace the draft
    # in place and re-publish the rule). None for plain rule/CE name conflicts.
    local_ce_id: Optional[int] = None


class PublishResponse(BaseModel):
    status: str  # "success" | "conflict" | "race" | "error"
    public_id: Optional[str] = None
    name: Optional[str] = None
    conflict_with: Optional[PublishConflictInfo] = None
    error: Optional[str] = None


@router.post("/publish/ce/{ce_id}", response_model=PublishResponse)
async def publish_ce_endpoint(
    ce_id: int,
    user_id: int = Depends(get_current_user),
):
    """Push a local CE draft (and its excitation dataset) to the HF
    registry atomically. The actual HF write goes through the central
    server, which holds the HF write token. The user's bearer token is
    forwarded so the central server can verify the request and bump
    that user's contribution counter."""
    from services.hf_publish import publish_ce
    try:
        result = publish_ce(ce_id, publisher_user_id=user_id)
        return PublishResponse(**result.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/publish/rule/{rule_id}", response_model=PublishResponse)
async def publish_rule_endpoint(
    rule_id: int,
    user_id: int = Depends(get_current_user),
):
    """Push a local rule draft + any draft CE dependencies to the HF
    registry atomically (via the central server's HF write proxy)."""
    from services.hf_publish import publish_rule
    try:
        result = publish_rule(rule_id, publisher_user_id=user_id)
        return PublishResponse(**result.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/publish/rule-set/{classifier_id}", response_model=PublishResponse)
async def publish_rule_set_endpoint(
    classifier_id: int,
    user_id: int = Depends(get_current_user),
):
    """Share a private rule set (a model-less guardrail / classifier) to the
    public registry as a model-agnostic collection of already-published rules.

    Only the rule selection is published — never the model or training. The
    private classifiers row is the caller's and stays untouched; the published
    artifact is a separate `rule_sets` record. Every member rule must already
    be public (the service refuses otherwise)."""
    from utils.ownership import assert_owns_classifier
    from services.hf_publish import publish_rule_set
    assert_owns_classifier(user_id, classifier_id)
    try:
        result = publish_rule_set(classifier_id, publisher_user_id=user_id)
        return PublishResponse(**result.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class AdoptCERequest(BaseModel):
    public_id: str


@router.post("/ce/{ce_id}/adopt-public")
def adopt_public_ce(ce_id: int, req: AdoptCERequest, _: int = Depends(get_current_user)):
    """Replace a local DRAFT cognitive element with the existing PUBLIC CE it
    name-clashed with, converting the row IN PLACE.

    There is no rule editor, so when a rule's draft CE collides with a public CE
    of the same name, the user can't manually swap it. This pulls the public CE's
    definition + training data + public_id into the existing draft row and flips
    it to non-draft. Because the rule's rule_ce_link still points at this ce_id,
    the rule now references the public CE automatically — no re-linking, no
    orphaned draft. Re-publishing the rule then only adds the new rule.
    """
    rows = execute_query_dict(
        "SELECT ce_id, name, is_local_draft FROM cognitive_elements WHERE ce_id = %s",
        (ce_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Cognitive element not found.")
    draft = rows[0]
    if not draft.get("is_local_draft"):
        raise HTTPException(status_code=400, detail="This CE is already published — nothing to adopt.")

    from services.hf_sync import _resolve_token, _fetch_record, ensure_excitation
    from services.library_schemas import CERecord
    from utils.DButils import normalize_and_upsert_categories
    from utils.embedding_utils import trigger_embedding

    token = _resolve_token()  # None is fine — public repo reads work anonymously
    try:
        payload = _fetch_record(token, f"public_ces/{req.public_id}.json")
        rec = CERecord.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch the public CE from the registry: {exc}")

    if rec.name != draft["name"]:
        raise HTTPException(
            status_code=409,
            detail=f"Name mismatch: this draft is '{draft['name']}' but the public CE is '{rec.name}'.",
        )

    final_categories = normalize_and_upsert_categories(list(rec.categories), allow_new=True)
    execute_query(
        """
        UPDATE cognitive_elements
        SET definition = %s, category = %s, categories = %s, examples = %s::jsonb,
            public_id = %s, published_at = %s, is_local_draft = FALSE,
            created_by_username = COALESCE(created_by_username, %s)
        WHERE ce_id = %s
        """,
        (rec.definition, rec.category, final_categories, json.dumps(rec.examples),
         rec.public_id, rec.published_at, rec.created_by_username, ce_id),
    )
    # Pull the public CE's training data so the adopted row is complete, and
    # refresh its embedding. Both best-effort — the adopt itself already landed.
    try:
        ensure_excitation(ce_id)
    except Exception:
        pass
    try:
        trigger_embedding("ce", ce_id, rec.name, rec.definition)
    except Exception:
        pass
    return {"status": "adopted", "ce_id": ce_id, "public_id": rec.public_id, "name": rec.name}


# -----------------------------------------------------------------------------
# NAME-CONFLICT LOOKUPS (used by AI pipeline early-detection + UI rename inputs)
# -----------------------------------------------------------------------------


class CheckNameResponse(BaseModel):
    """Result of a name-conflict probe against the registry."""
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


@router.get("/check-name", response_model=CheckNameResponse)
async def check_name(
    kind: str = Query(..., description="'rule' or 'ce'"),
    name: str = Query(..., description="Name to probe against the registry."),
    _: int = Depends(get_current_user),
):
    """Lightweight name-conflict probe. Used by the AI-pipeline early-check
    and the UI rename input.

    Looks up the name in the registry's manifest name index. If found,
    also returns a summary of the existing record so the UI can preview it
    inline. Cheap because the manifest is fetched into the local HF cache
    on every sync — this endpoint just rereads it.
    """
    if kind not in ("rule", "ce"):
        raise HTTPException(status_code=400, detail="kind must be 'rule' or 'ce'")
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="name is required")

    # Probe via the registry manifest, read ANONYMOUSLY (public repo) — the same
    # token-less path the sync uses. The old code routed this through the publish
    # helper and demanded an HF_TOKEN, which 500'd on backends that don't have one
    # even though the read needs no auth at all.
    try:
        from services.hf_sync import _fetch_manifest_bytes
        manifest = json.loads(_fetch_manifest_bytes())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read registry: {exc}")

    index_key = "rule_names" if kind == "rule" else "ce_names"
    name_index = manifest.get(index_key, {}) or {}
    pid = name_index.get(name.strip())
    if not pid:
        return CheckNameResponse(exists=False)

    summary = _lookup_conflict_summary(kind, pid)
    return CheckNameResponse(exists=True, public_id=pid, summary=summary)


class CleanupResponse(BaseModel):
    """Result of POST /library/cleanup-local-drafts."""
    rules_deleted: int = 0
    ces_deleted: int = 0
    kept_for_conflict: int = 0


@router.get("/drafts")
async def list_local_drafts(_: int = Depends(get_current_user)):
    """Return every rule and CE in the local DB that's still
    is_local_draft = TRUE — i.e., user-created content that hasn't been
    pushed to the HF registry yet.

    Powers the "My Drafts" page where the user reviews everything pending
    publish. Read-only; no DB writes happen here.
    """
    try:
        rule_rows = execute_query_dict(
            """
            SELECT
                r.rule_id, r.name, r.predicate, r.description,
                r.created_at,
                (SELECT array_agg(c.name) FROM categories c WHERE c.category_id = ANY(r.categories)) AS categories,
                COALESCE(
                    json_agg(
                        json_build_object(
                            'ce_id', ce.ce_id,
                            'name', ce.name,
                            'role', COALESCE(rl.role, 'necessary'),
                            'fallback_group', COALESCE(rl.fallback_group, 0)
                        )
                    ) FILTER (WHERE ce.ce_id IS NOT NULL),
                    '[]'
                ) AS active_ces
            FROM rules r
            LEFT JOIN rule_ce_link rl ON r.rule_id = rl.rule_id
            LEFT JOIN cognitive_elements ce ON rl.ce_id = ce.ce_id
            WHERE r.is_local_draft = TRUE AND r.is_ready = TRUE
              -- Hide a draft while its default test/calibration set is still
              -- being generated (in the background). It pops into Drafts/Browse
              -- only once the sets are ready, so the user never sees a
              -- half-generated rule (or a premature Publish button).
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

        ce_rows = execute_query_dict(
            """
            SELECT
                ce.ce_id, ce.name, ce.definition, ce.category, ce.created_at,
                (SELECT array_agg(c.name) FROM categories c WHERE c.category_id = ANY(ce.categories)) AS categories,
                ce.is_local_draft,
                ce.examples,
                CASE WHEN ed.dataset_id IS NOT NULL THEN TRUE ELSE FALSE END AS has_training_data
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
    published (is_local_draft = FALSE) — published rows live on HF and
    can't be removed from the local cache without a re-sync.

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

    A draft is "stranded" when its name is NOT in the public registry's
    name index — that means there is nothing on HF for the user to lose.
    Drafts whose name IS in the registry are left alone: those are either
    same-name collisions the user must resolve via the publish-time
    CONFLICT modal, or ghost-published rows that the recovery step in
    sync_library will heal forward on its own.

    Always called AFTER sync_library so the manifest cache is fresh and
    pending_public_id rows have already been recovered.
    """
    from services.hf_publish import _resolve_token, _fetch_head_sha_and_manifest

    token = _resolve_token()
    if not token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set on server")
    try:
        _sha, manifest = _fetch_head_sha_and_manifest()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read registry: {exc}")

    rule_names_idx = manifest.get("rule_names", {}) or {}
    ce_names_idx = manifest.get("ce_names", {}) or {}

    rules_deleted = 0
    ces_deleted = 0
    kept = 0

    drafts = execute_query_dict(
        "SELECT rule_id, name FROM rules WHERE is_local_draft = TRUE"
    ) or []
    for d in drafts:
        if d["name"] in rule_names_idx:
            kept += 1
            continue
        try:
            execute_query("DELETE FROM rules WHERE rule_id = %s", (d["rule_id"],))
            rules_deleted += 1
        except Exception:
            pass

    drafts = execute_query_dict(
        "SELECT ce_id, name FROM cognitive_elements WHERE is_local_draft = TRUE"
    ) or []
    for d in drafts:
        if d["name"] in ce_names_idx:
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