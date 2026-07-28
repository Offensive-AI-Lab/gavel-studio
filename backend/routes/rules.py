from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
from utils.auth import get_current_user
from utils.ownership import assert_owns_setup
# Import your existing scripts
from sql_scripts.model_scripts import (
    update_private_setup,
    delete_rule_setup,
    fork_setup_to_draft,
)
from sql_scripts.definition_scripts import (
    get_all_public_rules,
    get_all_public_rule_sets,
    get_rule_set_detail,
    create_ce,
)
from services.bookmarks import BookmarkService
from gavel_pipeline.db_access import upsert_rule_with_links
from sql_scripts.junction_scripts import (
    link_ce_to_setup,
    unlink_ce_from_setup,
    compute_rule_fingerprint_from_links,
    find_existing_rule_by_fingerprint,
)
from utils.PostgreSQL import execute_query, execute_query_dict
from utils.text_safety import clean_text

# Same BookmarkService used by routes/cognitive.py — see services/bookmarks.py.
_BOOKMARK_ASSET = "rule"

router = APIRouter()


def _mark_needs_retraining_for_setup(setup_id: int):
    """If the guardrail owning this setup is 'active', mark it as needing retraining."""
    row = execute_query_dict(
        "SELECT classifier_id FROM rule_setup WHERE setup_id = %s", (setup_id,)
    )
    if row:
        execute_query(
            "UPDATE classifiers SET status = 'needs_retraining' WHERE classifier_id = %s AND status = 'active'",
            (row[0]["classifier_id"],),
        )

# --- SCHEMAS ---

class UpdateLogicRequest(BaseModel):
    user_id: int
    predicate: str | None = Field(default=None, max_length=4000)
    active_ces: List[str] | None = None  # legacy path
    ce_links: List[dict] | None = None   # [{ce_id, role, fallback_group}]

    @field_validator("predicate", mode="before")
    @classmethod
    def _clean_predicate(cls, value):
        if value is None:
            return None
        return clean_text(value, field_name="predicate", max_length=4000, allow_newlines=False)

class SaveEditedRequest(BaseModel):
    """Body for POST /rules/setup/{setup_id}/save-edited.

    Fields beyond ce_links are only consulted when the source rule is
    public (forking is required). For in-place updates of the user's
    own draft, new_name and add_bookmark are ignored — we just patch
    the existing rules row + setup_ce_link rows."""
    user_id: int
    ce_links: List[dict]                   # [{ce_id, role, fallback_group}]
    new_name: str | None = Field(default=None, max_length=255)
    add_bookmark: bool = False


class CheckDuplicateRequest(BaseModel):
    """Body for POST /rules/check-duplicate. The rule editor sends the
    proposed CE/role/fallback shape; we return whether it collides with
    any rule the user could observe (a setup in the same guardrail OR
    a row in the global `rules` table). exclude_setup_id lets the
    editor dedup-check WITHOUT matching itself, so an unchanged save
    doesn't surface as a 'duplicate'."""
    ce_links: List[dict]                # [{ce_id, role, fallback_group}]
    classifier_id: int | None = None    # scope for the per-guardrail scan
    exclude_setup_id: int | None = None # the setup currently being edited


class LinkCERequest(BaseModel):
    ce_id: int

class CreateCERequest(BaseModel):
    name: str = Field(..., max_length=120)
    user_id: int
    definition: str = Field(default="", max_length=4000)  # Optional definition for better categorization

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, value):
        return clean_text(value, field_name="CE name", max_length=120)

    @field_validator("definition", mode="before")
    @classmethod
    def _clean_definition(cls, value):
        if value in (None, ""):
            return ""
        return clean_text(value, field_name="CE definition", max_length=4000, allow_newlines=True)

class CreatePublicRuleRequest(BaseModel):
    name: str = Field(..., max_length=120)
    predicate: str = Field(..., max_length=4000)
    ce_names: List[str] = []  # legacy payload: treated as necessary
    necessary: List[str] = []
    fallback: List[List[str]] = []
    sufficient: List[str] = []
    user_id: int
    definition: str = Field(default="", max_length=4000)  # Optional definition for better categorization
    categories: List[str] = []

    @field_validator("name", mode="before")
    @classmethod
    def _clean_rule_name(cls, value):
        return clean_text(value, field_name="rule name", max_length=120)

    @field_validator("predicate", mode="before")
    @classmethod
    def _clean_rule_predicate(cls, value):
        return clean_text(value, field_name="predicate", max_length=4000, allow_newlines=False)

    @field_validator("definition", mode="before")
    @classmethod
    def _clean_rule_definition(cls, value):
        if value in (None, ""):
            return ""
        return clean_text(value, field_name="rule definition", max_length=4000, allow_newlines=True)


class RuleBookmarkRequest(BaseModel):
    user_id: int
    rule_id: int


class RuleSetBookmarkRequest(BaseModel):
    user_id: int
    rule_set_id: int

# --- PRIVATE RULE INSTANCES ---

@router.put("/setup/{setup_id}")
def update_rule_logic(setup_id: int, req: UpdateLogicRequest, auth_uid: int = Depends(get_current_user)):
    assert_owns_setup(auth_uid, setup_id)
    try:
        # New role-aware path
        if req.ce_links:
            predicate = update_private_setup(setup_id, ce_roles=req.ce_links)
            _mark_needs_retraining_for_setup(setup_id)
            return {"status": "success", "predicate": predicate, "active_ces": req.ce_links}

        # Legacy path (kept for backward compatibility)
        ce_ids = []
        for name in req.active_ces or []:
            ce_record = create_ce(req.user_id, name)
            ce_ids.append(ce_record['ce_id'])
        predicate = req.predicate or ""
        predicate = update_private_setup(setup_id, predicate, ce_ids)
        _mark_needs_retraining_for_setup(setup_id)
        return {"status": "success", "predicate": predicate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/setup/{setup_id}/save-edited")
def save_edited_rule(setup_id: int, req: SaveEditedRequest, auth_uid: int = Depends(get_current_user)):
    """Save user-edited rule logic.

    Routes between two paths based on the source rule:

      * **In-place** when the setup's backing rule is the user's own
        existing draft (is_local_draft = TRUE). We just patch the
        rule's CE links and predicate — no new rule entity, no draft
        clutter. The current setup keeps pointing at the same rule.

      * **Fork** when the setup is purely manual (rule_id NULL) or its
        backing rule is published (is_local_draft = FALSE). A new
        rules row is created (is_local_draft = TRUE) under the
        user-supplied `new_name`, the setup is repointed at it, and
        — if `add_bookmark` is True — the user gets a bookmark for
        cross-guardrail reuse. The new draft surfaces in My Drafts.

    Either way, the setup_ce_link rows are replaced to match the new
    structure and the owning guardrail is flagged needs_retraining.
    """
    assert_owns_setup(auth_uid, setup_id)
    try:
        # Detect the source-rule state. We need is_local_draft and rule_id
        # to decide between in-place and fork. NULL rule_id is treated as
        # "fork" so a manual setup gets promoted to a backing draft on
        # the user's first edit.
        source = execute_query_dict(
            """
            SELECT rs.rule_id, r.is_local_draft
            FROM rule_setup rs
            LEFT JOIN rules r ON rs.rule_id = r.rule_id
            WHERE rs.setup_id = %s
            """,
            (setup_id,),
        )
        if not source:
            raise HTTPException(status_code=404, detail="Rule setup not found")

        rule_id = source[0]["rule_id"]
        is_local_draft = source[0]["is_local_draft"]

        # In-place path: existing draft, structure changed, no new entity
        if rule_id is not None and is_local_draft is True:
            predicate = update_private_setup(setup_id, ce_roles=req.ce_links)
            _mark_needs_retraining_for_setup(setup_id)
            return {
                "status": "success",
                "fork": False,
                "predicate": predicate,
            }

        # Fork path: must have a name. The helper validates structure /
        # name uniqueness and raises ValueError on conflict, which we
        # surface as a 409 so the frontend can prompt for a new name.
        if not req.new_name or not req.new_name.strip():
            raise HTTPException(
                status_code=400,
                detail="A rule name is required when saving the edit as a new draft.",
            )
        try:
            result = fork_setup_to_draft(
                setup_id=setup_id,
                user_id=req.user_id,
                new_name=req.new_name,
                ce_roles=req.ce_links,
                add_bookmark=req.add_bookmark,
            )
        except ValueError as ve:
            raise HTTPException(status_code=409, detail=str(ve))

        return {
            "status": "success",
            "fork": True,
            "rule_id": result["rule_id"],
            "predicate": result["predicate"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-duplicate")
def check_rule_duplicate(req: CheckDuplicateRequest):
    """Structural-fingerprint dedup probe used by the rule editor on Save.

    Returns the first rule that matches the proposed shape, or
    `{exists: False}` if the shape is unique. Two scopes are scanned:

    1. `rule_setup` rows in `req.classifier_id` (the user's local
       overrides for this guardrail). Excludes `req.exclude_setup_id`
       so a no-op save doesn't surface as a duplicate of itself.

    2. The global `rules` table — public library + every user's
       private fork. We also exclude the source rule_id of the setup
       being edited (read from rule_setup.rule_id) so a user editing
       a rule that was forked from public rule X can save unchanged
       without tripping a self-match against X.

    Both scopes use compute_rule_fingerprint_from_links so the result
    is comparable across the two storage shapes.
    """
    fingerprint = compute_rule_fingerprint_from_links(req.ce_links)

    # 1) Setups in the same guardrail
    if req.classifier_id is not None:
        rows = execute_query_dict(
            """
            SELECT rs.setup_id, rs.custom_name,
                   COALESCE(
                       json_agg(
                           json_build_object(
                               'ce_id', scl.ce_id,
                               'role', scl.role,
                               'fallback_group', scl.fallback_group
                           )
                       ) FILTER (WHERE scl.ce_id IS NOT NULL),
                       '[]'::json
                   ) AS links
            FROM rule_setup rs
            LEFT JOIN setup_ce_link scl ON scl.setup_id = rs.setup_id
            WHERE rs.classifier_id = %s
              AND (%s::int IS NULL OR rs.setup_id <> %s::int)
            GROUP BY rs.setup_id, rs.custom_name
            """,
            (req.classifier_id, req.exclude_setup_id, req.exclude_setup_id),
        ) or []
        for row in rows:
            if compute_rule_fingerprint_from_links(row["links"]) == fingerprint:
                return {
                    "exists": True,
                    "kind": "setup",
                    "name": row["custom_name"] or "(unnamed)",
                    "setup_id": row["setup_id"],
                }

    # 2) Global `rules` table — exclude the source rule_id of the setup
    # being edited so saving without changes is a no-op match, not a
    # spurious "this duplicates yourself" error.
    exclude_name = None
    if req.exclude_setup_id is not None:
        owner_row = execute_query_dict(
            """
            SELECT r.name AS rule_name
            FROM rule_setup rs
            LEFT JOIN rules r ON rs.rule_id = r.rule_id
            WHERE rs.setup_id = %s
            """,
            (req.exclude_setup_id,),
        ) or []
        if owner_row:
            exclude_name = owner_row[0].get("rule_name")

    duplicate = find_existing_rule_by_fingerprint(fingerprint, exclude_name=exclude_name)
    if duplicate is not None:
        return {
            "exists": True,
            "kind": "rule",
            "name": duplicate.get("name") or "(unnamed)",
            "rule_id": duplicate.get("rule_id"),
        }

    return {"exists": False}


@router.delete("/setup/{setup_id}")
def delete_rule_instance(setup_id: int, auth_uid: int = Depends(get_current_user)):
    assert_owns_setup(auth_uid, setup_id)
    try:
        _mark_needs_retraining_for_setup(setup_id)
        delete_rule_setup(setup_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- JUNCTION TABLE ENDPOINTS ---

@router.post("/setup/{setup_id}/link-ce")
def link_existing_ce(setup_id: int, req: LinkCERequest, auth_uid: int = Depends(get_current_user)):
    """Links an existing CE to a specific rule setup"""
    assert_owns_setup(auth_uid, setup_id)
    try:
        success = link_ce_to_setup(setup_id, req.ce_id)
        if not success:
            raise HTTPException(status_code=500, detail="Linking failed")

        # Also update the predicate to include the new CE so it persists after reload
        ce_row = execute_query_dict("SELECT name FROM cognitive_elements WHERE ce_id = %s", (req.ce_id,)) or []
        ce_name = ce_row[0]["name"] if ce_row else None

        pred_row = execute_query_dict("SELECT predicate FROM rule_setup WHERE setup_id = %s", (setup_id,)) or []
        current_pred = pred_row[0]["predicate"] if pred_row else ""

        placeholders = {"IF TRUE THEN BLOCK", "IF TRUE THEN BLOCK".lower(), ""}

        if ce_name:
            base = "" if current_pred in placeholders else current_pred
            new_pred = ce_name if not base else f"({base}) AND {ce_name}"
            execute_query("UPDATE rule_setup SET predicate = %s WHERE setup_id = %s", (new_pred, setup_id))

        return {"status": "linked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/setup/{setup_id}/create-ce")
def create_and_link_new_ce(setup_id: int, req: CreateCERequest, auth_uid: int = Depends(get_current_user)):
    """Creates a new CE and links it to the setup. Triggers Step 2B"""
    assert_owns_setup(auth_uid, setup_id)
    try:
        # 1. Create the CE record
        ce_record = create_ce(req.user_id, req.name, definition=req.definition)
        if not ce_record or 'ce_id' not in ce_record:
             raise HTTPException(status_code=500, detail="CE creation failed")
             
        ce_id = ce_record['ce_id']
        print(f"Created CE with ID: {ce_id}")
        
        # 2. Link it immediately to the setup instance
        success = link_ce_to_setup(setup_id, ce_id)
        print(f"Linking CE ID {ce_id} to Setup ID {setup_id}: {'Success' if success else 'Failed'}")
        
        if not success:
            raise HTTPException(status_code=500, detail="Linking failed after creation")

        # Update predicate to persist the new CE
        pred_row = execute_query_dict("SELECT predicate FROM rule_setup WHERE setup_id = %s", (setup_id,)) or []
        current_pred = pred_row[0]["predicate"] if pred_row else ""

        placeholders = {"IF TRUE THEN BLOCK", "IF TRUE THEN BLOCK".lower(), ""}
        base = "" if current_pred in placeholders else current_pred
        new_pred = req.name if not base else f"({base}) AND {req.name}"
        execute_query("UPDATE rule_setup SET predicate = %s WHERE setup_id = %s", (new_pred, setup_id))

        return {"ce_id": ce_id, "status": "created and linked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/setup/{setup_id}/ce/{ce_id}")
def remove_ce_link(setup_id: int, ce_id: int, auth_uid: int = Depends(get_current_user)):
    """Removes the specific link in the junction table"""
    assert_owns_setup(auth_uid, setup_id)
    try:
        success = unlink_ce_from_setup(setup_id, ce_id)
        if success:
            return {"status": "unlinked"}
        raise HTTPException(status_code=500, detail="Unlinking failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{rule_id}/detail")
def get_rule_detail(rule_id: int, _: int = Depends(get_current_user)):
    """Rule-scoped detail for the Rule page: name, predicate, and the rule's
    CEs with their roles, definitions and examples. Guardrail-independent —
    reads straight from rules + rule_ce_link + cognitive_elements."""
    import json as _json
    rows = execute_query_dict(
        "SELECT rule_id, name, predicate, description, public_id, created_by_username, is_local_draft "
        "FROM rules WHERE rule_id = %s",
        (rule_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Rule not found")
    rule = rows[0]
    ce_rows = execute_query_dict(
        """SELECT ce.ce_id, ce.name, ce.definition, ce.examples,
                  rcl.role, rcl.fallback_group
           FROM rule_ce_link rcl
           JOIN cognitive_elements ce ON ce.ce_id = rcl.ce_id
           WHERE rcl.rule_id = %s
           ORDER BY rcl.role, rcl.fallback_group, ce.name""",
        (rule_id,),
    ) or []
    ces = []
    for r in ce_rows:
        ex = r.get("examples")
        if isinstance(ex, str):
            try:
                ex = _json.loads(ex)
            except Exception:
                ex = []
        if not isinstance(ex, list):
            ex = []
        ces.append({
            "ce_id": r["ce_id"],
            "name": r.get("name"),
            "definition": r.get("definition") or "",
            "examples": ex,
            "role": r.get("role") or "necessary",
            "fallback_group": r.get("fallback_group") or 0,
        })
    return {
        "rule_id": rule["rule_id"],
        "name": rule.get("name"),
        "predicate": rule.get("predicate") or "",
        "description": rule.get("description") or "",
        "public_id": rule.get("public_id"),
        "created_by_username": rule.get("created_by_username"),
        "is_local_draft": rule.get("is_local_draft"),
        "ces": ces,
    }


# --- PUBLIC LIBRARY ---

@router.get("/public/library")
def get_public_rules_endpoint():
    try:
        rules = get_all_public_rules()
        return {"rules": rules}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/public/create")
def create_public_rule(req: CreatePublicRuleRequest, _: int = Depends(get_current_user)):
    """Create a public rule with automatic category generation"""
    try:
        # Normalize roles: if new fields are present, use them; otherwise treat ce_names as necessary
        necessary = req.necessary or req.ce_names or []
        fallback = req.fallback or []
        sufficient = req.sufficient or []

        # Ensure all referenced CEs exist (create_ce is idempotent)
        all_ce_names = set(necessary)
        for group in fallback:
            all_ce_names.update(group)
        all_ce_names.update(sufficient)

        categories = req.categories or []
        for name in all_ce_names:
            create_ce(req.user_id, name, categories=categories)

        # Persist rule with role-aware links
        rule_data = {
            "rule_name": req.name,
            "predicate": req.predicate,
            "necessary": necessary,
            "fallback": fallback,
            "sufficient": sufficient, 
            "description": req.definition,
            "categories": categories
        }

        rule_id = upsert_rule_with_links(rule_data)

        return {
            "rule_id": rule_id,
            "categories": categories,
            "categorization_source": "manual",
            "categorization_confidence": 1.0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Rule Bookmarks ---

@router.get("/public/bookmarks/{user_id}")
def get_rule_bookmarks(user_id: int):
    """List rule bookmarks (user_id in the path is ignored — there is one
    local user)."""
    try:
        return {"bookmarks": BookmarkService.list(_BOOKMARK_ASSET)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/public/bookmark")
def bookmark_rule(req: RuleBookmarkRequest):
    from services.bookmarks import BookmarkLookupError
    try:
        BookmarkService.add(_BOOKMARK_ASSET, req.rule_id)
        return {"status": "bookmarked"}
    except BookmarkLookupError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/public/bookmark/{user_id}/{rule_id}")
def remove_rule_bookmark_endpoint(user_id: int, rule_id: int):
    try:
        BookmarkService.remove(_BOOKMARK_ASSET, rule_id)
        return {"status": "removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PUBLIC RULE SETS ---
# A rule set is a model-agnostic, shareable collection of published rules.
# These mirror the public-rule endpoints above; bookmarks reuse the generic
# BookmarkService with asset_type "rule_set".
_RULE_SET_BOOKMARK_ASSET = "rule_set"


@router.get("/public/rule-sets")
def get_public_rule_sets_endpoint():
    try:
        return {"rule_sets": get_all_public_rule_sets()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/public/rule-set/bookmarks/{user_id}")
def get_rule_set_bookmarks(user_id: int):
    """List rule-set bookmarks (path user_id ignored — one local user)."""
    try:
        return {"bookmarks": BookmarkService.list(_RULE_SET_BOOKMARK_ASSET)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/public/rule-set/bookmark")
def bookmark_rule_set(req: RuleSetBookmarkRequest):
    from services.bookmarks import BookmarkLookupError
    try:
        BookmarkService.add(_RULE_SET_BOOKMARK_ASSET, req.rule_set_id)
        return {"status": "bookmarked"}
    except BookmarkLookupError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/public/rule-set/bookmark/{user_id}/{rule_set_id}")
def remove_rule_set_bookmark_endpoint(user_id: int, rule_set_id: int):
    try:
        BookmarkService.remove(_RULE_SET_BOOKMARK_ASSET, rule_set_id)
        return {"status": "removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/public/rule-set/{public_id}/detail")
def get_rule_set_detail_endpoint(public_id: str, _: int = Depends(get_current_user)):
    try:
        detail = get_rule_set_detail(public_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Rule set not found")
        return detail
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))