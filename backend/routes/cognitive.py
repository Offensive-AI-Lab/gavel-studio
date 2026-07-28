import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
from utils.auth import get_current_user
from utils.PostgreSQL import execute_query_dict
from sql_scripts.definition_scripts import (
    get_user_ces,
    create_ce,
)
from services.bookmarks import BookmarkService
from utils.text_safety import clean_text

# Same BookmarkService used by routes/rules.py — this is the single
# implementation; the asset_type literal is the only thing that differs.
_BOOKMARK_ASSET = "ce"

router = APIRouter()

class CreateCERequest(BaseModel):
    user_id: int
    name: str = Field(..., max_length=120)
    definition: str = Field(default="", max_length=4000)
    categories: List[str] = []

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


class CEBookmarkRequest(BaseModel):
    user_id: int
    ce_id: int

# Two-segment path — registered safely alongside the one-segment /{user_id}.
@router.get("/element/{ce_id}")
def get_cognitive_element(ce_id: int, _: int = Depends(get_current_user)):
    """A single CE's detail (definition + curated examples) for the rule page."""
    rows = execute_query_dict(
        "SELECT ce_id, name, definition, type, examples FROM cognitive_elements WHERE ce_id = %s",
        (ce_id,),
    ) or []
    if not rows:
        raise HTTPException(status_code=404, detail="Cognitive element not found")
    row = rows[0]
    ex = row.get("examples")
    if isinstance(ex, str):
        try:
            ex = json.loads(ex)
        except Exception:
            ex = []
    if not isinstance(ex, list):
        ex = []
    return {
        "ce_id": row["ce_id"],
        "name": row.get("name"),
        "definition": row.get("definition") or "",
        "type": row.get("type"),
        "examples": ex,
    }


@router.get("/{user_id}")
def get_ces(user_id: int):
    """Get CEs bookmarked by the user"""
    try:
        return get_user_ces()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create")
def create_new_ce(req: CreateCERequest, _: int = Depends(get_current_user)):
    """Create or Find a Cognitive Element"""
    try:
        # create_ce handles the "Find or Create" logic
        result = create_ce(req.user_id, req.name, definition=req.definition, categories=req.categories)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bookmarks/{user_id}")
def get_ce_bookmarks(user_id: int):
    """List CE bookmarks (user_id in the path is ignored — there is one
    local user)."""
    try:
        return {"bookmarks": BookmarkService.list(_BOOKMARK_ASSET)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bookmark")
def add_ce_bookmark_endpoint(req: CEBookmarkRequest):
    from services.bookmarks import BookmarkLookupError
    try:
        BookmarkService.add(_BOOKMARK_ASSET, req.ce_id)
        return {"status": "bookmarked"}
    except BookmarkLookupError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/bookmark/{user_id}/{ce_id}")
def remove_ce_bookmark_endpoint(user_id: int, ce_id: int):
    try:
        BookmarkService.remove(_BOOKMARK_ASSET, ce_id)
        return {"status": "removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))