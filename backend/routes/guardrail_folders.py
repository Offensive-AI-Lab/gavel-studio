"""Guardrail folder endpoints — the "library" grouping over the Guardrails page.

All routes are per-user (get_current_user) and operate only on the caller's own
folders/guardrails; the service layer enforces ownership.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from utils.auth import get_current_user
from services import guardrail_folders as gf

router = APIRouter()


class FolderCreate(BaseModel):
    name: str = "New folder"


class FolderRename(BaseModel):
    name: str


class AssignBody(BaseModel):
    classifier_id: int
    folder_id: Optional[int] = None   # None = ungroup


@router.get("")
def get_state(uid: int = Depends(get_current_user)):
    """The user's folders. Guardrail membership comes from the
    /classifiers/details/all payload (each guardrail carries its folder_id)."""
    return {"folders": gf.list_folders(uid)}


@router.post("")
def create(body: FolderCreate, uid: int = Depends(get_current_user)):
    return {"folder": gf.create_folder(uid, body.name)}


@router.patch("/{folder_id}")
def rename(folder_id: int, body: FolderRename, uid: int = Depends(get_current_user)):
    try:
        row = gf.rename_folder(uid, folder_id, body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not row:
        raise HTTPException(status_code=404, detail="Folder not found.")
    return {"folder": row}


@router.delete("/{folder_id}")
def delete(folder_id: int, uid: int = Depends(get_current_user)):
    if not gf.delete_folder(uid, folder_id):
        raise HTTPException(status_code=404, detail="Folder not found.")
    return {"ok": True}


@router.post("/assign")
def assign(body: AssignBody, uid: int = Depends(get_current_user)):
    """Move a guardrail into a folder, or out of one (folder_id null)."""
    try:
        gf.assign(uid, body.classifier_id, body.folder_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return {"ok": True}
