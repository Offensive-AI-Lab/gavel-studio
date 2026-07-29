"""Guardrail folders — a MANUAL "library" grouping over guardrails (classifiers).

A guardrail belongs to at most one folder (folder_id NULL = ungrouped). The user
decides what goes in each folder — there is no automatic arrangement. Deleting a
folder CASCADE-deletes the guardrails inside it (enforced by the FK in DButils),
so removing a folder removes its guardrails.
"""
from utils.sqlite_db import execute_query, execute_query_dict


# --- folder CRUD -----------------------------------------------------------

def list_folders(user_id: int):
    return execute_query_dict(
        "SELECT folder_id, name, created_at "
        "FROM guardrail_folders WHERE user_id = %s ORDER BY created_at, folder_id",
        (user_id,)) or []


def create_folder(user_id: int, name: str):
    name = (name or "").strip() or "New folder"
    rows = execute_query_dict(
        "INSERT INTO guardrail_folders (user_id, name) VALUES (%s, %s) "
        "RETURNING folder_id, name, created_at",
        (user_id, name))
    return rows[0] if rows else None


def rename_folder(user_id: int, folder_id: int, name: str):
    name = (name or "").strip()
    if not name:
        raise ValueError("Folder name can't be empty.")
    rows = execute_query_dict(
        "UPDATE guardrail_folders SET name = %s WHERE folder_id = %s AND user_id = %s "
        "RETURNING folder_id, name, created_at",
        (name, folder_id, user_id))
    return rows[0] if rows else None


def delete_folder(user_id: int, folder_id: int) -> bool:
    # ON DELETE CASCADE on classifiers.folder_id deletes the guardrails inside
    # the folder along with it.
    rows = execute_query_dict(
        "DELETE FROM guardrail_folders WHERE folder_id = %s AND user_id = %s "
        "RETURNING folder_id",
        (folder_id, user_id))
    return bool(rows)


# --- membership ------------------------------------------------------------

def _owns_guardrail(user_id: int, classifier_id: int) -> bool:
    return bool(execute_query_dict(
        "SELECT 1 FROM classifiers WHERE classifier_id = %s AND user_id = %s",
        (classifier_id, user_id)))


def _owns_folder(user_id: int, folder_id: int) -> bool:
    return bool(execute_query_dict(
        "SELECT 1 FROM guardrail_folders WHERE folder_id = %s AND user_id = %s",
        (folder_id, user_id)))


def assign(user_id: int, classifier_id: int, folder_id):
    """Place a guardrail in a folder, or take it out (folder_id None = ungroup)."""
    if not _owns_guardrail(user_id, classifier_id):
        raise PermissionError("That rule set isn't yours.")
    if folder_id is not None and not _owns_folder(user_id, folder_id):
        raise PermissionError("That folder isn't yours.")
    execute_query(
        "UPDATE classifiers SET folder_id = %s WHERE classifier_id = %s AND user_id = %s",
        (folder_id, classifier_id, user_id))
