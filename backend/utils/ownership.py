"""Object-level authorization guards (defense against IDOR).

The app-wide auth gate in main.py guarantees a request is *authenticated*; these
helpers enforce that the authenticated user actually *owns* the resource they
name by id. That matters because a single backend can be shared by a team
(see services.user._mirror_team_users / is_team users), so more than one user's
rows can live in one database.

Ownership chain in the schema:
    users(user_id)
      ├─ target_models(model_id, user_id)
      └─ classifiers(classifier_id, user_id, model_id NULLABLE)
                └─ rule_setup(setup_id, classifier_id)

A classifier (UI "guardrail") is owned DIRECTLY by a user via classifiers.user_id
(added in schema v15). That decoupling matters because a guardrail can exist
before a model is chosen — model_id is NULL until the user attaches one at train
time — so ownership can no longer be derived through the model.

Cognitive elements and the public `rules` registry have NO per-user owner column
(they are shared libraries), so they are protected by authentication alone — not
by these guards.

Guards raise 404 (not 403) on a miss so they don't leak whether an id exists to a
user who isn't allowed to see it.
"""
from fastapi import Depends, HTTPException, Path

from utils.sqlite_db import execute_query_dict
from utils.auth import get_current_user


def _exists(sql: str, params: tuple) -> bool:
    rows = execute_query_dict(sql, params)
    return bool(rows)


def assert_owns_model(user_id: int, model_id: int) -> None:
    if not _exists(
        "SELECT 1 FROM target_models WHERE model_id = %s AND user_id = %s",
        (model_id, user_id),
    ):
        raise HTTPException(status_code=404, detail="Model not found")


def assert_owns_classifier(user_id: int, classifier_id: int) -> None:
    # Direct owner check (classifiers.user_id) so it holds for unattached
    # guardrails too — model_id may be NULL before a model is picked.
    if not _exists(
        "SELECT 1 FROM classifiers WHERE classifier_id = %s AND user_id = %s",
        (classifier_id, user_id),
    ):
        raise HTTPException(status_code=404, detail="Classifier not found")


def assert_owns_setup(user_id: int, setup_id: int) -> None:
    if not _exists(
        "SELECT 1 FROM rule_setup rs "
        "JOIN classifiers c ON rs.classifier_id = c.classifier_id "
        "WHERE rs.setup_id = %s AND c.user_id = %s",
        (setup_id, user_id),
    ):
        raise HTTPException(status_code=404, detail="Rule setup not found")


# --- FastAPI dependencies -------------------------------------------------
# Drop-in route guards: they pull the id straight from the path and the user
# from the token, so an endpoint (or a whole router) becomes ownership-checked
# with a single `dependencies=[Depends(require_*_owner)]` — no body changes.

def require_model_owner(model_id: int = Path(...),
                        uid: int = Depends(get_current_user)) -> int:
    assert_owns_model(uid, model_id)
    return uid


def require_classifier_owner(classifier_id: int = Path(...),
                             uid: int = Depends(get_current_user)) -> int:
    assert_owns_classifier(uid, classifier_id)
    return uid


def require_setup_owner(setup_id: int = Path(...),
                        uid: int = Depends(get_current_user)) -> int:
    assert_owns_setup(uid, setup_id)
    return uid
