"""Export local artifacts as gavel-rules contribution files.

The studio is read-only against the registry, so "contributing" means handing
the user files they can drop into a gavel-rules pull request. Every endpoint
here is a READ that returns a downloadable file; nothing is published and
nothing leaves the machine.

Shape mirrors the registry layout:

    ces/<name>/       ce.yaml, excitation.json, calibration.json
    rules/<name>/     rule.yaml, tests/{positive,negative,positive_calibration}.json
    rulesets/         <name>.yaml

`/preflight` answers "what would I get, and what is wrong with it" before the
user commits to a download. Problems come back split:

- errors   — the registry's CI would reject the files themselves (missing
             role, empty definition, no datasets, unknown category, no GitHub
             username). While any exist, the download endpoints refuse with
             422 so an invalid contribution can't leave the studio.
- warnings — advisory only (a referenced CE/rule exists only locally); the
             legitimate fix can be "contribute both in the same PR", so these
             never block.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from services import registry_export as rx
from utils.auth import get_current_user
from utils.ownership import assert_owns_classifier
from utils.sqlite_db import execute_query_dict

router = APIRouter()

# Where the studio remembers the contributor's GitHub handle. The registry
# requires provenance.created_by, and its CI compares it against the PR author,
# so this is per-operator identity rather than a cosmetic preference.
_GITHUB_USER_KEY = "export.github_username"


# --- settings ---------------------------------------------------------------

def _get_setting(key: str) -> str:
    rows = execute_query_dict("SELECT value FROM _app_meta WHERE key = %s", (key,))
    return (rows[0]["value"] if rows else "") or ""


@router.get("/settings")
def get_export_settings(user_id: int = Depends(get_current_user)):
    """The remembered GitHub handle, so the export dialog only asks once."""
    return {"github_username": _get_setting(_GITHUB_USER_KEY)}


@router.put("/settings")
def put_export_settings(payload: dict, user_id: int = Depends(get_current_user)):
    username = (payload or {}).get("github_username", "").strip()
    execute_query_dict(
        "INSERT INTO _app_meta (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
        (_GITHUB_USER_KEY, username),
    )
    return {"github_username": username}


def _author(author: str | None) -> str:
    """Explicit handle wins; otherwise fall back to the remembered one."""
    return (author or "").strip() or _get_setting(_GITHUB_USER_KEY) or ""


def _yaml_response(doc: dict, filename: str) -> Response:
    return Response(
        content=rx.dump_yaml(doc),
        media_type="application/x-yaml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _json_response(doc: dict, filename: str) -> Response:
    return Response(
        content=json.dumps(doc, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _author_errors(author: str) -> list[str]:
    if not author:
        return ["no GitHub username — provenance.created_by is required and the "
                "registry checks it against the pull-request author"]
    return []


def _block_on(errors: list[str]) -> None:
    """Refuse a download while the export would be rejected upstream.

    The modal disables its button on the same error list, so hitting this
    means someone bypassed preflight — the message repeats what's wrong.
    """
    if errors:
        raise HTTPException(
            status_code=422,
            detail="Export blocked — fix first: " + "; ".join(errors),
        )


# --- row loaders ------------------------------------------------------------

def _load_ce(ce_id: int) -> dict:
    rows = execute_query_dict(
        "SELECT ce_id, name, title, definition, role, categories, tags, examples, "
        "       public_id, is_local_draft "
        "FROM cognitive_elements WHERE ce_id = %s",
        (ce_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Cognitive element not found")
    return rows[0]


def _load_rule(rule_id: int) -> dict:
    rows = execute_query_dict(
        "SELECT rule_id, name, title, description, categories, tags, ce_groups, "
        "       condition, public_id, is_local_draft "
        "FROM rules WHERE rule_id = %s",
        (rule_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Rule not found")
    return rows[0]


def _rule_category_names(rule: dict) -> list[str]:
    """rules.categories holds local integer ids; the registry wants names."""
    ids = [c for c in rx._coerce_json(rule.get("categories"), []) if isinstance(c, int)]
    if not ids:
        return []
    rows = execute_query_dict(
        "SELECT category_id, name FROM categories WHERE category_id = ANY(%s)", (ids,)
    )
    by_id = {r["category_id"]: r["name"] for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def _local_only_ces(rule: dict) -> list[str]:
    """Member CE names the registry cannot resolve — drafts or deleted rows.

    rule.yaml references CEs by bare name, so a member that never left this
    studio makes the contributed rule unresolvable until the CE is
    contributed too (same situation a rule set is in with local-only rules).
    """
    groups = rx._coerce_json(rule.get("ce_groups"), {})
    names = sorted({str(m) for members in groups.values() if isinstance(members, list)
                    for m in members if m})
    if not names:
        return []
    rows = execute_query_dict(
        "SELECT name, is_local_draft FROM cognitive_elements WHERE name = ANY(%s)",
        (names,),
    )
    published = {r["name"] for r in rows if not r["is_local_draft"]}
    return [n for n in names if n not in published]


def _load_classifier(classifier_id: int, user_id: int) -> dict:
    assert_owns_classifier(user_id, classifier_id)
    rows = execute_query_dict(
        "SELECT classifier_id, name FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Rule set not found")
    return rows[0]


def _classifier_members(classifier_id: int) -> tuple[list[str], list[str]]:
    """Member rule names, plus the subset that exists only in this studio.

    A rule set references rules BY NAME, so a member that never came from the
    library resolves to nothing upstream — the user has to contribute it
    separately or drop it from the set.
    """
    rows = execute_query_dict(
        "SELECT s.custom_name, r.name AS rule_name, r.is_local_draft "
        "FROM rule_setup s LEFT JOIN rules r ON r.rule_id = s.rule_id "
        "WHERE s.classifier_id = %s ORDER BY s.setup_id",
        (classifier_id,),
    )
    members, local_only = [], []
    for row in rows:
        name = row.get("rule_name") or row.get("custom_name")
        if not name:
            continue
        name = rx.slugify_name(name)
        members.append(name)
        # No backing library row, or a row that was never published upstream.
        if row.get("rule_name") is None or row.get("is_local_draft"):
            local_only.append(name)
    return members, local_only


# --- export assembly (shared by preflight and every download) ----------------

def _ce_export(ce_id: int, author: str | None, title: str | None,
               role: str | None) -> tuple[dict, list, list[str], list[str]]:
    """Returns (doc, files, errors, warnings) for one CE."""
    ce = _load_ce(ce_id)
    doc, errors, warnings = rx.build_ce_doc(ce, author=_author(author),
                                            title=title, role=role)
    errors = _author_errors(_author(author)) + errors

    files = [{"filename": "ce.yaml", "path": rx.repo_path("ce", doc["name"]), "kind": "yaml"}]
    for kind, table in (("excitation", "excitation_datasets"),
                        ("calibration", "calibration_datasets")):
        rows = execute_query_dict(
            f"SELECT dataset_id FROM {table} WHERE ce_id = %s LIMIT 1", (ce_id,)
        )
        if rows:
            files.append({
                "filename": f"{kind}.json",
                "path": rx.repo_path("ce", doc["name"], f"{kind}.json"),
                "kind": "json",
            })
        else:
            errors.append(f"no {kind} data generated yet — the registry requires "
                          f"{kind}.json alongside ce.yaml")
    return doc, files, errors, warnings


def _rule_export(rule_id: int, author: str | None,
                 title: str | None) -> tuple[dict, list, list[str], list[str]]:
    """Returns (doc, files, errors, warnings) for one rule."""
    rule = _load_rule(rule_id)
    doc, errors, warnings = rx.build_rule_doc(
        rule, author=_author(author), category_names=_rule_category_names(rule),
        title=title, local_only_ces=_local_only_ces(rule),
    )
    errors = _author_errors(_author(author)) + errors

    files = [{"filename": "rule.yaml", "path": rx.repo_path("rule", doc["name"]), "kind": "yaml"}]
    present = {
        r["dataset_type"] for r in execute_query_dict(
            "SELECT DISTINCT dataset_type FROM test_datasets "
            "WHERE rule_id = %s AND conversations IS NOT NULL", (rule_id,))
    }
    for bucket in rx.RULE_TEST_BUCKETS:
        if bucket in present:
            files.append({
                "filename": f"{bucket}.json",
                "path": rx.repo_path("rule", doc["name"], f"tests/{bucket}.json"),
                "kind": "json",
            })
        else:
            errors.append(f"no {bucket} test set generated yet — the registry "
                          f"requires tests/{bucket}.json")
    return doc, files, errors, warnings


def _ruleset_export(classifier_id: int, user_id: int, author: str | None,
                    title: str | None, description: str,
                    ) -> tuple[dict, list[str], list[str], list[str], list[str]]:
    """Returns (doc, members, local_only, errors, warnings) for one rule set."""
    classifier = _load_classifier(classifier_id, user_id)
    members, local_only = _classifier_members(classifier_id)
    try:
        doc, errors, warnings = rx.build_ruleset_doc(
            name=rx.slugify_name(classifier["name"]), members=members,
            author=_author(author), title=title or classifier["name"],
            description=description, local_only=local_only,
        )
    except rx.ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    errors = _author_errors(_author(author)) + errors
    return doc, members, local_only, errors, warnings


# --- cognitive elements -----------------------------------------------------

@router.get("/ce/{ce_id}/preflight")
def ce_preflight(ce_id: int, author: str | None = Query(None),
                 title: str | None = Query(None), role: str | None = Query(None),
                 user_id: int = Depends(get_current_user)):
    doc, files, errors, warnings = _ce_export(ce_id, author, title, role)
    return {
        "kind": "ce", "name": doc["name"], "title": doc["title"],
        "role": doc["role"], "roles": list(rx.CE_ROLES),
        "directory": f"ces/{doc['name']}", "files": files,
        "errors": errors, "warnings": warnings, "preview": rx.dump_yaml(doc),
    }


@router.get("/ce/{ce_id}/ce.yaml")
def ce_yaml(ce_id: int, author: str | None = Query(None), title: str | None = Query(None),
            role: str | None = Query(None), user_id: int = Depends(get_current_user)):
    doc, _files, errors, _warnings = _ce_export(ce_id, author, title, role)
    _block_on(errors)
    return _yaml_response(doc, "ce.yaml")


@router.get("/ce/{ce_id}/{kind}.json")
def ce_dataset(ce_id: int, kind: str, author: str | None = Query(None),
               role: str | None = Query(None), user_id: int = Depends(get_current_user)):
    if kind not in ("excitation", "calibration"):
        raise HTTPException(status_code=404, detail="Unknown dataset")
    doc, _files, errors, _warnings = _ce_export(ce_id, author, None, role)
    _block_on(errors)

    table = "excitation_datasets" if kind == "excitation" else "calibration_datasets"
    rows = execute_query_dict(
        f"SELECT dataset FROM {table} WHERE ce_id = %s ORDER BY dataset_id DESC LIMIT 1",
        (ce_id,),
    )
    if not rows:  # unreachable while a missing dataset is a blocking error
        raise HTTPException(status_code=404, detail=f"No {kind} data for this CE")

    payload = rx._coerce_json(rows[0]["dataset"], {})
    # Local excitation stores `samples`, calibration stores `conversations`;
    # the registry keys both on `samples`.
    samples = payload.get("samples") or payload.get("conversations") or []
    return _json_response(
        rx.build_ce_dataset(kind, ce_name=doc["name"],
                            ce_public_id=doc["public_id"], samples=samples),
        f"{kind}.json",
    )


# --- rules ------------------------------------------------------------------

@router.get("/rule/{rule_id}/preflight")
def rule_preflight(rule_id: int, author: str | None = Query(None),
                   title: str | None = Query(None),
                   user_id: int = Depends(get_current_user)):
    doc, files, errors, warnings = _rule_export(rule_id, author, title)
    return {
        "kind": "rule", "name": doc["name"], "title": doc["title"],
        "directory": f"rules/{doc['name']}", "files": files,
        "errors": errors, "warnings": warnings, "preview": rx.dump_yaml(doc),
    }


@router.get("/rule/{rule_id}/rule.yaml")
def rule_yaml(rule_id: int, author: str | None = Query(None),
              title: str | None = Query(None),
              user_id: int = Depends(get_current_user)):
    doc, _files, errors, _warnings = _rule_export(rule_id, author, title)
    _block_on(errors)
    return _yaml_response(doc, "rule.yaml")


@router.get("/rule/{rule_id}/tests/{bucket}.json")
def rule_test(rule_id: int, bucket: str, author: str | None = Query(None),
              user_id: int = Depends(get_current_user)):
    if bucket not in rx.RULE_TEST_BUCKETS:
        raise HTTPException(status_code=404, detail="Unknown test bucket")
    doc, _files, errors, _warnings = _rule_export(rule_id, author, None)
    _block_on(errors)

    rows = execute_query_dict(
        "SELECT conversations, config FROM test_datasets "
        "WHERE rule_id = %s AND dataset_type = %s AND conversations IS NOT NULL "
        "ORDER BY is_default DESC, dataset_id DESC LIMIT 1",
        (rule_id, bucket),
    )
    if not rows:  # unreachable while a missing test set is a blocking error
        raise HTTPException(status_code=404, detail=f"No {bucket} test set for this rule")

    conversations = rx._coerce_json(rows[0]["conversations"], [])
    config = rx._coerce_json(rows[0]["config"], {})
    instructions = (config.get("scenario_instructions")
                    or config.get("instructions")
                    or doc["definition"])
    return _json_response(
        rx.build_rule_test(bucket, rule_name=doc["name"], rule_public_id=doc["public_id"],
                           conversations=conversations, scenario_instructions=instructions),
        f"{bucket}.json",
    )


# --- rule sets --------------------------------------------------------------

@router.get("/ruleset/{classifier_id}/preflight")
def ruleset_preflight(classifier_id: int, author: str | None = Query(None),
                      title: str | None = Query(None),
                      description: str = Query(""),
                      user_id: int = Depends(get_current_user)):
    doc, members, local_only, errors, warnings = _ruleset_export(
        classifier_id, user_id, author, title, description)
    return {
        "kind": "ruleset", "name": doc["name"], "title": doc["title"],
        "directory": "rulesets", "members": members, "local_only": local_only,
        "files": [{"filename": f"{doc['name']}.yaml",
                   "path": rx.repo_path("ruleset", doc["name"]), "kind": "yaml"}],
        "errors": errors, "warnings": warnings, "preview": rx.dump_yaml(doc),
    }


@router.get("/ruleset/{classifier_id}/ruleset.yaml")
def ruleset_yaml(classifier_id: int, author: str | None = Query(None),
                 title: str | None = Query(None), description: str = Query(""),
                 user_id: int = Depends(get_current_user)):
    doc, _members, _local_only, errors, _warnings = _ruleset_export(
        classifier_id, user_id, author, title, description)
    _block_on(errors)
    return _yaml_response(doc, f"{doc['name']}.yaml")
