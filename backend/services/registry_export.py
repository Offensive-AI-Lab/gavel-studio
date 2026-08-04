"""Serialize local rules / CEs / rule sets into gavel-rules registry artifacts.

The registry (Offensive-AI-Lab/gavel-rules) stores every artifact as YAML —
`ces/<name>/ce.yaml`, `rules/<name>/rule.yaml`, `rulesets/<name>.yaml` — with
the companion datasets as JSON. `index.json` is a catalog their CI regenerates,
so it is never something we emit.

Everything here is pure: rows in, dicts/strings out. No DB, no FastAPI — the
route layer does the querying and this module does the shaping, which is what
makes the field-by-field format contract testable on its own.

Field order matters only cosmetically (the registry's own files read in this
order), but the *validator* contract is exact and is mirrored in EXPORT_RULES
below. Anything we cannot satisfy from local data is reported rather than
silently guessed, split two ways:

- errors   — the registry's validator rejects the file itself (bad role, no
             examples, dead groups, ...). The routes refuse to serve the files
             while any exist, so an invalid contribution can't be downloaded.
- warnings — the file is valid but references something the library doesn't
             have yet (a draft CE / rule). Advisory only: contributing both in
             the same pull request is legitimate, so this never blocks.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import yaml

# --- registry constants -----------------------------------------------------
# Mirrored from gavel-rules tools/validate.py. They are schema, not config:
# a mismatch here means CI rejects the PR, so they are pinned deliberately
# rather than fetched at runtime (the studio must work offline).
CE_ROLES = ("directive_to_user", "llm_task", "llm_behavior", "topic")

REGISTRY_CATEGORIES = (
    "Domain & Business Logic",
    "Fairness & Ethics",
    "Legal & Compliance",
    "Output Quality & Operational",
    "Privacy & Data Protection",
    "Resource & Cost Management",
    "Safety & Harm Prevention",
    "Security & Defense",
    "Tone & Style",
    "Utility & Tools",
)

CE_SCHEMA_VERSION = 2
RULE_SCHEMA_VERSION = 2
RULESET_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1

RULE_TEST_BUCKETS = ("positive", "negative", "positive_calibration")

# uuid5 namespace so a given artifact always exports with the same public_id.
# Minting a fresh uuid4 per export would silently fork an artifact's identity
# every time the user clicked the button.
_PUBLIC_ID_NS = uuid.UUID("6f9b1c1e-6f0a-4a1e-9a1f-7d3c5a2b8e40")


class ExportError(ValueError):
    """The artifact cannot be serialized at all (missing name, no members, ...)."""


def mint_public_id(kind: str, name: str) -> str:
    """Deterministic public_id for artifacts that never got one from the registry.

    Registry ids look like `ce_<32 hex>`; uuid5 keeps repeat exports stable.
    """
    if kind not in ("ce", "rule", "ruleset"):
        raise ExportError(f"unknown artifact kind {kind!r}")
    return f"{kind}_{uuid.uuid5(_PUBLIC_ID_NS, f'{kind}:{name}').hex}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce_json(value, default):
    """Rows arrive as dicts/lists (JSONB round-trip) or as raw JSON strings."""
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _title_from_name(name: str) -> str:
    return " ".join(part for part in str(name).replace("_", " ").split()).title()


def _provenance(author: str) -> dict:
    """Exports are always stamped with the current time — the registry treats
    the pull request as the publication event, not whatever the local row says."""
    return {
        "created_by": author,
        "published_at": _now_iso(),
    }


class _FlowMap(dict):
    """A mapping the emitter should write inline, e.g. {input: ..., output: 'YES'}."""


class _FlowList(list):
    """A sequence the emitter should write inline, e.g. [content_creation, click_or_enter]."""


class _RegistryDumper(yaml.SafeDumper):
    """Block style everywhere except mappings explicitly marked _FlowMap.

    That is the shape the registry's own artifacts use — block lists for
    categories/rules/examples, inline maps for each example — so exported files
    diff cleanly against hand-authored neighbours in a pull request.
    """


def _represent_flow_map(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


def _represent_flow_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_RegistryDumper.add_representer(_FlowMap, _represent_flow_map)
_RegistryDumper.add_representer(_FlowList, _represent_flow_list)


def dump_yaml(doc: dict) -> str:
    """Emit registry-style YAML: declared key order, block style, inline examples."""
    return yaml.dump(
        doc,
        Dumper=_RegistryDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )


# --- cognitive elements -----------------------------------------------------

def build_ce_doc(ce: dict, *, author: str, title: str | None = None,
                 role: str | None = None) -> tuple[dict, list[str], list[str]]:
    """Build ce.yaml. Returns (doc, errors, warnings) — see module docstring."""
    name = (ce.get("name") or "").strip()
    if not name:
        raise ExportError("cognitive element has no name")

    errors: list[str] = []
    resolved_title = (title or ce.get("title") or "").strip() or _title_from_name(name)
    resolved_role = (role or ce.get("role") or "").strip()
    if resolved_role not in CE_ROLES:
        errors.append(
            f"role {resolved_role or '(empty)'!r} is not one of {', '.join(CE_ROLES)} — "
            "the registry will reject this file until it is set"
        )

    examples = []
    for ex in _coerce_json(ce.get("examples"), []):
        if not isinstance(ex, dict):
            continue
        text = ex.get("input")
        output = ex.get("output")
        if not text or output not in ("YES", "NO"):
            continue
        examples.append(_FlowMap(input=text, output=output))
    if not examples:
        errors.append("no usable examples — the registry requires at least one")

    definition = (ce.get("definition") or "").strip()
    if not definition:
        errors.append("definition is empty — the registry requires one")

    doc = {
        "name": name,
        "title": resolved_title,
        "public_id": ce.get("public_id") or mint_public_id("ce", name),
        "schema_version": CE_SCHEMA_VERSION,
        "role": resolved_role,
        "definition": definition,
        "examples": examples,
    }
    tags = _coerce_json(ce.get("tags"), [])
    if tags:
        doc["tags"] = tags
    doc["provenance"] = _provenance(author)
    return doc, errors, []


def build_ce_dataset(kind: str, *, ce_name: str, ce_public_id: str,
                     samples: list, published_at: str | None = None) -> dict:
    """excitation.json / calibration.json — both keyed on `samples` upstream."""
    if kind not in ("excitation", "calibration"):
        raise ExportError(f"unknown CE dataset kind {kind!r}")
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "ce": ce_name,
        "ce_public_id": ce_public_id,
        "type": kind,
        "published_at": published_at or _now_iso(),
        "sample_count": len(samples),
        "samples": samples,
    }


# --- rules ------------------------------------------------------------------

def build_rule_doc(rule: dict, *, author: str, category_names: list[str] | None = None,
                   title: str | None = None,
                   local_only_ces: list[str] | None = None) -> tuple[dict, list[str], list[str]]:
    """Build rule.yaml. Returns (doc, errors, warnings) — see module docstring."""
    name = (rule.get("name") or "").strip()
    if not name:
        raise ExportError("rule has no name")

    errors: list[str] = []
    categories = [c for c in (category_names or []) if c]
    unknown = [c for c in categories if c not in REGISTRY_CATEGORIES]
    if unknown:
        errors.append(
            f"category {', '.join(repr(c) for c in unknown)} is not in the registry "
            "taxonomy and will be rejected"
        )
    if not categories:
        errors.append("no categories — the registry requires at least one")

    groups = _coerce_json(rule.get("ce_groups"), {})
    if not isinstance(groups, dict) or not groups:
        errors.append("no CE groups — the registry requires a non-empty groups map")
        groups = {}
    condition = (rule.get("condition") or "").strip()
    if not condition:
        errors.append("no condition — the registry requires one")

    referenced = {g for g in groups if g and g in condition}
    dead = sorted(set(groups) - referenced)
    if condition and dead:
        errors.append(
            f"group(s) {', '.join(repr(g) for g in dead)} are defined but never used "
            "in the condition — the registry rejects dead groups"
        )

    definition = (rule.get("description") or "").strip()
    if not definition:
        errors.append("definition is empty — the registry requires one")

    warnings = [
        f"CE {member!r} exists only in this studio — the exported rule will "
        "point at a CE the library does not have until you contribute it too"
        for member in (local_only_ces or [])
    ]

    doc = {
        "name": name,
        "title": (title or rule.get("title") or "").strip() or _title_from_name(name),
        "public_id": rule.get("public_id") or mint_public_id("rule", name),
        "schema_version": RULE_SCHEMA_VERSION,
        "categories": categories,
        "definition": definition,
        "groups": {k: _FlowList(v) for k, v in groups.items()},
        "condition": condition,
    }
    tags = _coerce_json(rule.get("tags"), [])
    if tags:
        doc["tags"] = tags
    doc["provenance"] = _provenance(author)
    return doc, errors, warnings


def build_rule_test(bucket: str, *, rule_name: str, rule_public_id: str,
                    conversations: list, scenario_instructions: str,
                    published_at: str | None = None) -> dict:
    """tests/<bucket>.json — `config` must be exactly {scenario_instructions}."""
    if bucket not in RULE_TEST_BUCKETS:
        raise ExportError(f"unknown rule test bucket {bucket!r}")
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "rule": rule_name,
        "rule_public_id": rule_public_id,
        "type": bucket,
        "published_at": published_at or _now_iso(),
        "config": {"scenario_instructions": scenario_instructions},
        "conversation_count": len(conversations),
        "conversations": conversations,
    }


# --- rule sets --------------------------------------------------------------

def build_ruleset_doc(*, name: str, members: list[str], author: str,
                      title: str | None = None, description: str = "",
                      public_id: str | None = None,
                      local_only: list[str] | None = None) -> tuple[dict, list[str], list[str]]:
    """Build rulesets/<name>.yaml — a title, a description and member rule names.

    Returns (doc, errors, warnings) — see module docstring."""
    name = (name or "").strip()
    if not name:
        raise ExportError("rule set has no name")

    errors: list[str] = []
    seen, ordered = set(), []
    for m in members:
        m = (m or "").strip()
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)
    if not ordered:
        raise ExportError("rule set has no rules to export")

    if not (description or "").strip():
        errors.append("description is empty — the registry requires one")
    warnings = [
        f"rule {member!r} exists only in this studio — the exported set will "
        "point at a rule the library does not have until you contribute it too"
        for member in (local_only or [])
    ]

    doc = {
        "name": name,
        "title": (title or "").strip() or _title_from_name(name),
        "public_id": public_id or mint_public_id("ruleset", name),
        "schema_version": RULESET_SCHEMA_VERSION,
        "description": (description or "").strip(),
        "rules": ordered,
        "provenance": _provenance(author),
    }
    return doc, errors, warnings


# --- repo placement ---------------------------------------------------------

def repo_path(kind: str, name: str, filename: str | None = None) -> str:
    """Where the file belongs inside a gavel-rules checkout."""
    if kind == "ce":
        return f"ces/{name}/{filename or 'ce.yaml'}"
    if kind == "rule":
        return f"rules/{name}/{filename or 'rule.yaml'}"
    if kind == "ruleset":
        return f"rulesets/{name}.yaml"
    raise ExportError(f"unknown artifact kind {kind!r}")


def slugify_name(name: str) -> str:
    """Registry artifact names are [A-Za-z0-9_]+ — used for local names."""
    cleaned = [c if (c.isalnum() or c == "_") else "_" for c in (name or "").strip()]
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.lower() or "unnamed"
