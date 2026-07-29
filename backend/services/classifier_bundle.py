"""Guardrail bundle export / import — the symmetric `gavel.classifier.bundle` format.

A bundle is a plain `.zip` that lets one user hand a trained guardrail to
another. It carries the trained RNN weights + architecture meta, REFERENCES the
policy (rules + CEs) by their HuggingFace `public_id`, and — depending on the
tier the exporter chose — the calibration thresholds and/or evaluation results.

Hard invariants (decided with the team):
  * A guardrail can only be exported once its ENTIRE policy is published to the
    public library. Manual (rule_id IS NULL) rules can never be exported. The
    export caller is expected to publish any draft rules first (with the user's
    approval) — `assess_export` reports exactly what's outstanding.
  * Export is only offered when the live policy still matches the policy the
    model was trained on (no drift). `assess_export` enforces this.
  * On import we SYNC with HF first, then resolve every referenced rule/CE by
    public_id. Anything that resolves is linked with its full library data;
    anything that does NOT resolve (a private rule that was never published) is
    a hard block — we name it and refuse, rather than import a broken guardrail.

Layout inside the zip:
    bundle_manifest.json            (always)
    model/trained_rnn.pth           (always)
    model/classifier_meta.json      (always)
    calibration/thresholds.json     (tier >= model+calibration)
    evaluation/results.json         (tier == full)

NOTE: this is ordinary backend code (NOT a Workflow script), so datetime is fine.
"""
import hashlib
import io
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone

from utils.sqlite_db import execute_query, execute_query_dict

FORMAT = "gavel.classifier.bundle"
FORMAT_VERSION = 1

TIER_MODEL = "model"
TIER_CALIBRATION = "model+calibration"
TIER_FULL = "full"
_TIERS = (TIER_MODEL, TIER_CALIBRATION, TIER_FULL)

# Zip-safety bounds. A bundle is tiny (one small RNN + a few JSON files), so
# these are generous ceilings purely to blunt a hostile upload.
_MAX_FILES = 64
_MAX_UNCOMPRESSED = 4 * 1024 * 1024 * 1024  # 4 GB
_EXEC_SIGNATURES = (b"\x7fELF", b"MZ", b"#!")


class BundleError(Exception):
    """A bundle build/parse/import failure with an HTTP-ish status code.

    Routes translate `.status_code` + `.message` into an HTTPException so the
    user sees a precise, actionable reason ("add base model X first", "rule Y
    isn't published", …) instead of a generic 500.
    """

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classifier_row(classifier_id: int):
    rows = execute_query_dict(
        """
        SELECT c.classifier_id, c.name, c.status, c.model_path,
               c.trained_policy_fingerprint, c.trained_at,
               m.model_id, m.user_id,
               m.storage_path AS base_storage_path, m.name AS base_model_name
        FROM classifiers c
        JOIN target_models m ON c.model_id = m.model_id
        WHERE c.classifier_id = %s
        """,
        (classifier_id,),
    )
    return rows[0] if rows else None


def _username_for(user_id: int):
    try:
        rows = execute_query_dict("SELECT username FROM users WHERE user_id = %s", (user_id,))
        return rows[0]["username"] if rows else None
    except Exception:
        return None


def _load_meta_from_disk(classifier_id: int, user_id: int):
    from classifier_engine.trainer import classifier_workdir
    workdir = classifier_workdir(classifier_id, user_id)
    meta_path = os.path.join(workdir, "classifier_meta.json")
    if not os.path.isfile(meta_path):
        return None, workdir
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f), workdir


def _latest_eval(classifier_id: int, eval_type: str, trained_at):
    """Most recent calibration/evaluation row produced for the CURRENT model
    (i.e. created at/after trained_at, so stale pre-retrain rows never leak)."""
    if trained_at is not None:
        rows = execute_query_dict(
            """
            SELECT thresholds, metrics, plots, created_at
            FROM evaluation_results
            WHERE classifier_id = %s AND eval_type = %s AND created_at >= %s
            ORDER BY created_at DESC, eval_id DESC LIMIT 1
            """,
            (classifier_id, eval_type, trained_at),
        )
    else:
        rows = execute_query_dict(
            """
            SELECT thresholds, metrics, plots, created_at
            FROM evaluation_results
            WHERE classifier_id = %s AND eval_type = %s
            ORDER BY created_at DESC, eval_id DESC LIMIT 1
            """,
            (classifier_id, eval_type),
        )
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Policy fingerprinting (machine-INDEPENDENT — keyed on CE public_id)
# ---------------------------------------------------------------------------

def _rule_fp_from_public_links(links: list) -> str:
    """Per-rule fingerprint over CE public_ids + roles + fallback grouping.

    Unlike `compute_rule_fingerprint_from_links` (which hashes local ce_ids and
    is therefore meaningless across machines), this hashes the stable public_ids
    so the exporter and importer compute the SAME value for the same published
    rule. Used purely as an integrity check on import.
    """
    necessary, sufficient, fb = [], [], {}
    for l in links or []:
        pid = l.get("ce_public_id")
        if not pid:
            continue
        role = (l.get("role") or "necessary").lower()
        if role == "sufficient":
            sufficient.append(pid)
        elif role == "fallback":
            g = int(l.get("fallback_group", 0) or 0)
            fb.setdefault(g, []).append(pid)
        else:
            necessary.append(pid)
    nec = sorted(necessary)
    suf = sorted(sufficient)
    fbn = sorted(tuple(sorted(g)) for g in fb.values())
    canonical = f"N:{tuple(nec)}|F:{fbn}|S:{tuple(suf)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _public_rule_links(rule_id: int) -> list:
    """The canonical CE composition of a PUBLISHED rule, by CE public_id.

    Reads the rule's own rule_ce_link (not any guardrail's local setup), so the
    fingerprint reflects the immutable published rule — which is exactly what the
    importer will re-derive from the pulled copy."""
    return execute_query_dict(
        """
        SELECT ce.public_id AS ce_public_id, ce.name AS name, rcl.role,
               COALESCE(rcl.fallback_group, 0) AS fallback_group
        FROM rule_ce_link rcl
        JOIN cognitive_elements ce ON rcl.ce_id = ce.ce_id
        WHERE rcl.rule_id = %s
        ORDER BY ce.name
        """,
        (rule_id,),
    ) or []


def _local_rule_public_fp(rule_id: int) -> str:
    return _rule_fp_from_public_links(_public_rule_links(rule_id))


# ---------------------------------------------------------------------------
# Policy inspection — the heart of "is this exportable / what's outstanding"
# ---------------------------------------------------------------------------

def _inspect_policy(classifier_id: int, meta: dict) -> dict:
    """Classify the guardrail's active policy for export.

    Returns:
        {
          "rules":   [manifest rule dicts that are FULLY published],
          "ces":     [manifest ce dicts (public_id+name+label_index)],
          "manual":  [rule names that can never be exported (rule_id IS NULL)],
          "unpublished_rules": [{"rule_id", "name"}],   # draft → publishable
          "unpublished_ces":   [ce names lacking a public_id, not covered above],
        }
    """
    from sql_scripts.model_scripts import get_classifier_rules

    rows = get_classifier_rules(classifier_id) or []
    manual, unpublished_rules = [], []
    manifest_rules = []

    for r in rows:
        if not r.get("is_active"):
            continue
        rid = r.get("source_rule_id")
        name = r.get("custom_name") or "(unnamed rule)"
        if rid is None:
            manual.append(name)
            continue
        if not r.get("public_id"):
            unpublished_rules.append({"rule_id": rid, "name": name})
            continue
        links = _public_rule_links(rid)
        if any(not l.get("ce_public_id") for l in links):
            # A published rule with an unpublished CE shouldn't happen, but never
            # emit a dangling reference — treat as still-needing-publish.
            unpublished_rules.append({"rule_id": rid, "name": name})
            continue
        manifest_rules.append({
            "public_id": r["public_id"],
            "name": name,
            "rule_fingerprint": _rule_fp_from_public_links(links),
            "ce_links": [
                {
                    "ce_public_id": l["ce_public_id"],
                    "name": l.get("name"),
                    "role": l.get("role") or "necessary",
                    "fallback_group": int(l.get("fallback_group") or 0),
                }
                for l in links
            ],
        })

    # policy.ces — every output head of the trained model, by name → public_id.
    label_map = meta.get("labels") or {}
    ce_names = list(label_map.keys())
    name_pub = {}
    if ce_names:
        for row in execute_query_dict(
            "SELECT name, public_id FROM cognitive_elements WHERE name = ANY(%s)",
            (ce_names,),
        ) or []:
            name_pub[row["name"]] = row["public_id"]

    ces, unpublished_ces = [], []
    for name, idx in label_map.items():
        pid = name_pub.get(name)
        if not pid:
            unpublished_ces.append(name)
            continue
        ces.append({"public_id": pid, "name": name, "label_index": int(idx)})

    return {
        "rules": manifest_rules,
        "ces": ces,
        "manual": manual,
        "unpublished_rules": unpublished_rules,
        "unpublished_ces": unpublished_ces,
    }


def assess_export(classifier_id: int) -> dict:
    """Decide whether a guardrail can be exported and what's outstanding.

    The Export button is only shown (and the export only proceeds) when
    `can_export` is True. When it's False, `reason` explains why; `unpublished`
    lists draft rules the caller can offer to publish; `blockers` lists things
    that can't be auto-fixed (manual rules, drift).
    """
    c = _classifier_row(classifier_id)
    if not c:
        raise BundleError("Rule Set not found.", 404)

    from sql_scripts.model_scripts import reconcile_classifier_status
    status = reconcile_classifier_status(classifier_id)

    result = {
        "classifier_id": classifier_id,
        "name": c["name"],
        "can_export": False,
        "reason": None,
        "drift": False,
        "tiers_available": [],
        "unpublished": [],
        "blockers": [],
        # internal payloads reused by build_bundle_zip (underscore-prefixed)
        "_calibration": None,
        "_evaluation": None,
    }

    if status not in ("active", "needs_retraining") or not c.get("trained_at"):
        result["reason"] = "This rule set hasn't been trained yet."
        return result
    if status == "needs_retraining":
        result["drift"] = True
        result["reason"] = ("The policy has changed since this rule set was trained. "
                            "Retrain it before exporting.")
        return result

    meta, _ = _load_meta_from_disk(classifier_id, c["user_id"])
    if not meta:
        result["reason"] = "Trained model metadata is missing on disk."
        return result

    pol = _inspect_policy(classifier_id, meta)
    for nm in pol["manual"]:
        result["blockers"].append(
            f"Rule “{nm}” was created manually and isn't in the public library, so it can't be exported."
        )
    for nm in pol["unpublished_ces"]:
        # Not tied to a draft rule we can publish → genuine blocker.
        result["blockers"].append(
            f"Cognitive element “{nm}” (an output of the trained model) isn't published and isn't covered by a draft rule."
        )
    result["unpublished"] = [
        {"type": "rule", "rule_id": r["rule_id"], "name": r["name"]}
        for r in pol["unpublished_rules"]
    ]

    # Tier availability from on-disk model + calibration/evaluation rows.
    tiers = [TIER_MODEL]
    cal = _latest_eval(classifier_id, "calibration", c.get("trained_at"))
    if cal and cal.get("thresholds"):
        tiers.append(TIER_CALIBRATION)
        result["_calibration"] = cal["thresholds"]
    ev = _latest_eval(classifier_id, "evaluation", c.get("trained_at"))
    if ev:
        tiers.append(TIER_FULL)
        result["_evaluation"] = {
            "thresholds": ev.get("thresholds"),
            "metrics": ev.get("metrics"),
            "plots": ev.get("plots"),
        }
    result["tiers_available"] = tiers

    if result["blockers"]:
        result["reason"] = result["blockers"][0]
        return result
    if result["unpublished"]:
        result["reason"] = ("Some rules in this rule set's policy aren't published yet. "
                            "Publish them to the library to export.")
        # can_export stays False until they're published, but it's a SOFT block:
        # the caller can publish then re-assess.
        return result

    result["can_export"] = True
    return result


# ---------------------------------------------------------------------------
# Build (export)
# ---------------------------------------------------------------------------

def build_bundle_zip(classifier_id: int, tier: str):
    """Return (zip_bytes, filename) for the requested tier.

    Re-validates everything (no drift, whole policy published, tier available)
    so it's safe to call directly — it never emits a partial/inconsistent bundle.
    """
    if tier not in _TIERS:
        raise BundleError(f"Unknown export tier '{tier}'.", 400)

    assessment = assess_export(classifier_id)
    if not assessment["can_export"]:
        raise BundleError(assessment["reason"] or "This rule set can't be exported right now.", 409)
    if tier not in assessment["tiers_available"]:
        need = "calibration" if tier == TIER_CALIBRATION else "evaluation"
        raise BundleError(f"Tier '{tier}' isn't available — this rule set has no {need} results yet.", 409)

    c = _classifier_row(classifier_id)
    meta, workdir = _load_meta_from_disk(classifier_id, c["user_id"])
    pth_path = os.path.join(workdir, "trained_rnn.pth")
    if not meta or not os.path.isfile(pth_path):
        raise BundleError("Trained model files were not found on disk.", 404)
    with open(pth_path, "rb") as f:
        pth_bytes = f.read()

    pol = _inspect_policy(classifier_id, meta)
    if pol["manual"] or pol["unpublished_rules"] or pol["unpublished_ces"]:
        raise BundleError("The rule set's policy isn't fully published; cannot export.", 409)

    files = {
        "model/trained_rnn.pth": pth_bytes,
        "model/classifier_meta.json": json.dumps(meta, indent=2).encode("utf-8"),
    }
    if tier in (TIER_CALIBRATION, TIER_FULL):
        files["calibration/thresholds.json"] = json.dumps(
            assessment["_calibration"], indent=2
        ).encode("utf-8")
    if tier == TIER_FULL:
        files["evaluation/results.json"] = json.dumps(
            assessment["_evaluation"], indent=2
        ).encode("utf-8")

    integrity = {p: _sha256(b) for p, b in files.items()}
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "tier": tier,
        "exported_at": _now_iso(),
        "exported_by_username": _username_for(c["user_id"]),
        "base_model": {
            "storage_path": c["base_storage_path"],
            "display_name": c["base_model_name"],
        },
        "source": {"classifier_name": c["name"]},
        "policy": {
            "trained_policy_fingerprint": c.get("trained_policy_fingerprint") or "",
            "ces": pol["ces"],
            "rules": pol["rules"],
        },
        "integrity": integrity,
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bundle_manifest.json", manifest_bytes)
        for path, data in files.items():
            zf.writestr(path, data)
    buf.seek(0)

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (c["name"] or "classifier"))
    fname = f"{safe}_{tier.replace('+', '_')}.gavel.zip"
    return buf.getvalue(), fname


# ---------------------------------------------------------------------------
# Parse + validate (import, read side)
# ---------------------------------------------------------------------------

def _safe_extract(zip_bytes: bytes) -> dict:
    """Extract a bundle zip into {path: bytes} with the same hardening the model
    upload path uses: reject path traversal, symlinks, oversized/over-count
    archives, and executable payloads (the torch .pth is exempt — it is itself a
    legitimate PK-zip container)."""
    src = io.BytesIO(zip_bytes)
    if not zipfile.is_zipfile(src):
        raise BundleError("Uploaded file is not a valid ZIP archive.", 400)
    src.seek(0)
    out = {}
    with zipfile.ZipFile(src, "r") as zf:
        members = [m for m in zf.infolist() if not m.filename.endswith("/")]
        if len(members) > _MAX_FILES:
            raise BundleError(f"Bundle has too many files ({len(members)}).", 400)
        total = 0
        for m in members:
            if m.filename.startswith("/") or ".." in m.filename:
                raise BundleError(f"Unsafe path in bundle: {m.filename}", 400)
            if m.external_attr >> 28 == 0xA:
                raise BundleError(f"Bundle contains a symlink ({m.filename}), which is not allowed.", 400)
            total += m.file_size
            if total > _MAX_UNCOMPRESSED:
                raise BundleError("Bundle is too large.", 400)
        for m in members:
            data = zf.read(m)
            _, ext = os.path.splitext(m.filename.lower())
            if ext != ".pth":
                for sig in _EXEC_SIGNATURES:
                    if data[: len(sig)] == sig:
                        raise BundleError(f"Bundle file '{m.filename}' appears to be an executable.", 400)
            out[m.filename.replace("\\", "/")] = data
    return out


def _validate_model_loads(meta: dict, pth_bytes: bytes) -> None:
    """Strictly load the weights (weights_only — no pickle code execution) into a
    TopicRNN built from the bundle's declared geometry. A mismatch means the
    bundle is corrupt or tampered, and we abort before any DB write."""
    import torch
    from classifier_engine.RNN import TopicRNN

    try:
        sd = torch.load(io.BytesIO(pth_bytes), map_location="cpu", weights_only=True)
    except Exception as e:
        raise BundleError(f"The model weights could not be safely loaded: {e}", 400)
    try:
        rnn = TopicRNN(
            input_dim=int(meta["readout_dim"]),
            num_layers=int(meta["n_layers"]),
            hidden_dim=int(meta["hidden_dim"]),
            num_rnn_layers=int(meta["num_rnn_layers"]),
            num_topics=int(meta["num_classes"]),
            rnn_type="GRU",
        )
        rnn.load_state_dict(sd)  # strict
    except Exception as e:
        raise BundleError(
            f"The model weights don't match the architecture declared in the bundle: {e}", 400
        )


def parse_and_validate_bundle(zip_bytes: bytes) -> dict:
    files = _safe_extract(zip_bytes)

    if "bundle_manifest.json" not in files:
        raise BundleError("This is not a GAVEL rule set bundle (no bundle_manifest.json).", 400)
    try:
        manifest = json.loads(files["bundle_manifest.json"])
    except Exception:
        raise BundleError("bundle_manifest.json is not valid JSON.", 400)

    if manifest.get("format") != FORMAT:
        raise BundleError("This file is not a GAVEL rule set bundle.", 400)
    if int(manifest.get("format_version", 0)) > FORMAT_VERSION:
        raise BundleError(
            f"This bundle was made by a newer version (format v{manifest.get('format_version')}). "
            "Update the app, then import again.", 400
        )
    tier = manifest.get("tier")
    if tier not in _TIERS:
        raise BundleError(f"Unknown bundle tier '{tier}'.", 400)

    for req in ("model/trained_rnn.pth", "model/classifier_meta.json"):
        if req not in files:
            raise BundleError(f"Bundle is missing required file '{req}'.", 400)

    # Integrity checksums — corruption / tamper detection.
    for path, expected in (manifest.get("integrity") or {}).items():
        if path not in files:
            raise BundleError(f"Bundle integrity error: '{path}' is referenced but missing.", 400)
        if _sha256(files[path]) != expected:
            raise BundleError(f"Bundle integrity error: '{path}' is corrupted or was modified.", 400)

    try:
        meta = json.loads(files["model/classifier_meta.json"])
    except Exception:
        raise BundleError("classifier_meta.json is not valid JSON.", 400)
    for k in ("labels", "readout_dim", "n_layers", "hidden_dim",
              "num_rnn_layers", "num_classes", "selected_layers", "rnn_sequence_length"):
        if k not in meta:
            raise BundleError(f"classifier_meta.json is missing '{k}'.", 400)

    if not isinstance(manifest.get("policy"), dict) or not isinstance(manifest["policy"].get("rules"), list):
        raise BundleError("Bundle manifest has no policy.", 400)
    if not isinstance(manifest.get("base_model"), dict) or not manifest["base_model"].get("storage_path"):
        raise BundleError("Bundle manifest is missing its base model reference.", 400)

    # No DB writes happen before this passes.
    _validate_model_loads(meta, files["model/trained_rnn.pth"])

    return {"manifest": manifest, "meta": meta, "files": files, "tier": tier}


# ---------------------------------------------------------------------------
# Import (write side)
# ---------------------------------------------------------------------------

def _unique_classifier_name(model_id: int, base: str) -> str:
    """Avoid two identically named guardrails under one base model."""
    existing = {
        r["name"]
        for r in (execute_query_dict(
            "SELECT name FROM classifiers WHERE model_id = %s", (model_id,)) or [])
    }
    if base not in existing:
        return base
    i = 2
    while f"{base} ({i})" in existing:
        i += 1
    return f"{base} ({i})"


def _base_model_key(storage_path: str) -> str:
    """Normalized identity for a base model — its basename, lowercased.

    Lets us recognize the *same* model registered under different storage paths,
    e.g. an HF repo id ("meta-llama/Llama-2-7b-chat") vs a local folder that ends
    in the same name ("D:/models/meta-llama/Llama-2-7b-chat"). Both reduce to
    "llama-2-7b-chat".
    """
    p = (storage_path or "").strip().replace("\\", "/").rstrip("/")
    return (p.rsplit("/", 1)[-1] if p else "").lower()


def _resolve_base_model(user_id: int, base_path: str):
    """Find the importer's copy of the bundle's base model.

    Prefer an exact storage_path match. Failing that, fall back to a basename
    match — but ONLY when it's unambiguous (exactly one of the user's models
    shares the basename), so we never silently bind a trained head to the wrong
    base model. Returns the model row (model_id, name, storage_path) or None.
    """
    exact = execute_query_dict(
        "SELECT model_id, name, storage_path FROM target_models "
        "WHERE user_id = %s AND storage_path = %s",
        (user_id, base_path),
    )
    if exact:
        return exact[0]
    want = _base_model_key(base_path)
    if not want:
        return None
    rows = execute_query_dict(
        "SELECT model_id, name, storage_path FROM target_models WHERE user_id = %s",
        (user_id,),
    ) or []
    candidates = [r for r in rows if _base_model_key(r.get("storage_path")) == want]
    return candidates[0] if len(candidates) == 1 else None


def import_bundle(zip_bytes: bytes, user_id: int, *, sync: bool = True,
                  on_phase=None, on_classifier_created=None) -> dict:
    """Import a guardrail bundle for `user_id`.

    Steps: sync library → validate bundle → require the base model locally →
    resolve every policy rule/CE by public_id (block on a private dep) →
    create the guardrail, attach the policy, drop in the weights, stamp the
    trained snapshot, and load whatever tier artifacts came along.

    `on_phase(text)` receives human-readable progress for a background job.
    `on_classifier_created(classifier_id)` fires the instant the guardrail row
    exists — the job records it so crash recovery can roll back a partial import.
    """
    _phase = on_phase or (lambda *_: None)

    if sync:
        _phase("Syncing the public library…")
        try:
            from services.hf_sync import sync_library
            sync_library(force=True)
        except Exception as e:
            raise BundleError(f"Couldn't sync with the public library before import: {e}", 503)

    _phase("Validating the bundle…")
    parsed = parse_and_validate_bundle(zip_bytes)
    manifest, meta, files = parsed["manifest"], parsed["meta"], parsed["files"]

    # 1. Base model precondition — inference reads live activations from it.
    base = manifest["base_model"]
    base_path = base["storage_path"]
    # Match the importer's base model. Exact storage_path first; otherwise an
    # unambiguous basename match (same model, different path — HF id vs local
    # folder). See _resolve_base_model.
    model_row = _resolve_base_model(user_id, base_path)
    if not model_row:
        label = base.get("display_name") or base_path
        raise BundleError(
            f"This bundle needs the base model “{label}” (source: {base_path}). "
            "Add that exact base model under your models first, then import again.",
            409,
        )
    model_id = model_row["model_id"]
    # The model this guardrail will run on — baked into the imported name so the
    # recipient can tell at a glance which model it's for.
    model_label = model_row.get("name") or base.get("display_name") or base_path

    # 2. Resolve the policy by public_id. Block on anything private/unpublished.
    _phase("Resolving the policy from the library…")
    resolved_rules = []
    for mr in manifest["policy"]["rules"]:
        pid = mr.get("public_id")
        nm = mr.get("name") or pid
        row = execute_query_dict("SELECT rule_id FROM rules WHERE public_id = %s", (pid,))
        if not row:
            raise BundleError(
                f"This bundle depends on rule “{nm}” which isn't in the public library. "
                "Ask the sender to publish it, then re-export.",
                409,
            )
        local_rule_id = row[0]["rule_id"]
        if _local_rule_public_fp(local_rule_id) != mr.get("rule_fingerprint"):
            raise BundleError(
                f"Integrity check failed for rule “{nm}”: the library version differs from the "
                "version this rule set was trained on.",
                409,
            )
        resolved_rules.append(local_rule_id)

    # Every output head the model has must exist locally after the sync.
    missing_ces = []
    for ce in (manifest["policy"].get("ces") or []):
        got = execute_query_dict(
            "SELECT ce_id FROM cognitive_elements WHERE public_id = %s", (ce.get("public_id"),)
        )
        if not got:
            missing_ces.append(ce.get("name") or ce.get("public_id"))
    if missing_ces:
        raise BundleError(
            "This bundle depends on cognitive element(s) that aren't in the public library: "
            f"{', '.join(missing_ces[:5])}{'…' if len(missing_ces) > 5 else ''}. "
            "Ask the sender to publish them, then re-export.",
            409,
        )

    # 3. Create the guardrail and assemble everything. On any failure, roll the
    #    whole thing back (delete the row + its workdir) so a botched import never
    #    leaves a half-built guardrail behind.
    from sql_scripts.model_scripts import (
        create_classifier,
        add_rule_to_classifier,
        commit_trained_policy_snapshot,
        delete_classifier,
    )
    from classifier_engine.trainer import classifier_workdir

    base_name = (manifest.get("source") or {}).get("classifier_name") or "Imported rule set"
    cname = _unique_classifier_name(model_id, f"{base_name} (imported · {model_label})")
    _phase("Building the rule set…")
    created = create_classifier(user_id, cname, model_id)
    classifier_id = created["classifier_id"]
    # Record the new guardrail id with the caller (the job) RIGHT AWAY, before
    # any further work — so a crash mid-import leaves a breadcrumb recovery can
    # use to delete this still-incomplete guardrail.
    if on_classifier_created:
        try:
            on_classifier_created(classifier_id)
        except Exception:
            pass
    workdir = classifier_workdir(classifier_id, user_id)

    try:
        os.makedirs(workdir, exist_ok=True)

        # Rewrite model_path to THIS user's base model location, then persist meta.
        out_meta = dict(meta)
        out_meta["model_path"] = base_path
        out_meta["_imported"] = True
        with open(os.path.join(workdir, "classifier_meta.json"), "w", encoding="utf-8") as f:
            json.dump(out_meta, f, indent=2)
        rnn_path = os.path.join(workdir, "trained_rnn.pth")
        with open(rnn_path, "wb") as f:
            f.write(files["model/trained_rnn.pth"])
        execute_query(
            "UPDATE classifiers SET model_path = %s WHERE classifier_id = %s",
            (rnn_path, classifier_id),
        )

        for rid in resolved_rules:
            add_rule_to_classifier(classifier_id, rid)

        # Stamp the trained snapshot (fingerprint + trained_at) so drift shows
        # "Up to date". This also clears any eval rows — harmless, none exist yet.
        commit_trained_policy_snapshot(classifier_id)
        # Back trained_at off slightly so the calibration/eval inserts below are
        # unambiguously newer and survive the post-train visibility filter.
        execute_query(
            "UPDATE classifiers SET trained_at = datetime(trained_at, '-5 seconds') WHERE classifier_id = %s",
            (classifier_id,),
        )

        if "calibration/thresholds.json" in files:
            thresholds = json.loads(files["calibration/thresholds.json"])
            execute_query(
                "INSERT INTO evaluation_results (classifier_id, eval_type, thresholds, metrics, plots) "
                "VALUES (%s, 'calibration', %s::jsonb, NULL, NULL)",
                (classifier_id, json.dumps(thresholds)),
            )
        if "evaluation/results.json" in files:
            ev = json.loads(files["evaluation/results.json"])
            execute_query(
                "INSERT INTO evaluation_results (classifier_id, eval_type, thresholds, metrics, plots) "
                "VALUES (%s, 'evaluation', %s::jsonb, %s::jsonb, %s::jsonb)",
                (
                    classifier_id,
                    json.dumps(ev.get("thresholds")),
                    json.dumps(ev.get("metrics")),
                    json.dumps(ev.get("plots")),
                ),
            )

        execute_query(
            "UPDATE classifiers SET status = 'active' WHERE classifier_id = %s",
            (classifier_id,),
        )

        return {
            "classifier_id": classifier_id,
            "name": cname,
            "tier": parsed["tier"],
            "rules": len(resolved_rules),
            "base_model": base.get("display_name") or base_path,
            "calibrated": "calibration/thresholds.json" in files,
            "evaluated": "evaluation/results.json" in files,
        }
    except BundleError:
        _rollback_import(classifier_id, workdir, delete_classifier)
        raise
    except Exception as e:
        _rollback_import(classifier_id, workdir, delete_classifier)
        raise BundleError(f"Import failed while assembling the rule set: {e}", 500)


def _rollback_import(classifier_id: int, workdir: str, delete_classifier) -> None:
    try:
        delete_classifier(classifier_id)
    except Exception:
        pass
    try:
        if workdir and os.path.isdir(workdir):
            shutil.rmtree(workdir, ignore_errors=True)
    except Exception:
        pass
