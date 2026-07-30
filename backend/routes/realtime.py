# backend/routes/realtime.py
# REST endpoints for realtime CE monitoring — two modes, reference-parity:
#
#   1. LIVE  (/analyze)         — chat with the LLM; classify its reply.
#   2. STORED (/analyze-stored) — classify an existing dialogue from a CE's
#                                 dataset (no generation), so you can inspect
#                                 which tokens/windows trigger which CEs on the
#                                 data the guardrail was trained/calibrated on.
#
# Both produce the SAME analysis shape:
#   * windows — non-overlapping blocks; feed the calibrated `compute_triggers`
#               (faithful to how training/calibration windowed the data) →
#               which CEs fired (patience-gated) → which RULES fired.
#   * tokens  — per-token, stride-1 (for the colored text + activation curve).
import json
import logging
import os
import threading
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from utils.auth import get_current_user
from utils.ownership import require_classifier_owner
from utils.sqlite_db import execute_query_dict

logger = logging.getLogger(__name__)
# Every endpoint here is /{classifier_id}/… — guard the whole router so the
# caller must own that guardrail (auth + ownership) before any handler runs.
router = APIRouter(dependencies=[Depends(require_classifier_owner)])


class AnalyzeRequest(BaseModel):
    system_prompt: str = "You are a helpful assistant."
    user_message: str
    history: Optional[List[dict]] = None
    max_new_tokens: int = 128


class AnalyzeStoredRequest(BaseModel):
    # A normalized conversation: [{"role": "...", "content": "..."}, ...].
    # The LAST assistant turn is the one classified.
    messages: List[dict]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _friendly_local_error(e: Exception) -> str:
    """Map a local-inference failure to a user-facing message. The common case
    on a weak client is the ~15 GB target model not fitting in RAM/GPU — surface
    that clearly (with the fix: use the cluster) instead of a raw torch trace."""
    s = str(e).lower()
    mem_markers = (
        "out of memory", "outofmemory", "can't allocate", "cannot allocate",
        "defaultcpuallocator", "cuda error", "cuda out", "oom", "killed",
        "not enough memory", "paging file",
    )
    if any(m in s for m in mem_markers):
        return "The model couldn't load on this machine."
    return f"Analysis failed: {e}"


def _require_trained_classifier(classifier_id: int):
    rows = execute_query_dict(
        "SELECT classifier_id, status FROM classifiers WHERE classifier_id = %s",
        (classifier_id,),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Rule Set not found")
    if rows[0]["status"] not in ("active", "needs_retraining"):
        raise HTTPException(status_code=400, detail="Rule Set must be trained first")


def _build_scoring_context(classifier_id: int, labels: dict):
    """Load the calibrated thresholds + rule tensors ONCE for a guardrail.

    Returns a context object reused to score one or many spans (a multi-turn
    dialogue scores every assistant turn against the same thresholds/ruleset),
    so we don't re-query calibration and rebuild the ruleset per turn.
    """
    import torch
    import classifier_engine.reference  # noqa: F401  — registers the gavel.* alias
    from gavel.evaluation.metrics import build_ruleset_logic
    from evaluation.ruleset_builder import build_unified_ruleset

    idx_to_label = {v: k for k, v in labels.items()}
    num_topics = len(labels)

    # --- Calibrated thresholds (latest post-train calibration row) ---
    from routes.evaluation import _POST_TRAIN_CLAUSE  # noqa: WPS433
    calib_rows = execute_query_dict(
        f"""SELECT thresholds FROM evaluation_results
           WHERE classifier_id = %s AND eval_type = 'calibration'
             AND thresholds IS NOT NULL
             {_POST_TRAIN_CLAUSE}
           ORDER BY created_at DESC, eval_id DESC LIMIT 1""",
        (classifier_id, classifier_id),
    )
    thresholds_dict = calib_rows[0]["thresholds"] if calib_rows else {}

    thr_vec = torch.full((num_topics,), 0.5, dtype=torch.float32)
    patience_vec = [1] * num_topics
    for ce_name, idx in labels.items():
        spec = (thresholds_dict or {}).get(ce_name) or {}
        thr_vec[idx] = float(spec.get("threshold", 0.5))
        patience_vec[idx] = int(spec.get("patience", 1))
    patience_rate = max(patience_vec) if patience_vec else 1

    unified_ruleset = build_unified_ruleset(classifier_id)
    ruleset_logic = build_ruleset_logic(unified_ruleset, labels)

    thresholds_used = {
        idx_to_label[i]: {"threshold": float(thr_vec[i].item()), "patience": int(patience_vec[i])}
        for i in range(num_topics)
    }
    return {
        "labels": labels,
        "idx_to_label": idx_to_label,
        "num_topics": num_topics,
        "thr_vec": thr_vec,
        "patience_vec": patience_vec,
        "patience_rate": patience_rate,
        "unified_ruleset": unified_ruleset,
        "ruleset_logic": ruleset_logic,
        "thresholds_used": thresholds_used,
    }


def _score_span(ctx: dict, windows: list, tokens: list):
    """Apply the prebuilt scoring context to one classified assistant span.

    Mutates `windows`/`tokens` in place to add per-CE probabilities, and returns
    that span's trigger/rule verdicts.
    """
    import numpy as np
    import torch
    from gavel.evaluation.metrics import compute_triggers

    idx_to_label = ctx["idx_to_label"]
    num_topics = ctx["num_topics"]
    thr_vec = ctx["thr_vec"]

    # --- Per-CE triggers (windowed + patience, like calibration) ---
    if windows:
        logits_matrix = torch.tensor([w["logits"] for w in windows], dtype=torch.float32)
        triggers_tensor = compute_triggers(logits_matrix, thresholds=thr_vec, patience_rate=ctx["patience_rate"])
    else:
        logits_matrix = torch.zeros((0, num_topics), dtype=torch.float32)
        triggers_tensor = torch.zeros(num_topics, dtype=torch.float32)
    triggered_ces = [idx_to_label[i] for i in range(num_topics) if bool(triggers_tensor[i].item())]

    # --- Annotate windows with per-CE probabilities ---
    win_probs = torch.sigmoid(logits_matrix).numpy() if logits_matrix.shape[0] else np.zeros((0, num_topics))
    for i, w in enumerate(windows):
        p = win_probs[i]
        w["probabilities"] = {idx_to_label.get(j, f"class_{j}"): round(float(v), 4) for j, v in enumerate(p)}
        w["window_triggered_ces"] = sorted(
            idx_to_label.get(j, f"class_{j}") for j, v in enumerate(p) if v >= float(thr_vec[j])
        )
        # Drop the raw logits now that probabilities are computed — the client
        # never reads them, and an unbounded NaN/Inf logit would serialize as a
        # literal NaN/Infinity token that browser JSON.parse rejects (breaking
        # the whole response). Probabilities are bounded sigmoid+round.
        w.pop("logits", None)

    # --- Annotate tokens with per-CE probabilities (drives chart + coloring) ---
    if tokens:
        tok_logits = torch.tensor([t["logits"] for t in tokens], dtype=torch.float32)
        tok_probs = torch.sigmoid(tok_logits).numpy()
        for i, t in enumerate(tokens):
            p = tok_probs[i]
            t["probabilities"] = {idx_to_label.get(j, f"class_{j}"): round(float(v), 4) for j, v in enumerate(p)}
            t["triggered_ces"] = sorted(
                idx_to_label.get(j, f"class_{j}") for j, v in enumerate(p) if v >= float(thr_vec[j])
            )
            t.pop("logits", None)   # see window note above

    # --- Rule conditions (groups + v2 grammar, evaluated over triggers) ---
    from gavel.evaluation.metrics import detect_usecase
    from utils.rule_condition import selectors as condition_selectors

    predicted_idxs = triggers_tensor.bool().nonzero(as_tuple=True)[0]
    triggered_set = {idx_to_label[i] for i in predicted_idxs.tolist()}
    rule_triggers: list = []
    for rule_name, spec in ctx["unified_ruleset"].items():
        if not spec.get("enabled", True):
            continue
        entry = ctx["ruleset_logic"].get(rule_name) or {"ast": None, "groups_idx": {}}
        fired = detect_usecase(ctx["ruleset_logic"], rule_name, predicted_idxs)

        # Per-group explanation for the UI: which members hit, what the
        # selector demands ('all' / k / None for unreferenced groups), and
        # whether it is satisfied. A group referenced by several selectors
        # reports its first positive selector's quantifier.
        quantifiers: dict = {}
        if entry["ast"] is not None:
            for _, quant, gname in condition_selectors(entry["ast"]):
                quantifiers.setdefault(gname, quant)
        groups_report: dict = {}
        unmet: list = []
        for gname, members in (spec.get("groups") or {}).items():
            hits = sorted(m for m in (members or []) if m in triggered_set)
            quant = quantifiers.get(gname)
            if quant is None:
                satisfied = True      # unreferenced (e.g. supporting) group
            else:
                k = len(members or []) if quant == "all" else int(quant)
                satisfied = len(hits) >= k
                if not satisfied:
                    unmet.append(gname)
            groups_report[gname] = {
                "members": list(members or []),
                "hits": hits,
                "quantifier": quant,
                "satisfied": satisfied,
            }
        rule_triggers.append({
            "rule_name": rule_name,
            "fired": bool(fired),
            "condition": spec.get("condition") or "",
            "groups": groups_report,
            "groups_unmet": unmet,
        })

    return {
        "windows": windows,
        "tokens": tokens,
        "triggered_ces": triggered_ces,
        "rule_triggers": rule_triggers,
        "num_windows": len(windows),
    }


def _score_and_evaluate(classifier_id: int, windows: list, tokens: list, labels: dict):
    """Score a SINGLE classified span (live path / single assistant turn)."""
    ctx = _build_scoring_context(classifier_id, labels)
    span = _score_span(ctx, windows, tokens)
    return {
        "labels": labels,
        "thresholds_used": ctx["thresholds_used"],
        **span,
    }


def _aggregate_rule_triggers(per_turn_rule_triggers: list) -> list:
    """Collapse per-turn rule verdicts into one conversation-level summary for
    the sidebar: a rule is 'fired' if it fired on ANY assistant turn."""
    agg: dict = {}
    order: list = []
    for turn_rules in per_turn_rule_triggers:
        for rt in (turn_rules or []):
            name = rt["rule_name"]
            if name not in agg:
                agg[name] = dict(rt)
                order.append(name)
            else:
                cur = agg[name]
                if rt.get("fired") and not cur.get("fired"):
                    agg[name] = dict(rt)  # prefer a firing turn's detail
                elif not cur.get("fired"):
                    # merge per-group hits/satisfied across turns for a
                    # clearer "how close did it get" hint
                    merged = dict(cur.get("groups") or {})
                    for gname, info in (rt.get("groups") or {}).items():
                        if gname in merged:
                            merged_hits = sorted(set(merged[gname].get("hits") or []) | set(info.get("hits") or []))
                            merged[gname] = {
                                **merged[gname],
                                "hits": merged_hits,
                                "satisfied": bool(merged[gname].get("satisfied") or info.get("satisfied")),
                            }
                        else:
                            merged[gname] = dict(info)
                    cur["groups"] = merged
                    cur["groups_unmet"] = [
                        g for g, info in merged.items()
                        if info.get("quantifier") is not None and not info.get("satisfied")
                    ]
    return [agg[n] for n in order]


def _extract_ce_conversations(ce_id: int, limit: int = 60) -> list:
    """Return the CE's stored dialogues as normalized message-lists.

    Prefers the per-CE calibration dialogues (clean conversations where the CE
    is present); falls back to the excitation/training set. Each returned item
    is a List[{"role","content"}] with at least one user + one assistant turn.
    """
    from evaluation.inference import _normalize_conversation

    raw = None
    for table in ("calibration_datasets", "excitation_datasets"):
        rows = execute_query_dict(f"SELECT dataset FROM {table} WHERE ce_id = %s", (ce_id,)) or []
        if rows and rows[0].get("dataset"):
            raw = rows[0]["dataset"]
            break
    if raw is None:
        return []

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []

    if isinstance(raw, dict):
        items = raw.get("conversations") or raw.get("samples") or raw.get("training_data") or []
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    out: list = []
    for item in items:
        # An item may be a conversation (list of messages) or a training pair.
        if isinstance(item, dict) and ("input" in item or "output" in item or "prompt" in item or "response" in item):
            user = item.get("input") or item.get("prompt") or ""
            asst = item.get("output") or item.get("response") or ""
            msgs = [{"role": "user", "content": str(user)}, {"role": "assistant", "content": str(asst)}]
        elif isinstance(item, dict) and isinstance(item.get("messages"), list):
            msgs = _normalize_conversation(item["messages"])
        else:
            msgs = _normalize_conversation(item)
        # Need a user turn before an assistant turn to classify.
        if any((m.get("role") or "").lower() == "assistant" and m.get("content") for m in msgs) and \
           any((m.get("role") or "").lower() == "user" for m in msgs):
            out.append(msgs)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/{classifier_id}/analyze")
def analyze_message(classifier_id: int, req: AnalyzeRequest, _: int = Depends(get_current_user)):
    """LIVE mode: generate the assistant reply and classify it."""
    _require_trained_classifier(classifier_id)
    try:
        import classifier_engine.reference  # noqa: F401  (registers gavel.* alias)
        from evaluation.model_cache import load_or_get
        from evaluation.realtime import generate_and_classify
        from utils.device import get_torch_device

        llm, tokenizer, rnn_model, meta = load_or_get(classifier_id, get_torch_device())
        generated_text, windows, tokens = generate_and_classify(
            user_input=req.user_message,
            system_prompt=req.system_prompt,
            model=llm, tokenizer=tokenizer, classifier=rnn_model, meta=meta,
            max_new_tokens=req.max_new_tokens, history=req.history,
        )
        scored = _score_and_evaluate(classifier_id, windows, tokens, meta["labels"])
        return {"generated_text": generated_text, **scored}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Realtime analysis failed: {e}")
        raise HTTPException(status_code=500, detail=_friendly_local_error(e))


@router.post("/{classifier_id}/analyze-stored")
def analyze_stored(classifier_id: int, req: AnalyzeStoredRequest, _: int = Depends(get_current_user)):
    """STORED mode: classify an existing dialogue (no generation)."""
    _require_trained_classifier(classifier_id)
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")
    try:
        import classifier_engine.reference  # noqa: F401  (registers gavel.* alias)
        from evaluation.model_cache import load_or_get
        from evaluation.realtime import classify_conversation_turns
        from utils.device import get_torch_device

        # Load the LLM (cached in-process for the session) and classify EVERY
        # assistant turn LIVE — nothing is persisted; the logits are recomputed
        # each request, matching the reference realtime monitor (it holds logits
        # only in session memory and discards them).
        llm, tokenizer, rnn_model, meta = load_or_get(classifier_id, get_torch_device())
        labels = meta["labels"]
        per_turn = classify_conversation_turns(
            messages=req.messages, model=llm, tokenizer=tokenizer, classifier=rnn_model, meta=meta,
        )

        ctx = _build_scoring_context(classifier_id, labels)

        turns: list = []
        all_triggered: set = set()
        per_turn_rules: list = []
        for i, m in enumerate(req.messages):
            role = (m.get("role") or "").lower()
            if role == "system":
                continue
            if role == "assistant" and i in per_turn:
                windows, tokens = per_turn[i]
                span = _score_span(ctx, windows, tokens)
                all_triggered.update(span["triggered_ces"])
                per_turn_rules.append(span["rule_triggers"])
                turns.append({
                    "role": "assistant",
                    "content": m.get("content", ""),
                    "thresholds_used": ctx["thresholds_used"],
                    **span,
                })
            else:
                turns.append({"role": role or "user", "content": m.get("content", "")})

        return {
            "generated_text": None,
            "labels": labels,
            "thresholds_used": ctx["thresholds_used"],
            "turns": turns,
            # Conversation-level aggregates for the shared sidebar.
            "triggered_ces": sorted(all_triggered),
            "rule_triggers": _aggregate_rule_triggers(per_turn_rules),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Stored analysis failed: {e}")
        raise HTTPException(status_code=500, detail=_friendly_local_error(e))


@router.get("/{classifier_id}/model-status")
def model_status(classifier_id: int, _: int = Depends(get_current_user)):
    """Is this guardrail's target LLM already loaded into the in-process cache?

    The 7B model is loaded lazily on the first realtime analysis (~15 GB, can take
    minutes on CPU/MPS) and cached afterward. The frontend reads this so it can show
    a one-time 'loading the model' notice only when the load is actually pending."""
    from evaluation.model_cache import get_cached_models
    return {"loaded": get_cached_models(classifier_id) is not None}


@router.post("/{classifier_id}/unload")
def unload_model(classifier_id: int, _: int = Depends(get_current_user)):
    """Free the realtime model: evict it from the in-process cache (RAM) AND delete
    its HuggingFace download from disk (reclaims ~15 GB). The next analysis
    re-downloads it. A LOCAL (user-uploaded) model file is left alone — only the
    downloaded HuggingFace cache is removed."""
    import os
    import shutil
    from evaluation.model_cache import evict

    evict(classifier_id)  # free RAM + device memory

    row = execute_query_dict(
        """SELECT tm.storage_path FROM classifiers c
           JOIN target_models tm ON c.model_id = tm.model_id
           WHERE c.classifier_id = %s""",
        (classifier_id,),
    )
    model_ref = (row[0]["storage_path"] if row else None) or ""

    # Only an HF repo id (e.g. 'mistralai/Mistral-7B-Instruct-v0.2') has a download
    # cache to delete; a path that exists on disk is the user's own uploaded model.
    if not model_ref or os.path.exists(model_ref):
        return {"ram_freed": True, "disk_deleted": False, "freed_bytes": 0,
                "model": model_ref or None,
                "note": "Local model file kept — only downloaded (HuggingFace) models are removed."}

    try:
        from huggingface_hub.constants import HF_HUB_CACHE as _HUB
    except Exception:
        _HUB = os.path.expanduser("~/.cache/huggingface/hub")
    cache_path = os.path.join(_HUB, "models--" + model_ref.replace("/", "--"))

    freed, deleted = 0, False
    if os.path.isdir(cache_path):
        # Sum the real bytes (blobs hold the weights; snapshots are just symlinks).
        blobs = os.path.join(cache_path, "blobs")
        if os.path.isdir(blobs):
            for f in os.listdir(blobs):
                fp = os.path.join(blobs, f)
                try:
                    if os.path.isfile(fp):
                        freed += os.path.getsize(fp)
                except OSError:
                    pass
        shutil.rmtree(cache_path, ignore_errors=True)
        deleted = not os.path.isdir(cache_path)
    return {"ram_freed": True, "disk_deleted": deleted, "freed_bytes": freed,
            "model": model_ref, "note": None if deleted else "Nothing was on disk to delete."}


# ---------------------------------------------------------------------------
# Warm cluster realtime SESSION — the LLM forward runs on the cluster GPU so
# realtime works on ANY client PC (Windows / Mac / weak laptops). The backend
# only orchestrates the session + applies the calibrated thresholds/rules to the
# raw logits the warm job returns (so recalibration needs no session restart).
# ---------------------------------------------------------------------------

# Warm realtime sessions, keyed by classifier_id, stored as (provider, session).
# A session here runs OFF this machine — on the remote GPU worker (HTTP) or the
# SLURM cluster (SSH). BOTH go through the compute-provider interface; this route
# never talks to realtime_session / cluster_direct directly.
_sessions: dict = {}
_sessions_lock = threading.Lock()


def _session(classifier_id: int):
    """In-memory (provider, session) record for this guardrail, or None."""
    with _sessions_lock:
        return _sessions.get(classifier_id)


def _active_session(classifier_id: int):
    """(provider, session) for a warm realtime session — remote worker OR cluster.
    Returns the in-memory record if present; otherwise, if the realtime provider is
    the (stateless) SLURM cluster, rebuilds a handle from the classifier_id so a
    still-running warm job stays reachable after a backend restart. None when there
    is no off-box session (local mode)."""
    s = _session(classifier_id)
    if s is not None:
        return s
    from services import compute
    from services.compute.base import RealtimeSession
    p = compute.get_provider(compute.Workload.REALTIME)
    if p.name == "slurm":
        return (p, RealtimeSession(provider="slurm", classifier_id=classifier_id,
                                   id=str(classifier_id)))
    return None


def _resolve_for_session(classifier_id: int):
    """Load the on-disk classifier_meta.json + locate trained_rnn.pth (needed to
    submit / score the warm session) — without loading any LLM locally."""
    from classifier_engine.trainer import classifier_workdir
    work_dir = classifier_workdir(classifier_id)
    meta_path = os.path.join(work_dir, "classifier_meta.json")
    rnn_path = os.path.join(work_dir, "trained_rnn.pth")
    if not (os.path.isfile(meta_path) and os.path.isfile(rnn_path)):
        raise HTTPException(status_code=400, detail="Rule Set has no trained model on disk.")
    with open(meta_path) as f:
        meta = json.load(f)
    return meta, rnn_path


@router.post("/{classifier_id}/session/start")
def session_start(classifier_id: int, _: int = Depends(get_current_user)):
    """Submit a warm realtime job on the cluster. Returns immediately; poll
    /session/status until it reports 'ready' (the model is loading).

    Returns {fallback:'local'} (NOT an error) when the warm session can't be
    used — the cluster isn't configured, or the guardrail uses a LOCALLY-uploaded
    model that exists only on this machine's disk (the cluster can't load it). The
    client uses that signal to load the model locally instead. Genuine problems
    (not trained / no model on disk) still raise a 4xx the client surfaces."""
    _require_trained_classifier(classifier_id)
    meta, rnn_path = _resolve_for_session(classifier_id)
    model_ref = meta.get("model_path") or ""

    # Warm-session providers run the model OFF this machine: the remote GPU worker
    # (HTTP) or the SLURM cluster (SSH). Local mode has no warm session (the client
    # runs the model in-process), so anything else returns {fallback:'local'}. A
    # user-uploaded (local-filesystem) model can't be loaded off-box either. The
    # provider resolution already accounts for reachability (an unreachable cluster
    # resolves to 'local' here), so this route stays transport-agnostic.
    from services import compute
    # Off-box failover ladder for the warm session: remote_worker -> slurm. We try
    # each in turn so a dead/unreachable worker degrades to the cluster (and only
    # THEN to local in-process mode) instead of jumping straight to local. The
    # viewer re-calls this endpoint on crash detection, so this also drives
    # mid-session recovery down the same ladder.
    offbox = [n for n in compute.failover_providers(compute.Workload.REALTIME)
              if n in ("remote_worker", "slurm")]
    if not offbox:
        return {"ok": True, "fallback": "local", "reason": "no_offload_provider"}
    if model_ref and os.path.exists(model_ref):
        return {"ok": True, "fallback": "local", "reason": "local_model"}

    # Start from the availability-aware pick (so a configured-but-down worker
    # doesn't cost a connect timeout), then descend the rest of the ladder.
    top = compute.get_provider(compute.Workload.REALTIME)
    order = offbox[offbox.index(top.name):] if top.name in offbox else []
    if not order:
        return {"ok": True, "fallback": "local", "reason": "no_offload_provider"}

    spec = compute.RealtimeSpec(
        classifier_id=classifier_id, model_hf_path=model_ref,
        classifier_meta=meta, rnn_path=rnn_path,
    )
    for name in order:
        provider = compute.provider_by_name(name)
        if provider is None:
            continue
        try:
            session = provider.start_realtime(spec)
        except Exception as e:
            # Busy / unreachable / submit failed → try the next tier, then local.
            logger.warning(f"[realtime] {name} session start failed for {classifier_id}: {e}")
            continue
        with _sessions_lock:
            _sessions[classifier_id] = (provider, session)
        if provider.name == "remote_worker":
            return {"ok": True, "mode": "remote_worker", "session_id": session.id, "status": "loading"}
        # Cluster: preserve the exact response shape the viewer already expects (the
        # rs.start_session info dict, carried on session.raw minus the internal 'mode').
        info = {k: v for k, v in session.raw.items() if k != "mode"}
        return {"ok": True, **info}

    # Every off-box tier failed → local in-process mode (model runs on this machine).
    return {"ok": True, "fallback": "local", "reason": "session_error"}


@router.get("/{classifier_id}/session/status")
def session_status_ep(classifier_id: int, _: int = Depends(get_current_user)):
    """queued | loading | ready | dead | stopped | none — drives the startup
    spinner and crash detection in the viewer."""
    s = _active_session(classifier_id)
    if s is None:
        return {"status": "none"}
    provider, session = s
    return {"status": provider.realtime_status(session)}


@router.post("/{classifier_id}/session/keepalive")
def session_keepalive(classifier_id: int, _: int = Depends(get_current_user)):
    """Client liveness ping — resets the job's idle clock; if these stop, the
    backend sweep (and the job's own idle timeout) reclaim the GPU."""
    s = _active_session(classifier_id)
    if s is None:
        return {"alive": False}
    provider, session = s
    return {"alive": provider.realtime_keepalive(session)}


@router.post("/{classifier_id}/session/end")
def session_end(classifier_id: int, _: int = Depends(get_current_user)):
    """Tear down the session (stop sentinel + scancel + cleanup). Called on every
    clean exit from realtime; also reachable via navigator.sendBeacon on unload."""
    s = _active_session(classifier_id)
    if s is None:
        return {"ok": True, "ended": False}
    provider, session = s
    try:
        provider.end_realtime(session)
    finally:
        with _sessions_lock:
            _sessions.pop(classifier_id, None)
    return {"ok": True, "ended": True}


@router.post("/{classifier_id}/session/analyze-stored")
def session_analyze_stored(classifier_id: int, req: AnalyzeStoredRequest, _: int = Depends(get_current_user)):
    """STORED mode over the warm session: the job classifies every assistant turn
    on the cluster GPU; the backend applies thresholds/rules. Same shape as
    /analyze-stored, so the viewer renders it identically."""
    _require_trained_classifier(classifier_id)
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages is required")
    meta, _rnn = _resolve_for_session(classifier_id)
    labels = meta["labels"]
    s = _active_session(classifier_id)
    if s is None:
        raise HTTPException(status_code=409, detail="No active realtime session.")
    provider, session = s
    try:
        result = provider.realtime_analyze(session, {"mode": "stored", "messages": req.messages})
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.warning(f"Realtime session request failed for classifier {classifier_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Realtime session request failed: {e}")

    # Scoring hop — guarded so a malformed job result or a post-session retrain
    # (logits/topic count mismatch) maps to a clean error instead of a bare 500.
    try:
        raw = result.get("per_turn") or {}
        per_turn = {int(k): (v.get("windows", []), v.get("tokens", [])) for k, v in raw.items()}

        ctx = _build_scoring_context(classifier_id, labels)
        turns: list = []
        all_triggered: set = set()
        per_turn_rules: list = []
        for i, m in enumerate(req.messages):
            role = (m.get("role") or "").lower()
            if role == "system":
                continue
            if role == "assistant" and i in per_turn:
                windows, tokens = per_turn[i]
                span = _score_span(ctx, windows, tokens)
                all_triggered.update(span["triggered_ces"])
                per_turn_rules.append(span["rule_triggers"])
                turns.append({
                    "role": "assistant",
                    "content": m.get("content", ""),
                    "thresholds_used": ctx["thresholds_used"],
                    **span,
                })
            else:
                turns.append({"role": role or "user", "content": m.get("content", "")})

        return {
            "generated_text": None,
            "labels": labels,
            "thresholds_used": ctx["thresholds_used"],
            "turns": turns,
            "triggered_ces": sorted(all_triggered),
            "rule_triggers": _aggregate_rule_triggers(per_turn_rules),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Realtime session scoring failed for classifier {classifier_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@router.post("/{classifier_id}/session/analyze")
def session_analyze(classifier_id: int, req: AnalyzeRequest, _: int = Depends(get_current_user)):
    """LIVE mode over the warm session: the job generates the reply + classifies
    it on the cluster GPU; the backend applies thresholds/rules."""
    _require_trained_classifier(classifier_id)
    meta, _rnn = _resolve_for_session(classifier_id)
    labels = meta["labels"]
    s = _active_session(classifier_id)
    if s is None:
        raise HTTPException(status_code=409, detail="No active realtime session.")
    provider, session = s
    _live_req = {"mode": "live", "user_message": req.user_message,
                 "system_prompt": req.system_prompt, "history": req.history,
                 "max_new_tokens": req.max_new_tokens}
    try:
        result = provider.realtime_analyze(session, _live_req)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.warning(f"Realtime session request failed for classifier {classifier_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Realtime session request failed: {e}")

    try:
        windows = result.get("windows") or []
        tokens = result.get("tokens") or []
        scored = _score_and_evaluate(classifier_id, windows, tokens, labels)
        return {"generated_text": result.get("generated_text"), **scored}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Realtime session scoring failed for classifier {classifier_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


def _usable_dialogue(msgs: list) -> bool:
    """A dialogue is classifiable if it has a user turn followed by an
    assistant turn with content."""
    return (
        any((m.get("role") or "").lower() == "assistant" and m.get("content") for m in msgs)
        and any((m.get("role") or "").lower() == "user" for m in msgs)
    )


def _list_sample_groups(classifier_id: int) -> list:
    """Catalog of browsable conversation groups for this guardrail — the cloud
    equivalent of the reference calibration / eval / ce_dataset folders:

      * Test sets — each attached rule's positive / negative / calibration
        dialogues (`test_datasets`), labelled like 'Test · positive · <rule>'.
      * CE calibration — each trained CE's dialogues (`calibration_datasets`).
    """
    from evaluation.ruleset_builder import get_classifier_labels

    groups: list = []

    # 1) Rule test/eval/calibration datasets attached to this guardrail.
    rows = execute_query_dict(
        """
        SELECT DISTINCT td.dataset_id, td.dataset_type,
               COALESCE(rs.custom_name, r.name) AS rule_name,
               COALESCE(json_array_length(td.conversations), 0) AS count
        FROM test_datasets td
        JOIN rule_setup rs ON rs.rule_id = td.rule_id AND rs.classifier_id = %s
        LEFT JOIN rules r ON r.rule_id = td.rule_id
        WHERE td.status = 'ready' AND td.conversations IS NOT NULL
        ORDER BY rule_name, td.dataset_type
        """,
        (classifier_id,),
    ) or []
    type_label = {"positive": "positive", "negative": "negative", "positive_calibration": "calibration"}
    for r in rows:
        if (r.get("count") or 0) <= 0:
            continue
        t = type_label.get(r["dataset_type"], r["dataset_type"])
        groups.append({
            "key": f"testds:{r['dataset_id']}",
            "label": f"Test · {t} · {r['rule_name']}",
            "count": int(r["count"]),
        })

    # 2) Per-CE calibration dialogues for the CEs the guardrail trained on.
    ce_names = list((get_classifier_labels(classifier_id) or {}).keys())
    if ce_names:
        ce_rows = execute_query_dict(
            "SELECT ce_id, name FROM cognitive_elements WHERE name = ANY(%s) ORDER BY name",
            (ce_names,),
        ) or []
        for c in ce_rows:
            n = len(_extract_ce_conversations(c["ce_id"]))
            if n > 0:
                groups.append({"key": f"ce:{c['ce_id']}", "label": f"CE · {c['name']}", "count": n})

    return groups


def _load_sample_group(key: str) -> list:
    """Return the conversations (normalized message-lists) for a group key."""
    from evaluation.inference import _normalize_conversation

    if key.startswith("testds:"):
        dataset_id = int(key.split(":", 1)[1])
        rows = execute_query_dict(
            "SELECT conversations FROM test_datasets WHERE dataset_id = %s", (dataset_id,),
        ) or []
        convos = rows[0]["conversations"] if rows else []
        if isinstance(convos, str):
            convos = json.loads(convos)
        out = []
        for conv in (convos or []):
            msgs = _normalize_conversation(conv)
            if _usable_dialogue(msgs):
                out.append(msgs)
        return out

    if key.startswith("ce:"):
        return _extract_ce_conversations(int(key.split(":", 1)[1]))

    return []


@router.get("/{classifier_id}/sample-groups")
def list_sample_groups(classifier_id: int, _: int = Depends(get_current_user)):
    """List browsable conversation groups (test sets + CE calibration)."""
    _require_trained_classifier(classifier_id)
    try:
        return {"groups": _list_sample_groups(classifier_id)}
    except Exception as e:
        logger.exception(f"Listing sample groups failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{classifier_id}/sample-group")
def get_sample_group(classifier_id: int, key: str, _: int = Depends(get_current_user)):
    """Return the dialogues in one group as selectable samples for mode 2."""
    _require_trained_classifier(classifier_id)
    conversations = _load_sample_group(key)
    samples = []
    for i, msgs in enumerate(conversations):
        user = next((m.get("content", "") for m in msgs if (m.get("role") or "").lower() == "user"), "")
        asst = next((m.get("content", "") for m in reversed(msgs) if (m.get("role") or "").lower() == "assistant"), "")
        # The conversation's OPENING line (whatever role it is), so the picker
        # label reflects where the dialogue starts rather than its final turn.
        first = (msgs[0].get("content", "") if msgs else "")
        samples.append({"index": i, "user_preview": user[:160], "assistant_preview": asst[:160],
                        "first_preview": first[:160], "messages": msgs})
    return {"key": key, "samples": samples}
