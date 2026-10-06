"""Single call-in point for every "generate text from a chat prompt" call in
the AI pipeline. Every call site used to call `litellm.completion(model=
"gpt-4.1", ...)` directly, which meant every one of them hard-required an
OpenAI key. This module decides which provider actually answers the request,
so adding a provider — or, here, adding a local-GPU model as an alternative
to OpenAI — means editing one file instead of every call site.

Routing: with only one of {OpenAI key, local model} configured, that one is
used. With both configured, utils/local_llm.get_active_provider() decides —
defaulting to OpenAI, so setting a local model never silently steals traffic
from an already-working key. The two are not combined or raced — nothing
here retries an OpenAI failure on the local model or vice versa.

The return value always exposes `.choices[0].message.content`, matching what
litellm's response already gave every call site, so swapping
`litellm.completion(...)` for `llm_client.complete(...)` is a one-line change
at each site.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from utils import local_llm, openai_key

logger = logging.getLogger(__name__)

# How many times a local JSON-mode call may be regenerated before giving up.
# OpenAI's json_object mode guarantees valid JSON; a local model has no such
# guarantee (nothing constrains its decoding), so one missing comma would
# otherwise fail the whole CE/rule build. Retries are cheap next to that.
_JSON_ATTEMPTS = 4

# Local model + tokenizer stay loaded across requests — reloading a model
# from disk on every "Generate" click would make the feature unusable. Keyed
# by model id so switching the configured path loads the new one instead of
# silently reusing the old weights.
_load_lock = threading.Lock()
_loaded: dict = {}  # model_id -> (model, tokenizer, device)

# Local generation is single-request-at-a-time: these are small models on a
# shared GPU (probe training uses the same device — utils/device.py), and
# HF `generate` isn't safe to call concurrently on one model instance.
_generate_lock = threading.Lock()

# Default cap on a local generation. 4096 covers most call sites; the rule-
# generation prompt (rule_generator_prompt.md) demands a long structured
# "reasoning" section BEFORE groups/condition/new_ces even appear, or the
# output truncates mid-`reasoning` — 800 (the original default) cut it off
# before `new_ces` was ever reached, so the rule's groups referenced CE names
# nothing had defined yet ("references unknown CE(s)"). Callers with an
# unusually verbose prompt can still override via `local_max_tokens=`.
_MAX_NEW_TOKENS = 4096

# Load progress, for the frontend's "loading model…" indicator
# (routes/settings.py's GET /local-llm/status). Separate from `_loaded`
# (which only knows "resident or not") — this also captures "in progress"
# and "the last load attempt failed, here's why".
_status_lock = threading.Lock()
_status: dict = {}  # model_id -> {"state": "loading"|"ready"|"error", "detail": str|None}


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message = field(default=None)


@dataclass
class _Response:
    choices: list


def _set_status(model_id: str, state: str, detail: str = None, device: str = None) -> None:
    with _status_lock:
        entry = {"state": state, "detail": detail}
        if device is not None:
            entry["device"] = device
        _status[model_id] = entry


def _record_progress(model_id: str, n, total, desc: str) -> None:
    if not total:
        return
    with _status_lock:
        cur = _status.get(model_id)
        # Only meaningful mid-load; a bar that fires after we've already
        # moved on (or before "loading" was recorded) is stale/irrelevant.
        if not cur or cur.get("state") != "loading":
            return
        cur["progress"] = {
            "desc": (desc or "").strip(),
            "n": n,
            "total": total,
            "percent": round(100 * n / total),
        }


@contextlib.contextmanager
def _track_progress(model_id: str):
    """Mirror every tqdm bar huggingface_hub/transformers draw during the
    wrapped call into `_status[model_id]["progress"]`, so the frontend's
    loading indicator has real percentages — first "fetching N files" (the
    download), then "loading weights" (fast, but still real progress) —
    instead of just an elapsed-seconds spinner.

    Patches the actual `tqdm.tqdm` class process-wide for the duration of
    the `with` block. Safe here specifically because `_load_local_model`
    only ever runs one load at a time under `_load_lock`, so nothing else
    is expected to be driving an unrelated tqdm bar concurrently; the patch
    is removed in `finally` regardless of how the block exits.
    """
    import tqdm as tqdm_pkg

    orig_update = tqdm_pkg.tqdm.update

    def patched_update(self, n=1):
        result = orig_update(self, n)
        _record_progress(model_id, self.n, self.total, self.desc)
        return result

    tqdm_pkg.tqdm.update = patched_update
    try:
        yield
    finally:
        tqdm_pkg.tqdm.update = orig_update


def _device_label(device) -> str:
    if device.type == "cuda":
        try:
            import torch
            idx = device.index if device.index is not None else 0
            return f"cuda:{idx} — {torch.cuda.get_device_name(device)}"
        except Exception:
            return str(device)
    if device.type == "mps":
        return "mps — Apple GPU"
    return "cpu"


def get_status(model_id: str) -> dict:
    """Load progress for `model_id`: 'not_loaded' (never attempted this
    process), 'loading' (optionally with a `progress` dict), 'ready' (with
    `device`), or 'error' (with `detail`)."""
    with _status_lock:
        return dict(_status.get(model_id, {"state": "not_loaded", "detail": None}))


def start_warmup(model_id: str) -> str:
    """Kick off loading `model_id` in a background thread if it isn't
    already resident or loading. Returns immediately with the resulting
    state ('ready' if already cached, 'loading' otherwise) — callers poll
    get_status() for progress instead of blocking on this."""
    if _loaded.get(model_id) is not None:
        return "ready"
    if get_status(model_id)["state"] == "loading":
        return "loading"

    def _run():
        try:
            _load_local_model(model_id)
        except Exception:
            pass  # _load_local_model already recorded the error status

    threading.Thread(target=_run, daemon=True).start()
    return "loading"


def _load_local_model(model_id: str):
    with _load_lock:
        cached = _loaded.get(model_id)
        if cached is not None:
            _set_status(model_id, "ready", device=_device_label(cached[2]))
            return cached

        _set_status(model_id, "loading")
        try:
            from classifier_engine.utils_train import load_model_and_tokenizer

            with _track_progress(model_id):
                model, tokenizer = load_model_and_tokenizer(model_id)
        except Exception as e:
            _set_status(model_id, "error", str(e))
            raise
        device = next(model.parameters()).device
        entry = (model, tokenizer, device)

        # Keep at most one local model resident — these routes are used one
        # at a time by one operator, and holding two would double the VRAM
        # a probe-training job on the same GPU has to work around.
        _loaded.clear()
        _loaded[model_id] = entry
        _set_status(model_id, "ready", device=_device_label(device))
        return entry


def _normalize_messages(messages: list, fold_system: bool = False) -> list:
    """Collapse into the strict system?/user/assistant/user/assistant/...
    shape several chat templates (Mistral's, notably) enforce via a Jinja
    {% raise_exception %}. Every call site was written against OpenAI's
    tolerant API — which accepts a system message anywhere, or two
    consecutive same-role messages (e.g. rule_generator.call_thinking_model
    turns a leading "system" into "user", which can land right next to an
    existing "user" turn) — so this is the one place that adapts to the
    stricter local format instead of teaching every call site about it.

    - Only the FIRST message may stay "system"; a later one is downgraded to
      "user" rather than dropped, since it usually carries a real
      instruction (e.g. "respond with only valid JSON").
    - An "assistant" message before any "user" message is dropped: a strict
      template requires the turn right after the optional system message to
      be "user", and a scripted opening line has no user turn to attach to
      (ai_pipeline.py's scenario/CE chats seed history as
      [system, assistant: <static greeting>] before the operator types
      anything). The greeting is still shown in the UI; it just isn't fed
      back as conversation history the local model must parse.
    - Consecutive messages of the same role are merged (content joined by a
      blank line) rather than dropped, so no instruction is lost.
    - `fold_system=True` moves a leading system message into the FIRST user
      turn instead of leaving it as a system role. For templates that don't
      keep the system prompt at the top of the conversation (see
      `_template_keeps_system_first`).
    """
    normalized = []
    seen_user = False
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system" and i != 0:
            role = "user"
        if role == "user":
            seen_user = True
        elif role == "assistant" and not seen_user:
            continue
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] = f"{normalized[-1]['content']}\n\n{content}"
        else:
            normalized.append({"role": role, "content": content})
    if fold_system and normalized and normalized[0]["role"] == "system":
        system = normalized.pop(0)
        if normalized and normalized[0]["role"] == "user":
            normalized[0]["content"] = f"{system['content']}\n\n{normalized[0]['content']}"
        else:
            normalized.insert(0, {"role": "user", "content": system["content"]})
    return normalized


_system_first_cache: dict = {}


def _template_keeps_system_first(tokenizer) -> bool:
    """Does this chat template keep the system prompt at the TOP of a
    multi-turn conversation, the way OpenAI-style models treat it?

    Mistral's template doesn't: it glues the system prompt onto the LAST user
    message, so on every turn the system prompt sits right next to the newest
    user text. The scenario chat's system prompt ends with "Now, greet the
    user and start the conversation!", so every reply became a fresh greeting
    that ignored what the operator had just said, and the final "scenario"
    was copied from the prompt's own example instead of the conversation.

    Probed by rendering a tiny conversation rather than keyed on model names:
    True only if the system text appears BEFORE the first user turn. A
    template that drops the system role, or rejects it, counts as False (the
    caller then folds the system prompt into the first user turn instead of
    losing it).
    """
    key = getattr(tokenizer, "name_or_path", None) or id(tokenizer)
    if key in _system_first_cache:
        return _system_first_cache[key]
    probe = [
        {"role": "system", "content": "SYSPROBE"},
        {"role": "user", "content": "USERPROBE1"},
        {"role": "assistant", "content": "ASSTPROBE"},
        {"role": "user", "content": "USERPROBE2"},
    ]
    try:
        text = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
        sys_at, first_user_at = text.find("SYSPROBE"), text.find("USERPROBE1")
        keeps = sys_at != -1 and first_user_at != -1 and sys_at < first_user_at
    except Exception:  # noqa: BLE001 - a template that raises can't carry a system role
        keeps = False
    _system_first_cache[key] = keeps
    return keeps


def _build_prompt(tokenizer, messages: list, want_json: bool) -> str:
    msgs = list(messages)
    if want_json:
        msgs = msgs + [{
            "role": "system",
            "content": "Respond with only valid JSON. No prose, no markdown fences.",
        }]
    has_template = bool(getattr(tokenizer, "chat_template", None))
    msgs = _normalize_messages(
        msgs, fold_system=has_template and not _template_keeps_system_first(tokenizer),
    )
    if has_template:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Plain fallback for a base model with no chat template.
    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs]
    lines.append("assistant:")
    return "\n".join(lines)


def _local_complete(
    messages: list, temperature: float, response_format: dict | None, max_tokens: int = None,
) -> _Response:
    import torch

    model_id = local_llm.get_model()
    model, tokenizer, device = _load_local_model(model_id)
    want_json = bool(response_format and response_format.get("type") == "json_object")

    def generate(temp: float, retry_feedback=None) -> str:
        msgs = messages
        if retry_feedback:
            # Show the model its own bad output and WHAT was wrong with it,
            # rather than resampling the same prompt: at a similar temperature
            # a blind resample tends to repeat the same slip (e.g. a missing
            # brace), which is how a 3-attempt retry loop used to fail 3 times.
            bad, hint = retry_feedback
            msgs = list(messages) + [
                {"role": "assistant", "content": bad},
                {"role": "user", "content": (
                    f"That was not valid JSON ({hint}). Reply with ONLY the complete, valid "
                    "JSON object — every brace and bracket closed, nothing before or after it."
                )},
            ]
        prompt = _build_prompt(tokenizer, msgs, want_json)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with _generate_lock, torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens or _MAX_NEW_TOKENS,
                do_sample=temp > 0,
                temperature=max(temp, 0.01),
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    if want_json:
        text = _generate_valid_json(generate, temperature)
    else:
        text = generate(temperature)
    return _Response(choices=[_Choice(message=_Message(content=text))])


def _close_unbalanced_json(s: str):
    """Repair JSON that stops before its closing braces/brackets — the most
    common way a local model's JSON is broken (it forgets one `}`) — by closing
    an unterminated string, dropping a dangling comma/colon, and appending the
    missing closers in the right order. Returns None when there is nothing to
    close or the structure is mismatched (not a simple truncation); it does NOT
    attempt to fix errors in the middle of a document."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
    if not stack and not in_str:
        return None
    out = s + ('"' if in_str else "")
    out = out.rstrip()
    while out.endswith(","):
        out = out[:-1].rstrip()
    if out.endswith(":"):
        out += " null"
    return out + "".join(reversed(stack))


# What a stock json.JSONDecodeError message usually MEANS in model output, in
# words a model can act on. Keyed by a lowercase substring of the message.
_JSON_HINTS = (
    ("control character",
     'a string value was probably not closed with a double quote (") before the line '
     'break — close every string with " and write line breaks inside strings as \\n'),
    ("unterminated string",
     'a string value was not closed with a double quote (")'),
    ("property name enclosed",
     "a trailing comma, a comment, or a placeholder like `...` is where a quoted key belongs"),
    ("delimiter",
     "a comma is missing between two items, or a quote/bracket was not closed"),
    ("expecting value",
     "a value is missing (for example after a trailing comma or an empty field)"),
)


def describe_json_error(blob: str, err) -> str:
    """Say WHERE and (usually) WHY `blob` failed to parse, for a model to act on.

    "Invalid control character at: line 22 column 121" means nothing to a model:
    it cannot count to column 121 of a 5,000-character answer, so a retry that
    quoted only that message came back byte-for-byte identical. Quoting the
    offending line and naming the usual cause gets it somewhere to look."""
    msg = (getattr(err, "msg", None) or str(err)).removesuffix(" at")
    lineno, colno = getattr(err, "lineno", None), getattr(err, "colno", None)
    parts = [f"{msg} at line {lineno}, column {colno}" if lineno else msg]
    if lineno and colno:
        lines = blob.splitlines()
        if 0 < lineno <= len(lines):
            line = lines[lineno - 1]
            # Plain text in backticks, NOT repr(): repr adds backslashes to the
            # quotes, i.e. shows the model text it never wrote.
            parts.append(f"near: `{line[max(0, colno - 90):colno + 60]}`")
    hint = next((h for k, h in _JSON_HINTS if k in msg.lower()), "")
    if hint:
        parts.append(hint)
    return "; ".join(parts)


def _json_error_hint(text: str) -> str:
    """Why `text` didn't parse, for the model's retry."""
    start = text.find("{")
    if start == -1:
        return "no JSON object found"
    sub = text[start:]
    try:
        json.JSONDecoder(strict=False).raw_decode(sub)
    except ValueError as e:
        hint = describe_json_error(sub, e)
        # A raw line break inside a string is tolerated on its own, but it is ALSO
        # what an UNCLOSED string looks like (the parser runs on to the next quote
        # and blames something later). If a strict parse chokes on one EARLIER than
        # where the lenient parse gave up, say so too: that located clue is what
        # made a model repair `'}` written instead of `"}`.
        try:
            json.JSONDecoder().raw_decode(sub)
        except ValueError as strict_err:
            if ("control character" in (getattr(strict_err, "msg", "") or "").lower()
                    and getattr(strict_err, "pos", 1 << 30) < getattr(e, "pos", 0)):
                hint += "; also, " + describe_json_error(sub, strict_err)
        return hint
    return "it is not a JSON object"


def _extract_json_object(text: str):
    """The outermost JSON object in `text` — bare or inside a ``` fence,
    ignoring prose before it and junk after it — or None if it doesn't parse.

    Only the FIRST `{` of each candidate is tried. If the outer object is
    broken, scanning on to later braces would happily parse a nested
    fragment and return it as if it were the whole answer."""
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)
    # strict=False: a raw line break / tab INSIDE a string value is a very common
    # model slip (a multi-sentence string written across lines). It is
    # unambiguous, and json.dumps() on the way out re-escapes it, so callers
    # that json.loads() our output (strictly) are fine.
    decoder = json.JSONDecoder(strict=False)
    for cand in candidates:
        start = cand.find("{")
        if start == -1:
            continue
        try:
            obj, _ = decoder.raw_decode(cand[start:])
        except ValueError:
            obj = None
            closed = _close_unbalanced_json(cand[start:])
            if closed is not None:
                try:
                    obj = json.loads(closed, strict=False)
                except ValueError:
                    obj = None
        if isinstance(obj, dict):
            return obj
    return None


def extract_json(text: str):
    """The outermost JSON object in a model's reply (bare or fenced; tolerates
    prose around it and a missing closing brace), or None. For call sites that
    ask for free text and parse the JSON themselves."""
    return _extract_json_object(text)


def _generate_valid_json(generate, temperature: float) -> str:
    """Emulate OpenAI's json_object mode on a model that can't enforce it:
    regenerate (at a lower, steadier temperature) until the output contains a
    parseable JSON object, then return just that object re-serialized. If
    every attempt fails, return the last raw text so the caller's own parse
    error surfaces exactly as it would have before."""
    text = generate(temperature)
    for attempt in range(1, _JSON_ATTEMPTS + 1):
        obj = _extract_json_object(text)
        if obj is not None:
            return json.dumps(obj)
        if attempt == _JSON_ATTEMPTS:
            break
        logger.warning(
            "[llm_client] local model returned invalid JSON (attempt %d/%d); regenerating",
            attempt, _JSON_ATTEMPTS,
        )
        # Steadier each time; the LAST try is greedy (temperature 0), the most
        # conservative decode, which tends to copy-and-fix its previous answer
        # rather than reshuffle it.
        retry_temp = 0.0 if attempt == _JSON_ATTEMPTS - 1 else min(temperature, 0.3)
        text = generate(retry_temp, (text, _json_error_hint(text)))
    return text


def _use_local() -> bool:
    if not local_llm.is_configured():
        return False
    if not openai_key.is_configured():
        return True
    return local_llm.get_active_provider() == "local"


def using_local() -> bool:
    """True when `complete()` would answer from the local model right now.
    For callers that add behavior only a local model needs (e.g. a corrective
    retry on a malformed answer), so the OpenAI path stays exactly as it was."""
    return _use_local()


def complete(
    messages: list,
    model: str = "gpt-4.1",
    temperature: float = 0.2,
    response_format: dict | None = None,
    local_max_tokens: int | None = None,
    **litellm_kwargs: Any,
):
    """Drop-in for `litellm.completion(...)`: routes to the configured local
    GPU model or litellm/OpenAI per `_use_local()`. `local_max_tokens`
    overrides the local path's default budget (`_MAX_NEW_TOKENS`) and is
    deliberately NOT forwarded to OpenAI — a cap there could starve a
    reasoning model (gpt-5.x counts hidden reasoning tokens against it)."""
    if _use_local():
        return _local_complete(messages, temperature, response_format, local_max_tokens)

    import litellm

    kwargs = {"model": model, "messages": messages, "temperature": temperature, **litellm_kwargs}
    if response_format is not None:
        kwargs["response_format"] = response_format
    return litellm.completion(**kwargs)
