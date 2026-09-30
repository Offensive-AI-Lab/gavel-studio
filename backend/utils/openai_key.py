"""The OpenAI credential: is it there, what do we raise when it isn't, and
how does the operator set one without leaving the app.

Every AI feature in the studio goes through the same key. Before this module
each route discovered its absence in its own way — an opaque 500, a silent
zero-sample dataset, a yellow warning with no next step — and the only cure
was "edit backend/.env and restart the backend", which the person using the
app can't reasonably be asked to do.

So there is exactly one shape for "this needs a key":

    HTTP 503  {"detail": {"code": "OPENAI_KEY_MISSING", "message": "..."}}

`code` is what the frontend keys on (never the prose) to open the "set your
key" prompt. Two situations produce it: the key is absent at request time
(`require_key`) and the provider rejects the credential we do have
(`is_auth_error`, checked where a route already catches LLM failures).

`save_key` closes the loop: it writes the key into backend/.env AND into the
running process, so the feature the operator was in the middle of using works
on the next click and still works after the next boot.
"""

import os
from pathlib import Path

from fastapi import HTTPException

from utils import env_file

ENV_VAR = "OPENAI_API_KEY"

# The machine-readable marker of the contract error. Shared with the frontend;
# changing it breaks the "set your key" prompt.
MISSING_CODE = "OPENAI_KEY_MISSING"
MISSING_MESSAGE = "This feature needs an OpenAI key. Add yours to continue."

# backend/.env — the one config file for both native and Docker runs (main.py
# loads it at import). backend/.env.example seeds it when it doesn't exist yet,
# so a file written here keeps the comments that explain every other setting.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = _BACKEND_DIR / ".env"
ENV_EXAMPLE_PATH = _BACKEND_DIR / ".env.example"

# Markers of "the provider refused this credential", matched case-insensitively
# against the exception text. Text matching is needed on top of the type/status
# checks because several call paths re-wrap the provider's exception in a plain
# RuntimeError before it reaches the route.
_AUTH_MARKERS = (
    "authenticationerror",
    "missing credentials",
    "invalid_api_key",
    "invalid api key",
    "incorrect api key",
    "no api key provided",
    "openai_api_key",
    "openaiexception - unauthorized",
)


def is_configured() -> bool:
    """True when the running process has a non-blank key."""
    return bool((os.environ.get(ENV_VAR) or "").strip())


def key_missing_error() -> HTTPException:
    """The one contract error. Raised by `require_key`, and by routes that
    catch an LLM failure `is_auth_error` identifies as a bad credential."""
    return HTTPException(
        status_code=503,
        detail={"code": MISSING_CODE, "message": MISSING_MESSAGE},
    )


def require_key() -> None:
    """Pre-flight for every LLM-dependent route. Call it before spawning a
    generation thread or writing the row that thread is supposed to fill —
    a route that fails here must leave nothing behind."""
    if not is_configured():
        raise key_missing_error()


def is_auth_error(exc) -> bool:
    """Is this failure the credential being refused, rather than a rate limit,
    a timeout, a missing model or a provider outage?

    Accepts an exception or an error string (some helpers report failure as
    `{"success": False, "error": "..."}` instead of raising).
    """
    if exc is None:
        return False

    status = getattr(exc, "status_code", None)
    try:
        if status is not None and int(status) == 401:
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(exc, BaseException):
        for base in type(exc).__mro__:
            if base.__name__.lower() == "authenticationerror":
                return True

    text = str(exc).lower()
    return any(marker in text for marker in _AUTH_MARKERS)


def save_key(value: str) -> None:
    """Persist the key to backend/.env and apply it to the running process.

    Writes atomically (temp file + os.replace) so a crash mid-write can never
    truncate the operator's config, and keeps every other line and comment in
    the file exactly as it was. A file we create is owner-only — it holds a
    paid credential; a file that already exists keeps the permissions and
    owner it came with.
    """
    key = (value or "").strip()
    if not key:
        raise ValueError("Enter your OpenAI key.")
    # One line per setting: a pasted value carrying a newline would otherwise
    # write a second line into the file and set something else.
    if "\n" in key or "\r" in key:
        raise ValueError("That doesn't look like a key — remove the line breaks.")

    env_file.write_var(ENV_PATH, ENV_VAR, key, fallback_path=ENV_EXAMPLE_PATH)
    os.environ[ENV_VAR] = key
