"""Set the OpenAI key from the app.

There is no settings page and no first-run wizard: the key is asked for at the
moment an AI feature needs it and can't run. These two endpoints back that
prompt.

The key is write-only. GET answers "is one set?" and nothing else — the value
never leaves the backend, the same rule the stored HF token follows
(`has_hf_token`, sql_scripts/model_scripts.py). PUT stores it in backend/.env
and in the running process, so the feature the operator was using works on the
next click without a restart.

No answer from these endpoints ever contains the submitted key — not the
success body, and not a rejection: an error says what is wrong, never what was
sent.

A key the provider refuses is rejected here (400) rather than saved and
discovered later. Anything else that goes wrong while checking — no network, a
rate limit, an outage — is NOT a reason to refuse a key the operator says is
good, so those save normally.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from utils import openai_key
from utils.auth import get_current_user

router = APIRouter()

# Long enough for any provider key format, short enough that a pasted document
# can't reach the file writer.
_MAX_KEY_LENGTH = 512


class OpenAiKeyRequest(BaseModel):
    # Deliberately untyped and optional: a pydantic failure would be answered
    # with a 422 whose body quotes the offending `input` back at the caller,
    # and the offending input here is the operator's key. Nothing about this
    # field is validated by the schema; the handler does it all, and says what
    # is wrong without repeating the value.
    api_key: Any = None


@router.get("/openai-key")
def get_openai_key_status(_: int = Depends(get_current_user)):
    """Whether an OpenAI key is configured. Never returns the key."""
    return {"configured": openai_key.is_configured()}


@router.put("/openai-key")
def put_openai_key(req: OpenAiKeyRequest, _: int = Depends(get_current_user)):
    """Save an OpenAI key and start using it immediately."""
    raw = req.api_key if isinstance(req.api_key, str) else ""
    key = raw.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Enter your OpenAI key.")
    if len(key) > _MAX_KEY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="That's too long to be a key. Paste just the key itself.",
        )

    if _provider_rejects(key):
        raise HTTPException(
            status_code=400,
            detail="OpenAI did not accept that key. Check it and try again.",
        )

    try:
        openai_key.save_key(key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save the key to backend/.env: {e}",
        )

    return {"configured": True}


def _provider_rejects(api_key: str) -> bool:
    """Ask the provider to honour the credential once, with the smallest call
    that can be made. Only an outright refusal counts — see module docstring."""
    try:
        import litellm

        litellm.completion(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            api_key=api_key,
            timeout=20,
        )
        return False
    except Exception as e:
        return openai_key.is_auth_error(e)
