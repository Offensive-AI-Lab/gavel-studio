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

from utils import local_llm, openai_key
from utils.auth import get_current_user

router = APIRouter()

# Long enough for any provider key format, short enough that a pasted document
# can't reach the file writer.
_MAX_KEY_LENGTH = 512


_MAX_MODEL_LENGTH = 256


class LocalLlmRequest(BaseModel):
    # Same reasoning as OpenAiKeyRequest below: left untyped so a pydantic
    # failure echoes `input` back, and the offending input is at most a model
    # path, never a secret — but the same "don't repeat what was sent on
    # error" discipline still applies for consistency.
    model: Any = None


@router.get("/local-llm")
def get_local_llm_status(_: int = Depends(get_current_user)):
    """Whether a local model is configured, and which one. Unlike the OpenAI
    key this isn't a secret, so (unlike GET /openai-key) the value itself is
    returned — the frontend prefills the field with it."""
    return {"configured": local_llm.is_configured(), "model": local_llm.get_model()}


@router.put("/local-llm")
def put_local_llm(req: LocalLlmRequest, _: int = Depends(get_current_user)):
    """Save a local Hugging Face model path and start using it immediately.

    No provider ping here (unlike PUT /openai-key): validating a HF path for
    real means downloading it, which can take minutes and isn't a fair thing
    to do inside a save click. A bad path instead surfaces as a normal
    generation error the first time an AI route runs."""
    raw = req.model if isinstance(req.model, str) else ""
    model_id = raw.strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="Enter a Hugging Face model path.")
    if len(model_id) > _MAX_MODEL_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="That's too long to be a model path.",
        )

    try:
        local_llm.save_model(model_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save the model path to backend/.env: {e}",
        )

    return {"configured": True, "model": model_id}


@router.delete("/local-llm")
def delete_local_llm(_: int = Depends(get_current_user)):
    """Clear the local model so AI routes fall back to the OpenAI key."""
    local_llm.clear_model()
    return {"configured": False, "model": ""}


@router.get("/local-llm/status")
def get_local_llm_load_status(_: int = Depends(get_current_user)):
    """Is the configured local model actually loaded onto the GPU yet?
    Distinct from GET /local-llm, which only reports whether a path is SET —
    this reports load progress, for the frontend's loading indicator."""
    model_id = local_llm.get_model()
    if not model_id:
        return {"state": "not_configured", "detail": None}
    from gavel_pipeline import llm_client
    return llm_client.get_status(model_id)


@router.post("/local-llm/warmup")
def post_local_llm_warmup(_: int = Depends(get_current_user)):
    """Start loading the configured local model in the background, so the
    frontend can show progress before the user's first chat message instead
    of that message silently taking minutes on a cold load."""
    model_id = local_llm.get_model()
    if not model_id:
        raise HTTPException(status_code=400, detail="No local model is set.")
    from gavel_pipeline import llm_client
    return {"state": llm_client.start_warmup(model_id)}


class AiProviderRequest(BaseModel):
    provider: Any = None


@router.get("/ai-provider")
def get_ai_provider_status(_: int = Depends(get_current_user)):
    """Everything the "which AI backs generation" UI needs in one call: is
    each credential set, which local model, and which one wins when both
    are. `active_provider` only matters when both are configured — with
    just one set, that one runs regardless of this value."""
    return {
        "openai_configured": openai_key.is_configured(),
        "local_configured": local_llm.is_configured(),
        "local_model": local_llm.get_model(),
        "active_provider": local_llm.get_active_provider(),
    }


@router.put("/ai-provider")
def put_ai_provider(req: AiProviderRequest, _: int = Depends(get_current_user)):
    """Pick which configured provider generation should use. Rejects picking
    one that isn't actually configured, rather than silently storing a
    preference nothing can satisfy yet."""
    provider = req.provider if isinstance(req.provider, str) else ""
    provider = provider.strip().lower()
    if provider not in local_llm.PROVIDERS:
        raise HTTPException(status_code=400, detail="Provider must be 'openai' or 'local'.")
    if provider == "openai" and not openai_key.is_configured():
        raise HTTPException(status_code=400, detail="No OpenAI key is set yet.")
    if provider == "local" and not local_llm.is_configured():
        raise HTTPException(status_code=400, detail="No local model is set yet.")

    local_llm.set_active_provider(provider)
    return {"active_provider": provider}


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
