"""Gate for every AI-generation route: is *some* way to talk to an LLM
configured — an OpenAI key, or a local model running on this machine's own
GPU (utils/local_llm.py + gavel_pipeline/llm_client.py)?

Reuses the OpenAI-key contract's error code (`openai_key.MISSING_CODE`)
rather than inventing a new one: the frontend's "you need to unlock AI
features" banner already keys on that code everywhere a route can 503, and
both ways to unlock (the existing "Set API key" button, and the new "Set
local LLM" button next to it) belong on the same banner — see
openai_key.py's module docstring for why `code` and not prose is what the
frontend matches on.
"""

from fastapi import HTTPException

from utils import local_llm, openai_key

MISSING_MESSAGE = "This feature needs an OpenAI key or a local model. Add one to continue."


def is_configured() -> bool:
    """True once either an OpenAI key or a local model is set."""
    return openai_key.is_configured() or local_llm.is_configured()


def require_llm() -> None:
    """Pre-flight for every LLM-dependent route — replaces the old direct
    `openai_key.require_key()` call so a configured local model also
    satisfies the gate."""
    if not is_configured():
        raise HTTPException(
            status_code=503,
            detail={"code": openai_key.MISSING_CODE, "message": MISSING_MESSAGE},
        )
