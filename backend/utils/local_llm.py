"""The local-GPU-model setting: is one configured, and how does the operator
set it without leaving the app.

Mirrors utils/openai_key.py's contract — write-only from the frontend's point
of view (GET only reports whether one is set), same backend/.env persistence —
but stores a Hugging Face model path instead of a credential, and generation
runs on this machine's own GPU (see gavel_pipeline/llm_client.py) instead of
calling out to OpenAI. See utils/llm_access.py for the combined "is some LLM
available at all" gate that AI routes actually check.
"""

import os
from pathlib import Path

from utils import env_file

ENV_VAR = "GAVEL_LOCAL_LLM_MODEL"

# Which provider `llm_client.complete()` uses when BOTH an OpenAI key and a
# local model are configured. Irrelevant (and ignored) when only one is set —
# that one is used regardless. Defaults to "openai" so setting a local model
# never silently steals traffic from an already-working key; switching to
# local is something the operator opts into.
PROVIDER_ENV_VAR = "GAVEL_LLM_PROVIDER"
PROVIDERS = ("openai", "local")

_BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = _BACKEND_DIR / ".env"
ENV_EXAMPLE_PATH = _BACKEND_DIR / ".env.example"


def is_configured() -> bool:
    """True when the running process has a non-blank local model path."""
    return bool(get_model())


def get_model() -> str:
    """The configured Hugging Face model path/repo id, or "" if none is set."""
    return (os.environ.get(ENV_VAR) or "").strip()


def save_model(value: str) -> None:
    """Persist the model path to backend/.env and apply it to the running
    process, the same way `openai_key.save_key` does for the OpenAI key."""
    model_id = (value or "").strip()
    if not model_id:
        raise ValueError("Enter a Hugging Face model path.")
    if "\n" in model_id or "\r" in model_id:
        raise ValueError("That doesn't look like a model path — remove the line breaks.")

    env_file.write_var(ENV_PATH, ENV_VAR, model_id, fallback_path=ENV_EXAMPLE_PATH)
    os.environ[ENV_VAR] = model_id


def clear_model() -> None:
    """Unset the local model so AI routes fall back to the OpenAI key."""
    os.environ.pop(ENV_VAR, None)
    env_file.write_var(ENV_PATH, ENV_VAR, "", fallback_path=ENV_EXAMPLE_PATH)


def get_active_provider() -> str:
    """'openai' or 'local'. Only meaningful when both are configured; see
    the module-level PROVIDER_ENV_VAR comment for the default."""
    value = (os.environ.get(PROVIDER_ENV_VAR) or "").strip().lower()
    return value if value in PROVIDERS else "openai"


def set_active_provider(value: str) -> None:
    provider = (value or "").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError("Provider must be 'openai' or 'local'.")
    env_file.write_var(ENV_PATH, PROVIDER_ENV_VAR, provider, fallback_path=ENV_EXAMPLE_PATH)
    os.environ[PROVIDER_ENV_VAR] = provider
