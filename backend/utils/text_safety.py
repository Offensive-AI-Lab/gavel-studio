import re

from fastapi import HTTPException

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# First char must be alphanumeric so we never end up with usernames like
# "_admin" or "-system" that look like sentinels. Length 3-30 including
# the first char. Case is normalized to lowercase by validate_username
# (the column is CITEXT but we still lowercase at the application layer
# so the stored value is canonical for URLs and display).
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,29}$")


def clean_text(value, *, field_name: str, max_length: int, allow_newlines: bool = False) -> str:
    if value is None:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a string")

    cleaned = CONTROL_CHARS.sub("", value)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    if allow_newlines:
        cleaned = cleaned.strip()
    else:
        cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty")
    if len(cleaned) > max_length:
        raise HTTPException(status_code=400, detail=f"{field_name} must be at most {max_length} characters")

    return cleaned


def clean_optional_text(value, *, field_name: str, max_length: int, allow_newlines: bool = False):
    if value is None:
        return None
    cleaned = clean_text(value, field_name=field_name, max_length=max_length, allow_newlines=allow_newlines)
    return cleaned or None


def validate_username(value) -> str:
    """Clean + format-check + lowercase. User can type "Abc" or "ABC" or
    "abc" — we always store "abc". This makes uniqueness case-insensitive
    at the application layer regardless of DB collation, gives URLs a
    canonical form, and means there's exactly one "true" representation
    of every username everywhere in the system."""
    cleaned = clean_text(value, field_name="username", max_length=30)
    if not USERNAME_PATTERN.fullmatch(cleaned):
        raise HTTPException(
            status_code=400,
            detail=(
                "Username must be 3-30 characters, start with a letter or digit, "
                "and use only letters, numbers, underscore, or dash."
            ),
        )
    return cleaned.lower()