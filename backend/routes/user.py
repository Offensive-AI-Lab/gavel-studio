"""User + contributor endpoints — LOCAL only, no accounts.

Login was removed: GAVEL Studio runs as a single-operator localhost app, so
there is no register/login/verify and no central user directory. What survives
here is everything that can be answered from the LOCAL database:

  * the local user's own profile (display name / bio / tutorial flag), so the
    Profile page and the onboarding modal keep working;
  * contributor profiles for OTHER authors, reconstructed from the
    `created_by_username` stamped on every rule/CE synced from the public
    HuggingFace registry — you can still browse who wrote what;
  * contributor discovery (search + leaderboard) over exactly those authors.

Ratings are gone with the accounts that gave them meaning, so every
rating-shaped field below reports 0 / None. The response SHAPES are unchanged
so the frontend needs no special-casing.
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from utils.auth import LOCAL_USER_ID, LOCAL_USERNAME
from utils.PostgreSQL import execute_query, execute_query_dict
from utils.text_safety import clean_text

router = APIRouter()


# ---------------------------------------------------------------------------
# Local identity
# ---------------------------------------------------------------------------

def _local_user_row() -> dict:
    rows = execute_query_dict(
        "SELECT user_id, username, email, display_name, bio, is_team, tutorial_seen "
        "FROM users WHERE user_id = %s",
        (LOCAL_USER_ID,),
    ) or []
    if rows:
        return rows[0]
    # Should be impossible (DButils.ensure_local_user runs at boot), but never
    # 500 the Profile page over it.
    return {
        "user_id": LOCAL_USER_ID, "username": LOCAL_USERNAME, "email": None,
        "display_name": None, "bio": None, "is_team": False, "tutorial_seen": False,
    }


@router.get("/me")
def get_me():
    """The local user. No token — there is only one."""
    return _local_user_row()


class UpdateMeRequest(BaseModel):
    """Editable profile fields. `username` stays fixed — it is the key every
    locally-authored rule/CE is stamped with, so renaming it would orphan your
    own contributions."""
    display_name: Optional[str] = Field(None, max_length=255)
    bio: Optional[str] = Field(None, max_length=2000)


@router.patch("/me")
def update_me(req: UpdateMeRequest):
    updates, params = [], []
    if req.display_name is not None:
        cleaned = clean_text(req.display_name, field_name="display_name", max_length=255)
        updates.append("display_name = %s")
        params.append((cleaned or "").strip() or None)
    if req.bio is not None:
        cleaned = clean_text(req.bio, field_name="bio", max_length=2000, allow_newlines=True)
        updates.append("bio = %s")
        params.append((cleaned or "").strip() or None)

    if updates:
        params.append(LOCAL_USER_ID)
        execute_query(f"UPDATE users SET {', '.join(updates)} WHERE user_id = %s", tuple(params))
    return _local_user_row()


@router.put("/tutorial-seen")
def mark_tutorial_seen():
    """Flip the onboarding flag. Used to live on the central server; now it is
    just a column on the local user row."""
    execute_query(
        "UPDATE users SET tutorial_seen = TRUE WHERE user_id = %s", (LOCAL_USER_ID,)
    )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Contributor profiles (reconstructed from the synced library)
# ---------------------------------------------------------------------------

class ProfileResponse(BaseModel):
    user_id: int
    username: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    is_team: bool
    member_since: Optional[str] = None
    contribution_count_rules: int = 0
    contribution_count_ces: int = 0
    # Ratings were removed along with accounts; kept in the shape so the
    # frontend's profile card renders unchanged.
    total_rating_count: int = 0
    avg_rating_received: Optional[float] = None
    last_published_at: Optional[str] = None


def _local_published_counts(usernames: List[str]) -> dict:
    """username(lowercased) -> {"rules": n, "ces": n} over PUBLISHED items in the
    local synced library."""
    unames = sorted({(u or "").strip().lower() for u in usernames if u and str(u).strip()})
    if not unames:
        return {}
    out = {u: {"rules": 0, "ces": 0} for u in unames}
    for table, key in (("rules", "rules"), ("cognitive_elements", "ces")):
        rows = execute_query_dict(
            f"SELECT LOWER(created_by_username) AS u, COUNT(*) AS n FROM {table} "
            "WHERE LOWER(created_by_username) = ANY(%s) AND is_local_draft = FALSE GROUP BY 1",
            (unames,),
        ) or []
        for r in rows:
            out.setdefault(r["u"], {"rules": 0, "ces": 0})[key] = r["n"]
    return out


def _last_published(username: str) -> Optional[str]:
    rows = execute_query_dict(
        """
        SELECT MAX(published_at) AS ts FROM (
            SELECT published_at FROM rules
            WHERE LOWER(created_by_username) = %s AND is_local_draft = FALSE
            UNION ALL
            SELECT published_at FROM cognitive_elements
            WHERE LOWER(created_by_username) = %s AND is_local_draft = FALSE
        ) x
        """,
        (username, username),
    ) or []
    ts = rows[0]["ts"] if rows else None
    return ts.isoformat() if ts else None


def _known_contributors() -> List[dict]:
    """Every author with at least one PUBLISHED rule or CE in the local library,
    enriched with their local `users` row when one exists (the local user always
    has one; remote authors usually don't, so they get username-only profiles).
    """
    rows = execute_query_dict(
        """
        SELECT LOWER(created_by_username) AS username FROM rules
        WHERE created_by_username IS NOT NULL AND is_local_draft = FALSE
        UNION
        SELECT LOWER(created_by_username) AS username FROM cognitive_elements
        WHERE created_by_username IS NOT NULL AND is_local_draft = FALSE
        """
    ) or []
    names = [r["username"] for r in rows if r.get("username")]
    if not names:
        return []

    counts = _local_published_counts(names)
    profiles = execute_query_dict(
        "SELECT user_id, LOWER(username) AS uname, username AS display_username, "
        "display_name, bio, is_team FROM users WHERE LOWER(username) = ANY(%s)",
        (names,),
    ) or []
    by_name = {p["uname"]: p for p in profiles}

    out = []
    for n in sorted(names):
        p = by_name.get(n, {})
        c = counts.get(n, {"rules": 0, "ces": 0})
        out.append({
            "user_id": p.get("user_id") or 0,
            "username": p.get("display_username") or n,
            "display_name": p.get("display_name"),
            "bio": p.get("bio"),
            "is_team": bool(p.get("is_team", False)),
            "contribution_count_rules": c["rules"],
            "contribution_count_ces": c["ces"],
            "total_rating_count": 0,
            "avg_rating_received": None,
        })
    return out


@router.get("/profile/{username}", response_model=ProfileResponse)
def get_profile(username: str):
    """A contributor's profile, assembled from the local library.

    Works for any author name that appears on a synced rule/CE — they never had
    to log into this machine (they can't; there is no login).
    """
    uname = (username or "").strip().lower()
    if not uname:
        raise HTTPException(status_code=400, detail="username required")

    rows = execute_query_dict(
        "SELECT user_id, username, email, display_name, bio, is_team, created_at "
        "FROM users WHERE LOWER(username) = %s",
        (uname,),
    ) or []
    counts = _local_published_counts([uname]).get(uname, {"rules": 0, "ces": 0})

    if not rows and counts["rules"] == 0 and counts["ces"] == 0:
        raise HTTPException(status_code=404, detail="User not found")

    row = rows[0] if rows else {}
    created = row.get("created_at")
    return ProfileResponse(
        user_id=row.get("user_id") or 0,
        username=row.get("username") or uname,
        email=row.get("email"),
        display_name=row.get("display_name"),
        bio=row.get("bio"),
        is_team=bool(row.get("is_team", False)),
        member_since=created.isoformat() if created else None,
        contribution_count_rules=counts["rules"],
        contribution_count_ces=counts["ces"],
        last_published_at=_last_published(uname),
    )


@router.get("/profile/{username}/contributions")
def get_profile_contributions(
    username: str,
    type: str = "rule",
    page: int = 1,
    page_size: int = 20,
):
    """Paginated list of an author's PUBLISHED rules or CEs. Drafts are excluded
    — this reflects the public library only. (Unchanged behaviour: this endpoint
    always read the local DB.)"""
    username = (username or "").strip().lower()
    if type not in ("rule", "ce"):
        raise HTTPException(status_code=400, detail="type must be 'rule' or 'ce'")
    if page < 1:
        raise HTTPException(status_code=400, detail="page must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="page_size must be 1-100")

    offset = (page - 1) * page_size
    if type == "rule":
        total = execute_query_dict(
            "SELECT COUNT(*) AS n FROM rules "
            "WHERE LOWER(created_by_username) = %s AND is_local_draft = FALSE",
            (username,),
        )[0]["n"]
        items = execute_query_dict(
            """
            SELECT rule_id AS id, name, predicate AS content,
                   public_id, published_at, categories
            FROM rules
            WHERE LOWER(created_by_username) = %s AND is_local_draft = FALSE
            ORDER BY published_at DESC NULLS LAST, rule_id DESC
            LIMIT %s OFFSET %s
            """,
            (username, page_size, offset),
        )
    else:
        total = execute_query_dict(
            "SELECT COUNT(*) AS n FROM cognitive_elements "
            "WHERE LOWER(created_by_username) = %s AND is_local_draft = FALSE",
            (username,),
        )[0]["n"]
        items = execute_query_dict(
            """
            SELECT ce_id AS id, name, definition AS content,
                   public_id, published_at, categories
            FROM cognitive_elements
            WHERE LOWER(created_by_username) = %s AND is_local_draft = FALSE
            ORDER BY published_at DESC NULLS LAST, ce_id DESC
            LIMIT %s OFFSET %s
            """,
            (username, page_size, offset),
        )

    for it in (items or []):
        if it.get("published_at"):
            it["published_at"] = it["published_at"].isoformat()
        it["categories"] = it.get("categories") or []

    return {
        "username": username,
        "type": type,
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items or [],
    }


# ---------------------------------------------------------------------------
# Contributor discovery (search + leaderboard) over the synced library
# ---------------------------------------------------------------------------

class ArtistSummary(BaseModel):
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    is_team: bool
    contribution_count_rules: int
    contribution_count_ces: int
    total_rating_count: int = 0
    avg_rating_received: Optional[float] = None


class ArtistListResponse(BaseModel):
    page: int
    page_size: int
    total: int
    items: List[ArtistSummary]


def _paginate(items: list, page: int, page_size: int) -> ArtistListResponse:
    start = (page - 1) * page_size
    return ArtistListResponse(
        page=page,
        page_size=page_size,
        total=len(items),
        items=[ArtistSummary(**it) for it in items[start:start + page_size]],
    )


@router.get("/search", response_model=ArtistListResponse)
def search_artists(q: str = "", page: int = 1, page_size: int = 20):
    """Find contributors in the locally-synced library. Substring match on
    username / display name."""
    if page < 1 or page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="invalid pagination")
    needle = (q or "").strip().lower()
    items = _known_contributors()
    if needle:
        items = [
            it for it in items
            if needle in it["username"].lower()
            or needle in (it.get("display_name") or "").lower()
        ]
    return _paginate(items, page, page_size)


@router.get("/leaderboard", response_model=ArtistListResponse)
def leaderboard(by: str = "avg_rating", page: int = 1, page_size: int = 20, min_ratings: int = 0):
    """Top contributors by published volume.

    `by` and `min_ratings` are accepted for frontend compatibility but no longer
    change the ordering: with ratings removed there is only one meaningful
    ranking left, so both modes sort by total contributions.
    """
    if by not in ("avg_rating", "count"):
        raise HTTPException(status_code=400, detail="by must be 'avg_rating' or 'count'")
    if page < 1 or page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="invalid pagination")

    items = _known_contributors()
    items.sort(
        key=lambda it: (it["contribution_count_rules"] + it["contribution_count_ces"]),
        reverse=True,
    )
    return _paginate(items, page, page_size)
