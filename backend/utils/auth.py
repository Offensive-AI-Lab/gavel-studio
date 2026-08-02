"""Single-user local identity.

GAVEL Studio runs as a localhost, single-operator application: one backend,
one SQLite file, one person. There is no login. Every request is attributed to
ONE fixed local user (`LOCAL_USER_ID`), seeded into the `users` table by
`utils.DButils.init_database`.

Why a constant instead of removing user_id entirely: the schema is built
around ownership. Six tables carry `user_id` with `ON DELETE CASCADE`
(target_models, classifiers, guardrail_folders, bundle_jobs, test_datasets,
pipeline_runs), trained artifacts live under
`trained_classifiers/<user_id>/classifier_<id>/`, and the ownership guards in
`utils.ownership` join through it. Keeping a stable id means none of that has
to change — the ~100 `Depends(get_current_user)` call sites across the routes
keep working verbatim, they just always resolve to the same person.

IMPORTANT — the deployment assumption: this is only safe because the backend
binds to 127.0.0.1 and serves exactly one human. If two people ever share one
backend process, they share this identity and therefore each other's models,
rule sets and trained weights. Do not expose this backend on a shared host.
"""
# The one and only local user. Seeded by init_database(); every FK that
# references users(user_id) points here.
LOCAL_USER_ID = 1
LOCAL_USERNAME = "local"
LOCAL_EMAIL = "local@gavel.local"
LOCAL_DISPLAY_NAME = "Local User"


def get_current_user() -> int:
    """FastAPI dependency: the acting user id.

    Deliberately takes no parameters and never raises, so every existing
    `Depends(get_current_user)` call site is unaffected and no route can 401.
    """
    return LOCAL_USER_ID

