"""Integration-test fixtures.

Owns everything that needs a live database:
  * The FastAPI TestClient.
  * Session-level fixtures that create a test user / model / classifier.
  * The DB snapshot+restore mechanism that cleans up after every test.

Unit tests do not see these fixtures because pytest's conftest discovery is
directory-scoped — only files under tests/integration/ pick up this file.

The cleanup strategy:

  1. At session start (`_session_baseline`) — snapshot the current PK state of
     every tracked table. This is the "safety net" that catches anything the
     session-scoped fixtures themselves create (test_user, test_model,
     test_classifier).

  2. Before every test (`_per_test_cleanup`) — snapshot the current state
     again, run the test, then delete any rows that appeared during the test.
     With per-test cleanup, even an interrupted run only pollutes the rows
     created by the single test that was in flight.

  3. At session end — restore to the session baseline so the dev DB is in the
     exact state we found it.

Foreign keys in the schema all use ON DELETE CASCADE, so deleting parent rows
automatically removes children. The deletion order in `_TRACKED_TABLES` is
children-first to avoid relying on cascade behavior, which keeps the cleanup
robust if a future migration weakens any FK to RESTRICT.
"""
import threading
import uuid

import pytest

# ---------------------------------------------------------------------------
# Boot-recovery barrier — MUST be installed BEFORE `from main import app`.
#
# Importing main spawns daemon threads; one of them runs full crash recovery
# ~20s later (after model warmup), and IncompletePipelineRecovery wipes every
# `is_ready = FALSE` rule/CE it finds. A test holding an in-flight draft at
# that moment loses it mid-test (observed: finalize-draft 404s because the
# draft vanished). We wrap run_all_recovery to flag completion, and the
# session fixture below blocks until the boot pass is done before any test
# creates state it cares about.
# ---------------------------------------------------------------------------
import utils.crash_recovery as _crash_recovery

_BOOT_RECOVERY_DONE = threading.Event()
_real_run_all_recovery = _crash_recovery.run_all_recovery


def _tracked_run_all_recovery(*args, **kwargs):
    try:
        return _real_run_all_recovery(*args, **kwargs)
    finally:
        _BOOT_RECOVERY_DONE.set()


_crash_recovery.run_all_recovery = _tracked_run_all_recovery

from fastapi.testclient import TestClient

from main import app
from utils.auth import (
    LOCAL_USER_ID, LOCAL_USERNAME, LOCAL_EMAIL,
)


# Tables we track for per-test cleanup, listed children-first so deletes never
# trip an FK constraint. Tables with seed data (categories) are intentionally
# excluded — wiping them would break every test.
#
# Each entry is (table_name, pk_columns) where pk_columns is a TUPLE of the
# column(s) forming the row identity. The junction tables (setup_ce_link,
# rule_ce_link) have COMPOSITE primary keys — there is no surrogate `id`
# column — so they must be tracked by their full key tuple. Tracking them by
# a non-existent "id" column silently broke their cleanup (the query raised
# "column id does not exist", the exception was swallowed, and orphaned link
# rows leaked across the whole session).
_TRACKED_TABLES = [
    # Junction tables (reference rules / cognitive_elements / classifiers).
    # COMPOSITE keys — see note above. Membership-only since schema v24:
    # the role/fallback_group columns died with the groups+condition model.
    ("setup_ce_link", ("setup_id", "ce_id")),
    ("rule_ce_link", ("rule_id", "ce_id")),
    # Bookmarks are local tables again (login removal moved them back off the
    # central server), so they need per-test cleanup. Ratings are gone entirely.
    ("rule_bookmarks", ("user_id", "rule_id")),
    ("ce_bookmarks", ("user_id", "ce_id")),
    ("rule_set_bookmarks", ("user_id", "rule_set_id")),
    # Datasets attached to CEs.
    ("excitation_datasets", ("dataset_id",)),
    ("calibration_datasets", ("dataset_id",)),
    # Datasets / results attached to classifiers.
    ("evaluation_results", ("eval_id",)),
    ("test_datasets", ("dataset_id",)),
    # Per-classifier rule wiring.
    ("rule_setup", ("setup_id",)),
    # Classifiers and the models they belong to.
    ("classifiers", ("classifier_id",)),
    ("target_models", ("model_id",)),
    # Top-level definitions.
    ("rules", ("rule_id",)),
    ("cognitive_elements", ("ce_id",)),
    # In-flight pipeline state (orphaned by interrupted AI-pipeline tests).
    ("pipeline_runs", ("run_id",)),
    # Users last.
    ("users", ("user_id",)),
]


def _snapshot_pks() -> dict:
    """Capture the set of primary keys currently present in every tracked
    table. Returned dict is `{table_name: set(pk_tuples)}` — each row's
    identity is the tuple of its pk column values, which supports both single-
    column and composite primary keys. Failures are swallowed (return empty
    set) because some tables may be missing in tests that initialise a partial
    schema."""
    from utils.sqlite_db import execute_query_dict
    snap: dict = {}
    for tbl, pk_cols in _TRACKED_TABLES:
        cols = ", ".join(pk_cols)
        try:
            rows = execute_query_dict(f"SELECT {cols} FROM {tbl}") or []
            snap[tbl] = {tuple(r[c] for c in pk_cols) for r in rows}
        except Exception:
            snap[tbl] = set()
    return snap


def _restore_to(snapshot: dict) -> None:
    """Delete any row whose pk tuple is not in the snapshot. Iterates the
    tracked tables in declared (children-first) order. We make two passes: the
    second catches anything that lingered because of weird FK interactions in
    the first pass. Errors are swallowed per-row so one stuck row never blocks
    the rest of the cleanup."""
    from utils.sqlite_db import execute_query, execute_query_dict
    for _ in range(2):
        for tbl, pk_cols in _TRACKED_TABLES:
            cols = ", ".join(pk_cols)
            where = " AND ".join(f"{c} = %s" for c in pk_cols)
            try:
                rows = execute_query_dict(f"SELECT {cols} FROM {tbl}") or []
                current = {tuple(r[c] for c in pk_cols) for r in rows}
                stale = current - snapshot.get(tbl, set())
                for pk_values in stale:
                    try:
                        execute_query(
                            f"DELETE FROM {tbl} WHERE {where}", tuple(pk_values)
                        )
                    except Exception:
                        pass
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _await_boot_recovery():
    """Block until main's boot-time crash-recovery daemon has finished, so it
    can't wipe a test's in-flight (is_ready=FALSE) draft mid-test. Bounded:
    a stuck warmup shouldn't deadlock the suite."""
    _BOOT_RECOVERY_DONE.wait(timeout=180)


@pytest.fixture(scope="session", autouse=True)
def _session_baseline(_await_boot_recovery):
    """Outer safety net. Snapshot at session start, restore at session end so
    the dev database returns exactly to the state we found it — including
    rows that the session-scoped fixtures (test_user et al.) created."""
    baseline = _snapshot_pks()
    yield baseline
    _restore_to(baseline)


@pytest.fixture(scope="session", autouse=True)
def _materialize_session_fixtures(_session_baseline, request):
    """Force the session-scoped resources into existence BEFORE any per-test
    snapshot is taken.

    This is load-bearing, and its absence was a latent bug. `_per_test_cleanup`
    deletes every row that appeared while a test ran. `test_model` /
    `test_classifier` are lazy session fixtures, so without this they are
    created *during* whichever test first asks for them — and that test's
    teardown then deletes them, leaving every later test holding a dangling
    model_id / classifier_id (404s cascading through evaluation, realtime and
    crash-recovery).

    It stayed hidden while those fixtures reused a PRE-EXISTING row (they used
    to return `models[0]`), because a pre-existing row is part of the session
    baseline and is never considered "new". Now that they mint dedicated rows,
    they must be materialized here instead.
    """
    request.getfixturevalue("test_classifier")   # pulls test_model -> test_user -> client


@pytest.fixture(autouse=True)
def _per_test_cleanup(_materialize_session_fixtures):
    """Inner cleanup. Captures DB state right before the test runs, then on
    teardown deletes anything new. Depends on `_materialize_session_fixtures`
    so the session-scoped rows already exist when this snapshot is taken and
    are therefore NOT deleted between tests."""
    pre = _snapshot_pks()
    yield
    _restore_to(pre)


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient — lives for the whole test session."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def test_user(client):
    """The single local user, read from the DATABASE.

    There is no registration or login: `DButils.ensure_local_user()` seeds a
    fixed row at boot and every request is attributed to it. We read the row
    rather than returning the seed constants, because `ensure_local_user()`
    uses ON CONFLICT DO NOTHING — on a database that predates the login removal,
    user_id=1 keeps whatever username it already had. Hardcoding the constants
    here would make every test that compares this fixture's username against an
    API response fail on such an install.
    """
    from utils.sqlite_db import execute_query_dict
    rows = execute_query_dict(
        "SELECT user_id, username, email FROM users WHERE user_id = %s",
        (LOCAL_USER_ID,),
    ) or []
    if rows:
        return {
            "user_id": rows[0]["user_id"],
            "username": rows[0]["username"],
            "email": rows[0]["email"],
        }
    return {"user_id": LOCAL_USER_ID, "username": LOCAL_USERNAME, "email": LOCAL_EMAIL}


@pytest.fixture(scope="session")
def auth_headers():
    """Kept so the many `headers=auth_headers` call sites stay valid. There is
    no authentication, so these are simply empty."""
    return {}


@pytest.fixture(scope="session")
def test_model(client, test_user, auth_headers):
    """Create a DEDICATED test model once per session.

    It must be dedicated, not "the user's first model". These fixtures used to
    reuse `models[0]`, which was harmless when every run registered a brand-new
    empty user — there was nothing to reuse. There is only one user now (login
    was removed), and it is the operator's real account, so reusing [0] would
    hand the suite whatever real model happens to sort first and let tests
    mutate it. The name is uniquified per run and the row is cleaned up by the
    session baseline restore.
    """
    name = f"pytest-model-{uuid.uuid4().hex[:8]}"
    res = client.post("/models/create", json={
        "user_id": test_user["user_id"],
        "name": name,
        "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
    }, headers=auth_headers)
    if res.status_code == 200:
        data = res.json()
        return data.get("model", data)
    pytest.skip(f"Could not create test model: {res.status_code} {res.text[:200]}")


@pytest.fixture(scope="session")
def test_classifier(client, test_model, auth_headers):
    """Create a DEDICATED test classifier once per session.

    Same reasoning as test_model: this used to return `classifiers[0]` for the
    model, which with a real account could hand the suite a guardrail that is
    mid-training or already calibrated — and evaluation / realtime / crash-
    recovery tests assume a clean one.
    """
    model_id = test_model.get("model_id")
    if not model_id:
        pytest.skip("No model_id in test_model")

    res = client.post("/classifiers/create", json={
        "model_id": model_id,
        "name": f"pytest-classifier-{uuid.uuid4().hex[:8]}",
    }, headers=auth_headers)
    if res.status_code == 200:
        data = res.json()
        return data.get("classifier", data)
    pytest.skip(f"Could not create test classifier: {res.status_code} {res.text[:200]}")
