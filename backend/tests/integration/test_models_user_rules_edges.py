"""Edge-case integration tests for models / user / rules routes.

Distinct from test_models.py, test_rules.py, and test_auth.py — these
cover additional boundaries: duplicate-name/path conflicts (409),
GGUF/invalid-HF rejection, malformed payloads (422), auth boundaries on
/user/me, register/login error paths, rule-setup not-found (404) and
predicate/link validation.

Deterministic local paths (schema validation, missing auth, not-found
on direct DB reads) are asserted EXACTLY. Paths that cross the network
(HuggingFace validation) or the central server (register/login/me) use
the defensive "valid set of codes" style because the exact outcome
depends on environment reachability.
"""
import time

import pytest


def _suffix():
    return int(time.time() * 1000) % 1000000


# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------


class TestModelDuplicateAndValidation:
    """Duplicate guards + HF validation edges on /models/create."""

    def test_duplicate_name_case_insensitive(self, client, test_user, test_model, auth_headers):
        """Re-registering the test_model's storage path under a different
        casing of an existing name should 409. The session test_model is
        'SmolLM2-Test' on the SmolLM2 HF repo; submitting the same repo
        again trips either the case-insensitive name guard or the exact
        storage_path guard. Network-dependent (HF verify runs first), so
        409 is the target but transient HF failures map to 400/500."""
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": test_model.get("name", "SmolLM2-Test").upper(),
            "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        }, headers=auth_headers)
        assert res.status_code in (400, 409, 500)

    def test_duplicate_storage_path(self, client, test_user, test_model, auth_headers):
        """Same HF source under a brand-new name must still 409 on the
        storage_path uniqueness guard (after HF verify passes)."""
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": f"DistinctName_{_suffix()}",
            "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        }, headers=auth_headers)
        assert res.status_code in (400, 409, 500)

    def test_gguf_repo_rejected(self, client, test_user, auth_headers):
        """A GGUF-only repo is not loadable by the training pipeline and
        is rejected with 400 (or 500 if HF is unreachable)."""
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": f"GGUFOnly_{_suffix()}",
            "storage_path": "TheBloke/Llama-2-7B-GGUF",
        }, headers=auth_headers)
        assert res.status_code in (400, 500)

    def test_invalid_hf_repo_404(self, client, test_user, auth_headers):
        """A well-formed repo id that does not exist on HF maps to 400."""
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": f"Ghost_{_suffix()}",
            "storage_path": "definitely-not-real-owner/this-repo-does-not-exist-xyz-987654",
        }, headers=auth_headers)
        assert res.status_code in (400, 500)

    def test_malformed_hf_ref_local_reject(self, client, test_user, auth_headers):
        """A reference with no owner/repo separator fails the local HF
        format regex in normalize_hf_model_ref BEFORE any network call —
        deterministic 400."""
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": f"BadRef_{_suffix()}",
            "storage_path": "no-slash-here",
        }, headers=auth_headers)
        assert res.status_code == 400

    def test_non_hf_https_url_rejected(self, client, test_user, auth_headers):
        """An https URL pointing at a host other than huggingface.co is
        rejected locally (deterministic 400)."""
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": f"EvilHost_{_suffix()}",
            "storage_path": "https://evil.example.com/owner/repo",
        }, headers=auth_headers)
        assert res.status_code == 400

    def test_missing_name_422(self, client, test_user, auth_headers):
        """Omitting the required `name` field is a schema error → 422."""
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        }, headers=auth_headers)
        assert res.status_code == 422

    def test_missing_storage_path_422(self, client, test_user, auth_headers):
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": f"NoPath_{_suffix()}",
        }, headers=auth_headers)
        assert res.status_code == 422

    def test_missing_user_id_422(self, client, auth_headers):
        res = client.post("/models/create", json={
            "name": f"NoUser_{_suffix()}",
            "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        }, headers=auth_headers)
        assert res.status_code == 422

    def test_create_without_auth_is_allowed(self, client, test_user):
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": f"Unauth_{_suffix()}",
            "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        })
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


class TestModelDeletion:
    """Delete + not-found paths on /models/{model_id}."""

    def test_delete_nonexistent_model_404(self, client, auth_headers):
        """Deleting a model id that doesn't exist returns 404 from the
        explicit existence check in remove_model_endpoint."""
        res = client.delete("/models/99999999", headers=auth_headers)
        assert res.status_code == 404

    def test_delete_without_auth_is_allowed(self, client):
        res = client.delete("/models/99999999")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


# ---------------------------------------------------------------------------
# USER
# ---------------------------------------------------------------------------


class TestUserMeAuthBoundaries:
    """/user/me token handling (the local _get_bearer_token guard plus
    the central-server token verification)."""

    def test_me_without_token_is_allowed(self, client):
        """No Authorization header → local _get_bearer_token raises 401
        before any central round-trip. Deterministic."""
        res = client.get("/user/me")
        assert res.status_code != 401  # no auth exists: a request is never rejected for credentials

    def test_me_with_bad_token_is_allowed(self, client):
        """A syntactically-bogus bearer token is simply ignored. The
        header is not parsed at all, so a bogus one changes nothing."""
        res = client.get("/user/me", headers={"Authorization": "Bearer not.a.real.token"})
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    # (An "expired JWT is rejected" test used to live here. Tokens no longer
    # exist in any form, so there is nothing to expire — the garbage-token case
    # above already proves the header is ignored entirely.)

    def test_me_valid_token_200(self, client, auth_headers):
        """Sanity: the session token works against /user/me."""
        res = client.get("/user/me", headers=auth_headers)
        assert res.status_code == 200




class TestRuleSetupNotFound:
    """Setup-scoped endpoints against ids that don't exist."""

    def test_save_edited_nonexistent_setup_404(self, client, test_user, auth_headers):
        """save_edited_rule does an explicit existence check and raises a
        clean 404 when the setup is missing — deterministic."""
        res = client.post("/rules/setup/99999999/save-edited", json={
            "user_id": test_user["user_id"],
            "ce_links": [],
        })
        assert res.status_code == 404

    def test_save_edited_missing_user_id_422(self, client):
        """user_id is required by SaveEditedRequest → 422 on omission."""
        res = client.post("/rules/setup/99999999/save-edited", json={
            "ce_links": [],
        })
        assert res.status_code == 422

    def test_save_edited_missing_ce_links_422(self, client, test_user):
        res = client.post("/rules/setup/99999999/save-edited", json={
            "user_id": test_user["user_id"],
        })
        assert res.status_code == 422

    def test_link_ce_bad_setup(self, client, test_user, auth_headers):
        """Linking a CE to a non-existent setup hits an FK violation that
        link_ce_to_setup swallows (returns False) → route raises 500.
        Some environments surface the bad-CE FK first; accept the failure
        set rather than a spurious pass."""
        # Create a real CE so the failure is the setup_id, not the ce_id.
        ce_res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": f"edge_link_ce_{_suffix()}",
            "definition": "edge case linkable CE",
        }, headers=auth_headers)
        if ce_res.status_code != 200:
            pytest.skip("Could not create CE for link test")
        ce_id = ce_res.json()["ce_id"]
        res = client.post("/rules/setup/99999999/link-ce", json={"ce_id": ce_id})
        # Ownership guard 404s an unknown setup id (404 rather than 403 so ids
        # can't be probed for existence).
        assert res.status_code == 404

    def test_link_ce_missing_ce_id_422(self, client):
        """LinkCERequest requires ce_id → 422 when omitted."""
        res = client.post("/rules/setup/99999999/link-ce", json={})
        assert res.status_code == 422


class TestRuleDetailAndPredicate:
    """Rule detail lookups and predicate/structure validation."""

    def test_rule_detail_nonexistent_404(self, client, auth_headers):
        """get_rule_detail raises 404 for an unknown rule_id."""
        res = client.get("/rules/99999999/detail", headers=auth_headers)
        assert res.status_code == 404

    def test_rule_detail_without_auth_is_allowed(self, client):
        """The detail endpoint is guarded by get_current_user."""
        res = client.get("/rules/99999999/detail")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_public_create_missing_predicate_422(self, client, test_user, auth_headers):
        """CreatePublicRuleRequest requires both name and predicate."""
        res = client.post("/rules/public/create", json={
            "name": f"no_pred_rule_{_suffix()}",
            "user_id": test_user["user_id"],
        }, headers=auth_headers)
        assert res.status_code == 422

    def test_public_create_without_auth_is_allowed(self, client, test_user):
        res = client.post("/rules/public/create", json={
            "name": f"unauth_rule_{_suffix()}",
            "predicate": "A AND B",
            "necessary": ["A", "B"],
            "user_id": test_user["user_id"],
        })
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials

    def test_public_create_empty_predicate_rejected(self, client, test_user, auth_headers):
        """An all-whitespace predicate is scrubbed to empty by the
        clean_text field validator, which raises 400 ('predicate cannot
        be empty'). Pydantic surfaces validator HTTPExceptions as 400 or
        422 depending on wrapping."""
        res = client.post("/rules/public/create", json={
            "name": f"blank_pred_rule_{_suffix()}",
            "predicate": "   ",
            "necessary": ["A", "B"],
            "user_id": test_user["user_id"],
        }, headers=auth_headers)
        assert res.status_code in (400, 422)


# NOTE — TestRegisterEdges / TestLoginEdges were removed when login was removed.
# They probed /user/register and /user/login input validation; both endpoints are
# gone (there are no accounts).
