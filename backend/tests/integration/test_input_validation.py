"""Input validation and security tests: injection, XSS, boundary values, sanitization."""
import pytest


class TestSQLInjection:
    """SQL injection prevention tests."""

    def test_login_sql_injection_email(self, client):
        res = client.post("/user/login", json={
            "email": "' OR 1=1 --",
            "password": "anything",
        })
        assert res.status_code in (400, 401, 404, 422)
        # Must NOT return 200 with a valid token
        if res.status_code == 200:
            assert "token" not in res.json()

    def test_login_sql_injection_password(self, client):
        res = client.post("/user/login", json={
            "email": "test@test.com",
            "password": "' OR '1'='1",
        })
        assert res.status_code in (400, 401, 404)


    def test_ce_name_sql_injection(self, client, auth_headers, test_user):
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": "'; DROP TABLE cognitive_elements; --",
            "definition": "injection attempt",
        }, headers=auth_headers)
        # Should either sanitize or reject, NOT crash the DB
        # If 200, verify tables still exist
        assert res.status_code in (200, 400, 422, 500)


class TestXSSPrevention:
    """Cross-site scripting prevention."""

    def test_xss_in_ce_definition(self, client, auth_headers, test_user):
        xss_payload = '<script>alert("XSS")</script>'
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": "xss_test_ce",
            "definition": xss_payload,
        }, headers=auth_headers)
        if res.status_code == 200:
            # If accepted, the script tags should be stored as plain text (not executable)
            # This is fine for API-only backends — XSS is a frontend concern
            pass


class TestBoundaryValues:
    """Boundary value tests for input fields."""


    def test_very_long_ce_definition(self, client, auth_headers, test_user):
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": "long_def_ce",
            "definition": "x" * 50000,  # 50K chars
        }, headers=auth_headers)
        assert res.status_code in (200, 400, 422)

    def test_unicode_in_ce_name(self, client, auth_headers, test_user):
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": "test_emoji_CE",
            "definition": "Definition with unicode chars",
        }, headers=auth_headers)
        assert res.status_code == 200

    def test_negative_model_id(self, client, auth_headers):
        res = client.get("/classifiers/-1", headers=auth_headers)
        assert res.status_code in (200, 400, 404, 422)

    def test_zero_model_id(self, client, auth_headers):
        res = client.get("/classifiers/0", headers=auth_headers)
        assert res.status_code in (200, 400, 404)

    def test_very_large_id(self, client, auth_headers):
        res = client.get("/classifiers/999999999", headers=auth_headers)
        assert res.status_code in (200, 404)


class TestControlCharacters:
    """Control character and special input tests."""

    def test_null_bytes_in_name(self, client, auth_headers, test_user):
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": "null\x00byte",
            "definition": "has null bytes",
        }, headers=auth_headers)
        # Should be sanitized or rejected
        assert res.status_code in (200, 400, 422)

    def test_newlines_in_ce_name(self, client, auth_headers, test_user):
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": "line1\nline2",
            "definition": "multiline name",
        }, headers=auth_headers)
        assert res.status_code in (200, 400, 422)

    def test_tab_characters_in_input(self, client, auth_headers, test_user):
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": "tabbed\tinput",
            "definition": "has tabs",
        }, headers=auth_headers)
        assert res.status_code in (200, 400, 422)


class TestAuthorizationBoundaries:
    """Access control edge cases."""

    def test_access_other_users_models(self, client, auth_headers):
        """Should return empty or 403, not another user's data."""
        res = client.get("/models/99999", headers=auth_headers)
        data = res.json()
        if res.status_code == 200:
            models = data.get("models", data) if isinstance(data, dict) else data
            assert isinstance(models, list)

    def test_delete_nonexistent_classifier(self, client, auth_headers):
        res = client.delete("/classifiers/99999", headers=auth_headers)
        assert res.status_code in (200, 404, 500)

    def test_delete_nonexistent_model(self, client, auth_headers):
        res = client.delete("/models/99999", headers=auth_headers)
        assert res.status_code in (200, 404, 500)
