"""Unit tests for services.central_server — the HTTP client for the central
server's HuggingFace WRITE proxy.

Since login was removed, that proxy is the module's only remaining
responsibility: no auth, no user directory, no ratings, no bookmarks, and no
tokens to forward. What is left to test is request shaping and error mapping.

Pure-logic: no real network, no live central server. The module uses a single
module-level httpx.Client (`_client`) and a module-level `CENTRAL_SERVER_URL`;
we monkeypatch both:

  * `central_server._client` -> a fake client whose `.request(...)` returns a
    canned `_FakeResponse` (or raises httpx.RequestError to model a timeout /
    connection failure).
  * `central_server.CENTRAL_SERVER_URL` -> a known base URL so the method,
    path, JSON body and params can be asserted on.
"""
import httpx
import pytest

from services import central_server as cs


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Mimics the slice of httpx.Response that _request() touches:
    status_code, headers (dict-like with .get), .json(), .text."""

    def __init__(self, status_code=200, json_body=None, text="",
                 content_type="application/json", raise_on_json=False):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text
        self.headers = {"content-type": content_type}
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("malformed JSON")
        return self._json_body


class _RecordingClient:
    """Stands in for the module-level httpx.Client. Records each request and
    returns a queued response (or raises a queued exception)."""

    def __init__(self, response=None, exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    def request(self, method, url, headers=None, json=None, params=None):
        self.calls.append({
            "method": method,
            "url": url,
            "headers": headers,
            "json": json,
            "params": params,
        })
        if self.exc is not None:
            raise self.exc
        return self.response

    @property
    def last(self):
        return self.calls[-1]


@pytest.fixture
def base_url(monkeypatch):
    """Configure a known central server base URL on the module."""
    url = "https://central.example.test"
    monkeypatch.setattr(cs, "CENTRAL_SERVER_URL", url)
    return url


def _install_client(monkeypatch, response=None, exc=None):
    client = _RecordingClient(response=response, exc=exc)
    monkeypatch.setattr(cs, "_client", client)
    return client


# ---------------------------------------------------------------------------
# _headers — auth header shaping
# ---------------------------------------------------------------------------



class TestIsEnabled:
    def test_enabled_when_url_set(self, monkeypatch):
        monkeypatch.setattr(cs, "CENTRAL_SERVER_URL", "https://x")
        assert cs.is_enabled() is True

    def test_disabled_when_url_empty(self, monkeypatch):
        monkeypatch.setattr(cs, "CENTRAL_SERVER_URL", "")
        assert cs.is_enabled() is False


# ---------------------------------------------------------------------------
# _request — core request shaping + response handling
# ---------------------------------------------------------------------------


class TestRequestCore:
    def test_unconfigured_url_raises_503(self, monkeypatch):
        monkeypatch.setattr(cs, "CENTRAL_SERVER_URL", "")
        # The client should never be hit if URL is unset.
        client = _install_client(monkeypatch, response=_FakeResponse())
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/anything")
        assert ei.value.status_code == 503
        assert client.calls == []

    def test_builds_full_url_and_passes_method(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"ok": True}))
        out = cs._request("GET", "/hf/head-sha")
        assert out == {"ok": True}
        assert client.last["method"] == "GET"
        assert client.last["url"] == base_url + "/hf/head-sha"


    def test_never_sends_an_authorization_header(self, base_url, monkeypatch):
        # There is no login, so nothing can ever be authenticated to central.
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs._request("GET", "/hf/head-sha")
        assert "Authorization" not in client.last["headers"]

    def test_json_body_and_params_forwarded(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs._request("POST", "/x", json={"a": 1}, params={"p": "q"})
        assert client.last["json"] == {"a": 1}
        assert client.last["params"] == {"p": "q"}

    def test_parses_json_when_content_type_json(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            response=_FakeResponse(json_body={"hello": "world"}))
        assert cs._request("GET", "/x") == {"hello": "world"}

    def test_returns_text_for_non_json_content_type(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                json_body=None, text="plain body",
                content_type="text/plain"))
        assert cs._request("GET", "/x") == "plain body"

    def test_missing_content_type_returns_text(self, base_url, monkeypatch):
        # headers.get("content-type", "") -> "" -> not application/json -> text
        resp = _FakeResponse(text="raw", content_type="")
        # Force an empty content-type header.
        resp.headers = {}
        _install_client(monkeypatch, response=resp)
        assert cs._request("GET", "/x") == "raw"


# ---------------------------------------------------------------------------
# _request — error branches
# ---------------------------------------------------------------------------


class TestRequestErrors:
    def test_non_200_raises_with_detail_from_payload(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                status_code=404, json_body={"detail": "not found"}))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert ei.value.status_code == 404
        assert "not found" in str(ei.value)
        assert ei.value.payload == {"detail": "not found"}

    def test_error_payload_without_detail_falls_back_to_http_code(self, base_url, monkeypatch):
        # payload is a dict but has no "detail" key -> detail is None ->
        # falls back to "HTTP {status}".
        _install_client(
            monkeypatch,
            response=_FakeResponse(status_code=500, json_body={"other": "x"}))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert "HTTP 500" in str(ei.value)
        assert ei.value.status_code == 500

    def test_error_with_malformed_json_uses_text(self, base_url, monkeypatch):
        # resp.json() raises -> payload = {"detail": resp.text}
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                status_code=400, raise_on_json=True,
                text="gateway exploded"))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert "gateway exploded" in str(ei.value)
        assert ei.value.status_code == 400

    def test_error_with_non_dict_json_payload(self, base_url, monkeypatch):
        # JSON parses but is a list, not a dict -> detail = str(payload).
        _install_client(
            monkeypatch,
            response=_FakeResponse(status_code=422, json_body=["bad", "input"]))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert ei.value.status_code == 422
        # detail derived via str(payload) since payload isn't a dict
        assert "bad" in str(ei.value)

    def test_connection_error_raises_502(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            exc=httpx.ConnectError("connection refused"))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert ei.value.status_code == 502
        assert "unreachable" in str(ei.value)

    def test_timeout_raises_502(self, base_url, monkeypatch):
        # httpx.TimeoutException is a subclass of httpx.RequestError.
        assert issubclass(httpx.TimeoutException, httpx.RequestError)
        _install_client(
            monkeypatch,
            exc=httpx.ConnectTimeout("timed out"))
        with pytest.raises(cs.CentralServerError) as ei:
            cs._request("GET", "/x")
        assert ei.value.status_code == 502

    def test_400_is_error_boundary(self, base_url, monkeypatch):
        # status_code >= 400 is the error boundary; 400 raises.
        _install_client(
            monkeypatch,
            response=_FakeResponse(status_code=400, json_body={"detail": "bad"}))
        with pytest.raises(cs.CentralServerError):
            cs._request("GET", "/x")

    def test_399_is_not_error(self, base_url, monkeypatch):
        # Just below the boundary: treated as success. 399 isn't a real HTTP
        # code but it exercises the < 400 branch precisely.
        _install_client(
            monkeypatch,
            response=_FakeResponse(status_code=399, json_body={"ok": 1}))
        assert cs._request("GET", "/x") == {"ok": 1}


# ---------------------------------------------------------------------------
# CentralServerError construction
# ---------------------------------------------------------------------------


class TestCentralServerError:
    def test_defaults(self):
        e = cs.CentralServerError("boom")
        assert e.status_code == 500
        assert e.payload == {}
        assert str(e) == "boom"

    def test_payload_none_becomes_empty_dict(self):
        e = cs.CentralServerError("boom", status_code=404, payload=None)
        assert e.payload == {}

    def test_custom_payload_retained(self):
        e = cs.CentralServerError("boom", status_code=409, payload={"k": "v"})
        assert e.status_code == 409
        assert e.payload == {"k": "v"}


# ---------------------------------------------------------------------------
# Bookmarks — request shaping + list parsing
# ---------------------------------------------------------------------------






class TestHfProxy:
    def test_hf_head_sha_extracts_sha(self, base_url, monkeypatch):
        _install_client(
            monkeypatch, response=_FakeResponse(json_body={"sha": "deadbeef"}))
        assert cs.hf_head_sha() == "deadbeef"

    def test_hf_head_sha_missing_returns_none(self, base_url, monkeypatch):
        _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        assert cs.hf_head_sha() is None

    def test_hf_head_sha_non_dict_returns_none(self, base_url, monkeypatch):
        _install_client(
            monkeypatch,
            response=_FakeResponse(
                json_body=None, text="x", content_type="text/plain"))
        assert cs.hf_head_sha() is None

    def test_hf_commit_base64_encodes_content(self, base_url, monkeypatch):
        import base64
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={"commit": "c1"}))
        ops = [{"path": "a.txt", "content": b"hello"}]
        cs.hf_commit(operations=ops, commit_message="msg")
        sent = client.last["json"]
        assert sent["commit_message"] == "msg"
        assert sent["parent_commit"] is None
        # Content is base64-encoded for transport; decode round-trips.
        enc = sent["operations"][0]["content_b64"]
        assert base64.b64decode(enc) == b"hello"
        assert sent["operations"][0]["path"] == "a.txt"

    def test_hf_commit_passes_parent_commit(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.hf_commit(operations=[{"path": "p", "content": b""}],
                     commit_message="m", parent_commit="abc")
        assert client.last["json"]["parent_commit"] == "abc"

    def test_hf_commit_empty_operations(self, base_url, monkeypatch):
        client = _install_client(
            monkeypatch, response=_FakeResponse(json_body={}))
        cs.hf_commit(operations=[], commit_message="m")
        assert client.last["json"]["operations"] == []


# ---------------------------------------------------------------------------
# RPC logging breadcrumbs
# ---------------------------------------------------------------------------


class TestRpcLogging:
    def test_success_logs_breadcrumb(self, base_url, monkeypatch, caplog):
        _install_client(
            monkeypatch, response=_FakeResponse(status_code=200, json_body={}))
        with caplog.at_level("INFO", logger="central-rpc"):
            cs._request("GET", "/hf/head-sha")
        # The one-line breadcrumb records method, path, and status code.
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "GET" in msgs
        assert "/hf/head-sha" in msgs
        assert "200" in msgs

    def test_failure_logs_warning(self, base_url, monkeypatch, caplog):
        _install_client(monkeypatch, exc=httpx.ConnectError("refused"))
        with caplog.at_level("WARNING", logger="central-rpc"):
            with pytest.raises(cs.CentralServerError):
                cs._request("GET", "/hf/head-sha")
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "FAIL" in msgs
        assert "/hf/head-sha" in msgs


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_calls_client_close(self, monkeypatch):
        calls = []

        class _Closeable:
            def close(self):
                calls.append(True)

        monkeypatch.setattr(cs, "_client", _Closeable())
        cs.close()
        assert calls == [True]

    def test_close_swallows_exceptions(self, monkeypatch):
        class _Exploding:
            def close(self):
                raise RuntimeError("already closed")

        monkeypatch.setattr(cs, "_client", _Exploding())
        # Must not raise.
        cs.close()
