"""Tests for the read-side port + GitHubReader adapter.

`requests.get` is monkeypatched, so these exercise the adapter's real logic
(URL/header construction, ref pinning, status mapping, json convenience)
with no network.
"""
import json

import pytest

from services.registry_sync.reader import (
    GitHubReader,
    RegistryNotFound,
    RegistryReader,
    RegistryReadError,
    build_reader,
)


class _Resp:
    def __init__(self, status_code=200, content=b"", json_data=None, text=""):
        self.status_code = status_code
        self.content = content
        self._json = json_data
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _patch_get(monkeypatch, responder, capture=None):
    """Install a fake requests.get. `responder(url, **kwargs)` -> _Resp.
    `capture` (list) records (url, kwargs) per call."""
    def fake_get(url, **kwargs):
        if capture is not None:
            capture.append((url, kwargs))
        return responder(url, **kwargs)

    monkeypatch.setattr("requests.get", fake_get)


def _reader(token=None):
    return GitHubReader("Offensive-AI-Lab/gavel-rules", "main", token=token)


def test_adapter_conforms_to_the_port():
    assert isinstance(_reader(), RegistryReader)
    assert _reader().name == "github"


# --------------------------------------------------------------------------- #
# head_version — commits API, returns the HEAD sha
# --------------------------------------------------------------------------- #
def test_head_version_parses_commit_sha(monkeypatch):
    cap = []
    _patch_get(monkeypatch, lambda url, **kw: _Resp(json_data={"sha": "abc123"}), cap)
    assert _reader().head_version() == "abc123"
    url, kwargs = cap[0]
    assert url == "https://api.github.com/repos/Offensive-AI-Lab/gavel-rules/commits/main"
    assert kwargs["headers"]["Accept"] == "application/vnd.github+json"


def test_head_version_failure_maps_to_read_error(monkeypatch):
    _patch_get(monkeypatch, lambda url, **kw: _Resp(status_code=500))
    with pytest.raises(RegistryReadError):
        _reader().head_version()


def test_head_version_network_error_maps_to_read_error(monkeypatch):
    def boom(url, **kw):
        raise ConnectionError("dns down")
    _patch_get(monkeypatch, boom)
    with pytest.raises(RegistryReadError):
        _reader().head_version()


# --------------------------------------------------------------------------- #
# fetch_bytes — contents API with the raw accept header, pinned to a ref
# --------------------------------------------------------------------------- #
def test_fetch_bytes_uses_raw_accept_and_default_ref(monkeypatch):
    cap = []
    _patch_get(monkeypatch, lambda url, **kw: _Resp(content=b'{"rules":{}}'), cap)
    assert _reader().fetch_bytes("index.json") == b'{"rules":{}}'
    url, kwargs = cap[0]
    assert url == "https://api.github.com/repos/Offensive-AI-Lab/gavel-rules/contents/index.json"
    assert kwargs["headers"]["Accept"] == "application/vnd.github.raw"
    # No explicit revision -> the reader's own ref.
    assert kwargs["params"] == {"ref": "main"}


def test_fetch_bytes_pins_to_explicit_revision(monkeypatch):
    cap = []
    _patch_get(monkeypatch, lambda url, **kw: _Resp(content=b"x"), cap)
    _reader().fetch_bytes("ces/foo/excitation.json", revision="deadbeef")
    assert cap[0][1]["params"] == {"ref": "deadbeef"}


def test_fetch_bytes_anonymous_when_no_token(monkeypatch):
    cap = []
    _patch_get(monkeypatch, lambda url, **kw: _Resp(content=b"x"), cap)
    _reader().fetch_bytes("index.json")
    assert "Authorization" not in cap[0][1]["headers"]


def test_fetch_bytes_sends_bearer_token_when_set(monkeypatch):
    cap = []
    _patch_get(monkeypatch, lambda url, **kw: _Resp(content=b"x"), cap)
    _reader(token="ghp_secret").fetch_bytes("index.json")
    assert cap[0][1]["headers"]["Authorization"] == "Bearer ghp_secret"


def test_404_maps_to_registry_not_found_with_private_repo_hint(monkeypatch):
    _patch_get(monkeypatch, lambda url, **kw: _Resp(status_code=404))
    with pytest.raises(RegistryNotFound) as ei:
        _reader().fetch_bytes("ces/nope/excitation.json")
    # The #1 setup mistake (private repo, missing token) must be in the message.
    msg = str(ei.value)
    assert "private" in msg and "GITHUB_TOKEN" in msg


def test_not_found_is_a_read_error_subclass():
    assert issubclass(RegistryNotFound, RegistryReadError)


def test_other_http_status_maps_to_read_error(monkeypatch):
    _patch_get(monkeypatch, lambda url, **kw: _Resp(status_code=403, text="rate limited"))
    with pytest.raises(RegistryReadError) as ei:
        _reader().fetch_bytes("index.json")
    assert not isinstance(ei.value, RegistryNotFound)
    assert "403" in str(ei.value)


def test_transport_exception_maps_to_read_error(monkeypatch):
    def boom(url, **kw):
        raise ConnectionError("connection reset")
    _patch_get(monkeypatch, boom)
    with pytest.raises(RegistryReadError):
        _reader().fetch_bytes("index.json")


# --------------------------------------------------------------------------- #
# fetch_json — provided by the base class for free
# --------------------------------------------------------------------------- #
def test_fetch_json_parses_via_fetch_bytes(monkeypatch):
    payload = json.dumps({"id": 1, "name": "x"}).encode("utf-8")
    _patch_get(monkeypatch, lambda url, **kw: _Resp(content=payload))
    assert _reader().fetch_json("index.json") == {"id": 1, "name": "x"}


def test_fetch_json_is_provided_on_any_adapter():
    # A trivial in-memory adapter only implements fetch_bytes/head_version;
    # fetch_json must work for free from the base class.
    class MemReader(RegistryReader):
        name = "mem"

        def head_version(self):
            return "v0"

        def fetch_bytes(self, path, *, revision=None):
            return b'{"ok": true}'

    assert MemReader().fetch_json("anything.json") == {"ok": True}


# --------------------------------------------------------------------------- #
# build_reader — env resolution
# --------------------------------------------------------------------------- #
def test_build_reader_defaults(monkeypatch):
    monkeypatch.delenv("GAVEL_RULES_REPO", raising=False)
    monkeypatch.delenv("GAVEL_RULES_REF", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    r = build_reader()
    assert isinstance(r, GitHubReader)
    assert r.repo == "Offensive-AI-Lab/gavel-rules"
    assert r.ref == "main"
    assert r._token is None


def test_build_reader_respects_env(monkeypatch):
    monkeypatch.setenv("GAVEL_RULES_REPO", "me/my-rules")
    monkeypatch.setenv("GAVEL_RULES_REF", "v2-branch")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_tok")
    r = build_reader()
    assert r.repo == "me/my-rules"
    assert r.ref == "v2-branch"
    assert r._token == "ghp_tok"


def test_build_reader_blank_env_falls_back(monkeypatch):
    # Empty strings (a commented-out .env line rendered as "") must not
    # produce an empty ref or a whitespace token.
    monkeypatch.delenv("GAVEL_RULES_REPO", raising=False)
    monkeypatch.setenv("GAVEL_RULES_REF", "  ")
    monkeypatch.setenv("GITHUB_TOKEN", " ")
    r = build_reader()
    assert r.ref == "main"
    assert r._token is None
