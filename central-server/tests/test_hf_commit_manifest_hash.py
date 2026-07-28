"""The /hf/commit response carries manifest_sha256 = the sha256 of the manifest
it ACTUALLY committed (after augment_manifest version-stamps it).

The publisher caches that value so its own next reconcile short-circuits and it
never flags itself "behind" over the stamp the central server (not the
publisher) computed. HfApi is mocked, so no network / token is needed.
"""
import base64
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.manifest_versions import augment_manifest


@pytest.fixture
def client(monkeypatch):
    from app.routes import hf as hf_route
    monkeypatch.setattr(hf_route, "HF_TOKEN", "fake-token")

    class _Info:
        oid = "commit-sha-1"

    class _Api:
        def __init__(self, *a, **k):
            pass

        def create_commit(self, **kw):
            return _Info()

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def _b64(obj) -> str:
    return base64.b64encode(json.dumps(obj).encode("utf-8")).decode("ascii")


def test_commit_returns_stamped_manifest_hash(client):
    # Empty record indices -> no referential-integrity guard -> no list_repo_files.
    manifest = {"rules": {}, "ces": {}, "neutral": {}}
    resp = client.post("/hf/commit", json={
        "operations": [{"path": "manifest.json", "content_b64": _b64(manifest)}],
        "commit_message": "test publish",
        "parent_commit": None,
    })

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"

    # The hash must be over the STAMPED bytes, serialized exactly as the route does.
    stamped = json.dumps(augment_manifest(json.loads(json.dumps(manifest))),
                         ensure_ascii=False).encode("utf-8")
    assert body["manifest_sha256"] == hashlib.sha256(stamped).hexdigest()


def test_commit_without_manifest_has_no_hash(client):
    # A batch that doesn't touch manifest.json carries no manifest hash.
    resp = client.post("/hf/commit", json={
        "operations": [{"path": "public_ces/x.json", "content_b64": _b64({"id": "x"})}],
        "commit_message": "ce only",
        "parent_commit": None,
    })
    assert resp.status_code == 200, resp.text
    assert resp.json().get("manifest_sha256") is None
