"""Tests for the /infer spec transport — specifically the gzipped `spec_gz`
path that keeps large evaluation requests (the neutral corpus is several MB of
inlined JSON) under hosted-proxy body limits, plus the plain-`spec` back-compat
path. ORCH.submit_batch is stubbed so no GPU/torch is needed.
"""
import gzip
import json

import pytest
from fastapi.testclient import TestClient

from gavel_gpu_worker import app as app_mod
from gavel_gpu_worker.auth import require_token


@pytest.fixture
def client(monkeypatch):
    captured = {}

    def fake_submit(kind, payload, rnn_bytes, dataset_files=None):
        captured["kind"] = kind
        captured["payload"] = payload
        captured["rnn_bytes"] = rnn_bytes
        return "job_test123"

    monkeypatch.setattr(app_mod.ORCH, "submit_batch", fake_submit)
    # Bypass bearer auth for the unit test.
    app_mod.app.dependency_overrides[require_token] = lambda: None
    c = TestClient(app_mod.app)
    c.captured = captured
    c.app_mod = app_mod
    yield c
    app_mod.app.dependency_overrides.clear()


def _payload():
    return {"classifier_id": 99, "dialogues": [{"conversation": [{"role": "user", "content": "hi"}]}]}


def test_infer_accepts_gzipped_spec(client):
    payload = _payload()
    spec_gz = gzip.compress(json.dumps(payload).encode("utf-8"))
    r = client.post(
        "/infer",
        files={
            "rnn": ("trained_rnn.pth", b"\x00weights", "application/octet-stream"),
            "spec_gz": ("spec.json.gz", spec_gz, "application/gzip"),
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["job_id"] == "job_test123"
    # The worker must have gunzipped the spec back to the exact payload.
    assert client.captured["payload"] == payload
    assert client.captured["rnn_bytes"] == b"\x00weights"


def test_infer_still_accepts_plain_spec(client):
    payload = _payload()
    r = client.post(
        "/infer",
        data={"spec": json.dumps(payload)},
        files={"rnn": ("trained_rnn.pth", b"\x00weights", "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    assert client.captured["payload"] == payload


def test_infer_rejects_missing_spec(client):
    r = client.post(
        "/infer",
        files={"rnn": ("trained_rnn.pth", b"\x00weights", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "spec" in r.json()["detail"].lower()


def test_infer_rejects_corrupt_gzip(client):
    r = client.post(
        "/infer",
        files={
            "rnn": ("trained_rnn.pth", b"\x00weights", "application/octet-stream"),
            "spec_gz": ("spec.json.gz", b"not actually gzip", "application/gzip"),
        },
    )
    assert r.status_code == 400
    assert "gzip" in r.json()["detail"].lower()


def test_infer_cleanup_deletes_job(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(client.app_mod.ORCH, "cleanup_batch", lambda jid: seen.setdefault("id", jid))
    r = client.delete("/infer/job_abc")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert seen["id"] == "job_abc"


def test_train_cleanup_deletes_job(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(client.app_mod.ORCH, "cleanup_batch", lambda jid: seen.setdefault("id", jid))
    r = client.delete("/train/train_xyz")
    assert r.status_code == 200
    assert seen["id"] == "train_xyz"


def test_infer_rejects_empty_rnn(client):
    spec_gz = gzip.compress(json.dumps(_payload()).encode("utf-8"))
    r = client.post(
        "/infer",
        files={
            "rnn": ("trained_rnn.pth", b"", "application/octet-stream"),
            "spec_gz": ("spec.json.gz", spec_gz, "application/gzip"),
        },
    )
    assert r.status_code == 400
    assert "rnn" in r.json()["detail"].lower()
