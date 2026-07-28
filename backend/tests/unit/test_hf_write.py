"""Unit tests for services.hf_write — the direct HuggingFace WRITE client that
replaced the central-server proxy.

Pure-logic: no real network. `huggingface_hub.HfApi` is monkeypatched with a
fake that records calls and returns canned results (or raises). The repo ref
is patched too, so these tests never import-couple to hf_sync's env handling.

What must hold (the contract hf_publish relies on):
  * response shape {"status": "success"|"race"|"error", "commit_sha",
    "manifest_sha256", "error"},
  * no token -> HFWriteError(503) BEFORE anything touches HF (publish then
    fails at the head-sha step, keeping the local draft),
  * race (stale parent_commit) -> {"status": "race"} with no retry,
  * transient failures retry the whole commit; permanent ones don't,
  * the referential-integrity guard refuses a manifest that references record
    files that are neither in the batch nor already on HF,
  * manifest_sha256 is the sha256 of exactly the manifest bytes committed
    (nothing rewrites the manifest anymore — no version stamping).
"""
import hashlib
import json
import time

import pytest

from services import hf_write
from services.hf_write import HFWriteError


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeHfApi:
    """Stands in for huggingface_hub.HfApi. Class-level knobs are reset by the
    `fake_api` fixture; instances record constructor tokens and call args."""

    instances = []
    repo_info_result = _Obj(sha="head-sha-1")
    repo_info_exc = None
    create_commit_results = None      # list; exceptions are raised, others returned
    list_repo_files_result = []
    list_repo_files_exc = None

    def __init__(self, token=None):
        self.token = token
        self.create_commit_calls = []
        self.list_repo_files_calls = 0
        FakeHfApi.instances.append(self)

    def repo_info(self, repo_id, repo_type):
        if FakeHfApi.repo_info_exc:
            raise FakeHfApi.repo_info_exc
        return FakeHfApi.repo_info_result

    def create_commit(self, *, repo_id, repo_type, operations, commit_message,
                      parent_commit=None):
        self.create_commit_calls.append({
            "repo_id": repo_id, "repo_type": repo_type,
            "operations": operations, "commit_message": commit_message,
            "parent_commit": parent_commit,
        })
        result = FakeHfApi.create_commit_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def list_repo_files(self, repo_id, repo_type):
        self.list_repo_files_calls += 1
        if FakeHfApi.list_repo_files_exc:
            raise FakeHfApi.list_repo_files_exc
        return FakeHfApi.list_repo_files_result


@pytest.fixture
def fake_api(monkeypatch):
    FakeHfApi.instances = []
    FakeHfApi.repo_info_result = _Obj(sha="head-sha-1")
    FakeHfApi.repo_info_exc = None
    FakeHfApi.create_commit_results = [_Obj(oid="commit-1")]
    FakeHfApi.list_repo_files_result = []
    FakeHfApi.list_repo_files_exc = None
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(hf_write, "_repo_ref", lambda: ("test/repo", "dataset"))
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(time, "sleep", lambda s: None)   # no real backoff waits
    return FakeHfApi


def _ops(*pairs):
    return [{"path": p, "content": c} for p, c in pairs]


def _last_api():
    return FakeHfApi.instances[-1]


# ---------------------------------------------------------------------------
# is_enabled / token handling
# ---------------------------------------------------------------------------


def test_is_enabled_reflects_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert hf_write.is_enabled() is True
    monkeypatch.setenv("HF_TOKEN", "")
    assert hf_write.is_enabled() is False
    monkeypatch.setenv("HF_TOKEN", "   ")
    assert hf_write.is_enabled() is False
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert hf_write.is_enabled() is False


def test_head_sha_without_token_raises_503(fake_api, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "")
    with pytest.raises(HFWriteError) as ei:
        hf_write.hf_head_sha()
    assert ei.value.status_code == 503
    assert FakeHfApi.instances == []          # nothing touched HF


def test_commit_without_token_raises_503(fake_api, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "")
    with pytest.raises(HFWriteError) as ei:
        hf_write.hf_commit(operations=_ops(("a.json", b"{}")), commit_message="m")
    assert ei.value.status_code == 503
    assert FakeHfApi.instances == []


# ---------------------------------------------------------------------------
# head sha
# ---------------------------------------------------------------------------


def test_head_sha_returns_repo_head(fake_api):
    assert hf_write.hf_head_sha() == "head-sha-1"
    assert _last_api().token == "hf_test_token"


def test_head_sha_hf_failure_maps_to_502(fake_api):
    FakeHfApi.repo_info_exc = RuntimeError("boom")
    with pytest.raises(HFWriteError) as ei:
        hf_write.hf_head_sha()
    assert ei.value.status_code == 502


# ---------------------------------------------------------------------------
# commit — happy path
# ---------------------------------------------------------------------------


def test_commit_success_shape_and_manifest_hash(fake_api):
    manifest = {"ces": {"ce_1": "2026-01-01"}, "rules": {}}
    manifest_bytes = json.dumps(manifest).encode()
    ops = _ops(
        ("public_ces/ce_1.json", b'{"name": "x"}'),
        ("manifest.json", manifest_bytes),
    )

    resp = hf_write.hf_commit(operations=ops, commit_message="Publish CE: x",
                              parent_commit="head-sha-1")

    assert resp["status"] == "success"
    assert resp["commit_sha"] == "commit-1"
    assert resp["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    call = _last_api().create_commit_calls[0]
    assert call["repo_id"] == "test/repo"
    assert call["parent_commit"] == "head-sha-1"
    assert len(call["operations"]) == 2


def test_commit_without_manifest_has_no_hash(fake_api):
    resp = hf_write.hf_commit(operations=_ops(("neutral/x.json", b"[]")),
                              commit_message="m")
    assert resp["status"] == "success"
    assert resp["manifest_sha256"] is None


# ---------------------------------------------------------------------------
# commit — race / retry semantics
# ---------------------------------------------------------------------------


def test_commit_race_is_not_retried(fake_api):
    FakeHfApi.create_commit_results = [
        RuntimeError("412 Client Error: Precondition Failed"),
        _Obj(oid="never"),
    ]
    resp = hf_write.hf_commit(operations=_ops(("a.json", b"{}")), commit_message="m")
    assert resp["status"] == "race"
    assert len(_last_api().create_commit_calls) == 1


def test_commit_transient_error_retries_then_succeeds(fake_api):
    FakeHfApi.create_commit_results = [
        RuntimeError("Connection reset by peer"),
        _Obj(oid="commit-2"),
    ]
    resp = hf_write.hf_commit(operations=_ops(("a.json", b"{}")), commit_message="m")
    assert resp["status"] == "success"
    assert resp["commit_sha"] == "commit-2"
    assert len(_last_api().create_commit_calls) == 2


def test_commit_permanent_error_fails_without_retry(fake_api):
    FakeHfApi.create_commit_results = [
        RuntimeError("401 Unauthorized: bad token"),
        _Obj(oid="never"),
    ]
    resp = hf_write.hf_commit(operations=_ops(("a.json", b"{}")), commit_message="m")
    assert resp["status"] == "error"
    assert "401" in resp["error"]
    assert len(_last_api().create_commit_calls) == 1


def test_commit_transient_errors_exhaust_attempts(fake_api):
    FakeHfApi.create_commit_results = [
        RuntimeError("timeout"), RuntimeError("timeout"), RuntimeError("timeout"),
    ]
    resp = hf_write.hf_commit(operations=_ops(("a.json", b"{}")), commit_message="m")
    assert resp["status"] == "error"
    assert len(_last_api().create_commit_calls) == 3   # HF_COMMIT_ATTEMPTS default


# ---------------------------------------------------------------------------
# commit — referential-integrity guard (#7: no catalog entry without its file)
# ---------------------------------------------------------------------------


def test_integrity_refuses_manifest_with_dangling_record(fake_api):
    manifest = {"rules": {"rule_x": "2026-01-01"}}   # rule_x's file nowhere
    with pytest.raises(HFWriteError) as ei:
        hf_write.hf_commit(
            operations=_ops(("manifest.json", json.dumps(manifest).encode())),
            commit_message="m",
        )
    assert ei.value.status_code == 409
    assert "public_rules/rule_x.json" in str(ei.value)
    assert _last_api().create_commit_calls == []      # nothing was committed


def test_integrity_passes_when_record_in_batch(fake_api):
    manifest = {"rules": {"rule_x": "2026-01-01"}}
    resp = hf_write.hf_commit(
        operations=_ops(
            ("public_rules/rule_x.json", b"{}"),
            ("manifest.json", json.dumps(manifest).encode()),
        ),
        commit_message="m",
    )
    assert resp["status"] == "success"
    assert _last_api().list_repo_files_calls == 0     # no HF probe needed


def test_integrity_passes_when_record_already_on_hf(fake_api):
    manifest = {"ces": {"ce_old": "2025-01-01"}, "rules": {}}
    FakeHfApi.list_repo_files_result = ["public_ces/ce_old.json"]
    resp = hf_write.hf_commit(
        operations=_ops(("manifest.json", json.dumps(manifest).encode())),
        commit_message="m",
    )
    assert resp["status"] == "success"
    assert _last_api().list_repo_files_calls == 1


def test_integrity_unverifiable_registry_refuses_commit(fake_api):
    manifest = {"ces": {"ce_old": "2025-01-01"}}
    FakeHfApi.list_repo_files_exc = RuntimeError("HF down")
    resp = hf_write.hf_commit(
        operations=_ops(("manifest.json", json.dumps(manifest).encode())),
        commit_message="m",
    )
    assert resp["status"] == "error"
    assert "verify registry state" in resp["error"]
    assert _last_api().create_commit_calls == []


def test_invalid_manifest_json_raises_400(fake_api):
    with pytest.raises(HFWriteError) as ei:
        hf_write.hf_commit(
            operations=_ops(("manifest.json", b"{not json")), commit_message="m")
    assert ei.value.status_code == 400
    assert _last_api().create_commit_calls == []


def test_rule_set_records_are_guarded_too(fake_api):
    manifest = {"rule_sets": {"ruleset_z": "2026-01-01"}}
    with pytest.raises(HFWriteError) as ei:
        hf_write.hf_commit(
            operations=_ops(("manifest.json", json.dumps(manifest).encode())),
            commit_message="m",
        )
    assert "public_rule_sets/ruleset_z.json" in str(ei.value)
