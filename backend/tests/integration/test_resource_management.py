"""Resource management tests — temp files, memory, cleanup, locks.

Important for a clone-and-run repo where users have limited local resources.
All test artifacts are cleaned up in teardown.
"""
import gc
import io
import os
import struct
import tempfile
import pytest


class TestTempFileCleanup:
    """Verify uploaded files don't leak into the LLM uploads directory on errors."""

    def _malicious_payload(self):
        """ELF binary header — should be rejected."""
        return b"\x7fELF" + b"\x00" * 100

    def test_rejected_upload_does_not_leave_file(self, client, auth_headers, test_user):
        """When upload validation rejects a file, no remnant should remain on disk."""
        from routes.models import UPLOAD_DIRECTORY

        user_dir = os.path.join(UPLOAD_DIRECTORY, str(test_user["user_id"]))
        files_before = set(os.listdir(user_dir)) if os.path.isdir(user_dir) else set()

        # Upload a malicious file (ELF disguised as .pth)
        res = client.post(
            "/models/upload",
            data={"user_id": str(test_user["user_id"]), "name": "ShouldFail"},
            files={"file": ("evil.pth", io.BytesIO(self._malicious_payload()), "application/octet-stream")},
            headers=auth_headers,
        )
        assert res.status_code == 400

        # Verify no new files were created
        files_after = set(os.listdir(user_dir)) if os.path.isdir(user_dir) else set()
        new_files = files_after - files_before
        assert not new_files, f"Rejected upload left file(s) on disk: {new_files}"

    def test_invalid_extension_does_not_create_file(self, client, auth_headers, test_user):
        """A .py upload should be rejected before any file write happens."""
        from routes.models import UPLOAD_DIRECTORY

        user_dir = os.path.join(UPLOAD_DIRECTORY, str(test_user["user_id"]))
        files_before = set(os.listdir(user_dir)) if os.path.isdir(user_dir) else set()

        res = client.post(
            "/models/upload",
            data={"user_id": str(test_user["user_id"]), "name": "Py"},
            files={"file": ("script.py", io.BytesIO(b"import os"), "text/x-python")},
            headers=auth_headers,
        )
        assert res.status_code == 400

        files_after = set(os.listdir(user_dir)) if os.path.isdir(user_dir) else set()
        assert files_after == files_before


class TestUploadCleanupOnSuccess:
    """If upload succeeds, the extracted model directory should exist at the
    registered location. Cleanup must remove both the files and the DB row."""

    def _model_zip(self):
        import io as _io
        import json as _json
        import struct as _struct
        import zipfile as _zipfile
        buf = _io.BytesIO()
        with _zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("config.json", _json.dumps({"model_type": "llama"}))
            hdr = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
            zf.writestr("model.safetensors", _struct.pack("<Q", len(hdr)) + hdr + b"\x00" * 4)
            zf.writestr("tokenizer.json", "{}")
        return buf.getvalue()

    def test_successful_upload_creates_file_and_cleanup_works(self, client, auth_headers, test_user):
        import shutil
        from utils.sqlite_db import execute_query

        model_name = f"ResourceTestModel_{int.from_bytes(os.urandom(2), 'big')}"
        res = client.post(
            "/models/upload",
            data={"user_id": str(test_user["user_id"]), "name": model_name},
            files={"file": ("model.zip", io.BytesIO(self._model_zip()), "application/zip")},
            headers=auth_headers,
        )
        assert res.status_code == 200, res.text

        # The extracted model directory is registered as the storage path.
        local_path = res.json().get("local_path")
        assert local_path and os.path.isdir(local_path)
        assert os.path.exists(os.path.join(local_path, "config.json"))

        # Cleanup: delete the extracted dir and the DB record.
        try:
            shutil.rmtree(os.path.dirname(local_path) if os.path.basename(local_path) != model_name
                          else local_path, ignore_errors=True)
        except OSError:
            pass
        model_data = res.json().get("model", {})
        if model_data.get("model_id"):
            execute_query("DELETE FROM target_models WHERE model_id = %s", (model_data["model_id"],))


class TestDatabaseConnectionPool:
    """The connection pool should not leak connections."""

    def test_many_queries_dont_exhaust_pool(self):
        """Run 100 queries — pool should release connections back."""
        from utils.sqlite_db import execute_query_dict
        for _ in range(100):
            res = execute_query_dict("SELECT 1 AS v")
            assert res[0]["v"] == 1


class TestMemoryStability:
    """Light-touch memory test — make sure repeated requests don't grow forever.
    Skip if `psutil` is not installed."""

    def test_repeated_search_no_obvious_leak(self, client, auth_headers, test_user):
        psutil = pytest.importorskip("psutil")
        process = psutil.Process(os.getpid())
        gc.collect()
        mem_before = process.memory_info().rss

        for _ in range(30):
            res = client.get("/library/search", params={"q": "test", "user_id": test_user["user_id"]}, headers=auth_headers)
            assert res.status_code == 200

        gc.collect()
        mem_after = process.memory_info().rss
        # Allow up to 200MB growth — anything more suggests a leak
        growth_mb = (mem_after - mem_before) / (1024 * 1024)
        assert growth_mb < 200, f"Memory grew by {growth_mb:.1f} MB across 30 search requests"


class TestConcurrentRequests:
    """Verify the system handles concurrent requests without DB deadlock."""

    def test_concurrent_dashboard_reads(self, client, test_user, auth_headers):
        """10 concurrent dashboard reads — should all succeed."""
        import concurrent.futures
        uid = test_user["user_id"]

        def fetch():
            return client.get(f"/dashboard/{uid}", headers=auth_headers).status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(lambda _: fetch(), range(10)))
        assert all(r == 200 for r in results)


class TestLargePayloadHandling:
    """The system should reject overly large payloads gracefully."""

    def test_huge_ce_definition_handled(self, client, auth_headers, test_user):
        """A 1MB definition string — system should either accept or reject cleanly."""
        big_text = "x" * (1024 * 1024)
        res = client.post("/cognitive/create", json={
            "user_id": test_user["user_id"],
            "name": "huge_def_ce_test",
            "definition": big_text,
        }, headers=auth_headers)
        # Should reject (too large) or accept (truncate via validator)
        assert res.status_code in (200, 400, 413, 422)
        # Cleanup if accepted
        if res.status_code == 200:
            ce_id = res.json().get("ce_id")
            if ce_id:
                from utils.sqlite_db import execute_query
                execute_query("DELETE FROM cognitive_elements WHERE ce_id = %s", (ce_id,))

    def test_empty_payload_to_post_endpoint(self, client, auth_headers):
        """POST with empty JSON body should return 422, not crash."""
        res = client.post("/cognitive/create", json={}, headers=auth_headers)
        assert res.status_code == 422
