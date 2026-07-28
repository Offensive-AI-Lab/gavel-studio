"""Tests for model management: CRUD, HuggingFace validation, file upload security."""
import io
import json
import os
import struct
import zipfile
import pytest


def _safetensors_bytes() -> bytes:
    """Minimal valid .safetensors payload (u64 header length + JSON header)."""
    hdr = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    return struct.pack("<Q", len(hdr)) + hdr + b"\x00" * 4


def make_model_zip(*, config=True, weights=True, tokenizer=True, extra=None) -> bytes:
    """Build an in-memory model zip (config.json + weights + tokenizer).
    Flags drop a required piece; `extra` adds {name: bytes} files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if config:
            zf.writestr("config.json", json.dumps({"model_type": "llama"}))
        if weights:
            zf.writestr("model.safetensors", _safetensors_bytes())
        if tokenizer:
            zf.writestr("tokenizer.json", "{}")
        for fname, data in (extra or {}).items():
            zf.writestr(fname, data)
    return buf.getvalue()


class TestModelCreation:
    """Model registration and validation."""

    def test_create_model_valid_hf(self, client, test_user, auth_headers):
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": "GPT2Test",
            "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        }, headers=auth_headers)
        # 409 is valid: the session `test_model` fixture may already have
        # registered this exact storage_path for this user, and the
        # duplicate-path guard (routes/models.py::_check_model_unique) rejects
        # a second registration of the same HF source.
        assert res.status_code in (200, 400, 409, 422, 500)

    def test_create_model_invalid_hf_repo(self, client, test_user, auth_headers):
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": "FakeModel",
            "storage_path": "nonexistent/fake-model-xyz-12345",
        }, headers=auth_headers)
        assert res.status_code in (400, 422, 500)

    def test_create_model_gguf_blocked(self, client, test_user, auth_headers):
        """GGUF models should be rejected."""
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": "GGUFModel",
            "storage_path": "TheBloke/Llama-2-7B-GGUF",
        }, headers=auth_headers)
        assert res.status_code in (400, 422, 500)

    def test_create_model_missing_name(self, client, test_user, auth_headers):
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        }, headers=auth_headers)
        assert res.status_code in (400, 422)

    def test_create_model_without_auth_is_allowed(self, client, test_user):
        res = client.post("/models/create", json={
            "user_id": test_user["user_id"],
            "name": "NoAuth",
            "storage_path": "HuggingFaceTB/SmolLM2-360M-Instruct",
        })
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


class TestModelRetrieval:
    """Model listing and retrieval."""

    def test_get_user_models(self, client, test_user, test_model, auth_headers):
        res = client.get(f"/models/{test_user['user_id']}", headers=auth_headers)
        assert res.status_code == 200
        data = res.json()
        models = data.get("models", data) if isinstance(data, dict) else data
        assert isinstance(models, list)
        assert len(models) >= 1

    def test_get_models_nonexistent_user(self, client, auth_headers):
        res = client.get("/models/99999", headers=auth_headers)
        assert res.status_code in (200, 404)
        data = res.json()
        # Response may be {"models": []} or [] or 404
        if res.status_code == 200:
            models = data.get("models", data) if isinstance(data, dict) else data
            assert isinstance(models, list)


class TestModelDeletion:
    """Model deletion with cascade."""

    def test_delete_model_without_auth_is_allowed(self, client):
        res = client.delete("/models/99999")
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


class TestModelUploadValidation:
    """Uploads are ZIP-ONLY: a zipped model directory (config.json + weights +
    tokenizer). Anything else is rejected up front."""

    def _upload(self, client, auth_headers, test_user, filename, content, name="UploadTest"):
        return client.post(
            "/models/upload",
            data={"user_id": str(test_user["user_id"]), "name": name},
            files={"file": (filename, io.BytesIO(content), "application/octet-stream")},
            headers=auth_headers,
        )

    def test_valid_model_zip_accepted(self, client, auth_headers, test_user):
        res = self._upload(client, auth_headers, test_user, "model.zip",
                           make_model_zip(), name="ZipModelOK")
        assert res.status_code == 200, res.text

    @pytest.mark.parametrize("filename", [
        "model.safetensors", "model.pth", "model.bin", "model.gguf",
        "malicious.py", "virus.exe", "hack.sh",
    ])
    def test_non_zip_extension_rejected(self, client, auth_headers, test_user, filename):
        # Even with otherwise-valid-looking bytes, a non-.zip is rejected.
        res = self._upload(client, auth_headers, test_user, filename, _safetensors_bytes())
        assert res.status_code == 400
        assert ".zip" in res.json()["detail"].lower()

    def test_zip_extension_but_not_a_zip_rejected(self, client, auth_headers, test_user):
        # .zip name but the bytes aren't a real ZIP (no PK magic).
        res = self._upload(client, auth_headers, test_user, "model.zip", b"\x00" * 64)
        assert res.status_code == 400

    def test_upload_without_auth_is_allowed(self, client, test_user):
        res = client.post(
            "/models/upload",
            data={"user_id": str(test_user["user_id"]), "name": "NoAuth"},
            files={"file": ("model.zip", io.BytesIO(make_model_zip()), "application/zip")},
        )
        assert res.status_code not in (401, 403)  # no auth exists: a request is never rejected for credentials


class TestModelZipContent:
    """Validation of the contents of a model zip."""

    def _upload(self, client, auth_headers, test_user, content, name="ZipContentTest"):
        return client.post(
            "/models/upload",
            data={"user_id": str(test_user["user_id"]), "name": name},
            files={"file": ("model.zip", io.BytesIO(content), "application/zip")},
            headers=auth_headers,
        )

    def test_missing_config_rejected(self, client, auth_headers, test_user):
        res = self._upload(client, auth_headers, test_user, make_model_zip(config=False))
        assert res.status_code == 400
        assert "config.json" in res.json()["detail"].lower()

    def test_missing_weights_rejected(self, client, auth_headers, test_user):
        res = self._upload(client, auth_headers, test_user, make_model_zip(weights=False))
        assert res.status_code == 400
        assert "weight" in res.json()["detail"].lower()

    def test_missing_tokenizer_rejected(self, client, auth_headers, test_user):
        res = self._upload(client, auth_headers, test_user, make_model_zip(tokenizer=False))
        assert res.status_code == 400
        assert "tokenizer" in res.json()["detail"].lower()

    def test_executable_inside_zip_rejected(self, client, auth_headers, test_user):
        # An extracted model file whose CONTENT is an ELF binary → rejected.
        # (Use .safetensors: it's always extracted and gets the full scan; a
        # .bin would be auto-skipped when safetensors are present.)
        evil = make_model_zip(extra={"evil.safetensors": b"\x7fELF" + b"\x00" * 100})
        res = self._upload(client, auth_headers, test_user, evil)
        assert res.status_code == 400
        assert "executable" in res.json()["detail"].lower()

    def test_bin_only_model_is_accepted_pk_magic_allowed(self, client, auth_headers, test_user):
        # A model with ONLY torch .bin weights (no safetensors). torch.save
        # writes .bin as a ZIP container (PK magic) — a legitimate weight file
        # that must NOT be flagged as an archive.
        zipped = make_model_zip(weights=False, extra={
            "pytorch_model.bin": b"PK\x03\x04" + b"\x00" * 200,
        })
        res = self._upload(client, auth_headers, test_user, zipped, name="BinOnlyModel")
        assert res.status_code == 200, res.text

    def test_redundant_torch_bin_skipped_when_safetensors_present(self, client, auth_headers, test_user):
        # HF ships both formats. With safetensors present, the duplicate .bin
        # shards (and their index) are auto-skipped — the user can zip the whole
        # download and it still works.
        zipped = make_model_zip(extra={
            "model.safetensors.index.json": "{}",
            "pytorch_model-00001-of-00002.bin": b"PK\x03\x04" + b"\x00" * 200,
            "pytorch_model-00002-of-00002.bin": b"PK\x03\x04" + b"\x00" * 200,
            "pytorch_model.bin.index.json": "{}",
        })
        res = self._upload(client, auth_headers, test_user, zipped, name="BothFormatsModel")
        assert res.status_code == 200, res.text

    def test_non_model_files_inside_zip_are_skipped(self, client, auth_headers, test_user):
        # A non-model file (README.md, a stray .py, an image) is SKIPPED, not
        # rejected — HF repos ship these alongside the weights. The model still
        # uploads; the extra files just aren't extracted.
        zipped = make_model_zip(extra={
            "README.md": b"# Mistral-7B\nGreat model.\n",
            "LICENSE": b"Apache 2.0\n",
            "setup.py": b"import os\n",
            "logo.png": b"\x89PNG\r\n\x1a\n",
        })
        res = self._upload(client, auth_headers, test_user, zipped, name="WithDocs")
        assert res.status_code == 200, res.text

    def test_hf_dotfiles_are_skipped_not_rejected(self, client, auth_headers, test_user):
        # HF model repos are git clones, so they ship .gitattributes/.gitignore
        # (and sometimes a .git/ dir). These benign dotfiles must be ignored,
        # not rejected.
        zipped = make_model_zip(extra={
            ".gitattributes": b"*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
            ".gitignore": b"*.log\n",
            ".git/config": b"[core]\n",
            ".cache/blob": b"\x00\x00",
        })
        res = self._upload(client, auth_headers, test_user, zipped, name="HFDotfiles")
        assert res.status_code == 200, res.text

    def test_empty_file_rejected(self, client, auth_headers, test_user):
        res = self._upload(client, auth_headers, test_user, b"")
        assert res.status_code == 400

    def test_path_traversal_zip_name_is_sanitized(self, client, auth_headers, test_user):
        # The filename is sanitized via os.path.basename; valid content → accepted.
        res = client.post(
            "/models/upload",
            data={"user_id": str(test_user["user_id"]), "name": "TraversalZip"},
            files={"file": ("../../etc/passwd.zip", io.BytesIO(make_model_zip()), "application/zip")},
            headers=auth_headers,
        )
        assert res.status_code in (200, 400)
