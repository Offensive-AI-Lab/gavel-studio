import os
import re
import shutil
import json
import zipfile
from typing import List, Optional
from urllib.parse import urlparse
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, field_validator
from sql_scripts.model_scripts import get_user_models, register_model, delete_model, update_model_layers
from utils.auth import get_current_user
from utils.ownership import require_model_owner
from utils.sqlite_db import execute_query_dict
from utils.text_safety import clean_text
router = APIRouter()

# --- Configuration ---
# This directory will be created in your 'backend' folder
UPLOAD_DIRECTORY = "LLMs"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

HF_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
HF_ALLOWED_HOST = "huggingface.co"
HF_API_TIMEOUT_SECONDS = 10

# Weight formats that can be loaded by AutoModelForCausalLM (transformers).
# GGUF and ONNX are explicitly excluded — they are NOT loadable by the training pipeline.
HF_COMPATIBLE_WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth")

# Required fields in config.json (or nested text_config) for the training pipeline
# to be able to extract attention representations from the model.
HF_REQUIRED_CONFIG_FIELDS = ("num_hidden_layers", "num_attention_heads", "hidden_size")


def verify_hf_model_repository(repo_id: str, token: Optional[str] = None) -> None:
    """
    Validates that a HuggingFace repo is fully compatible with the training pipeline.

    All conditions a model MUST meet:
      1. The repository must be publicly accessible on huggingface.co.
      2. If pipeline_tag is set, it must be "text-generation".
      3. Must contain at least one .safetensors, .bin, .pt, or .pth weight file
         (GGUF and ONNX are NOT supported — they cannot be loaded by transformers).
      4. Must contain a config.json.
      5. config.json must list at least one architecture ending in "ForCausalLM"
         (e.g. LlamaForCausalLM, MistralForCausalLM, GPT2LMHeadModel is excluded).
      6. config.json (or its nested text_config) must have: num_hidden_layers,
         num_attention_heads, and hidden_size — these are used to build the RNN head.
      7. If quantization_config is present it must include a "quant_method" field.
         Models whose quantization_config lacks quant_method cause a fatal error when
         transformers tries to instantiate them (AttributeError at load time).

    Security note: Only two fixed, hardcoded URLs are ever fetched — the HF model
    API endpoint and the raw config.json. No user-supplied URL is ever opened.
    """
    # ------------------------------------------------------------------
    # Step 1 — Fetch repo metadata from the HF API
    # ------------------------------------------------------------------
    _auth_headers = {"Authorization": f"Bearer {token}"} if token else {}
    api_url = f"https://{HF_ALLOWED_HOST}/api/models/{quote(repo_id, safe='/')}"
    req = Request(api_url, headers={"Accept": "application/json", "User-Agent": "gavel-model-validator/1.0", **_auth_headers})

    try:
        with urlopen(req, timeout=HF_API_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise HTTPException(
                status_code=400,
                detail=(
                    "This Hugging Face repository is private or gated. "
                    "Provide a Hugging Face token (with access to this model) when linking it."
                    if not token else
                    "The provided Hugging Face token doesn't have access to this repository."
                ),
            )
        if exc.code == 404:
            raise HTTPException(
                status_code=400,
                detail="Hugging Face model repository was not found. Double-check the link.",
            )
        raise HTTPException(status_code=400, detail="Failed to reach Hugging Face. Please try again.")
    except URLError:
        raise HTTPException(
            status_code=400,
            detail="Could not connect to Hugging Face. Check your internet connection and try again.",
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Unexpected response from Hugging Face while validating the model.")

    # ------------------------------------------------------------------
    # Step 2 — pipeline_tag must be "text-generation" (if set)
    # ------------------------------------------------------------------
    pipeline_tag = payload.get("pipeline_tag")
    if pipeline_tag and pipeline_tag != "text-generation":
        raise HTTPException(
            status_code=400,
            detail=(
                f"This model is tagged as '{pipeline_tag}', not 'text-generation'. "
                "Only causal language models (text generation) are supported by the training pipeline."
            ),
        )

    # ------------------------------------------------------------------
    # Step 3 — Must have compatible weight files (not GGUF/ONNX only)
    # ------------------------------------------------------------------
    siblings = payload.get("siblings") or []
    if not siblings:
        raise HTTPException(status_code=400, detail="Repository appears empty — no files were found.")

    file_names = [item.get("rfilename", "") for item in siblings if isinstance(item, dict)]

    has_compatible_weights = any(
        name.lower().endswith(ext) for name in file_names for ext in HF_COMPATIBLE_WEIGHT_EXTENSIONS
    )

    if not has_compatible_weights:
        has_gguf = any(name.lower().endswith(".gguf") for name in file_names)
        has_onnx = any(name.lower().endswith(".onnx") for name in file_names)
        if has_gguf:
            raise HTTPException(
                status_code=400,
                detail=(
                    "This repository only contains GGUF quantized weights, which cannot be loaded by the "
                    "training pipeline. Please use a standard HuggingFace model with .safetensors or .bin weights."
                ),
            )
        if has_onnx:
            raise HTTPException(
                status_code=400,
                detail=(
                    "This repository only contains ONNX weights, which are not supported. "
                    "Please use a standard HuggingFace model with .safetensors or .bin weights."
                ),
            )
        raise HTTPException(
            status_code=400,
            detail=(
                "No compatible weight files found. "
                "Supported formats: .safetensors, .bin, .pt, .pth"
            ),
        )

    # ------------------------------------------------------------------
    # Step 4 — Must have a config.json
    # ------------------------------------------------------------------
    has_config = any(name.lower() == "config.json" for name in file_names)
    if not has_config:
        raise HTTPException(
            status_code=400,
            detail="Repository does not contain a config.json. This file is required to verify model architecture.",
        )

    # ------------------------------------------------------------------
    # Steps 5, 6, 7 — Fetch config.json and validate its contents
    # ------------------------------------------------------------------
    config_url = f"https://{HF_ALLOWED_HOST}/{quote(repo_id, safe='/')}/resolve/main/config.json"
    config_req = Request(
        config_url,
        headers={"Accept": "application/json", "User-Agent": "gavel-model-validator/1.0", **_auth_headers},
    )
    try:
        with urlopen(config_req, timeout=HF_API_TIMEOUT_SECONDS) as config_resp:
            config = json.loads(config_resp.read().decode("utf-8"))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read config.json from the repository. The model may be gated or inaccessible.",
        )

    # Step 5 — Must declare a ForCausalLM architecture
    architectures = config.get("architectures") or []
    if not architectures:
        raise HTTPException(
            status_code=400,
            detail=(
                "config.json does not declare any model architecture. "
                "Cannot verify compatibility with the training pipeline."
            ),
        )

    has_causal_lm = any("ForCausalLM" in arch for arch in architectures)
    if not has_causal_lm:
        arch_list = ", ".join(architectures)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model architecture '{arch_list}' is not a causal language model. "
                "Only models with a 'ForCausalLM' architecture (e.g. LlamaForCausalLM, MistralForCausalLM) "
                "are supported."
            ),
        )

    # Step 6 — Must have required structural fields for attention extraction
    # Some models (e.g. Gemma) nest these under a "text_config" sub-object.
    effective_config = config.get("text_config") or config
    missing_fields = [f for f in HF_REQUIRED_CONFIG_FIELDS if effective_config.get(f) is None]
    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail=(
                f"config.json is missing required fields: {', '.join(missing_fields)}. "
                "These are needed by the training pipeline to extract attention representations."
            ),
        )

    # Step 7 — quantization_config, if present, must include quant_method
    quant_cfg = config.get("quantization_config")
    if quant_cfg and isinstance(quant_cfg, dict):
        if "quant_method" not in quant_cfg:
            raise HTTPException(
                status_code=400,
                detail=(
                    "This model has an incompatible quantization configuration (the 'quant_method' field is missing). "
                    "When loaded, transformers will crash with an AttributeError. "
                    "Please use a non-quantized model or one with a standard GPTQ/bitsandbytes quantization config."
                ),
            )


def normalize_hf_model_ref(value: str) -> str:
    """
    Accepts either:
    - Hugging Face repo ID: owner/repo
    - Hugging Face URL: https://huggingface.co/owner/repo (or /models/owner/repo)
    Returns normalized repo ID: owner/repo
    """
    cleaned = (value or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Model reference is required")

    if HF_REPO_ID_PATTERN.fullmatch(cleaned):
        return cleaned

    parsed = urlparse(cleaned)
    if parsed.scheme != "https":
        raise HTTPException(status_code=400, detail="Model URL must use https")

    if parsed.username or parsed.password or parsed.port:
        raise HTTPException(status_code=400, detail="Invalid Hugging Face URL format")

    host = (parsed.hostname or "").lower()
    if host != HF_ALLOWED_HOST:
        raise HTTPException(status_code=400, detail="Only https://huggingface.co links are allowed")

    parts = [part for part in (parsed.path or "").split("/") if part]
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Invalid Hugging Face model link")

    owner, repo = parts[0], parts[1]
    if parts[0].lower() == "models":
        if len(parts) < 3:
            raise HTTPException(status_code=400, detail="Invalid Hugging Face model link")
        owner, repo = parts[1], parts[2]

    repo_id = f"{owner}/{repo}"
    if not HF_REPO_ID_PATTERN.fullmatch(repo_id):
        raise HTTPException(status_code=400, detail="Invalid Hugging Face model link format")

    return repo_id

# --- Schemas ---
class ModelCreate(BaseModel):
    user_id: int
    name: str = Field(..., max_length=120)
    storage_path: str = Field(..., max_length=2048)
    # Optional HF token for gated / private models. Stored on the model row
    # and passed to transformers when the model is downloaded.
    hf_token: Optional[str] = Field(default=None, max_length=512)
    # Optional per-model LLM layer info (e.g. for the demo presets, whose layer
    # counts we know up front). num_layers = total transformer layers;
    # selected_layers = the [start, end) range the user picked.
    num_layers: Optional[int] = Field(default=None, ge=1, le=512)
    selected_layers: Optional[List[int]] = None

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, value):
        return clean_text(value, field_name="model name", max_length=120)

    @field_validator("storage_path", mode="before")
    @classmethod
    def _clean_storage_path(cls, value):
        return clean_text(value, field_name="model reference", max_length=2048, allow_newlines=False)

# --- Helpers ---

def _check_model_unique(user_id: int, name: str, storage_path: str):
    """Reject duplicate model names or storage paths for the same user.

    Name comparison is case-insensitive — "GPT2" and "gpt2" should both
    be considered duplicates. Storage path comparison is exact — for HF
    refs that's `org/repo`, for uploads it's the absolute local path."""
    by_name = execute_query_dict(
        "SELECT model_id FROM target_models WHERE user_id = %s AND LOWER(name) = LOWER(%s)",
        (user_id, name),
    )
    if by_name:
        raise HTTPException(
            status_code=409,
            detail=f"You already have a model named '{name}'.",
        )
    by_path = execute_query_dict(
        "SELECT model_id, name FROM target_models WHERE user_id = %s AND storage_path = %s",
        (user_id, storage_path),
    )
    if by_path:
        existing_name = by_path[0]["name"]
        raise HTTPException(
            status_code=409,
            detail=f"This model source is already registered as '{existing_name}'.",
        )


# --- Endpoints ---

@router.get("/{user_id}")
def get_models_endpoint(user_id: int, auth_uid: int = Depends(get_current_user)):
    """Get all models for a user. You can only list your OWN models — the path
    user_id must match the authenticated user (prevents enumerating others')."""
    if user_id != auth_uid:
        raise HTTPException(status_code=404, detail="Not found")
    try:
        models = get_user_models(user_id)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create")
def create_model_endpoint(model: ModelCreate, _: int = Depends(get_current_user)):
    """
    Scenario 2: Register a Hugging Face link.
    Only stores the string path in the DB.
    """
    try:
        name = (model.name or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Model name is required")

        hf_token = (model.hf_token or "").strip() or None
        normalized_repo_id = normalize_hf_model_ref(model.storage_path)
        verify_hf_model_repository(normalized_repo_id, token=hf_token)
        _check_model_unique(model.user_id, name, normalized_repo_id)
        result = register_model(
            model.user_id, name, normalized_repo_id, hf_token=hf_token,
            num_layers=model.num_layers, selected_layers=model.selected_layers,
        )
        return {"success": True, "model": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LayersUpdate(BaseModel):
    selected_layers: List[int]


@router.patch("/{model_id}/layers", dependencies=[Depends(require_model_owner)])
def update_model_layers_endpoint(model_id: int, body: LayersUpdate, uid: int = Depends(get_current_user)):
    """Update which LLM layers a model uses (the [start, end) range). Saved on
    the model so the user doesn't re-pick each time and training reads it."""
    layers = body.selected_layers
    if not (isinstance(layers, list) and len(layers) == 2 and layers[0] >= 0 and layers[1] > layers[0]):
        raise HTTPException(status_code=400, detail="selected_layers must be [start, end) with end > start >= 0.")
    row = update_model_layers(model_id, uid, layers)
    if not row:
        raise HTTPException(status_code=404, detail="Model not found.")
    return {"success": True, "model": row}

# Uploads are ZIP-ONLY: a lone .safetensors/.pth is just weights (no config or
# tokenizer), so it can't load as a runnable model. Require a zipped model
# directory instead.
ALLOWED_UPLOAD_EXTENSIONS = (".zip",)

# Magic bytes for detecting potentially malicious files disguised as model weights.
# pickle-based files (.pth) start with \x80 (pickle protocol marker).
# safetensors files start with a JSON header length (little-endian u64),
# so the first 8 bytes are a small integer — never an ELF/PE/script header.
# Truly-executable signatures — never legitimate in a model file.
_EXECUTABLE_SIGNATURES = [
    b"\x7fELF",        # Linux ELF executable
    b"MZ",             # Windows PE executable
    b"#!",             # Shell script shebang
]
# Full set adds archive markers. NOTE: torch.save writes .bin/.pth as a ZIP
# container, so PK is EXPECTED for those — see _TORCH_CONTAINER_EXTENSIONS.
_DANGEROUS_SIGNATURES = _EXECUTABLE_SIGNATURES + [
    b"PK\x03\x04",    # ZIP archive (could hide code)
    b"\x1f\x8b",       # gzip (could wrap anything)
]
# Weight files that are LEGITIMATELY ZIP containers (torch serialization).
_TORCH_CONTAINER_EXTENSIONS = {".bin", ".pth"}


def _validate_upload_file(file: UploadFile) -> str:
    """Validate uploaded model file for type and safety.

    Returns the sanitized filename. Raises HTTPException on failure.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    safe_filename = os.path.basename(file.filename)

    # Block path traversal attempts
    if ".." in safe_filename or "/" in safe_filename or "\\" in safe_filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Uploads are ZIP-ONLY.
    _, ext = os.path.splitext(safe_filename.lower())
    if ext != ".zip":
        raise HTTPException(
            status_code=400,
            detail=f"Only .zip is accepted (you uploaded '{ext or 'a file with no extension'}'). "
                   "Upload a zipped model directory containing config.json, the weights "
                   "(.safetensors/.pth/.bin) and the tokenizer files.",
        )

    # Confirm the bytes really are a ZIP (magic number), not just the extension.
    header = file.file.read(16)
    file.file.seek(0)  # Reset for later reading
    if len(header) < 8:
        raise HTTPException(status_code=400, detail="File is too small to be a valid model")
    if header[:4] != b"PK\x03\x04":
        raise HTTPException(status_code=400, detail="File does not appear to be a valid ZIP archive")

    return safe_filename


# Big models (e.g. a 22 GB Mistral) need a generous cap. Configurable via env so
# a deployment can tighten/loosen it without a code change.
_MAX_ZIP_GB = float(os.getenv("MAX_MODEL_ZIP_GB", "64"))
_MAX_ZIP_UNCOMPRESSED = int(_MAX_ZIP_GB * 1024 * 1024 * 1024)
_MAX_ZIP_FILES = 500  # sharded large models can have many weight shards
_WEIGHT_EXTENSIONS = {".safetensors", ".pth", ".bin"}
_ALLOWED_ZIP_EXTENSIONS = {
    ".json", ".safetensors", ".bin", ".pth",
    ".txt", ".model", ".vocab", ".tiktoken",
}
_REQUIRED_CONFIG_KEYS = {"model_type"}


def _extract_and_validate_zip(zip_source, extract_dir: str) -> str:
    """Extract a model zip and validate it is a clean model directory.

    `zip_source` is a path OR a seekable file-like object (the uploaded
    stream). Extracting straight from the upload stream avoids writing the
    whole (multi-GB) zip to disk a second time before extraction.

    Checks: zip bomb, path traversal, symlinks, file count, extension
    allowlist, executable content, config.json validity, required files.
    Returns the path to the directory containing config.json + weights.
    """
    # is_zipfile + ZipFile both consume the stream, so rewind before each read.
    def _rewind():
        try:
            zip_source.seek(0)
        except (AttributeError, OSError):
            pass

    _rewind()
    if not zipfile.is_zipfile(zip_source):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive")

    _rewind()
    with zipfile.ZipFile(zip_source, "r") as zf:
        members = zf.infolist()

        # HF model repos commonly ship the weights in BOTH formats:
        # *.safetensors AND pytorch_model*.bin. They're the same weights, and
        # transformers loads safetensors when present, so the .bin set is dead
        # weight (often doubling a 14 GB → 28 GB upload). When safetensors are
        # present, auto-skip the torch .bin shards + their index so the user can
        # just zip the whole download and we extract only what's needed.
        _has_safetensors = any(
            os.path.basename(m.filename).lower().endswith(".safetensors")
            and not m.filename.endswith("/")
            and not any(p.startswith(".") for p in m.filename.split("/"))
            for m in members
        )

        def _is_redundant_torch_bin(basename_lower: str) -> bool:
            return _has_safetensors and (
                basename_lower.endswith(".bin")
                or basename_lower == "pytorch_model.bin.index.json"
            )

        safe_members = []
        for m in members:
            # --- Path traversal ---
            if m.filename.startswith("/") or ".." in m.filename:
                raise HTTPException(status_code=400, detail=f"ZIP contains unsafe path: {m.filename}")

            # --- Symlinks ---
            if m.external_attr >> 28 == 0xA:
                raise HTTPException(status_code=400, detail=f"ZIP contains a symlink ({m.filename}), which is not allowed")

            # Skip directories
            if m.filename.endswith("/"):
                continue

            # --- Hidden files/dirs: SKIP, don't reject ---
            # HF model repos are git clones, so they ship benign dotfiles like
            # .gitattributes / .gitignore (and sometimes a whole .git/ dir).
            # These are harmless metadata — ignore them instead of failing the
            # whole upload. Anything inside a hidden directory is skipped too.
            if any(part.startswith(".") for part in m.filename.split("/")):
                continue

            # --- Extension allowlist: SKIP non-model files, don't reject ---
            # HF model repos ship docs/metadata (README.md, LICENSE, *.png …)
            # alongside the weights. We only EXTRACT recognized model files and
            # silently skip the rest — a file that's never written to disk can't
            # do harm, and the executable-signature scan below still guards the
            # model files we do extract. This avoids rejecting a perfectly good
            # model over a README.
            basename = os.path.basename(m.filename)
            _, ext = os.path.splitext(basename.lower())
            if ext not in _ALLOWED_ZIP_EXTENSIONS:
                continue  # skip non-model file
            if _is_redundant_torch_bin(basename.lower()):
                continue  # safetensors present → skip the duplicate torch .bin
            safe_members.append(m)

        # --- Zip bomb / DOS protection (only what we actually extract) ---
        if len(safe_members) > _MAX_ZIP_FILES:
            raise HTTPException(status_code=400, detail=f"ZIP contains too many files ({len(safe_members)}). Max {_MAX_ZIP_FILES}.")
        total_size = sum(m.file_size for m in safe_members)
        if total_size > _MAX_ZIP_UNCOMPRESSED:
            raise HTTPException(status_code=400, detail=f"ZIP too large (>{_MAX_ZIP_GB:g} GB uncompressed)")

        # Extract only the validated, non-hidden members (dotfiles are skipped).
        zf.extractall(extract_dir, members=safe_members)

    # Resolve model directory (files at root or inside a single subfolder)
    entries = [e for e in os.listdir(extract_dir) if not e.startswith(".")]
    model_dir = extract_dir
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
        model_dir = os.path.join(extract_dir, entries[0])

    files = set(os.listdir(model_dir))

    # --- Scan extracted files for executable content ---
    # NB: PyTorch weight files (.bin/.pth) saved by torch.save are LEGITIMATELY
    # ZIP-format containers, so they start with the PK signature. For those we
    # only reject true executables (ELF/PE/shebang) — flagging the ZIP/gzip
    # signature would falsely reject every modern .bin checkpoint. Other files
    # get the full archive check.
    for fname in files:
        fpath = os.path.join(model_dir, fname)
        if os.path.isdir(fpath):
            continue
        with open(fpath, "rb") as f:
            header = f.read(16)
        _, fext = os.path.splitext(fname.lower())
        sigs = _EXECUTABLE_SIGNATURES if fext in _TORCH_CONTAINER_EXTENSIONS else _DANGEROUS_SIGNATURES
        for sig in sigs:
            if header[:len(sig)] == sig:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{fname}' inside ZIP appears to be an executable, not a model file.",
                )

    # --- Required: config.json ---
    if "config.json" not in files:
        raise HTTPException(
            status_code=400,
            detail="ZIP must contain config.json (model architecture definition). "
                   "Zip the full model directory, not just the weights file.",
        )

    # --- Validate config.json structure ---
    config_path = os.path.join(model_dir, "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="config.json is not valid JSON")
    missing_keys = _REQUIRED_CONFIG_KEYS - set(config.keys())
    if missing_keys:
        raise HTTPException(
            status_code=400,
            detail=f"config.json is missing required fields: {', '.join(sorted(missing_keys))}. "
                   "This doesn't look like a valid model config.",
        )

    # --- Required: weight file(s) ---
    has_weights = any(f for f in files if os.path.splitext(f.lower())[1] in _WEIGHT_EXTENSIONS)
    if not has_weights:
        raise HTTPException(
            status_code=400,
            detail="ZIP must contain model weights (.safetensors, .pth, or .bin)",
        )

    # --- Required: tokenizer ---
    has_tokenizer = any(f for f in files if "tokenizer" in f.lower())
    if not has_tokenizer:
        raise HTTPException(
            status_code=400,
            detail="ZIP must contain tokenizer files (tokenizer.json and/or tokenizer_config.json). "
                   "Without a tokenizer, the model cannot process text.",
        )

    return model_dir


@router.post("/upload")
async def upload_model_file(
    user_id: int = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    _: int = Depends(get_current_user),
):
    """
    Upload a model as a .zip of the full model directory (must include
    config.json + weights + tokenizer files).

    Extracts to LLMs/{user_id}/{slug}/ and registers the directory path so
    from_pretrained() works directly. The zip is extracted STRAIGHT from the
    upload stream — we never write the whole (multi-GB) zip to disk first.
    """
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Model name is required")

    _validate_upload_file(file)  # zip-only; raises on anything else

    # Pre-compute the storage path so we can reject a duplicate name/path
    # BEFORE writing any bytes — a rejected upload leaves no orphan files.
    user_dir = os.path.join(UPLOAD_DIRECTORY, str(user_id))
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:80]
    prospective_path = os.path.join(user_dir, slug)
    _check_model_unique(user_id, name, prospective_path)

    model_dir = None
    try:
        os.makedirs(user_dir, exist_ok=True)
        model_dir = prospective_path
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        # Extract directly from the uploaded stream (no intermediate zip copy).
        resolved_dir = _extract_and_validate_zip(file.file, model_dir)
        result = register_model(user_id, name, resolved_dir)
        return {"success": True, "model": result, "local_path": resolved_dir}
    except HTTPException:
        if model_dir and os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        raise
    except Exception:
        if model_dir and os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    

@router.delete("/{model_id}", dependencies=[Depends(require_model_owner)])
def remove_model_endpoint(model_id: int, _: int = Depends(get_current_user)):
    """
    Removes a model:
    1. Deletes physical file if it exists in the UPLOAD_DIRECTORY.
    2. Deletes the database record (cascades to guardrails/rules).
    3. Sweeps each cascaded guardrail's on-disk workdir so deleted
       models don't leak gigabytes of weights / datasets to disk until
       the next boot-time orphan recovery.
    """
    try:
        model_data = execute_query_dict("SELECT storage_path FROM target_models WHERE model_id = %s", (model_id,))

        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")

        file_path = model_data[0]['storage_path']

        # Capture (classifier_id, user_id) for every guardrail under
        # this model BEFORE the cascade fires — once the rows are gone
        # we can't resolve their workdirs from the DB anymore.
        classifier_rows = execute_query_dict(
            """
            SELECT c.classifier_id, m.user_id
            FROM classifiers c
            JOIN target_models m ON c.model_id = m.model_id
            WHERE c.model_id = %s
            """,
            (model_id,),
        ) or []

        # Stop any in-flight work across child guardrails BEFORE the DB cascade
        # wipes their pointers (training_log / '*_running' rows). Covers remote
        # TRAINING, remote CALIBRATION/EVALUATION, and warm REALTIME sessions.
        # Local in-process tasks self-abort once delete_model removes their rows
        # (trainer.TrainingCancelled / inference_core.InferenceCancelled).
        # Best-effort — failures don't block the delete.
        if classifier_rows:
            from routes.classifiers import (
                _cancel_remote_training_for_classifier,
                _cancel_remote_inference_for_classifier,
                _end_realtime_session_for_classifier,
            )
            for row in classifier_rows:
                cid = row["classifier_id"]
                _cancel_remote_training_for_classifier(cid)
                _cancel_remote_inference_for_classifier(cid)
                _end_realtime_session_for_classifier(cid)

        # 2. If the path points to our local UPLOAD_DIRECTORY, delete it
        if file_path.startswith(UPLOAD_DIRECTORY) and os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)

        # 3. Delete from Database (Cascades automatically)
        delete_model(model_id)

        # 4. Sweep the now-orphaned guardrail workdirs. Best-effort —
        # boot-time orphan recovery is the safety net.
        if classifier_rows:
            import logging
            from classifier_engine.trainer import delete_classifier_workdir
            for row in classifier_rows:
                try:
                    delete_classifier_workdir(row["classifier_id"], row["user_id"])
                except Exception as cleanup_err:
                    logging.getLogger(__name__).warning(
                        f"classifier {row['classifier_id']} workdir cleanup failed: {cleanup_err}"
                    )

        return {"success": True, "message": f"Model {model_id} and associated files removed."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))