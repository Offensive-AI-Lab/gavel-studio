# classifier_engine/inference_core.py
#
# The PURE inference core: target-LLM attention readouts -> sliding-window RNN
# logits. Shared by BOTH the local path (evaluation/inference.py) and the
# REMOTE path (compute_jobs/infer_job.py) so the per-window logits are byte-for-byte
# identical no matter where the compute ran — calibration/evaluation must not
# drift between a remote run and a local run.
#
# Hard rule: NO database / FastAPI / network imports here. It depends only on
# torch, numpy, and DB-free classifier_engine modules, so it imports cleanly on
# a remote worker (which has no DB config) exactly like train_job.py / eval_job.py.
from __future__ import annotations
import logging
from typing import List, Optional

import numpy as np
import torch

from classifier_engine.dialogue_dataset import DialogueDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention readout extraction
# ---------------------------------------------------------------------------

def _extract_value_cache(past_kvs, layer_idx: int) -> torch.Tensor:
    """Extract value cache from past_key_values, handling both transformers 4.x and 5.x."""
    # transformers 4.x: tuple of tuples
    if isinstance(past_kvs, tuple):
        return past_kvs[layer_idx][1]
    # transformers 5.x: DynamicCache with .layers API
    if hasattr(past_kvs, "layers"):
        return past_kvs.layers[layer_idx].values
    # transformers 5.x fallback: value_cache attribute
    if hasattr(past_kvs, "value_cache"):
        return past_kvs.value_cache[layer_idx]
    raise TypeError(f"Unsupported past_key_values type: {type(past_kvs)}")


@torch.no_grad()
def extract_attention_readouts(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    selected_layers: range,
) -> torch.Tensor:
    """Run LLM forward pass and extract attention-weighted value readouts.

    Args:
        model: Causal LM.
        input_ids: (B, S) token IDs.
        attention_mask: (B, S) mask.
        selected_layers: range of layer indices to extract.

    Returns:
        Tensor of shape (B, S, num_layers, readout_dim).
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        use_cache=True,
    )

    # Detect head geometry
    cfg = model.config
    if hasattr(cfg, "text_config"):
        # Gemma-style
        n_v_heads = cfg.text_config.num_key_value_heads
        head_dim = cfg.text_config.head_dim
        n_q_heads = cfg.text_config.num_attention_heads
    else:
        n_q_heads = cfg.num_attention_heads
        n_v_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_q_heads)

    group_size = n_q_heads // n_v_heads
    B, S = input_ids.shape

    per_layer = []
    for layer_idx in selected_layers:
        # Attention: (B, Hq, S, S)
        A = outputs.attentions[layer_idx]
        # Values: (B, Hv, S, D)
        V = _extract_value_cache(outputs.past_key_values, layer_idx)

        # Group query heads: (B, Hv, group_size, S, S) -> mean -> (B, Hv, S, S)
        if group_size > 1:
            A_grouped = A.view(B, n_v_heads, group_size, S, S).mean(dim=2)
        else:
            A_grouped = A

        # Readout: r_{b,i,h,d} = sum_j A_{b,h,i,j} * V_{b,h,j,d}
        readout = torch.einsum("bhij,bhjd->bihd", A_grouped, V)
        # Flatten heads: (B, S, readout_dim)
        readout = readout.reshape(B, S, n_v_heads * head_dim)
        per_layer.append(readout)

    # Stack layers: (B, S, num_layers, readout_dim)
    return torch.stack(per_layer, dim=2)


# ---------------------------------------------------------------------------
# Sliding window RNN inference
# ---------------------------------------------------------------------------

def _contiguous_true_runs(mask_1d):
    """1D BoolTensor -> list of (start, end) index pairs for True-runs (end-exclusive).

    Verbatim from the reference preprocessing/utils.py so eval/calibration
    windows match the reference exactly (one run == one contiguous assistant turn)."""
    runs = []
    L = mask_1d.numel()
    i = 0
    while i < L:
        if mask_1d[i]:
            s = i
            i += 1
            while i < L and mask_1d[i]:
                i += 1
            runs.append((s, i))
        else:
            i += 1
    return runs


def _windows(start: int, end: int, W: int, S: int):
    """[start, end) token indices -> list of (s, e) windows, step S. The trailing
    window is KEPT at its natural (short) length — never zero-padded, never dropped.
    Verbatim from the reference (_windows)."""
    out = []
    i = start
    while i < end:
        j = min(i + W, end)
        out.append((i, j))
        i += S
    return out


def build_assistant_windows(assistant_mask: torch.Tensor, window_size: int, window_stride: int):
    """Windows built PER contiguous assistant run, so a single window never spans
    two different assistant turns — matching the reference build_assistant_windows."""
    runs = _contiguous_true_runs(assistant_mask)
    windows = []
    for s_abs, e_abs in runs:
        windows.extend(_windows(s_abs, e_abs, window_size, window_stride))
    return windows


@torch.no_grad()
def rnn_logits_over_windows(
    rnn_model,
    token_vecs: torch.Tensor,        # (T, F) FULL-sequence token features
    windows,                         # list of (start, end) absolute-coordinate windows
    device: torch.device = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Run the RNN over a list of (start, end) windows into token_vecs.

    Ported verbatim from the reference preprocessing/utils.py: each
    window is scored at its NATURAL length (windows are grouped by length and run
    WITHOUT zero-padding), so the short trailing window of every assistant run is
    handled exactly like the reference. Returns (num_windows, num_topics)."""
    if device is None:
        device = next(rnn_model.parameters()).device
    num_w = len(windows)
    out = []
    for i in range(0, num_w, batch_size):
        batch_windows = windows[i:i + batch_size]
        # Group by window length to avoid any cross-length zero-padding.
        length_groups = {}
        for j, (s, e) in enumerate(batch_windows):
            length_groups.setdefault(e - s, []).append((j, s, e))
        batch_logits = [None] * len(batch_windows)
        for length, group in length_groups.items():
            if not group:
                continue
            F = token_vecs.size(1)
            x = torch.zeros((len(group), length, F), dtype=token_vecs.dtype, device=device)
            for k, (orig_idx, s, e) in enumerate(group):
                x[k] = token_vecs[s:e].to(device)
            out_logits = rnn_model(x)
            if isinstance(out_logits, tuple):
                out_logits = out_logits[0]
            for k, (orig_idx, s, e) in enumerate(group):
                batch_logits[orig_idx] = out_logits[k].detach().float().cpu().numpy()
        out.append(np.stack(batch_logits))
    return np.concatenate(out, axis=0) if out else np.array([])


# ---------------------------------------------------------------------------
# Conversation normalization
# ---------------------------------------------------------------------------

def _normalize_conversation(conv):
    """Coerce any conversation shape into List[{"role": str, "content": str}].

    Tolerates plain strings, lists of strings, lists of message dicts, and
    anything else (degenerate but valid), so one malformed message can't crash
    the whole batch's inference (the chat template raises on non-dict messages).
    """
    if isinstance(conv, str):
        return [{"role": "assistant", "content": conv}]
    if not isinstance(conv, list):
        return [{"role": "user", "content": ""}]

    normalized = []
    for i, msg in enumerate(conv):
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("from") or ("assistant" if i % 2 == 1 else "user")
            content = msg.get("content") or msg.get("value") or msg.get("text") or ""
            if not isinstance(content, str):
                content = str(content)
            normalized.append({"role": str(role), "content": content})
        elif isinstance(msg, str):
            role = "user" if i % 2 == 0 else "assistant"
            normalized.append({"role": role, "content": msg})
        else:
            normalized.append({"role": "user" if i % 2 == 0 else "assistant", "content": str(msg)})
    return normalized


# ---------------------------------------------------------------------------
# The shared inference loop
# ---------------------------------------------------------------------------

# Re-exported for callers that import it from here (e.g. evaluation/inference.py).
# Defined in a torch-free module so route modules can `except InferenceCancelled`
# without pulling torch into app startup.
from classifier_engine.cancellation import InferenceCancelled  # noqa: E402,F401


def run_inference_core(
    rnn_model,
    meta: dict,
    llm,
    tokenizer,
    dialogues: List[dict],
    device: torch.device,
    max_length: Optional[int] = None,   # None = keep the WHOLE dialogue (reference-parity). The per-dialogue OOM fallback below truncates only the specific long dialogues that don't fit the GPU (retrying 1024 -> 512), so None is safe even on a 24GB card.
    window_stride: int = 0,   # 0/None => non-overlapping (stride = window_size)
    progress_callback=None,
    cancel_check=None,   # optional callable() that raises (e.g. InferenceCancelled) to abort mid-loop
) -> List[dict]:
    """Run windowed inference for already-loaded models — the shared body of the
    local and remote inference paths.

    Args:
        rnn_model: trained TopicRNN (already on `device`).
        meta: classifier_meta.json dict — needs `rnn_sequence_length` and
            `selected_layers` ([start, end_exclusive]).
        llm, tokenizer: loaded target model + tokenizer.
        dialogues: [{conversation|text, metadata}], same shape as the local
            run_inference_on_dialogues input.
        device: torch device.

    Returns:
        [{"logits": np.ndarray (num_windows, num_topics), "metadata": {...}}],
        one entry per dialogue that produced assistant tokens (others dropped),
        order preserved.
    """
    window_size = meta["rnn_sequence_length"]
    # Non-overlapping windows (stride == window_size) match BOTH how the RNN was
    # trained AND the reference calibration/eval. A stride < 1 (the
    # default) means "auto" -> non-overlapping. Overlapping windows (stride 1)
    # with patience=1 ("a CE fires if ANY window crosses its threshold") inflate
    # per-CE firing on long dialogues, dragging calibrated thresholds far too low
    # and making eval over-fire (e.g. scamazon FPR ~0.7). An explicit stride >= 1
    # is still honored (the realtime per-token path uses its own windowing).
    if window_stride is None or window_stride < 1:
        window_stride = window_size
    selected_layers = range(meta["selected_layers"][0], meta["selected_layers"][1])

    def _progress(stage, detail=""):
        if progress_callback:
            try:
                progress_callback(stage, detail)
            except Exception:
                pass
        logger.info(f"[InferenceCore] [{stage}] {detail}")

    results: List[dict] = []
    total = len(dialogues)
    for i, dialogue in enumerate(dialogues):
        conv = dialogue.get("conversation", dialogue.get("text", ""))
        metadata = dialogue.get("metadata", {})
        conv = _normalize_conversation(conv)

        if i % 10 == 0:
            # Cooperative cancellation: if the guardrail was deleted while this
            # calibration/eval run is in flight, stop at this checkpoint instead
            # of finishing inference on a guardrail that's gone.
            if cancel_check is not None:
                cancel_check()
            _progress("inference", f"Dialogue {i + 1}/{total}")

        def _run_one(eff_max_length):
            """Tokenize -> LLM attention-value readout -> per-run windows -> RNN
            for one dialogue at a given truncation length. Returns the window
            logits, or None if the dialogue yields no assistant tokens/windows."""
            dataset = DialogueDataset([conv], tokenizer, max_length=eff_max_length)
            item = dataset[0]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=device)
            assistant_mask = item["assistant_mask"]  # (S,) boolean

            readouts = extract_attention_readouts(llm, input_ids, attention_mask, selected_layers)

            asst_mask_1d = assistant_mask[: input_ids.shape[1]].bool()
            if int(asst_mask_1d.sum().item()) == 0:
                return None

            # token_vecs is the FULL sequence; windows are built PER assistant run
            # with ABSOLUTE token coords, so a window never spans two assistant
            # turns and each run's short tail is kept un-padded — exactly the
            # reference build_assistant_windows + _windows behavior.
            S_len = readouts.shape[1]
            token_vecs = readouts[0, :S_len].reshape(S_len, -1).float()
            windows = build_assistant_windows(asst_mask_1d, window_size, window_stride)
            if not windows:
                return None
            return rnn_logits_over_windows(rnn_model, token_vecs, windows, device=device)

        # max_length=None keeps the WHOLE dialogue (reference-parity). But on a
        # memory-limited GPU, output_attentions over a very long dialogue can
        # CUDA-OOM (the per-layer S×S attention tensors). On OOM, empty the cache
        # and retry THIS dialogue truncated — otherwise one OOM wedges the CUDA
        # context and silently drops every dialogue after it (=> a partial eval).
        attempts = [max_length] + [c for c in (1024, 512) if c < (max_length or 1 << 30)]
        logits = None
        for eff in attempts:
            try:
                logits = _run_one(eff)
                break
            except RuntimeError as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if "out of memory" in str(e).lower() and eff is not attempts[-1]:
                    logger.warning(f"Dialogue {i}: CUDA OOM at max_length={eff}; retrying truncated")
                    continue
                logger.error(f"Dialogue {i} failed: {e}")
                break
            except Exception as e:
                logger.error(f"Dialogue {i} failed: {e}")
                break
        if logits is not None:
            results.append({"logits": logits, "metadata": metadata})

    _progress("done", f"Inference complete: {len(results)}/{total} dialogues processed")
    return results
