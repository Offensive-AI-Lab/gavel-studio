# classifier_engine/realtime_core.py
# The PURE realtime CE-classification core: given a loaded LLM + tokenizer + RNN
# + classifier meta, turn a conversation into per-window and per-token CE logits
# for every assistant turn (stored mode) OR generate a reply and classify it
# (live mode) — NO DB, NO FastAPI.
#
# It lives under classifier_engine/ (the package shipped to remote workers) so
# the SAME code runs in two places, byte-for-byte:
#   * the backend's local realtime path (evaluation/realtime.py re-exports these),
#   * the warm remote realtime job (compute_jobs/realtime_job.py), which keeps the
#     model loaded and serves session requests — this is what lets realtime work
#     on any PC (the heavy LLM forward runs on the remote GPU, not the laptop).
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def extract_value_cache(past_kvs, layer_idx):
    """Extract the value cache for one layer from past_key_values.

    Handles both the legacy tuple format (transformers 4.x) and the DynamicCache
    object (transformers 5.x), plus the various internal layouts those have used.
    """
    DynamicCache = None
    try:
        from transformers.cache_utils import DynamicCache as _DC
        DynamicCache = _DC
    except ImportError:
        pass

    # DynamicCache object (transformers 5.x)
    if DynamicCache is not None and isinstance(past_kvs, DynamicCache):
        if hasattr(past_kvs, 'layers'):
            layer_cache = past_kvs.layers[layer_idx]
            if hasattr(layer_cache, 'value_cache'):
                return layer_cache.value_cache
            elif hasattr(layer_cache, 'values'):
                return layer_cache.values
            elif isinstance(layer_cache, (list, tuple)) and len(layer_cache) >= 2:
                return layer_cache[1]
            else:
                layer_attrs = dir(layer_cache)
                for attr in ['value_cache', 'values', 'value', 'v_cache']:
                    if attr in layer_attrs:
                        return getattr(layer_cache, attr)
                raise ValueError(f"Unexpected layer_cache structure: {type(layer_cache)}, attributes: {layer_attrs}")
        elif hasattr(past_kvs, 'value_cache'):
            return past_kvs.value_cache[layer_idx]
        elif hasattr(past_kvs, 'to_legacy_cache') and callable(past_kvs.to_legacy_cache):
            legacy_cache = past_kvs.to_legacy_cache()
            return legacy_cache[layer_idx][1]
        else:
            raise ValueError(f"DynamicCache object has unexpected structure: {dir(past_kvs)}")
    elif hasattr(past_kvs, 'value_cache'):
        return past_kvs.value_cache[layer_idx]
    elif type(past_kvs).__name__ == 'DynamicCache':
        if hasattr(past_kvs, 'layers'):
            layer_cache = past_kvs.layers[layer_idx]
            if hasattr(layer_cache, 'value_cache'):
                return layer_cache.value_cache
            elif hasattr(layer_cache, 'values'):
                return layer_cache.values
            elif isinstance(layer_cache, (list, tuple)) and len(layer_cache) >= 2:
                return layer_cache[1]
            else:
                raise ValueError(f"Unexpected layer_cache structure: {type(layer_cache)}")
        elif hasattr(past_kvs, 'value_cache'):
            return past_kvs.value_cache[layer_idx]
        else:
            raise ValueError(f"DynamicCache type detected but no value_cache or layers attribute found")
    else:
        try:
            return past_kvs[layer_idx][1]
        except (TypeError, IndexError) as e:
            cache_type = type(past_kvs).__name__
            if layer_idx < len(past_kvs):
                elem_type = type(past_kvs[layer_idx]).__name__
                raise TypeError(f"Cannot extract value cache: past_kvs is {cache_type}, past_kvs[{layer_idx}] is {elem_type}. Error: {e}")
            else:
                raise IndexError(f"layer_idx {layer_idx} out of range for cache of length {len(past_kvs)}")


def extract_assistant_reps(model, tokenizer, prompt_messages, full_messages, meta):
    """Run the LLM forward over the FULL conversation and return the attention-
    readout representations for the ASSISTANT span only, plus the assistant token
    ids. `prompt_messages` is everything up to (but not including) the assistant
    turn; `full_messages` is the same with that turn appended (locates the span)."""
    device = next(model.parameters()).device
    selected_layers = range(meta["selected_layers"][0], meta["selected_layers"][1])

    complete_chat = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False,
    )
    complete_encoded = tokenizer(complete_chat, return_tensors="pt").to(device)
    complete_input_ids = complete_encoded["input_ids"]
    complete_attention_mask = complete_encoded["attention_mask"]

    prompt_only_chat = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_only_encoded = tokenizer(prompt_only_chat, return_tensors="pt").to(device)
    assistant_start_idx = prompt_only_encoded["input_ids"].shape[1]

    B, S = complete_input_ids.size()
    outputs = model(
        input_ids=complete_input_ids,
        attention_mask=complete_attention_mask,
        output_attentions=True,
        use_cache=True,
    )
    attns = outputs.attentions
    past_kvs = outputs.past_key_values

    layer_outputs = []
    for layer_idx in selected_layers:
        A = attns[layer_idx]
        V = extract_value_cache(past_kvs, layer_idx).to(A.device)
        Hq, Hv = A.size(1), V.size(1)
        if Hq != Hv:
            group_size = Hq // Hv
            A = A.view(B, Hv, group_size, S, S).mean(dim=2)
        r = torch.einsum("bhij,bhjd->bihd", A, V)
        layer_outputs.append(r.flatten(start_dim=2))

    stacked_rep = torch.stack(layer_outputs, dim=0)              # (n_layers, B, S, readout_dim)
    assistant_reps = stacked_rep[:, 0, assistant_start_idx:, :]  # (n_layers, T_asst, readout_dim)
    assistant_tokens = complete_input_ids[0, assistant_start_idx:]
    return assistant_reps, assistant_tokens


def classify_spans(classifier, assistant_reps, assistant_tokens, tokenizer, window_size):
    """Classify the assistant span by NON-OVERLAPPING windows and return
    (windows, tokens) — EXACTLY the reference scheme.

    It computes ONE logit vector per window (stride = window_size) and assigns
    that SAME vector to every token in the window. So:
      * windows — the per-window logits (feed the calibrated compute_triggers).
      * tokens  — the window logits BROADCAST to each token in the window, so the
                  colored text + the activation graph are a per-window step
                  function (NOT a separate stride-1 per-token pass). Each token's
                  text is `decode([id]).strip()` like the reference render_colored_tokens.
    """
    T = int(assistant_reps.shape[1])
    if T == 0:
        return [], []

    ids = [int(x) for x in assistant_tokens.tolist()]
    # Per-token display text WITH natural spacing: the incremental decode diff
    # gives each token the EXACT text it contributes, so a word-initial token keeps
    # its leading space and a sub-word continuation has none. The viewer renders
    # the leading space OUTSIDE the colour, so the reply reads as flowing text
    # ("provocative", not "provoc ative") while still colouring per token. Done by
    # diffing prefixes (clean_up off → strictly additive → consistent diffs).
    token_texts: List[str] = []
    prev = ""
    for i in range(T):
        cur = tokenizer.decode(ids[: i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        token_texts.append(cur[len(prev):] if cur.startswith(prev) else cur)
        prev = cur

    windows: List[Dict] = []
    tokens: List[Dict] = []
    for window_index, start_idx in enumerate(range(0, T, window_size)):
        end_idx = min(start_idx + window_size, T)
        reps = torch.stack(
            [assistant_reps[:, start_idx + i, :] for i in range(end_idx - start_idx)], dim=0,
        )                                                       # (len, n_layers, readout_dim)
        x = reps.flatten(1).unsqueeze(0).float()
        logits = classifier(x).cpu().detach().numpy().squeeze(0)
        logits_list = logits.tolist()
        windows.append({
            "window_index": window_index,
            "text": tokenizer.decode(ids[start_idx:end_idx], skip_special_tokens=True),
            "logits": logits_list,
            "token_count": end_idx - start_idx,
        })
        # Broadcast this window's logits to each of its tokens (the reference step function).
        for i in range(start_idx, end_idx):
            tokens.append({"token_index": i, "token": token_texts[i], "logits": logits_list})

    return windows, tokens


@torch.no_grad()
def classify_conversation_turns(
    messages: List[Dict[str, str]],
    model,
    tokenizer,
    classifier,
    meta: dict,
) -> Dict[int, Tuple[List[Dict], List[Dict]]]:
    """Classify every USER→ASSISTANT pair of a conversation (no generation).
    Returns a dict mapping the assistant turn's index in `messages` to its
    (windows, tokens).

    Mirrors the reference static-mode handling: only assistant turns that
    directly follow a user turn are scored, and the prefix is a CLEANED
    alternating history — consecutive same-role messages are collapsed and any
    LEADING assistant turns are dropped — so the chat template is valid even when
    the stored dialogue starts with an assistant greeting (otherwise Mistral's
    template rejects the assistant-first prefix and every turn is skipped)."""
    window_size = meta.get("rnn_sequence_length", 5)
    norm = [{"role": (m.get("role") or "").lower(), "content": m.get("content") or ""} for m in messages]
    out: Dict[int, Tuple[List[Dict], List[Dict]]] = {}
    for i in range(len(norm) - 1):
        cur, nxt = norm[i], norm[i + 1]
        if cur["role"] != "user" or nxt["role"] != "assistant":
            continue
        if not nxt["content"].strip():
            continue
        # Clean alternating history (collapse consecutive same-role runs, then drop
        # leading assistant turns) so apply_chat_template gets a valid user-first
        # conversation — exactly the reference static-mode fix.
        history: List[Dict[str, str]] = []
        for h in norm[:i]:
            if not history or history[-1]["role"] != h["role"]:
                history.append(h)
        while history and history[0]["role"] == "assistant":
            history = history[1:]
        prompt_messages = history + [cur]
        full_messages = prompt_messages + [nxt]
        try:
            assistant_reps, assistant_tokens = extract_assistant_reps(
                model, tokenizer, prompt_messages, full_messages, meta,
            )
        except Exception as e:  # one bad turn shouldn't kill the whole dialogue
            logger.warning(f"Failed to classify assistant turn {i + 1}: {e}")
            continue
        if assistant_reps.shape[1] == 0:
            continue
        # Key by the assistant turn's index in the ORIGINAL messages so the viewer
        # attaches the analysis to the right bubble.
        out[i + 1] = classify_spans(
            classifier, assistant_reps, assistant_tokens, tokenizer, window_size,
        )
    return out


@torch.no_grad()
def generate_and_classify(
    user_input: str,
    system_prompt: str,
    model,
    tokenizer,
    classifier,
    meta: dict,
    max_new_tokens: int = 128,
    history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Dict], List[Dict]]:
    """Generate the assistant reply, then classify it per-window and per-token.
    Returns (generated_text, windows, tokens)."""
    device = next(model.parameters()).device
    window_size = meta.get("rnn_sequence_length", 5)

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    chat_for_generation = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    encoded = tokenizer(chat_for_generation, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = encoded["input_ids"].shape[1]
    generated_text = tokenizer.decode(
        generated_ids[0, prompt_len:], skip_special_tokens=True,
    )

    full_messages = messages + [{"role": "assistant", "content": generated_text}]
    assistant_reps, assistant_tokens = extract_assistant_reps(
        model, tokenizer, messages, full_messages, meta,
    )
    if assistant_reps.shape[1] == 0:
        return generated_text, [], []
    windows, tokens = classify_spans(
        classifier, assistant_reps, assistant_tokens, tokenizer, window_size,
    )
    return generated_text, windows, tokens
