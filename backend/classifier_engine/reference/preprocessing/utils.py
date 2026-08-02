import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gavel.training.utils import LLAMA_CLEAN_TEMPLATE

logger = logging.getLogger(__name__)


def _kmp_build(pattern):
    # prefix function for KMP
    lps = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j and pattern[i] != pattern[j]:
            j = lps[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            lps[i] = j
    return lps


def _kmp_find_all(sequence, pattern):
    """Yield start indices where 'pattern' occurs in 'sequence' (both lists of ints)."""
    if not pattern or len(pattern) > len(sequence):
        return
    lps = _kmp_build(pattern)
    j = 0
    for i, x in enumerate(sequence):
        while j and x != pattern[j]:
            j = lps[j - 1]
        if x == pattern[j]:
            j += 1
            if j == len(pattern):
                yield i - j + 1
                j = lps[j - 1]


class DialogueDataset(Dataset):
    """PyTorch Dataset for multi-turn dialogue processing and inference.

    Processes chat-formatted conversations with role-aware masking, supporting
    multiple assistant responses within a single dialogue. Creates a boolean
    mask identifying all assistant tokens for logits extraction.

    Supports multiple model families:
        - Llama-3 style (<|start_header_id|>, <|end_header_id|>, <|eot_id|>)
        - Mistral style ([INST]...[/INST] blocks)
        - Gemma style (<start_of_turn>, <end_of_turn>)
        - Qwen style (<|im_start|>, <|im_end|>, with <think> tag stripping)

    Note:
        For training with single assistant responses, use CognitiveElementDataset
        from gavel.training instead (more efficient, returns start index not mask).
    """

    def __init__(
        self,
        conversations,
        tokenizer,
        max_length=None,
        llama_clean_template=True,
        strip_think_toks=True,
    ):
        self.data = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.strip_think = strip_think_toks

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

        # --- Resolve Llama header tokens (single-token ids) ---
        def tokid(s):
            tid = self.tokenizer.convert_tokens_to_ids(s)
            return None if tid in (None, -1) else tid

        # --- Llama single-token turn markers ---
        self.S_HDR = tokid("<|start_header_id|>")
        self.E_HDR = tokid("<|end_header_id|>")
        self.EOT_ID = tokid("<|eot_id|>")

        # --- Mistral multi-token markers ---
        def enc(s):
            return self.tokenizer.encode(s, add_special_tokens=False)

        self.M_BOS = self.tokenizer.bos_token_id
        self.M_EOS = self.tokenizer.eos_token_id
        self.M_INST = enc("[INST]")
        self.M_ENDI = enc("[/INST]")

        # --- Gemma single-token turn markers ---
        self.G_BOS = tokid("<bos>")
        self.G_SOT = tokid("<start_of_turn>")
        self.G_EOT = tokid("<end_of_turn>")

        # --- Qwen single-token turns ---
        self.Q_SOT = tokid("<|im_start|>")
        self.Q_EOT = tokid("<|im_end|>")
        # Optional: single-id think tags (if present in vocab)
        self.Q_THINK_S = tokid("<think>")
        self.Q_THINK_E = tokid("</think>")

        # newline sequence as plain tokens for splitting "role\ncontent"
        self.NL_SEQ = enc("\n")  # could be one token or multi-token depending on vocab

        # families
        name = (getattr(self.tokenizer, "name_or_path", "") or "").lower()
        self.is_mistral_like = any(x in name for x in ("mistral", "mixtral"))
        self.is_gemma_like = "gemma" in name
        self.is_llama_like = "llama" in name
        self.is_qwen_like = "qwen" in name  # handles qwen / qwen2

        self.prepend_user_role = (
            self.is_mistral_like or self.is_gemma_like
        )  # qwen doesn't need this

        if llama_clean_template and self.is_llama_like:
            self.tokenizer.chat_template = LLAMA_CLEAN_TEMPLATE

        # Pre-encode role tokens for fast comparison (avoid decode in hot path)
        self._role_tokens = {
            "assistant": enc("assistant"),
            "user": enc("user"),
            "system": enc("system"),
            # Gemma uses "model" for assistant
            "model": enc("model"),
        }

        # Single dispatch: store reference to appropriate segmentation method
        if self.is_llama_like and self.S_HDR is not None:
            self._segment_fn = self._segment_by_headers_llama
        elif self.is_mistral_like:
            self._segment_fn = self._segment_by_mistral_blocks
        elif self.is_gemma_like and self.G_SOT is not None:
            self._segment_fn = self._segment_by_gemma_turns
        elif self.is_qwen_like and self.Q_SOT is not None:
            self._segment_fn = self._segment_by_qwen_turns
        else:
            self._segment_fn = None  # will use fallback

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _norm_role(r):
        r = (r or "user").lower()
        return r if r in {"user", "assistant", "system"} else "user"

    def _match_role(self, token_ids: list) -> str:
        """Match a token sequence to a role using pre-encoded role tokens.

        Compares token_ids against pre-encoded role sequences. Returns the
        matching role or 'user' as default. This avoids expensive decode calls.
        """
        for role, role_toks in self._role_tokens.items():
            if len(token_ids) >= len(role_toks):
                if token_ids[: len(role_toks)] == role_toks:
                    # "model" is Gemma's alias for assistant
                    return "assistant" if role == "model" else role
        return "user"

    def _encode_full(self, messages):
        result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        ids = result.input_ids if hasattr(result, "input_ids") else list(result)
        return torch.tensor(ids, dtype=torch.long)

    def _segment_by_headers_llama(self, ids):
        spans = []
        i, n = 0, ids.numel()
        id_list = ids.tolist()
        while i < n:
            if id_list[i] == self.S_HDR:
                j = i + 1
                while j < n and id_list[j] != self.E_HDR:
                    j += 1
                if j >= n:
                    break
                # Use token comparison instead of decode
                role_tokens = id_list[i + 1 : j]
                role = self._match_role(role_tokens)
                content_start = j + 1
                k = content_start
                while k < n and id_list[k] != self.EOT_ID:
                    k += 1
                content_end = k
                if content_end > content_start:
                    spans.append({"role": role, "start": content_start, "end": content_end})
                i = k + 1 if (k < n and id_list[k] == self.EOT_ID) else k
            else:
                i += 1
        return spans

    def _segment_by_mistral_blocks(self, ids):
        """
        Find blocks of: <s> [INST] ... [/INST] assistant ... </s>
        Everything between [INST] and [/INST] is 'user' (includes system prologue if present).
        Everything after [/INST] until </s> (or next <s>) is 'assistant'.
        Returns list of spans {role,start,end} in token index space.
        """
        x = ids.tolist()
        L = len(x)
        spans = []

        # positions of BOS to help bound assistant spans cleanly
        bos_pos = [i for i, t in enumerate(x) if t == self.M_BOS]
        bos_pos.append(L)  # sentinel

        # iterate over each BOS-delimited segment
        for b in range(len(bos_pos) - 1):
            s_bos = bos_pos[b]
            e_bos = bos_pos[b + 1]

            seg = x[s_bos:e_bos]
            # locate [INST] and [/INST] inside this segment (could be exactly one pair)
            inst_starts = list(_kmp_find_all(seg, self.M_INST))
            if not inst_starts:
                continue
            # usually one, but if multiple appear, pair each with the nearest following [/INST]
            endi_starts = list(_kmp_find_all(seg, self.M_ENDI))

            ei_idx = 0
            for ist in inst_starts:
                # find the first [/INST] after this [INST]
                while ei_idx < len(endi_starts) and endi_starts[ei_idx] <= ist:
                    ei_idx += 1
                if ei_idx >= len(endi_starts):
                    break
                i_inst = s_bos + ist
                i_endi = s_bos + endi_starts[ei_idx]
                ei_idx += 1

                # user content span: after [INST] tokens to start of [/INST]
                user_start = i_inst + len(self.M_INST)
                user_end = i_endi
                if user_end > user_start:
                    spans.append({"role": "user", "start": user_start, "end": user_end})

                # assistant span: after [/INST] tokens to EOS (or next BOS)
                asst_start = i_endi + len(self.M_ENDI)
                # bound by next EOS or next BOS inside this outer segment
                # prefer EOS if present; else cut at seg end (next BOS)
                # Search for EOS in [asst_start, e_bos)
                k = asst_start
                while k < e_bos and x[k] != self.M_EOS:
                    k += 1
                asst_end = k  # exclusive; if EOS found, k points to EOS; else k == e_bos
                if asst_end > asst_start:
                    spans.append({"role": "assistant", "start": asst_start, "end": asst_end})

        return spans

    def _segment_by_gemma_turns(self, ids):
        """
        Gemma chat template:
          <bos><start_of_turn>{role}\n{content}<end_of_turn>\n...
        We:
          - locate each <start_of_turn> ... <end_of_turn> block
          - inside, split on the FIRST occurrence of '\n' (tokenized as NL_SEQ)
          - left side is role string; right side is content
        Returns list of spans {role,start,end} in token index space (content only).
        """
        if self.G_SOT is None or self.G_EOT is None:
            return []

        x = ids.tolist()
        # L = len(x)
        spans = []

        # find all <start_of_turn> and <end_of_turn> indices
        sot_positions = [i for i, t in enumerate(x) if t == self.G_SOT]
        eot_positions = [i for i, t in enumerate(x) if t == self.G_EOT]
        if not sot_positions or not eot_positions:
            return spans

        # walk pairs in order (assume well-formed template)
        ei = 0
        for si in sot_positions:
            # find the first end_of_turn after this start
            while ei < len(eot_positions) and eot_positions[ei] < si:
                ei += 1
            if ei >= len(eot_positions):
                break
            ej = eot_positions[ei]
            ei += 1
            if ej <= si + 1:  # nothing between markers
                continue

            # inside block: [si+1 : ej) == "{role}\n{content}"
            inner_start = si + 1
            inner_end = ej
            inner_tokens = x[inner_start:inner_end]

            # find FIRST occurrence of NL_SEQ
            nl_idx = None
            if self.NL_SEQ:
                rel_matches = list(_kmp_find_all(inner_tokens, self.NL_SEQ))
                if rel_matches:
                    nl_idx = rel_matches[0]

            # set content bounds based on newline position
            if nl_idx is not None:
                role_tok_start = inner_start
                role_tok_end = inner_start + nl_idx  # exclusive
                content_start = role_tok_end + len(self.NL_SEQ)
            else:
                # no newline—treat entire block as content with unknown role
                role_tok_start = role_tok_end = inner_start
                content_start = inner_start

            # Use token comparison instead of decode
            role_tokens = x[role_tok_start:role_tok_end]
            role = self._match_role(role_tokens)

            content_end = inner_end  # up to <end_of_turn>
            if content_end > content_start:
                spans.append({"role": role, "start": content_start, "end": content_end})

        return spans

    def _segment_by_qwen_turns(self, ids):
        """
        Qwen chat template:
          <|im_start|>{role}\n{content}<|im_end|>\n
        Roles are literal tokens ('system', 'user', 'assistant').
        Returns content spans only: list of {role, start, end} in token index space.
        """
        if self.Q_SOT is None or self.Q_EOT is None:
            return []

        x = ids.tolist()
        # L = len(x)
        spans = []

        # collect all <|im_start|> and <|im_end|> indices
        s_idx = [i for i, t in enumerate(x) if t == self.Q_SOT]
        e_idx = [i for i, t in enumerate(x) if t == self.Q_EOT]
        if not s_idx or not e_idx:
            return spans

        ei = 0
        for si in s_idx:
            while ei < len(e_idx) and e_idx[ei] < si:
                ei += 1
            if ei >= len(e_idx):
                break
            ej = e_idx[ei]
            ei += 1
            if ej <= si + 1:
                continue

            # inside block: [si+1 : ej) == "{role}\n{content}"
            inner_start = si + 1
            inner_end = ej
            inner_tokens = x[inner_start:inner_end]

            # find FIRST newline token sequence
            nl_rel = None
            rel_matches = list(_kmp_find_all(inner_tokens, self.NL_SEQ)) if self.NL_SEQ else []
            if rel_matches:
                nl_rel = rel_matches[0]

            # role + content split
            if nl_rel is not None:
                role_tok_start = inner_start
                role_tok_end = inner_start + nl_rel
                content_start = role_tok_end + len(self.NL_SEQ)
            else:
                # if no newline, treat whole block as content with unknown role
                role_tok_start = role_tok_end = inner_start
                content_start = inner_start

            # Use token comparison instead of decode
            role_tokens = x[role_tok_start:role_tok_end]
            role = self._match_role(role_tokens)

            content_end = inner_end

            # optional: trim out <think>...</think> from assistant content
            if (
                self.strip_think
                and role == "assistant"
                and self.Q_THINK_S is not None
                and self.Q_THINK_E is not None
            ):
                # scan for think blocks inside [content_start, content_end)
                i = content_start
                while i < content_end:
                    if x[i] == self.Q_THINK_S:
                        j = i + 1
                        while j < content_end and x[j] != self.Q_THINK_E:
                            j += 1
                        # yield any content before the think block
                        if i > content_start:
                            spans.append({"role": role, "start": content_start, "end": i})
                        # skip the think block (including closing tag if present)
                        i = j + 1 if (j < content_end and x[j] == self.Q_THINK_E) else j
                        content_start = i
                    else:
                        i += 1
                if content_end > content_start:
                    spans.append({"role": role, "start": content_start, "end": content_end})
            else:
                if content_end > content_start:
                    spans.append({"role": role, "start": content_start, "end": content_end})

        return spans

    def _tokenize_with_assistant_mask(self, messages):
        # Patch for Mistral alternation if needed
        if self.prepend_user_role and messages[0]["role"] == "assistant":
            messages = [{"role": "user", "content": ""}] + messages

        input_ids = self._encode_full(messages)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        spans = []

        # Single dispatch: use pre-determined segmentation method
        if self._segment_fn is not None:
            spans = self._segment_fn(input_ids)
            for sp in spans:
                if sp["role"] == "assistant":
                    assistant_mask[sp["start"] : sp["end"]] = True

        # Fallback: use message roles directly (avoids O(N²) re-tokenization)
        if not spans:
            prev = torch.empty(0, dtype=torch.long)
            tmp_spans, out, am = [], [], []
            for i in range(1, len(messages) + 1):
                # In transformers 5.x+, apply_chat_template returns a BatchEncoding.
                result = self.tokenizer.apply_chat_template(
                    messages[:i],
                    tokenize=True,
                    return_dict=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                ids = result.input_ids if hasattr(result, "input_ids") else list(result)
                cur = torch.tensor(ids, dtype=torch.long)
                delta = cur[len(prev) :]
                start = sum(x.numel() for x in out)
                end = start + delta.numel()
                role = self._norm_role(messages[i - 1].get("role"))
                out.append(delta)
                am.append(torch.full((delta.numel(),), role == "assistant", dtype=torch.bool))
                tmp_spans.append({"role": role, "start": start, "end": end})
                prev = cur
            input_ids = torch.cat(out) if out else torch.empty(0, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            assistant_mask = torch.cat(am) if am else torch.empty(0, dtype=torch.bool)
            spans = tmp_spans

        # Optional left-truncation
        if self.max_length is not None and input_ids.numel() > self.max_length:
            cut = input_ids.numel() - self.max_length
            input_ids = input_ids[-self.max_length :]
            attention_mask = attention_mask[-self.max_length :]
            assistant_mask = assistant_mask[-self.max_length :]
            new_spans = []
            for sp in spans:
                s = min(max(0, sp["start"] - cut), self.max_length)
                e = min(max(0, sp["end"] - cut), self.max_length)
                if e > s:
                    new_spans.append({"role": sp["role"], "start": s, "end": e})
            spans = new_spans

        return input_ids, attention_mask, assistant_mask, spans

    def __getitem__(self, idx):
        messages = self.data[idx]
        input_ids, attention_mask, assistant_mask, spans = self._tokenize_with_assistant_mask(
            messages
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_mask,
            "pad_token_id": self.tokenizer.pad_token_id,
        }


def collate_fn(batch):
    pad_id = batch[0]["pad_token_id"]
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    assistant_mask = [b["assistant_mask"] for b in batch]
    B = len(batch)
    L = max(x.size(0) for x in input_ids)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    amask = torch.zeros((B, L), dtype=torch.bool)
    for i, (ii, aa, am) in enumerate(zip(input_ids, attention_mask, assistant_mask)):
        n = ii.size(0)
        ids[i, :n] = ii
        attn[i, :n] = aa
        amask[i, :n] = am
    return {"input_ids": ids, "attention_mask": attn, "assistant_mask": amask}


def _normalize_to_conversations(data):
    """
    Returns: list of conversations.
    A conversation is: list[ {role: 'user'|'assistant'|'system', content: str} ]
    Handles:
      - top-level list of messages
      - top-level list of dicts with only content (assume 'user')
      - top-level dict with 'conversation' or 'converstation'
      - legacy 'Statement' fallback
    Skips empty/invalid items.
    """

    def norm_role(x):
        r = (x or "user").lower()
        if r not in {"user", "assistant", "system"}:
            return "user"
        return r

    def to_msg(obj):
        # prefer 'content', fallback to 'Statement'
        if not isinstance(obj, dict):
            return None
        text = obj.get("content", obj.get("Statement"))
        if not text or not isinstance(text, str) or not text.strip():
            return None
        role = norm_role(obj.get("role"))
        return {"role": role, "content": text}

    conversations = []
    if isinstance(data, list):
        # If items look like chat messages with roles → treat as a single conversation
        if all(isinstance(x, dict) and ("content" in x or "Statement" in x) for x in data):
            # If at least one has a role, assume entire list is one conversation
            has_any_role = any("role" in x for x in data if isinstance(x, dict))
            if has_any_role:
                conv = [m for m in (to_msg(x) for x in data) if m]
                if conv:
                    conversations.append(conv)
                return conversations
            # Else: treat each item as an independent single-turn user message
            for x in data:
                msg = to_msg(x)
                if msg:
                    conversations.append([msg])
            return conversations

    # Fallback: nothing usable
    return conversations


def load_with_dataloaders(
    data_directory_path: str,
    batch_size: int = 16,
    shuffle: bool = False,
    tokenizer=None,
):
    """
    Produces: dict[name -> DataLoader], each over conversations for that JSON file.
    """
    dataloaders = {}

    for filename in os.listdir(data_directory_path):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(data_directory_path, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                logger.debug(f"Empty file skipped: {filepath}")
                continue

            data = json.loads(content)
            conversations = _normalize_to_conversations(data)

            if not conversations:
                logger.debug(f"Skipped file with unexpected/empty format: {filepath}")
                continue

            ds = DialogueDataset(
                conversations, tokenizer=tokenizer, max_length=None
            )  # let collate pad
            dataloaders[os.path.splitext(filename)[0]] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
            )
            logger.debug(f"Created dataloader for {os.path.splitext(filename)[0]}: {len(ds)} convs")

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error processing file {filepath}: {e}. Skipping.")
            continue

    return dataloaders


def _contiguous_true_runs(mask_1d):
    """
    mask_1d: 1D BoolTensor of length L
    returns list of (start, end) index pairs for True-runs, end-exclusive
    """
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


def extract_attention_readouts(
    model_outputs,
    selected_layers: range,
    batch_size: int,
    seq_length: int,
) -> torch.Tensor:
    """Extract attention-weighted value readouts from model outputs.

    Args:
        model_outputs: Output from model forward pass with output_attentions=True, use_cache=True
        selected_layers: Range of layer indices to extract from
        batch_size: Batch size (B)
        seq_length: Sequence length (S)

    Returns:
        Tensor of shape (B, S, num_layers, readout_dim) containing attention-weighted readouts
    """
    attns = model_outputs.attentions
    past_kv = model_outputs.past_key_values

    layer_outputs = []
    for layer_idx in selected_layers:
        A = attns[layer_idx]  # (B, Hq, S, S)
        # Handle both old tuple format and new DynamicCache format (transformers 5.x+)
        if hasattr(past_kv, "layers"):
            V = past_kv.layers[layer_idx].values  # (B, Hv, S, D)
        else:
            V = past_kv[layer_idx][1]  # (B, Hv, S, D)
        V = V.to(A.device)

        Hq, Hv = A.size(1), V.size(1)
        if Hq != Hv:
            group = Hq // Hv
            A = A.view(batch_size, Hv, group, seq_length, seq_length).mean(dim=2)

        r = torch.einsum("bhij,bhjd->bihd", A, V)  # (B, S, Hv, D)
        readout = r.flatten(start_dim=2)  # (B, S, Hv*D)
        layer_outputs.append(readout)

    # (B, S, num_layers, Hv*D)
    return torch.stack(layer_outputs, dim=2)


def build_assistant_windows(
    assistant_mask: torch.Tensor,
    window_size: int,
    window_stride: int,
) -> List[Tuple[int, int]]:
    """Build windows from assistant mask spans.

    Args:
        assistant_mask: 1D boolean tensor indicating assistant tokens
        window_size: Size of each window
        window_stride: Stride between windows

    Returns:
        List of (start, end) tuples for each window
    """
    runs = _contiguous_true_runs(assistant_mask)
    windows = []
    for s_abs, e_abs in runs:
        ws = _windows(s_abs, e_abs, window_size, window_stride)
        windows.extend(ws)
    return windows


def decode_windows(
    input_ids: torch.Tensor,
    windows: List[Tuple[int, int]],
    tokenizer,
) -> List[dict]:
    """Decode token windows to text entries.

    Args:
        input_ids: 1D tensor of token IDs
        windows: List of (start, end) tuples
        tokenizer: Tokenizer for decoding

    Returns:
        List of dicts with abs_start, abs_end, and text
    """
    entries = []
    for s, e in windows:
        tok_span = input_ids[s:e].tolist()
        text = tokenizer.decode(
            tok_span,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        entries.append(
            {
                "abs_start": int(s),
                "abs_end": int(e),
                "text": text,
            }
        )
    return entries


def _flatten_token_rep(rep_token: torch.Tensor) -> torch.Tensor:
    # rep_token shape: (num_layers, readout_dim) → (num_layers*readout_dim,)
    return rep_token.flatten()


def _windows(start: int, end: int, W: int, S: int) -> List[Tuple[int, int]]:
    # [start, end) token indices → list of (s,e) windows with step S
    out = []
    i = start
    while i < end:
        j = min(i + W, end)
        out.append((i, j))
        i += S
    return out


@torch.no_grad()
def rnn_logits_over_windows(
    rnn_model,
    token_vecs: torch.Tensor,  # (T, F) float tensor on CPU or CUDA
    windows: List[Tuple[int, int]],
    batch_size: int = 128,
) -> np.ndarray:  # (num_windows, num_topics)
    device = next(rnn_model.parameters()).device
    num_w = len(windows)
    out = []

    # Process windows in batches, but group by similar lengths to minimize padding
    for i in range(0, num_w, batch_size):
        batch_windows = windows[i : i + batch_size]

        # Group windows by length to minimize padding waste
        length_groups = {}
        for j, (s, e) in enumerate(batch_windows):
            length = e - s
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append((j, s, e))

        batch_logits = [None] * len(batch_windows)

        # Process each length group separately
        for length, group in length_groups.items():
            if not group:
                continue

            # Create batch tensor for this length group
            group_size = len(group)
            F = token_vecs.size(1)
            x = torch.zeros((group_size, length, F), dtype=token_vecs.dtype, device=device)

            for k, (orig_idx, s, e) in enumerate(group):
                seq = token_vecs[s:e].to(device)  # (length, F)
                x[k] = seq

            # Forward pass for this length group
            out_logits = rnn_model(x)  # (group_size, num_topics)
            if isinstance(out_logits, tuple):
                out_logits = out_logits[0]

            # Store results in original order
            for k, (orig_idx, s, e) in enumerate(group):
                batch_logits[orig_idx] = out_logits[k].detach().float().cpu().numpy()

        # Concatenate results maintaining original order
        out.append(np.stack(batch_logits))

    return np.concatenate(out, axis=0) if out else np.array([])


def process_dialogue_batch(
    batch: dict,
    model,
    rnn_model,
    tokenizer,
    selected_layers: range,
    window_size: int,
    window_stride: int,
) -> List[dict]:
    """Process a batch of dialogues and extract logits.

    Args:
        batch: Batch dict with input_ids, attention_mask, assistant_mask
        model: LLM model for attention extraction
        rnn_model: RNN guardrail
        tokenizer: Tokenizer for decoding
        selected_layers: Range of layers to extract
        window_size: Window size for logits
        window_stride: Window stride

    Returns:
        List of dicts with 'logits' (numpy array) and 'windows' (list of window entries)
    """
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    assistant_mask = batch["assistant_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=True,
        )

    B, S = input_ids.size()
    stacked = extract_attention_readouts(outputs, selected_layers, B, S)

    results = []
    for b in range(B):
        L_valid = int(attention_mask[b].sum().item())
        am = assistant_mask[b, :L_valid]
        x = stacked[b, :L_valid]

        token_vecs = x.contiguous().view(x.size(0), -1).float().cpu()
        windows = build_assistant_windows(am, window_size, window_stride)

        if not windows:
            results.append(None)
            continue

        ids_valid = input_ids[b, :L_valid].detach().cpu()
        window_entries = decode_windows(ids_valid, windows, tokenizer)
        logits = rnn_logits_over_windows(rnn_model, token_vecs, windows, batch_size=128)

        results.append(
            {
                "logits": logits,
                "windows": window_entries,
            }
        )

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def extract_logits_for_directory(
    data_dir: str,
    output_dir: str,
    model,
    rnn_model,
    tokenizer,
    selected_layers: range,
    window_size: int,
    window_stride: int,
    split: str,
    usecase_path: str,
    batch_size: int = 32,
    logger=None,
    max_samples: Optional[int] = None,
) -> int:
    """Extract logits for dialogues in a directory.

    Args:
        data_dir: Directory containing dialogue JSON files
        output_dir: Root directory for saving logits
        model: LLM model
        rnn_model: RNN guardrail
        tokenizer: Tokenizer
        selected_layers: Range of layers to extract
        window_size: Window size
        window_stride: Window stride
        split: Split name (positive, negative, etc.)
        usecase_path: Relative usecase path
        batch_size: Batch size for dataloader
        logger: Optional logger
        max_samples: Optional maximum number of dialogues to process (None = all)

    Returns:
        Number of dialogues processed
    """
    dataloaders = load_with_dataloaders(
        data_directory_path=data_dir,
        batch_size=batch_size,
        shuffle=False,
        tokenizer=tokenizer,
    )

    processed = 0
    for dialogue_id, dl in dataloaders.items():
        # Check if we've reached the limit
        if max_samples is not None and processed >= max_samples:
            if logger:
                logger.debug(
                    f"Reached max_samples limit ({max_samples}) for directory {usecase_path}"
                )
            break

        save_dir = os.path.join(output_dir, split, usecase_path, dialogue_id)
        os.makedirs(save_dir, exist_ok=True)

        for batch in dl:
            try:
                results = process_dialogue_batch(
                    batch, model, rnn_model, tokenizer, selected_layers, window_size, window_stride
                )
            except torch.cuda.OutOfMemoryError:
                if logger:
                    logger.warning("CUDA OOM - skipping batch")
                torch.cuda.empty_cache()
                continue

            for i, result in enumerate(results):
                if result is None:
                    continue

                out_npy = os.path.join(save_dir, f"{dialogue_id}.npy")
                out_json = os.path.join(save_dir, f"{dialogue_id}.json")

                np.save(out_npy, result["logits"])
                meta = {
                    "dialogue_id": dialogue_id,
                    "split": split,
                    "usecase_path": usecase_path,
                    "window_size": window_size,
                    "window_stride": window_stride,
                    "num_topics": int(result["logits"].shape[1]),
                    "windows": result["windows"],
                }
                with open(out_json, "w") as f:
                    json.dump(meta, f, indent=2)

                processed += 1

                # Check limit after each dialogue
                if max_samples is not None and processed >= max_samples:
                    break

            # Break from batch loop if limit reached
            if max_samples is not None and processed >= max_samples:
                break

    return processed


def _collect_directories_for_split(
    split_dir: str,
    split: str,
) -> List[Tuple[str, str]]:
    """Collect all directories to process for a split (internal)."""
    dirs_to_process = []
    for root, _, files in os.walk(split_dir):
        if not any(f.endswith(".json") for f in files):
            continue
        rel_usecase_path = os.path.relpath(root, split_dir)
        # if (rel_usecase_path.startswith("_") or rel_usecase_path.startswith("neutral")) and split == "negative":
        #     continue
        dirs_to_process.append((root, rel_usecase_path))
    return dirs_to_process


def extract_logits_for_split(
    dataset_root: str,
    output_dir: str,
    split: str,
    model,
    rnn_model,
    tokenizer,
    selected_layers: range,
    window_size: int,
    window_stride: int,
    batch_size: int = 32,
    logger=None,
    show_progress: bool = True,
    max_samples: Optional[int] = None,
    usecases: Optional[List[str]] = None,
) -> int:
    """Extract logits for dialogues in a split.

    Args:
        dataset_root: Root directory of dataset (contains data/<split>/)
        output_dir: Root directory for saving logits
        split: Split name (positive, negative, calibration_usecase_level, etc.)
        model: LLM model
        rnn_model: RNN guardrail
        tokenizer: Tokenizer
        selected_layers: Range of layers to extract
        window_size: Window size
        window_stride: Window stride
        batch_size: Batch size for dataloader
        logger: Optional logger
        show_progress: Show tqdm progress bar
        max_samples: Optional maximum number of dialogues to process (None = all)
        usecases: Optional list of usecase names/directories to process (None = all)

    Returns:
        Total number of dialogues processed
    """
    from tqdm import tqdm

    split_dir = os.path.join(dataset_root, "data", split)
    dirs_to_process = _collect_directories_for_split(split_dir, split)

    # Filter by usecases if specified
    if usecases is not None:
        usecases_set = set(usecases)
        dirs_to_process = [
            (data_dir, usecase_path)
            for data_dir, usecase_path in dirs_to_process
            if usecase_path in usecases_set or any(uc in usecase_path for uc in usecases_set)
        ]
        if logger:
            logger.debug(
                f"Filtered to {len(dirs_to_process)} directories matching usecases: {usecases}"
            )

    if logger:
        logger.info(f"Processing split: {split} ({len(dirs_to_process)} directories)")
        if max_samples:
            logger.info(f"Limiting to {max_samples} dialogues total")

    iterator = (
        tqdm(dirs_to_process, desc=f"Processing {split}") if show_progress else dirs_to_process
    )

    total_processed = 0
    for data_dir, usecase_path in iterator:
        if max_samples is not None and total_processed >= max_samples:
            if logger:
                logger.info(f"Reached max_samples limit ({max_samples}), stopping")
            break

        if logger:
            logger.debug(f"Processing: {usecase_path}")

        # Calculate how many samples we can still process
        remaining = max_samples - total_processed if max_samples is not None else None

        processed = extract_logits_for_directory(
            data_dir=data_dir,
            output_dir=output_dir,
            model=model,
            rnn_model=rnn_model,
            tokenizer=tokenizer,
            selected_layers=selected_layers,
            window_size=window_size,
            window_stride=window_stride,
            split=split,
            usecase_path=usecase_path,
            batch_size=batch_size,
            logger=logger,
            max_samples=remaining,
        )
        total_processed += processed

    return total_processed


def extract_dialogues_in_memory(
    dataset_root: str,
    splits: List[str],
    model,
    rnn_model,
    tokenizer,
    selected_layers: range,
    window_size: int,
    window_stride: int,
    batch_size: int = 32,
    max_samples_per_usecase: Optional[int] = None,
    logger=None,
    show_progress: bool = True,
) -> List[dict]:
    """Extract logits for dialogues in-memory without saving to disk.

    Args:
        dataset_root: Root directory of dataset
        splits: List of split names to process
        model: LLM model
        rnn_model: RNN guardrail
        tokenizer: Tokenizer
        selected_layers: Range of layers to extract
        window_size: Window size
        window_stride: Window stride
        batch_size: Batch size for dataloader
        max_samples_per_usecase: Optional limit per usecase
        logger: Optional logger
        show_progress: Show progress bar

    Returns:
        List of dicts with 'logits' (numpy array), 'metadata' dict
    """
    from tqdm import tqdm

    all_dialogues = []

    for split in splits:
        split_dir = os.path.join(dataset_root, "data", split)
        dirs_to_process = _collect_directories_for_split(split_dir, split)

        if logger:
            logger.info(f"Processing split '{split}' ({len(dirs_to_process)} directories)")

        iterator = (
            tqdm(dirs_to_process, desc=f"Extracting {split}") if show_progress else dirs_to_process
        )

        for data_dir, usecase_path in iterator:
            if max_samples_per_usecase is not None and max_samples_per_usecase <= 0:
                continue

            # Load dataloaders for this directory
            dataloaders = load_with_dataloaders(
                data_directory_path=data_dir,
                batch_size=batch_size,
                shuffle=False,
                tokenizer=tokenizer,
            )

            processed_in_usecase = 0
            for dialogue_id, dl in dataloaders.items():
                if (
                    max_samples_per_usecase is not None
                    and processed_in_usecase >= max_samples_per_usecase
                ):
                    break

                for batch in dl:
                    try:
                        results = process_dialogue_batch(
                            batch,
                            model,
                            rnn_model,
                            tokenizer,
                            selected_layers,
                            window_size,
                            window_stride,
                        )
                    except torch.cuda.OutOfMemoryError:
                        if logger:
                            logger.warning(
                                f"CUDA OOM - skipping batch in {usecase_path}/{dialogue_id}"
                            )
                        torch.cuda.empty_cache()
                        continue

                    for i, result in enumerate(results):
                        if result is None:
                            continue

                        # Store in-memory with metadata
                        all_dialogues.append(
                            {
                                "logits": result["logits"],  # numpy array
                                "metadata": {
                                    "dialogue_id": dialogue_id,
                                    "split": split,
                                    "usecase_path": usecase_path,
                                    "window_size": window_size,
                                    "window_stride": window_stride,
                                    "num_topics": int(result["logits"].shape[1]),
                                    "windows": result["windows"],
                                },
                            }
                        )

                        processed_in_usecase += 1
                        if (
                            max_samples_per_usecase is not None
                            and processed_in_usecase >= max_samples_per_usecase
                        ):
                            break

                    if (
                        max_samples_per_usecase is not None
                        and processed_in_usecase >= max_samples_per_usecase
                    ):
                        break

    if logger:
        logger.info(f"Extracted {len(all_dialogues)} dialogues in-memory")

    return all_dialogues
