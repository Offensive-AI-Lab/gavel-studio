# classifier_engine/dialogue_dataset.py
# Multi-turn dialogue preprocessing with role-aware assistant masking.
# Ported from gavel/gavel/preprocessing/utils.py (DialogueDataset).
# Supports Llama-3, Mistral, Gemma, and Qwen model families.
from __future__ import annotations

import logging
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from classifier_engine.utils_train import LLAMA_CLEAN_TEMPLATE

logger = logging.getLogger(__name__)


# ---- KMP helpers for multi-token marker matching ----

def _kmp_build(pattern):
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
    """Yield start indices where *pattern* occurs in *sequence* (both lists of ints)."""
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
    multiple assistant responses within a single dialogue.  Creates a boolean
    mask identifying all assistant tokens for logits extraction.

    Supports:
        - Llama-3  (<|start_header_id|>, <|end_header_id|>, <|eot_id|>)
        - Mistral  ([INST]...[/INST] blocks)
        - Gemma    (<start_of_turn>, <end_of_turn>)
        - Qwen     (<|im_start|>, <|im_end|>, optional <think> stripping)
    """

    def __init__(
        self,
        conversations: List[list],
        tokenizer,
        max_length: Optional[int] = None,
        llama_clean_template: bool = True,
        strip_think_toks: bool = True,
    ):
        self.data = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.strip_think = strip_think_toks

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)

        # Helper to resolve single-token special ids
        def tokid(s):
            tid = self.tokenizer.convert_tokens_to_ids(s)
            return None if tid in (None, -1) else tid

        def enc(s):
            return self.tokenizer.encode(s, add_special_tokens=False)

        # --- Llama single-token turn markers ---
        self.S_HDR = tokid("<|start_header_id|>")
        self.E_HDR = tokid("<|end_header_id|>")
        self.EOT_ID = tokid("<|eot_id|>")

        # --- Mistral multi-token markers ---
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
        self.Q_THINK_S = tokid("<think>")
        self.Q_THINK_E = tokid("</think>")

        # Newline sequence for role/content splitting
        self.NL_SEQ = enc("\n")

        # Detect model family
        name = (getattr(self.tokenizer, "name_or_path", "") or "").lower()
        self.is_mistral_like = any(x in name for x in ("mistral", "mixtral"))
        self.is_gemma_like = "gemma" in name
        self.is_llama_like = "llama" in name
        self.is_qwen_like = "qwen" in name

        self.prepend_user_role = self.is_mistral_like or self.is_gemma_like

        if llama_clean_template and self.is_llama_like:
            self.tokenizer.chat_template = LLAMA_CLEAN_TEMPLATE

        # Pre-encode role tokens for fast comparison
        self._role_tokens = {
            "assistant": enc("assistant"),
            "user": enc("user"),
            "system": enc("system"),
            "model": enc("model"),  # Gemma uses "model" for assistant
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
            self._segment_fn = None  # fallback

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _norm_role(r):
        r = (r or "user").lower()
        return r if r in {"user", "assistant", "system"} else "user"

    def _match_role(self, token_ids: list) -> str:
        for role, role_toks in self._role_tokens.items():
            if len(token_ids) >= len(role_toks):
                if token_ids[:len(role_toks)] == role_toks:
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

    # ---- Llama segmentation ----

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
                role_tokens = id_list[i + 1:j]
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

    # ---- Mistral segmentation ----

    def _segment_by_mistral_blocks(self, ids):
        x = ids.tolist()
        L = len(x)
        spans = []

        bos_pos = [i for i, t in enumerate(x) if t == self.M_BOS]
        bos_pos.append(L)

        for b in range(len(bos_pos) - 1):
            s_bos = bos_pos[b]
            e_bos = bos_pos[b + 1]
            seg = x[s_bos:e_bos]

            inst_starts = list(_kmp_find_all(seg, self.M_INST))
            if not inst_starts:
                continue
            endi_starts = list(_kmp_find_all(seg, self.M_ENDI))

            ei_idx = 0
            for ist in inst_starts:
                while ei_idx < len(endi_starts) and endi_starts[ei_idx] <= ist:
                    ei_idx += 1
                if ei_idx >= len(endi_starts):
                    break
                i_inst = s_bos + ist
                i_endi = s_bos + endi_starts[ei_idx]
                ei_idx += 1

                user_start = i_inst + len(self.M_INST)
                user_end = i_endi
                if user_end > user_start:
                    spans.append({"role": "user", "start": user_start, "end": user_end})

                asst_start = i_endi + len(self.M_ENDI)
                k = asst_start
                while k < e_bos and x[k] != self.M_EOS:
                    k += 1
                asst_end = k
                if asst_end > asst_start:
                    spans.append({"role": "assistant", "start": asst_start, "end": asst_end})

        return spans

    # ---- Gemma segmentation ----

    def _segment_by_gemma_turns(self, ids):
        if self.G_SOT is None or self.G_EOT is None:
            return []

        x = ids.tolist()
        spans = []

        sot_positions = [i for i, t in enumerate(x) if t == self.G_SOT]
        eot_positions = [i for i, t in enumerate(x) if t == self.G_EOT]
        if not sot_positions or not eot_positions:
            return spans

        ei = 0
        for si in sot_positions:
            while ei < len(eot_positions) and eot_positions[ei] < si:
                ei += 1
            if ei >= len(eot_positions):
                break
            ej = eot_positions[ei]
            ei += 1
            if ej <= si + 1:
                continue

            inner_start = si + 1
            inner_end = ej
            inner_tokens = x[inner_start:inner_end]

            nl_idx = None
            if self.NL_SEQ:
                rel_matches = list(_kmp_find_all(inner_tokens, self.NL_SEQ))
                if rel_matches:
                    nl_idx = rel_matches[0]

            if nl_idx is not None:
                role_tok_start = inner_start
                role_tok_end = inner_start + nl_idx
                content_start = role_tok_end + len(self.NL_SEQ)
            else:
                role_tok_start = role_tok_end = inner_start
                content_start = inner_start

            role_tokens = x[role_tok_start:role_tok_end]
            role = self._match_role(role_tokens)
            content_end = inner_end

            if content_end > content_start:
                spans.append({"role": role, "start": content_start, "end": content_end})

        return spans

    # ---- Qwen segmentation ----

    def _segment_by_qwen_turns(self, ids):
        if self.Q_SOT is None or self.Q_EOT is None:
            return []

        x = ids.tolist()
        spans = []

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

            inner_start = si + 1
            inner_end = ej
            inner_tokens = x[inner_start:inner_end]

            nl_rel = None
            rel_matches = list(_kmp_find_all(inner_tokens, self.NL_SEQ)) if self.NL_SEQ else []
            if rel_matches:
                nl_rel = rel_matches[0]

            if nl_rel is not None:
                role_tok_start = inner_start
                role_tok_end = inner_start + nl_rel
                content_start = role_tok_end + len(self.NL_SEQ)
            else:
                role_tok_start = role_tok_end = inner_start
                content_start = inner_start

            role_tokens = x[role_tok_start:role_tok_end]
            role = self._match_role(role_tokens)
            content_end = inner_end

            # Optional: trim <think>...</think> blocks from assistant content
            if (
                self.strip_think
                and role == "assistant"
                and self.Q_THINK_S is not None
                and self.Q_THINK_E is not None
            ):
                i = content_start
                while i < content_end:
                    if x[i] == self.Q_THINK_S:
                        j = i + 1
                        while j < content_end and x[j] != self.Q_THINK_E:
                            j += 1
                        if i > content_start:
                            spans.append({"role": role, "start": content_start, "end": i})
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

    # ---- Tokenization with assistant mask ----

    def _tokenize_with_assistant_mask(self, messages):
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
                    assistant_mask[sp["start"]:sp["end"]] = True

        # Fallback: incremental tokenization
        if not spans:
            prev = torch.empty(0, dtype=torch.long)
            tmp_spans, out, am = [], [], []
            for i in range(1, len(messages) + 1):
                result = self.tokenizer.apply_chat_template(
                    messages[:i],
                    tokenize=True,
                    return_dict=True,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                ids = result.input_ids if hasattr(result, "input_ids") else list(result)
                cur = torch.tensor(ids, dtype=torch.long)
                delta = cur[len(prev):]
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
            input_ids = input_ids[-self.max_length:]
            attention_mask = attention_mask[-self.max_length:]
            assistant_mask = assistant_mask[-self.max_length:]
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


def dialogue_collate_fn(batch):
    """Pad variable-length sequences to max length in batch."""
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
