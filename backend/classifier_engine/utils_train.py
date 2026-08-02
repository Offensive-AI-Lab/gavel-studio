# classifier_engine/utils_train.py
# Adapted from the reference GAVEL utils_train.py for platform integration.
# Key change: LABELS dict is not used here; callers pass their own labels dict.
import os
import re
import json
import warnings
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

import random
from typing import Dict, Iterable, Tuple, Optional
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

warnings.simplefilter("ignore", FutureWarning)


LLAMA_CLEAN_TEMPLATE = r"""
{{ bos_token }}
{%- for m in messages %}
<|start_header_id|>{{ m['role'] }}<|end_header_id|>
{{ m['content'] | trim }}
<|eot_id|>
{%- endfor %}

{%- if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>
{%- endif %}
"""


class TopicDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512, clean=True):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if clean and "llama" in tokenizer.name_or_path.lower():
            self.tokenizer.chat_template = LLAMA_CLEAN_TEMPLATE

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]

        if not conversation or conversation[-1].get("role") != "assistant":
            trimmed = [m for m in conversation if m.get("role") in {"system", "user", "assistant"}]
            while trimmed and trimmed[-1].get("role") != "assistant":
                trimmed = trimmed[:-1]
            if not trimmed:
                input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                attention_mask = torch.zeros((self.max_length,), dtype=torch.long)
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "assistant_token_start_index": torch.tensor(0, dtype=torch.long),
                }
            conversation = trimmed

        full_ids = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False, return_dict=False
        )
        prompt_ids = self.tokenizer.apply_chat_template(
            conversation[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        assistant_start = len(prompt_ids)

        if len(full_ids) > self.max_length:
            overflow = len(full_ids) - self.max_length
            full_ids = full_ids[overflow:]
            assistant_start = max(0, assistant_start - overflow)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        padding_len = self.max_length - input_ids.size(0)
        if padding_len > 0:
            pad_ids = torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_mask = torch.cat([attention_mask, torch.zeros((padding_len,), dtype=torch.long)], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_token_start_index": torch.tensor(assistant_start, dtype=torch.long),
        }


def collate_fn(batch):
    if "assistant_token_start_index" in batch[0]:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "assistant_token_start_index": torch.stack([b["assistant_token_start_index"] for b in batch]),
        }
    else:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }


def create_dataloaders_from_directory(base_directory, tokenizer, batch_size, max_length=512):
    """
    Reads JSON files from 'train' and 'val' subdirectories, groups by topic,
    and creates PyTorch DataLoaders.
    """
    all_data = {"train": {}, "val": {}}
    allowed_roles = {"system", "user", "assistant"}

    for set_type in ["train", "val"]:
        set_path = os.path.join(base_directory, set_type)
        if not os.path.exists(set_path):
            continue

        for filename in os.listdir(set_path):
            if not filename.endswith(".json"):
                continue
            base_name = filename.replace('.json', '')
            match = re.match(r'(.+?)_\d+$', base_name)
            topic_name = match.group(1) if match else base_name

            file_path = os.path.join(set_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    continue

                filtered_conversations = []
                for conversation in data:
                    if isinstance(conversation, list):
                        filtered_messages = [
                            msg for msg in conversation if msg.get("role") in allowed_roles
                        ]
                        if filtered_messages:
                            filtered_conversations.append(filtered_messages)

                if topic_name not in all_data[set_type]:
                    all_data[set_type][topic_name] = []
                all_data[set_type][topic_name].extend(filtered_conversations)

            except (json.JSONDecodeError, IOError):
                pass

    train_dataloaders = {
        topic: DataLoader(
            TopicDataset(data, tokenizer=tokenizer, max_length=max_length),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        for topic, data in all_data["train"].items()
    }
    val_dataloaders = {
        topic: DataLoader(
            TopicDataset(data, tokenizer=tokenizer, max_length=max_length),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        for topic, data in all_data["val"].items()
    }

    return {"train_dataloaders": train_dataloaders, "val_dataloaders": val_dataloaders}


def split_dataset_into_train_val(dataset_root_path, train_ratio=0.8, random_seed=42):
    """
    Splits JSON files in dataset_root_path into train/val subdirectories.
    """
    train_folder = os.path.join(dataset_root_path, "train")
    val_folder = os.path.join(dataset_root_path, "val")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    all_json_files = [
        f for f in os.listdir(dataset_root_path)
        if f.endswith('.json') and os.path.isfile(os.path.join(dataset_root_path, f))
    ]
    if not all_json_files:
        return

    random.seed(random_seed)

    for file_name in all_json_files:
        original_filepath = os.path.join(dataset_root_path, file_name)
        train_filepath = os.path.join(train_folder, file_name)
        val_filepath = os.path.join(val_folder, file_name)

        try:
            with open(original_filepath, 'r') as f:
                conversations = json.load(f)
            if not isinstance(conversations, list) or not conversations:
                continue

            random.shuffle(conversations)
            split_index = int(len(conversations) * train_ratio)
            train_conversations = conversations[:split_index]
            val_conversations = conversations[split_index:]

            if train_conversations:
                with open(train_filepath, 'w') as f:
                    json.dump(train_conversations, f, indent=2)
            if val_conversations:
                with open(val_filepath, 'w') as f:
                    json.dump(val_conversations, f, indent=2)

        except Exception:
            pass


def _resolve_model_token(model_name_or_path):
    """Best-effort lookup of a stored HF token for a model ref. Returns None
    if there's no DB (e.g. on a remote worker), no match, or any error — the
    load then proceeds anonymously (fine for public models)."""
    try:
        from utils.sqlite_db import execute_query_dict
        rows = execute_query_dict(
            "SELECT hf_token FROM target_models "
            "WHERE storage_path = %s AND hf_token IS NOT NULL LIMIT 1",
            (model_name_or_path,),
        )
        return rows[0]["hf_token"] if rows else None
    except Exception:
        return None


def load_model_and_tokenizer(model_name_or_path, device_map=None, token=None):
    """
    Load any causal LM and its tokenizer.
    Supports local paths and HuggingFace repo IDs.

    `token` is an optional HF access token for gated / private models. When
    not given, we best-effort resolve it from the target_models row matching
    this ref, so callers don't have to thread it through.
    """
    import logging
    logger = logging.getLogger(__name__)
    if device_map is None:
        from utils.device import get_llm_device_map
        device_map = get_llm_device_map()
    if token is None:
        token = _resolve_model_token(model_name_or_path)
    logger.info(f"Loading model: {model_name_or_path} (device_map={device_map}, token={'yes' if token else 'no'})")

    if "gemma" in model_name_or_path.lower():
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    def _from_pretrained(dm):
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=data_type,
            device_map=dm,
            attn_implementation="eager",
            token=token,
        ).eval()

    _wants_mps = device_map == "mps" or (
        isinstance(device_map, dict) and device_map.get("") == "mps"
    )

    try:
        if _wants_mps:
            # accelerate's device_map placement deadlocks on MPS (observed with
            # torch 2.13 / transformers 5.14: from_pretrained hangs forever at
            # "Loading weights: 0%"). Load on CPU and move instead — same
            # placement, and a too-large model raises a catchable MPS error
            # (handled below) instead of hanging.
            model = _from_pretrained(None).to("mps")
        else:
            model = _from_pretrained(device_map)
    except Exception as e:
        # Apple Silicon: MPS caps a single buffer to a fraction of unified memory,
        # so a 7B fp16 model (~14 GB) can't load onto the GPU on a small Mac
        # ("Invalid buffer size" / MPS OOM). Fall back to CPU so the feature still
        # works (slower) instead of crashing. Only retry for an MPS placement.
        _msg = str(e).lower()
        if _wants_mps and ("buffer size" in _msg or "out of memory" in _msg
                        or "invalid buffer" in _msg or "mps" in _msg):
            logger.warning(
                f"MPS load failed for {model_name_or_path} ({e}); the model is too "
                f"large for this Mac's GPU — falling back to CPU (this will be slower)."
            )
            try:
                from utils.device import empty_device_cache
                empty_device_cache()
            except Exception:
                pass
            model = _from_pretrained("cpu")
        else:
            raise

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side="left", legacy=False, token=token
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "<pad>"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    logger.info(f"Loaded model type: {model.config.model_type}")
    return model, tokenizer


def _head_geometry(model):
    if "gemma" in model.config.model_type:
        n_q_heads = model.config.text_config.num_attention_heads
        n_v_heads = model.config.text_config.num_key_value_heads
        head_dim = model.config.text_config.head_dim
    else:
        n_q_heads = model.config.num_attention_heads
        n_v_heads = model.config.num_key_value_heads
        head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // n_q_heads)
    group_size = n_q_heads // n_v_heads
    return n_q_heads, n_v_heads, head_dim, group_size


@torch.no_grad()
def extract_per_sequence_reps(
    *,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    model,
    tokenizer,
    selected_layers: Iterable[int],
    save_root: str,
    dtype: torch.dtype = torch.float16,
    start_index_per_topic: Optional[Dict[str, int]] = None,
) -> None:
    """
    For each (topic, sequence):
      - compute attention-weighted value readouts on selected layers
      - slice to assistant span using assistant_token_start_index
      - drop special tokens
      - save ONE tensor per sequence: shape (T, num_layers, readout_dim) in float16
    """
    os.makedirs(save_root, exist_ok=True)
    model.eval()

    n_q_heads, n_v_heads, head_dim, group_size = _head_geometry(model)
    readout_dim = n_v_heads * head_dim

    for topic, loader in dataloaders.items():
        topic_dir = os.path.join(save_root, topic)
        os.makedirs(topic_dir, exist_ok=True)
        index_path = os.path.join(topic_dir, "index.jsonl")

        if start_index_per_topic and topic in start_index_per_topic:
            next_id = int(start_index_per_topic[topic])
        else:
            existing = [f for f in os.listdir(topic_dir) if f.startswith("seq_") and f.endswith(".pt")]
            if existing:
                last = max(int(x[4:-3]) for x in existing)
                next_id = last + 1
            else:
                next_id = 1

        with open(index_path, "a") as index_f:
            for batch in loader:
                device = model.device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_indices = batch["assistant_token_start_index"].to(device)
                B, L = input_ids.shape

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=True,
                    output_hidden_states=False,
                )
                attns = outputs.attentions
                past_kvs = outputs.past_key_values
                # transformers 5.x: DynamicCache with .layers[i].values
                # transformers 4.x: DynamicCache with .value_cache[i]
                # older:            tuple-of-tuples cache[i][1]
                def _get_v(cache, idx):
                    if hasattr(cache, "layers"):
                        return cache.layers[idx].values
                    if hasattr(cache, "value_cache"):
                        return cache.value_cache[idx]
                    return cache[idx][1]

                per_layer = []
                for layer_idx in selected_layers:
                    v = _get_v(past_kvs, layer_idx)
                    a = attns[layer_idx]
                    a_grouped = a.view(B, n_v_heads, group_size, L, L).mean(dim=2)
                    r_heads = torch.matmul(a_grouped.to(device), v.to(device))
                    r_layer = r_heads.permute(0, 2, 1, 3).reshape(B, L, readout_dim)
                    per_layer.append(r_layer)

                token_stack = torch.stack(per_layer, dim=2)

                special_ids = set(tokenizer.all_special_ids)
                special_tensor = torch.tensor(sorted(list(special_ids)), device=token_stack.device) if special_ids else None

                for b in range(B):
                    start = int(start_indices[b].item())
                    seq_len = int(attention_mask[b].sum().item())
                    if start >= seq_len:
                        continue
                    reps_b = token_stack[b, start:seq_len]
                    ids_b = input_ids[b, start:seq_len]

                    if special_tensor is not None and special_tensor.numel() > 0:
                        keep = ~torch.isin(ids_b, special_tensor)
                        reps_b = reps_b[keep]
                        ids_b = ids_b[keep]
                    T = reps_b.size(0)
                    if T == 0:
                        continue

                    fname = f"seq_{next_id:06d}.pt"
                    fpath = os.path.join(topic_dir, fname)
                    next_id += 1
                    torch.save(reps_b.to(dtype=dtype, device="cpu"), fpath)

                    decoded_text = tokenizer.decode(ids_b.tolist(), skip_special_tokens=True)
                    row = {
                        "file": fname,
                        "path": fpath,
                        "length": int(T),
                        "token_ids": [int(x) for x in ids_b.tolist()],
                        "text": decoded_text,
                    }
                    index_f.write(json.dumps(row) + "\n")

                from utils.device import empty_device_cache
                empty_device_cache()


class WindowedSequenceDataset(Dataset):
    def __init__(self, data_dir, label, config, num_classes, stride=None, pad_last=True):
        self.data_dir = data_dir
        self.label = label
        self.num_classes = num_classes
        self.window_len = int(config["RNN_sequence_length"])
        self.stride = self.window_len if stride is None else int(stride)
        self.pad_last = bool(pad_last)

        self.files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith("seq_") and f.endswith(".pt")]
        )
        if not self.files:
            raise RuntimeError(f"No sequence files in {data_dir}")

        self.index = []
        for fi, fname in enumerate(self.files):
            fpath = os.path.join(self.data_dir, fname)
            reps = torch.load(fpath, map_location="cpu", weights_only=True)
            T = reps.shape[0]

            pos = 0
            while pos + self.window_len <= T:
                self.index.append((fi, pos))
                pos += self.stride

            rem = T - pos
            if rem > 0 and self.pad_last:
                self.index.append((fi, pos))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, start = self.index[idx]
        fname = self.files[fi]
        fpath = os.path.join(self.data_dir, fname)

        reps = torch.load(fpath, map_location="cpu", weights_only=True).float()
        T, L, D = reps.shape
        reps = reps.reshape(T, L * D)

        end = min(start + self.window_len, T)
        chunk = reps[start:end]

        if chunk.size(0) < self.window_len:
            pad = torch.zeros(self.window_len - chunk.size(0), chunk.size(1), dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=0)

        y = torch.zeros(self.num_classes, dtype=torch.float32)
        y[self.label] = 1.0
        return chunk, y


def _build_window_datasets_per_class(split_root, labels, config, num_classes):
    datasets = {}
    for topic, label in labels.items():
        topic_dir = os.path.join(split_root, topic)
        if not os.path.isdir(topic_dir):
            continue
        files = [f for f in os.listdir(topic_dir) if f.startswith("seq_") and f.endswith(".pt")]
        if not files:
            continue
        ds = WindowedSequenceDataset(topic_dir, label, config, num_classes)
        datasets[topic] = ds
    return datasets


def build_stratified_concat_dataset(split_root, labels, config, num_classes, per_class_cap=None, seed=42):
    rng = random.Random(seed)
    ds_per_class = _build_window_datasets_per_class(split_root, labels, config, num_classes)

    if not ds_per_class:
        return {}, 0

    if per_class_cap is None:
        per_class_cap = min(len(ds) for ds in ds_per_class.values())

    per_class_datasets = {}
    for topic, label in labels.items():
        ds = ds_per_class.get(topic)
        if ds is None:
            continue
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:per_class_cap]
        per_class_datasets[topic] = Subset(ds, idxs)

    return per_class_datasets, per_class_cap


def concat_per_class_datasets(per_class_datasets):
    return ConcatDataset(list(per_class_datasets.values()))


def create_dataloaders_for_sequences(base_directory, labels, batch_size, config, seed=42, num_workers=0):
    """
    Expects extractor outputs at:
      {base_directory}/sequences/train/{topic}/seq_*.pt
      {base_directory}/sequences/val/{topic}/seq_*.pt

    Returns:
      dataloaders: {"train": DataLoader, "val": DataLoader}
      class_counts: {"train": {...}, "val": {...}}
      used_min: {"train": int, "val": int}
    """
    splits = {}
    used_min = {}
    class_counts = {"train": {}, "val": {}}
    num_classes = len(labels)

    for split in ["train", "val"]:
        split_root = os.path.join(base_directory, "sequences", split)

        per_class, mincount = build_stratified_concat_dataset(
            split_root=split_root,
            labels=labels,
            config=config,
            num_classes=num_classes,
            per_class_cap=None,
            seed=seed,
        )
        used_min[split] = mincount

        for topic, subset_ds in per_class.items():
            class_counts[split][topic] = len(subset_ds)

        if per_class:
            splits[split] = concat_per_class_datasets(per_class)
        else:
            from torch.utils.data import TensorDataset
            splits[split] = TensorDataset(torch.zeros(0), torch.zeros(0))

    dataloaders = {
        "train": DataLoader(splits["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(splits["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return dataloaders, class_counts, used_min
