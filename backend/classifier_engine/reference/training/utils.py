"""Training utilities for GAVEL.

This module provides utilities for data loading, preprocessing, and
representation extraction for training the GAVEL guardrail.
"""

import json
import logging
import os
import random
import re
import warnings
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.simplefilter("ignore", FutureWarning)

logger = logging.getLogger(__name__)

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


class CognitiveElementDataset(Dataset):
    """PyTorch Dataset for training cognitive element guardrails.

    Processes chat-formatted conversations for training, where each conversation
    ends with a single assistant response. Computes the token index where the
    assistant response begins, which is used to extract representations for
    cognitive element classification.

    Features:
        - Uses tokenizer.apply_chat_template for proper formatting
        - Computes assistant_token_start_index using add_generation_prompt=True
        - Left-truncates to keep assistant tokens visible when exceeding max_length

    Note:
        For multi-turn inference with multiple assistant responses, use
        DialogueDataset from gavel.preprocessing instead.
    """

    def __init__(self, conversations, tokenizer, max_length=512, clean=True):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token exists (common for decoder-only models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if clean and "llama" in tokenizer.name_or_path.lower():
            self.tokenizer.chat_template = LLAMA_CLEAN_TEMPLATE

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]

        # Ensure the last turn is the assistant; otherwise trim trailing turns
        if not conversation or conversation[-1].get("role") != "assistant":
            trimmed = [m for m in conversation if m.get("role") in {"system", "user", "assistant"}]
            while trimmed and trimmed[-1].get("role") != "assistant":
                trimmed = trimmed[:-1]
            if not trimmed:
                # Return an empty padded sample if no usable assistant target remains
                input_ids = torch.full(
                    (self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long
                )
                attention_mask = torch.zeros((self.max_length,), dtype=torch.long)
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "assistant_token_start_index": torch.tensor(0, dtype=torch.long),
                }
            conversation = trimmed

        # 1) Full sequence (prompt + assistant response) with chat template
        #    This includes whatever special tokens the template defines.
        #    Note: In transformers 5.x+, apply_chat_template returns a BatchEncoding.
        #    With return_dict=True, .input_ids gives us the token list directly.
        full_result = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False, return_dict=True
        )
        full_ids = full_result.input_ids if hasattr(full_result, "input_ids") else list(full_result)

        # 2) Prompt-only with an assistant "generation prompt" to find the response start
        #    The length of this is exactly where the assistant tokens begin in the full sequence.
        prompt_result = self.tokenizer.apply_chat_template(
            conversation[:-1], tokenize=True, add_generation_prompt=True, return_dict=True
        )
        prompt_ids = (
            prompt_result.input_ids if hasattr(prompt_result, "input_ids") else list(prompt_result)
        )
        assistant_start = len(prompt_ids)

        # 3) Left-truncate to keep the tail visible
        if len(full_ids) > self.max_length:
            overflow = len(full_ids) - self.max_length
            full_ids = full_ids[overflow:]
            assistant_start = max(0, assistant_start - overflow)

        # 4) Tensorize + pad
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        padding_len = self.max_length - input_ids.size(0)
        if padding_len > 0:
            pad_ids = torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_mask = torch.cat(
                [attention_mask, torch.zeros((padding_len,), dtype=torch.long)], dim=0
            )

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
            "assistant_token_start_index": torch.stack(
                [b["assistant_token_start_index"] for b in batch]
            ),
        }
    else:
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }


def create_dataloaders_from_directory(base_directory, tokenizer, batch_size, max_length=512):
    """
    Reads JSON files from 'train' and 'val' subdirectories of a base directory,
    groups them by topic, filters out unwanted roles, and creates PyTorch DataLoaders.

    Args:
        base_directory (str): Path to the root dataset folder (e.g., 'datasets/new_datasets_llama').
        tokenizer: The tokenizer to use.
        batch_size (int): The batch size for the DataLoaders.
        max_length (int): The maximum sequence length for the CognitiveElementDataset.

    Returns:
        dict: A dictionary containing 'train_dataloaders' and 'val_dataloaders'.
              Each is a dictionary mapping a topic name to its DataLoader.
    """
    all_data = {"train": {}, "val": {}}
    allowed_roles = {"system", "user", "assistant"}

    logger.debug("Searching for datasets...")
    for set_type in ["train", "val"]:
        set_path = os.path.join(base_directory, set_type)
        if not os.path.exists(set_path):
            logger.warning(f"Directory '{set_path}' not found. Skipping.")
            continue

        logger.debug(f"Processing directory: {set_path}")
        for filename in os.listdir(set_path):
            if filename.endswith(".json"):
                base_name = filename.replace(".json", "")
                match = re.match(r"(.+?)_\d+$", base_name)
                topic_name = match.group(1) if match else base_name

                file_path = os.path.join(set_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if not isinstance(data, list):
                            logger.warning(f"Data in {filename} is not a list. Skipping.")
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

                        logger.debug(
                            f"Loaded {len(filtered_conversations)} conversations from {filename} into topic '{topic_name}'."
                        )

                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Error processing file {filename}: {e}")

    # Create PyTorch DataLoaders from the filtered conversation data
    train_dataloaders = {
        topic: DataLoader(
            CognitiveElementDataset(data, tokenizer=tokenizer, max_length=max_length),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        for topic, data in all_data["train"].items()
    }

    validation_dataloaders = {
        topic: DataLoader(
            CognitiveElementDataset(data, tokenizer=tokenizer, max_length=max_length),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        for topic, data in all_data["val"].items()
    }

    logger.debug("Dataloader creation complete.")
    return {
        "train_dataloaders": train_dataloaders,
        "val_dataloaders": validation_dataloaders,
    }


def split_dataset_into_train_val(
    dataset_root_path,  # This will be 'datasets/new_datasets_llama'
    train_ratio=0.8,
    random_seed=42,  # For reproducibility of content split
):
    """
    Splits the content (list of conversations) within each JSON file
    into train and val sets, creating new files in 'train' and 'val' subfolders.
    The original files in dataset_root_path are kept unchanged.

    Args:
        dataset_root_path (str): The path to the folder containing the original dataset JSON files.
        train_ratio (float): The ratio of conversations to be allocated to the training set (e.g., 0.8 for 80%).
        random_seed (int): Seed for random shuffling to ensure reproducible splits of content.
    """

    train_folder = os.path.join(dataset_root_path, "train")
    val_folder = os.path.join(dataset_root_path, "val")

    # Create train and val subfolders
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    logger.debug(f"Created train folder: {train_folder}")
    logger.debug(f"Created val folder: {val_folder}")

    # Get a list of all JSON files in the dataset root (excluding train/val subfolders)
    # We explicitly exclude the newly created train/val folders from the list of files to process
    all_json_files = [
        f
        for f in os.listdir(dataset_root_path)
        if f.endswith(".json") and os.path.isfile(os.path.join(dataset_root_path, f))
    ]

    if not all_json_files:
        logger.warning(f"No JSON files found in {dataset_root_path} to split by content. Exiting.")
        return

    logger.debug(f"Processing {len(all_json_files)} JSON files for content-based split...")

    # Set random seed for reproducibility of content shuffle
    random.seed(random_seed)

    for file_name in all_json_files:
        original_filepath = os.path.join(dataset_root_path, file_name)
        train_filepath = os.path.join(train_folder, file_name)
        val_filepath = os.path.join(val_folder, file_name)

        try:
            with open(original_filepath, "r") as f:
                conversations = json.load(f)

            if not isinstance(conversations, list):
                logger.warning(
                    f"File {file_name} does not contain a JSON list. Skipping content split for this file."
                )
                continue

            if not conversations:
                logger.debug(f"File {file_name} is empty. Skipping content split for this file.")
                continue

            # Shuffle the conversations within this file
            random.shuffle(conversations)

            # Calculate split point for the content of this file
            split_index = int(len(conversations) * train_ratio)

            train_conversations = conversations[:split_index]
            val_conversations = conversations[split_index:]

            # Save train part
            if train_conversations:
                with open(train_filepath, "w") as f:
                    json.dump(train_conversations, f, indent=2)
                logger.debug(f"Saved {len(train_conversations)} conversations to {train_filepath}")
            else:
                logger.debug(
                    f"No training conversations generated for {file_name}. Skipping train file creation."
                )

            # Save val part
            if val_conversations:
                with open(val_filepath, "w") as f:
                    json.dump(val_conversations, f, indent=2)
                logger.debug(f"Saved {len(val_conversations)} conversations to {val_filepath}")
            else:
                logger.debug(
                    f"No validation conversations generated for {file_name}. Skipping val file creation."
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding JSON from {file_name}: {e}. Skipping this file.")
        except IOError as e:
            logger.warning(f"Error reading/writing file {file_name}: {e}. Skipping this file.")
        except Exception as e:
            logger.warning(
                f"An unexpected error occurred processing {file_name}: {e}. Skipping this file."
            )

    logger.debug("Content-based dataset splitting complete.")


def load_model_and_tokenizer(model_name_or_path, token=None):
    """
    Generic function to load any causal language model and its tokenizer.
    Handles different model architectures and tokenizer configurations.

    `token` is an optional HF access token for gated / private models;
    best-effort resolved from the target_models row when not supplied.
    """
    import os

    logger.debug(f"Loading model: {model_name_or_path}...")

    # Check if it's a local path
    is_local = os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path)

    if token is None and not is_local:
        try:
            from utils.sqlite_db import execute_query_dict
            rows = execute_query_dict(
                "SELECT hf_token FROM target_models "
                "WHERE storage_path = %s AND hf_token IS NOT NULL LIMIT 1",
                (model_name_or_path,),
            )
            token = rows[0]["hf_token"] if rows else None
        except Exception:
            token = None

    if "gemma" in model_name_or_path.lower():
        data_type = torch.bfloat16
    else:
        data_type = torch.float16
    # Load the model with generic parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=data_type,
        device_map="auto",
        attn_implementation="eager",
        local_files_only=is_local,
        token=token,
    ).eval()

    # Load tokenizer with model-specific handling
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side="left", legacy=False, local_files_only=is_local, token=token
    )

    # Generic padding token handling
    if tokenizer.pad_token is None:
        # Try to use EOS token as pad token (common for many models)
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug(f"Using EOS token as pad token: {tokenizer.eos_token}")
        else:
            # Fallback to a default pad token
            tokenizer.pad_token = "<pad>"
            logger.debug("Using default pad token: <pad>")

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    logger.debug(f"Model: {model.config.model_type}")
    logger.debug(f"Tokenizer pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    return model, tokenizer


def _head_geometry(model):
    if "gemma" in model.config.model_type:
        n_q_heads = model.config.text_config.num_attention_heads
        n_v_heads = model.config.text_config.num_key_value_heads
        head_dim = model.config.text_config.head_dim
        group_size = n_q_heads // n_v_heads
    else:
        n_q_heads = model.config.num_attention_heads
        n_v_heads = model.config.num_key_value_heads
        head_dim = getattr(model.config, "head_dim", None) or (
            model.config.hidden_size // n_q_heads
        )
        group_size = n_q_heads // n_v_heads
    return n_q_heads, n_v_heads, head_dim, group_size


@torch.no_grad()
def extract_per_sequence_reps(
    *,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    model,
    tokenizer,
    selected_layers: Iterable[int],
    save_root: str,  # e.g. f"{base_directory}/sequences/train"
    dtype: torch.dtype = torch.float16,  # compact on disk
    start_index_per_topic: Dict[str, int] | None = None,  # optional resume support
) -> None:
    """
    For each (topic, sequence) in dataloaders:
      • compute attention-weighted value readouts on the selected layers
      • slice to assistant span using 'assistant_token_start_index'
      • drop special tokens
      • save ONE tensor per sequence: shape (T, num_layers, readout_dim) in float16 (by default)
      • append an index row (jsonl) per sequence with filename and token_ids and decoded text

    Files are named sequentially per topic: seq_000001.pt, seq_000002.pt, ...
    """
    os.makedirs(save_root, exist_ok=True)
    model.eval()

    n_q_heads, n_v_heads, head_dim, group_size = _head_geometry(model)
    readout_dim = n_v_heads * head_dim
    logger.debug(
        f"[extract] n_q_heads={n_q_heads} n_v_heads={n_v_heads} head_dim={head_dim} readout_dim={readout_dim}"
    )

    for topic, loader in dataloaders.items():
        logger.debug(f"[extract] Topic: {topic}")
        topic_dir = os.path.join(save_root, topic)
        os.makedirs(topic_dir, exist_ok=True)
        index_path = os.path.join(topic_dir, "index.jsonl")

        # Determine next sequence id for this topic (so we can resume safely).
        if start_index_per_topic and topic in start_index_per_topic:
            next_id = int(start_index_per_topic[topic])
        else:
            # scan for existing seq_*.pt to continue numbering
            existing = [
                f for f in os.listdir(topic_dir) if f.startswith("seq_") and f.endswith(".pt")
            ]
            if existing:
                # files like seq_000123.pt
                last = max(int(x[4:-3]) for x in existing)
                next_id = last + 1
            else:
                next_id = 1

        with open(index_path, "a") as index_f:  # append if resuming
            for batch in loader:
                device = model.device
                input_ids = batch["input_ids"].to(device)  # (B, L)
                attention_mask = batch["attention_mask"].to(device)  # (B, L)
                start_indices = batch["assistant_token_start_index"].to(device)  # (B,)
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

                # Build readouts for ALL tokens at selected layers: (B, L, num_layers, readout_dim)
                per_layer = []
                for layer_idx in selected_layers:
                    if hasattr(past_kvs, "layers"):
                        v = past_kvs.layers[layer_idx].values  # (B, H_v, L, D)
                    else:
                        v = past_kvs[layer_idx][1]  # (B, H_v, L, D)
                    a = attns[layer_idx]  # (B, H_q, L, L)
                    a_grouped = a.view(B, n_v_heads, group_size, L, L).mean(dim=2)  # (B, H_v, L, L)
                    r_heads = torch.matmul(a_grouped.to(device), v.to(device))  # (B, H_v, L, D)
                    r_layer = r_heads.permute(0, 2, 1, 3).reshape(
                        B, L, readout_dim
                    )  # (B, L, H_v*D)
                    per_layer.append(r_layer)

                token_stack = torch.stack(per_layer, dim=2)  # (B, L, num_layers, readout_dim)

                special_ids = set(tokenizer.all_special_ids)
                special_tensor = (
                    torch.tensor(sorted(list(special_ids)), device=token_stack.device)
                    if special_ids
                    else None
                )

                for b in range(B):
                    start = int(start_indices[b].item())
                    seq_len = int(attention_mask[b].sum().item())
                    if start >= seq_len:
                        continue
                    reps_b = token_stack[b, start:seq_len]  # (T_all, num_layers, readout_dim)
                    ids_b = input_ids[b, start:seq_len]  # (T_all,)

                    if special_tensor is not None and special_tensor.numel() > 0:
                        keep = ~torch.isin(ids_b, special_tensor)
                        reps_b = reps_b[keep]
                        ids_b = ids_b[keep]
                    T = reps_b.size(0)
                    if T == 0:
                        continue

                    # file name: seq_000001.pt
                    fname = f"seq_{next_id:06d}.pt"
                    fpath = os.path.join(topic_dir, fname)
                    next_id += 1

                    # Save (float16 on disk for space)
                    torch.save(reps_b.to(device="cpu"), fpath)

                    # Decode text from ids_b (removed special tokens above)
                    decoded_text = tokenizer.decode(ids_b.tolist(), skip_special_tokens=True)

                    # Append index
                    row = {
                        "file": fname,
                        "path": fpath,
                        "length": int(T),
                        "token_ids": [int(x) for x in ids_b.tolist()],
                        "text": decoded_text,
                    }
                    index_f.write(json.dumps(row) + "\n")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.debug(f"[extract] Wrote/updated index: {index_path}")


class WindowedSequenceDataset(Dataset):
    """
    Expands each seq_*.pt (T, num_layers, readout_dim) into fixed-length windows.

    Produces windows of length `window_len` with stride=`stride` (default: non-overlapping).
    If `pad_last=True`, the last short window is zero-padded; otherwise it is dropped.

    Each __getitem__ returns:
      - chunk: Tensor of shape (window_len, num_layers * readout_dim), dtype=float32
      - y:     Multi-hot label vector of shape (num_classes,), dtype=float32
    """

    def __init__(
        self,
        data_dir: str,
        label: int,
        sequence_length: int,
        num_classes: int,
        stride: Optional[int] = None,
        pad_last: bool = True,
    ):
        self.data_dir = data_dir
        self.label = label
        self.num_classes = num_classes

        self.window_len = int(sequence_length)  # e.g., 5
        self.stride = self.window_len if stride is None else int(stride)
        self.pad_last = bool(pad_last)

        # collect sequence files
        self.files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith("seq_") and f.endswith(".pt")]
        )
        if not self.files:
            raise RuntimeError(f"No sequence files in {data_dir}")

        # Build an index of (file_idx, start_pos) in *token* coordinates
        self.index = []
        for fi, fname in enumerate(self.files):
            fpath = os.path.join(self.data_dir, fname)
            reps = torch.load(
                fpath, map_location="cpu"
            )  # (T, num_layers, readout_dim), stored as fp16
            T = reps.shape[0]

            pos = 0
            # full windows with given stride
            while pos + self.window_len <= T:
                self.index.append((fi, pos))
                pos += self.stride

            # tail handling
            rem = T - pos
            if rem > 0 and self.pad_last:
                self.index.append((fi, pos))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, start = self.index[idx]
        fname = self.files[fi]
        fpath = os.path.join(self.data_dir, fname)

        # load and upcast for training compute
        reps = torch.load(fpath, map_location="cpu").float()  # (T, L, D) -> fp32 in memory

        T, L, D = reps.shape  # L=num_layers, D=readout_dim
        reps = reps.reshape(
            T, L * D
        )  # flatten layer axis for the RNN input used by your current code

        end = min(start + self.window_len, T)
        chunk = reps[start:end]  # (<=window_len, L*D)

        # pad final short window if needed
        if chunk.size(0) < self.window_len:
            pad = torch.zeros(self.window_len - chunk.size(0), chunk.size(1), dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=0)

        # one-vs-all multi-label target (single positive)
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        y[self.label] = 1.0
        return chunk, y


def count_sequences_per_class(split_root: str) -> Tuple[Dict[str, int], int]:
    """
    Returns per-topic file counts and min count across topics.
    split_root: e.g. {base}/sequences/train
    """
    counts: Dict[str, int] = {}
    min_count = float("inf")
    for topic in os.listdir(split_root):
        topic_dir = os.path.join(split_root, topic)
        if not os.path.isdir(topic_dir):
            continue
        n = len([f for f in os.listdir(topic_dir) if f.startswith("seq_") and f.endswith(".pt")])
        counts[topic] = n
        if n < min_count:
            min_count = n
    if min_count == float("inf"):
        min_count = 0
    return counts, min_count


def build_per_class_window_datasets(
    split_root: str,
    labels: Dict[str, int],
    sequence_length: int,
    num_classes: int,
) -> Dict[str, WindowedSequenceDataset]:
    """
    Build a WindowedSequenceDataset for each available class (topic).
    If a class folder doesn't exist or is empty, it is skipped.
    """
    datasets: Dict[str, WindowedSequenceDataset] = {}
    for topic, label in labels.items():
        topic_dir = os.path.join(split_root, topic)
        if not os.path.isdir(topic_dir):
            continue
        files = [f for f in os.listdir(topic_dir) if f.startswith("seq_") and f.endswith(".pt")]
        if not files:
            continue
        ds = WindowedSequenceDataset(topic_dir, label, sequence_length, num_classes)
        datasets[topic] = ds
    return datasets


def build_stratified_concat_dataset(
    split_root: str,
    labels: Dict[str, int],
    sequence_length: int,
    num_classes: int,
    per_class_cap: Optional[int] = None,  # if None, use min window count across classes
    seed: int = 42,
) -> Tuple[Dict[str, Subset], int]:
    """
    For each class/topic under split_root, cap to the same number of *windows*
    (NOT files). The cap is computed as min(len(ds) over classes) unless provided.

    Returns:
      per_class_datasets: dict of {topic: Subset(ds, idxs)}
      actual_min: the cap value actually used
    """
    rng = random.Random(seed)

    # 1) Build all window datasets first (so lengths are window counts).
    ds_per_class = build_per_class_window_datasets(
        split_root=split_root,
        labels=labels,
        sequence_length=sequence_length,
        num_classes=num_classes,
    )

    if not ds_per_class:
        # no data available; return empties
        return {}, 0

    # 2) Determine the cap from window counts if not provided.
    if per_class_cap is None:
        per_class_cap = min(len(ds) for ds in ds_per_class.values())

    # 3) Build balanced Subsets capped by window count.
    per_class_datasets: Dict[str, Subset] = {}
    for topic, label in labels.items():
        ds = ds_per_class.get(topic)
        if ds is None:
            # create an empty subset for missing/empty classes
            per_class_datasets[topic] = Subset(
                WindowedSequenceDataset(
                    os.path.join(split_root, topic),
                    label,
                    sequence_length,
                    num_classes,
                    # The constructor will raise if no files exist, so avoid calling it when ds is None.
                ),
                [],
            )  # not actually used; just to keep keys uniform
            continue

        idxs = list(range(len(ds)))  # indices over windows
        rng.shuffle(idxs)
        idxs = idxs[:per_class_cap]
        per_class_datasets[topic] = Subset(ds, idxs)

    actual_min = per_class_cap
    return per_class_datasets, actual_min


def concat_per_class_datasets(per_class_datasets: Dict[str, Subset]) -> ConcatDataset:
    return ConcatDataset(list(per_class_datasets.values()))


def create_dataloaders_for_sequences(
    base_directory: str,
    labels: Dict[str, int],
    batch_size: int,
    sequence_length: int,
    seed: int = 42,
    num_workers: int = 4,
):
    """
    Expects extractor outputs at:
      {base_directory}/.embeddings/train/{topic}/seq_*.pt
      {base_directory}/.embeddings/val/{topic}/seq_*.pt

    Args:
        base_directory: Root directory containing .embeddings/ subdirectory
        labels: Dictionary mapping label names to indices
        batch_size: Batch size for DataLoaders
        sequence_length: Window length for sequence processing
        seed: Random seed for stratification
        num_workers: Number of DataLoader workers

    Returns:
      dataloaders: {"train": DataLoader, "val": DataLoader}
      class_counts: {"train": {...}, "val": {...}}  (per-class window counts actually used)
      used_min: {"train": int, "val": int}
    """
    splits = {}
    used_min = {}
    class_counts = {"train": {}, "val": {}}

    for split in ["train", "val"]:
        split_root = os.path.join(base_directory, ".embeddings", split)

        per_class, mincount = build_stratified_concat_dataset(
            split_root=split_root,
            labels=labels,
            sequence_length=sequence_length,
            num_classes=len(labels),
            per_class_cap=None,  # use min window count across classes
            seed=seed,
        )
        used_min[split] = mincount

        # materialize counts actually used (per-class window counts)
        for topic, subset_ds in per_class.items():
            class_counts[split][topic] = len(subset_ds)

        if per_class:
            splits[split] = concat_per_class_datasets(per_class)
        else:
            # empty concat is not allowed; make an empty dataset by hand
            splits[split] = Subset(WindowedSequenceDataset.__new__(WindowedSequenceDataset), [])

    dataloaders = {
        "train": DataLoader(
            splits["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            splits["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    return dataloaders, class_counts, used_min
