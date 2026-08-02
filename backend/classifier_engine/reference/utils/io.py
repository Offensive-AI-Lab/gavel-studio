"""Shared I/O utilities for GAVEL.

This module contains common file iteration and I/O functions used across
evaluation, calibration, and other modules.
"""

import json
import logging
from pathlib import Path
from typing import Iterator, Tuple

logger = logging.getLogger(__name__)


def iter_dialogue_files(root_dir: str) -> Iterator[Tuple[Path, Path, dict]]:
    """Yield (meta_path, npy_path, meta_json) for every dialogue folder that has both.

    Iterates through all JSON files in the given directory tree and yields
    tuples containing the metadata file path, corresponding numpy file path,
    and loaded metadata dictionary.

    Args:
        root_dir: Root directory to search for dialogue files.

    Yields:
        Tuple of (meta_path, npy_path, meta_dict) where:
            - meta_path: Path to the JSON metadata file
            - npy_path: Path to the corresponding .npy logits file
            - meta_dict: Loaded JSON metadata as a dictionary
    """
    root = Path(root_dir)
    for meta_path in root.rglob("*.json"):
        npy_path = meta_path.with_suffix(".npy")
        if not npy_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        yield meta_path, npy_path, meta
