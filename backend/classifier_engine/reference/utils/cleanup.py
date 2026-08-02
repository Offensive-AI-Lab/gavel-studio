"""Cleanup utilities for GAVEL."""

import logging
import os
import shutil
from typing import Optional


def cleanup_embeddings(embeddings_dir: str, logger: Optional[logging.Logger] = None) -> bool:
    """Delete embeddings directory if it exists.

    This function is designed to be called in a finally block to ensure
    cleanup runs even if training fails.

    Args:
        embeddings_dir: Path to the embeddings directory to delete.
        logger: Optional logger for status messages.

    Returns:
        True if directory was deleted, False if it didn't exist.
    """
    if os.path.exists(embeddings_dir):
        try:
            shutil.rmtree(embeddings_dir)
            if logger:
                logger.info(f"Cleaned up embeddings directory: {embeddings_dir}")
            return True
        except Exception as e:
            if logger:
                logger.warning(f"Failed to cleanup embeddings: {e}")
            return False
    return False
