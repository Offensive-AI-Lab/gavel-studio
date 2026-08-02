"""Centralized logging utilities for GAVEL pipeline.

Provides consistent logging configuration across all scripts with
professional, publication-ready output.
"""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color-coded log levels for terminal output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[34m",  # Blue
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(name: str, verbose: bool = False) -> logging.Logger:
    """Configure and return a logger with consistent formatting.

    Args:
        name: Name of the logger (typically __name__)
        verbose: If True, set level to DEBUG; otherwise INFO

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Set level based on verbosity
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Use colored formatter with clean, simple format
    formatter = ColoredFormatter(fmt="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def add_verbose_arg(parser):
    """Add --verbose argument to an ArgumentParser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (DEBUG level) logging"
    )
