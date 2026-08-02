"""Configuration management for GAVEL.

This module provides dataclasses and utilities for loading and managing
configuration from JSON files.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the base LLM model."""

    name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.2"
    selected_layers_range: Tuple[int, int] = (13, 27)

    @property
    def selected_layers(self) -> range:
        """Return range object for selected layers."""
        return range(self.selected_layers_range[0], self.selected_layers_range[1])


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    batch_size: int = 64
    batch_size_text: int = 4
    max_length: int = 256
    epochs: int = 25
    learning_rate: float = 3e-4
    patience: int = 3
    early_stopping: bool = True
    use_wandb: bool = True
    cleanup_embeddings: bool = True


@dataclass
class RNNConfig:
    """Configuration for the RNN model architecture."""

    hidden_dim: int = 256
    num_rnn_layers: int = 3
    rnn_type: str = "GRU"
    sequence_length: int = 5
    dropout: float = 0.3
    proj_dim: Optional[int] = None


@dataclass
class PathsConfig:
    """Configuration for file paths.

    User-configurable paths are stored as fields. Derived paths are computed
    as properties based on base_dir to enforce consistent naming conventions.
    """

    base_dir: str = ""
    train_dataset: str = ""
    eval_dataset: str = ""
    calibration_dataset: str = ""
    labels_path: str = "labels.json"
    ruleset_path: Optional[str] = None

    @property
    def logits_dir(self) -> str:
        """Directory for evaluation logits: {base_dir}/logits"""
        return os.path.join(self.base_dir, "logits") if self.base_dir else ""

    @property
    def model_dir(self) -> str:
        """Directory for trained models: {base_dir}/model"""
        return os.path.join(self.base_dir, "model") if self.base_dir else ""

    @property
    def rnn_model_path(self) -> str:
        """Path to trained RNN model: {base_dir}/model/trained_model_rnn.pth"""
        return os.path.join(self.model_dir, "trained_model_rnn.pth") if self.model_dir else ""

    @property
    def calibration_dir(self) -> str:
        """Directory for calibration results: {base_dir}/calibration_results"""
        return os.path.join(self.base_dir, "calibration_results") if self.base_dir else ""

    @property
    def thresholds_path(self) -> str:
        """Path to optimal thresholds JSON: {base_dir}/calibration_results/thresholds.json"""
        return os.path.join(self.calibration_dir, "thresholds.json") if self.calibration_dir else ""

    @property
    def results_dir(self) -> str:
        """Directory for evaluation results: {base_dir}/results"""
        return os.path.join(self.base_dir, "results") if self.base_dir else ""

    @property
    def embeddings_dir(self) -> str:
        """Directory for embeddings: {base_dir}/.embeddings (hidden, temporary)"""
        return os.path.join(self.base_dir, ".embeddings") if self.base_dir else ""


@dataclass
class EvalConfig:
    """Configuration for evaluation settings (non-path)."""

    window_size: int = 5
    window_stride: int = 5
    max_samples_per_usecase: Optional[int] = None
    save_logits: bool = False  # For debugging/backward compatibility


@dataclass
class GavelConfig:
    """Main configuration container for GAVEL.

    Attributes:
        model_name: Name/identifier for the model (e.g., 'mistral_7b')
        labels_path: Path to labels.json file
        labels: Dictionary mapping label names to indices (loaded from labels_path)
        model: Model configuration
        training: Training configuration
        rnn: RNN architecture configuration
        paths: File paths configuration
        eval: Evaluation configuration
    """

    model_name: str = "mistral_7b"
    labels_path: str = "labels.json"
    ruleset_path: Optional[str] = None
    labels: Dict[str, int] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Legacy fields for backward compatibility
    hugging_face_token: Optional[str] = None

    @property
    def num_labels(self) -> int:
        """Return the number of labels."""
        return len(self.labels)

    @property
    def index_to_label(self) -> Dict[int, str]:
        """Return reverse mapping from index to label name."""
        return {v: k for k, v in self.labels.items()}


def load_labels(path: str, base_dir: Optional[str] = None) -> Dict[str, int]:
    """Load labels from a JSON file.

    Args:
        path: Path to the labels JSON file
        base_dir: Optional base directory for resolving relative paths

    Returns:
        Dictionary mapping label names to indices
    """
    if base_dir:
        labels_path = Path(base_dir) / path
    else:
        labels_path = Path(path)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    logger.debug(f"Loaded {len(labels)} labels from {labels_path}")
    return labels


def load_config(config_path: str = "config.json") -> GavelConfig:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the config JSON file

    Returns:
        GavelConfig object with all configuration loaded
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Parse nested configs
    model_cfg = (
        ModelConfig(
            name_or_path=raw.get("model", {}).get("name_or_path", ModelConfig.name_or_path),
            selected_layers_range=tuple(
                raw.get("model", {}).get(
                    "selected_layers_range", list(ModelConfig.selected_layers_range)
                )
            ),
        )
        if "model" in raw
        else ModelConfig(
            # Legacy support: flat config structure
            name_or_path=raw.get("model_name_or_path", ModelConfig.name_or_path),
            selected_layers_range=tuple(
                raw.get("selected_layers_range", list(ModelConfig.selected_layers_range))
            ),
        )
    )

    training_cfg = TrainingConfig(
        batch_size=raw.get("training", {}).get(
            "batch_size", raw.get("batch_size", TrainingConfig.batch_size)
        ),
        batch_size_text=raw.get("training", {}).get(
            "batch_size_text", TrainingConfig.batch_size_text
        ),
        max_length=raw.get("training", {}).get("max_length", TrainingConfig.max_length),
        epochs=raw.get("training", {}).get("epochs", raw.get("epochs", TrainingConfig.epochs)),
        learning_rate=raw.get("training", {}).get("learning_rate", TrainingConfig.learning_rate),
        patience=raw.get("training", {}).get("patience", TrainingConfig.patience),
        early_stopping=raw.get("training", {}).get("early_stopping", TrainingConfig.early_stopping),
        use_wandb=raw.get("training", {}).get("use_wandb", TrainingConfig.use_wandb),
        cleanup_embeddings=raw.get("training", {}).get(
            "cleanup_embeddings", TrainingConfig.cleanup_embeddings
        ),
    )

    rnn_cfg = RNNConfig(
        hidden_dim=raw.get("rnn", {}).get("hidden_dim", RNNConfig.hidden_dim),
        num_rnn_layers=raw.get("rnn", {}).get("num_rnn_layers", RNNConfig.num_rnn_layers),
        rnn_type=raw.get("rnn", {}).get("rnn_type", RNNConfig.rnn_type),
        sequence_length=raw.get("rnn", {}).get(
            "sequence_length", raw.get("RNN_sequence_length", RNNConfig.sequence_length)
        ),
        dropout=raw.get("rnn", {}).get("dropout", RNNConfig.dropout),
        proj_dim=raw.get("rnn", {}).get("proj_dim", RNNConfig.proj_dim),
    )
    # Determine model name first (needed for paths)
    if "model_name" in raw:
        model_name = raw["model_name"]
    else:
        # Derive from model path/name (e.g. "org/Mistral-7B" -> "Mistral-7B")
        full_path = str(model_cfg.name_or_path).rstrip("/")
        model_name = os.path.basename(full_path)
        if not model_name:  # Handle edge case of root path
            model_name = "unknown_model"
        logger.info(f"Derived model_name '{model_name}' from path '{model_cfg.name_or_path}'")

    # Append derived model_name to base_dir to create effective base directory
    raw_base_dir = raw.get("paths", {}).get("base_dir", PathsConfig.base_dir)
    effective_base_dir = os.path.join(raw_base_dir, model_name) if raw_base_dir else ""

    paths_cfg = PathsConfig(
        base_dir=effective_base_dir,
        train_dataset=raw.get("paths", {}).get("train_dataset", PathsConfig.train_dataset),
        eval_dataset=raw.get("paths", {}).get("eval_dataset", PathsConfig.eval_dataset),
        calibration_dataset=raw.get("paths", {}).get(
            "calibration_dataset", PathsConfig.calibration_dataset
        ),
        labels_path=raw.get("paths", {}).get(
            "labels_path", raw.get("labels_path", PathsConfig.labels_path)
        ),
        ruleset_path=raw.get("paths", {}).get(
            "ruleset_path", raw.get("ruleset_path", PathsConfig.ruleset_path)
        ),
    )

    eval_cfg = EvalConfig(
        window_size=raw.get("eval", {}).get("window_size", EvalConfig.window_size),
        window_stride=raw.get("eval", {}).get("window_stride", EvalConfig.window_stride),
        max_samples_per_usecase=raw.get("eval", {}).get(
            "max_samples_per_usecase", EvalConfig.max_samples_per_usecase
        ),
        save_logits=raw.get("eval", {}).get("save_logits", EvalConfig.save_logits),
    )

    labels_path = paths_cfg.labels_path

    # Load labels from separate file
    config_dir = config_path.parent
    try:
        labels = load_labels(labels_path, base_dir=str(config_dir))
    except FileNotFoundError:
        # Try absolute path
        try:
            labels = load_labels(labels_path)
        except FileNotFoundError:
            logger.warning(f"Labels file not found at {labels_path}, using empty labels")
            labels = {}

    config = GavelConfig(
        model_name=model_name,
        labels_path=labels_path,
        ruleset_path=paths_cfg.ruleset_path,
        labels=labels,
        model=model_cfg,
        training=training_cfg,
        rnn=rnn_cfg,
        paths=paths_cfg,
        eval=eval_cfg,
        hugging_face_token=raw.get("hugging_face_token"),
    )

    logger.debug(f"Loaded config from {config_path}")
    return config
