# Standard Libraries
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt

# Third-party Libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Transformers and Metrics
from ignite.metrics import Accuracy, MultiLabelConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Optional wandb import
_wandb_available = False
try:
    import wandb

    _wandb_available = True
except ImportError:
    pass


def parse_token_filename_indices(filename: str) -> Tuple[int, int, int]:
    """Extract sorting indices from filenames.

    Supports two filename formats:
    1. "train_batch_<batch_num>_sequence_<sequence_num>_token_<token_num>_output.pt"
    2. "token_<token_num>.pt" (simpler format)

    Args:
        filename: The filename to extract indices from.

    Returns:
        Tuple of (batch_num, sequence_num, token_num) for sorting.
    """

    # Format 1: Full filename with batch & sequence info
    match_full = re.search(r"batch_(\d+)_sequence_(\d+)_token_(\d+)_output\.pt", filename)
    if match_full:
        batch_num, sequence_num, token_num = map(int, match_full.groups())
        return batch_num, sequence_num, token_num

    # Format 2: Simple filename ("token_0000.pt")
    match_simple = re.search(r"token_(\d+)\.pt", filename)
    if match_simple:
        token_num = int(match_simple.group(1))
        return (0, 0, token_num)  # Assign default batch & sequence for sorting

    # If format is unknown, return high values to send it to the end
    logger.warning(f"Unexpected filename format '{filename}', skipping sorting.")
    return (9999, 9999, 9999)


class TokenRepresentationDataset(Dataset):
    """
    Returns sequences of shape (seq_len, projection_dim).
    Ensures batching across time for the RNN.

    Args:
        data_dir (str): Directory where sequence subdirectories are stored.
        label (int): Class label for this dataset.
        config (dict): Configurations for dataset processing.
        num_classes (int): Number of classes for multi-label classification.
    """

    def __init__(self, data_dir, label, sequence_length: int, num_classes: int):
        self.data_dir = data_dir
        self.label = label
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # List all sequence directories
        self.sequence_dirs = sorted(
            [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        )

    def __len__(self):
        return len(self.sequence_dirs)  # Each directory is one sequence

    def __getitem__(self, idx):
        """
        Loads all token representations from a sequence directory.

        Returns:
            token_sequence (Tensor): Shape (seq_len, projection_dim)
            label_vector (Tensor): Shape (num_classes,)
        """
        sequence_path = os.path.join(self.data_dir, self.sequence_dirs[idx])
        token_files = [f for f in os.listdir(sequence_path) if f.endswith(".pt")]
        token_files.sort(key=parse_token_filename_indices)

        token_representations = []

        for token_file in token_files:
            token_path = os.path.join(sequence_path, token_file)
            token_tensor = torch.load(token_path)
            token_representations.append(token_tensor)

        # Ensure the sequence has exactly `sequence_length` tokens (pad if necessary)
        seq_len = len(token_representations)
        if seq_len < self.sequence_length:
            pad_tensor = torch.zeros_like(token_representations[0])  # Padding token
            token_representations.extend([pad_tensor] * (self.sequence_length - seq_len))

        # Stack into a single tensor of shape (seq_len, projection_dim)
        token_sequence = torch.stack(token_representations[: self.sequence_length])

        # Create multi-label vector
        label_vector = torch.zeros(self.num_classes)
        label_vector[self.label] = 1  # Multi-label format

        return token_sequence, label_vector


class TopicRNN(nn.Module):
    """RNN-based model for multi-label topic classification.

    Uses a bidirectional GRU/LSTM to process sequences of LLM layer representations
    and output per-topic logits for multi-label classification.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        num_layers: int = 16,
        hidden_dim: int = 256,
        num_rnn_layers: int = 3,
        num_topics: int = 5,
        rnn_type: str = "GRU",
        proj_dim: Optional[int] = None,
    ) -> None:
        """Initialize TopicRNN model.

        Args:
            input_dim: Dimension of each layer's output (default 1024).
            num_layers: Number of LLM layers stacked in input representation.
            hidden_dim: Hidden size of the RNN.
            num_rnn_layers: Number of RNN layers.
            num_topics: Number of topic categories for classification.
            rnn_type: Type of RNN - "GRU" or "LSTM".
            proj_dim: Optional projection dimension. If provided, projects input
                before RNN processing.
        """
        super(TopicRNN, self).__init__()

        self.input_dim = input_dim * num_layers
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_topics = num_topics
        self.rnn_type = rnn_type

        self.proj = nn.Linear(self.input_dim, proj_dim) if proj_dim is not None else None
        rnn_input_dim = proj_dim if proj_dim is not None else self.input_dim

        # RNN layer (GRU or LSTM)
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,  # Now expecting flattened input: (batch_size, seq_len, 49152)
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,  # Using bidirectional GRU
            dropout=0.3,
        )

        # Output layer (Multi-label classification)
        self.fc = nn.Linear(hidden_dim * 2, num_topics)  # *2 because bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RNN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim * num_layers).

        Returns:
            Output logits of shape (batch_size, num_topics).
        """
        # Ensure input is in float32
        x = x.float()  # Convert from float16 to float32 if needed

        # Forward through GRU
        rnn_out, _ = self.rnn(x)  # Shape: (batch_size, seq_len, hidden_dim*2)

        # Take the last hidden state
        final_hidden_state = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_dim*2)

        # Guardrail
        output = self.fc(final_hidden_state)  # Shape: (batch_size, num_topics)

        return output


def load_trained_classifier(
    model_path,
    num_topics,
    selected_layers_length,
    input_dim: int = 1024,
    hidden_dim: Optional[int] = None,
    num_rnn_layers: Optional[int] = None,
    rnn_type: Optional[str] = None,
    rnn_config=None,
):
    """Load a trained TopicRNN guardrail from a checkpoint.

    Args:
        model_path: Path to the saved model state dict (.pth file)
        num_topics: Number of topic categories the model was trained on
        selected_layers_length: Number of layers from LLM representation
        input_dim: Dimension of each layer's output (default 1024)
        hidden_dim: Hidden dimension of the RNN (default 256). Explicit values
            take precedence over rnn_config.
        num_rnn_layers: Number of RNN layers (default 3). Explicit values
            take precedence over rnn_config.
        rnn_type: Type of RNN - "GRU" or "LSTM" (default "GRU"). Explicit
            values take precedence over rnn_config.
        rnn_config: Optional RNNConfig object from gavel.config. Used as
            fallback for parameters not explicitly provided.

    Returns:
        TopicRNN model loaded with weights and set to eval mode
    """
    # Resolve parameters: explicit args > config > defaults
    if hidden_dim is None:
        hidden_dim = rnn_config.hidden_dim if rnn_config else 256
    if num_rnn_layers is None:
        num_rnn_layers = rnn_config.num_rnn_layers if rnn_config else 3
    if rnn_type is None:
        rnn_type = rnn_config.rnn_type if rnn_config else "GRU"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rnn_model = TopicRNN(
        num_layers=selected_layers_length,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_rnn_layers=num_rnn_layers,
        num_topics=num_topics,
        rnn_type=rnn_type,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    rnn_model.load_state_dict(checkpoint)
    rnn_model.eval()

    logger.debug("RNN Model loaded successfully.")
    return rnn_model


def evaluate_rnn_model(
    model: nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    index_to_label: Dict[int, str],
) -> Dict[str, Any]:
    """Evaluate the RNN model on the validation set.

    Args:
        model: Trained RNN model to evaluate.
        val_loader: DataLoader for validation data.
        num_classes: Number of topic classes.
        index_to_label: Mapping from class indices to topic names.

    Returns:
        Dictionary containing evaluation metrics including accuracy, loss,
        confusion matrix statistics, and per-class metrics.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    confusion_metric = MultiLabelConfusionMatrix(num_classes=num_classes)
    accuracy_metric = Accuracy(is_multilabel=True)

    # Collect predictions for sklearn metrics
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            val_outputs = model(batch_data)
            loss = criterion(val_outputs, batch_labels)
            total_loss += loss.item()

            probabilities = torch.sigmoid(val_outputs)
            predictions = (probabilities > 0.9).int()

            confusion_metric.update((predictions, batch_labels.int()))
            accuracy_metric.update((predictions, batch_labels.int()))

            all_preds.append(predictions.cpu().numpy())
            all_labels.append(batch_labels.int().cpu().numpy())

    # Stack for sklearn metrics
    import numpy as np

    if all_preds:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
    else:
        all_preds = np.array([])
        all_labels = np.array([])

    confusion_matrices = confusion_metric.compute()
    tp = confusion_matrices[:, 1, 1].cpu().numpy()
    fp = confusion_matrices[:, 0, 1].cpu().numpy()
    fn = confusion_matrices[:, 1, 0].cpu().numpy()
    tn = confusion_matrices[:, 0, 0].cpu().numpy()
    class_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    total_accuracy = accuracy_metric.compute()

    # Generate confusion matrix plots for each class
    cm_per_class_images = []
    for i in range(num_classes):
        cm = confusion_matrices[i].cpu().numpy()

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))  # Increase figure size

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap="Blues", ax=ax, colorbar=True)

        # Add custom cell annotations
        cell_texts = [
            ["TN: {}".format(cm[0, 0]), "FP: {}".format(cm[0, 1])],
            ["FN: {}".format(cm[1, 0]), "TP: {}".format(cm[1, 1])],
        ]
        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row - 0.2,  # Adjust 'row - 0.2' to move text higher
                    cell_texts[row][col],
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

        # Move X-axis (True Labels) to the top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label", rotation=90, labelpad=15)  # Rotate y-axis label and add padding

        # Save and close the figure to avoid memory issues
        cm_fig = fig
        cm_per_class_images.append(cm_fig)
        plt.close(cm_fig)

    logger.debug(
        f"Evaluation complete. Overall Accuracy: {total_accuracy:.2f}, Avg Val Loss: {total_loss / len(val_loader):.4f}"
    )
    for i, acc in enumerate(class_accuracy):
        class_name = index_to_label[i]
        logger.debug(f"  - Class {i} ({class_name}) Val Accuracy: {acc:.2f}")

    return {
        "accuracy": total_accuracy,
        "loss": total_loss / len(val_loader),
        "tp": tp.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist(),
        "tn": tn.tolist(),
        "class_accuracy": class_accuracy.tolist(),
        "cm_per_class": cm_per_class_images,
        "precision": precision_score(all_labels, all_preds, average=None, zero_division=0).tolist()
        if len(all_labels) > 0
        else [0] * num_classes,
        "recall": recall_score(all_labels, all_preds, average=None, zero_division=0).tolist()
        if len(all_labels) > 0
        else [0] * num_classes,
        "f1": f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
        if len(all_labels) > 0
        else [0] * num_classes,
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0)
        if len(all_labels) > 0
        else 0.0,
    }


def train_rnn_model(
    model: nn.Module,
    labels_dict: Dict[str, int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_class_counts: Dict[int, int],
    val_class_counts: Dict[int, int],
    checkpoint_dir: str,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    patience: Optional[int] = None,
    early_stopping: Optional[bool] = None,
    training_config: Optional[Any] = None,
    use_wandb: bool = True,
) -> nn.Module:
    """Train the RNN model with optional W&B logging and evaluation.

    Args:
        model: TopicRNN model instance to train.
        labels_dict: Dictionary mapping topic names to class indices.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        train_class_counts: Class sample counts for training set.
        val_class_counts: Class sample counts for validation set.
        checkpoint_dir: Directory for saving model checkpoints.
        epochs: Number of training epochs (default 25). Explicit values
            take precedence over training_config.
        learning_rate: Learning rate for optimizer (default 3e-4). Explicit
            values take precedence over training_config.
        patience: Epochs to wait before early stopping (default 3). Explicit
            values take precedence over training_config.
        early_stopping: Whether to use early stopping (default True). Explicit
            values take precedence over training_config.
        training_config: Optional TrainingConfig object from gavel.config.
            Used as fallback for parameters not explicitly provided.
        use_wandb: Whether to log to Weights & Biases. Defaults to True.
            Requires wandb to be installed.

    Returns:
        Trained RNN model.

    Raises:
        ImportError: If use_wandb is True but wandb is not installed.
    """
    # Check wandb availability
    if use_wandb and not _wandb_available:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb` or set use_wandb=False."
        )

    # Resolve parameters: explicit args > config > defaults
    if epochs is None:
        epochs = training_config.epochs if training_config else 25
    if learning_rate is None:
        learning_rate = training_config.learning_rate if training_config else 3e-4
    if patience is None:
        patience = training_config.patience if training_config else 3
    if early_stopping is None:
        early_stopping = training_config.early_stopping if training_config else True

    # Dataset details
    total_train_samples = len(train_loader.dataset)
    total_val_samples = len(val_loader.dataset)
    num_classes = len(labels_dict)
    train_class_distribution = dict(train_class_counts)
    val_class_distribution = dict(val_class_counts)

    # Build descriptive run name
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"gavel_rnn_{num_classes}cls_{epochs}ep_{timestamp}"

    # Initialize W&B (optional)
    if use_wandb:
        wandb.init(
            project="GAVEL-classifier-training",
            name=run_name,
            tags=["rnn", "multi-label", f"{num_classes}_classes"],
            settings=wandb.Settings(console="wrap"),
            config={
                "model/hidden_dim": model.hidden_dim,
                "model/num_rnn_layers": model.num_rnn_layers,
                "model/num_topics": model.num_topics,
                "model/rnn_type": model.rnn_type,
                "model/input_dim": model.input_dim,
                "model/total_params": sum(p.numel() for p in model.parameters()),
                "model/trainable_params": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "training/epochs": epochs,
                "training/learning_rate": learning_rate,
                "training/patience": patience,
                "training/early_stopping": early_stopping,
                "data/train_samples": total_train_samples,
                "data/val_samples": total_val_samples,
                "data/num_classes": num_classes,
                "data/train_class_distribution": train_class_distribution,
                "data/val_class_distribution": val_class_distribution,
                "data/class_names": list(labels_dict.keys()),
            },
        )

    # Log the stats for debugging
    logger.debug("Dataset Summary:")
    logger.debug(f"- Total train samples: {total_train_samples}")
    logger.debug(f"- Total validation samples: {total_val_samples}")
    logger.debug(f"- Number of classes: {num_classes}")
    logger.debug(f"- Train class distribution: {train_class_distribution}")
    logger.debug(f"- Validation class distribution: {val_class_distribution}")
    logger.debug(
        f"- Train-Validation Split Ratio: {round(total_train_samples / (total_train_samples + total_val_samples), 2)}"
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer, Loss, Scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Ignite Metrics
    accuracy_metric = Accuracy(is_multilabel=True)
    confusion_matrix_metric = MultiLabelConfusionMatrix(num_classes=num_classes)

    os.makedirs(checkpoint_dir, exist_ok=True)
    # Removing wandb.watch because it causes high overhead and log clutter
    # if use_wandb:
    #    wandb.watch(model, log="gradients", log_freq=100)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    global_step = 0
    prev_val_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        accuracy_metric.reset()
        confusion_matrix_metric.reset()

        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            train_outputs = model(batch_data)  # Shape: (batch_size, num_topics)

            # Compute loss
            loss = criterion(train_outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probabilities = torch.sigmoid(train_outputs)
            predictions = (probabilities > 0.5).int()

            # Update Ignite Metrics
            accuracy_metric.update((predictions, batch_labels.int()))
            confusion_matrix_metric.update((predictions, batch_labels.int()))

            # Batch-level logging
            global_step += 1
            if use_wandb:
                wandb.log({"train/batch_loss": loss.item(), "epoch": epoch + 1}, step=global_step)

        # Calculate epoch-level metrics
        avg_train_loss = total_loss / len(train_loader)
        epoch_accuracy = accuracy_metric.compute()
        logger.info(
            f"Epoch {epoch + 1} completed. Overall Accuracy: {epoch_accuracy:.4f}, Avg Train Loss: {avg_train_loss:.4f}"
        )

        # Compute class-wise accuracy
        confusion_matrices = confusion_matrix_metric.compute()
        tp = confusion_matrices[:, 1, 1].cpu().numpy()
        fp = confusion_matrices[:, 0, 1].cpu().numpy()
        fn = confusion_matrices[:, 1, 0].cpu().numpy()
        tn = confusion_matrices[:, 0, 0].cpu().numpy()
        class_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        index_to_label = {v: k for k, v in labels_dict.items()}
        for i, acc in enumerate(class_accuracy):
            class_name = index_to_label[i]
            logger.debug(f"  - Class {i} ({class_name}) Train Accuracy: {acc:.2f}")

        # Evaluation
        eval_metrics = evaluate_rnn_model(model, val_loader, num_classes, index_to_label)
        logger.info(
            f"  Val Loss: {eval_metrics['loss']:.4f}, Val Accuracy: {eval_metrics['accuracy']:.4f}"
        )

        # Log epoch-level metrics with hierarchical naming
        if use_wandb:
            # ONLY CONCISE AGGREGATES PER EPOCH
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "val/accuracy": eval_metrics["accuracy"],
                    "val/loss": eval_metrics["loss"],
                    "val/macro_f1": eval_metrics.get("macro_f1", 0.0),
                },
                step=global_step,
            )

            # Log confusion matrix images - ONLY aggregated or sample
            # (Logging 23 images per epoch is also too much, let's skip per-epoch CM logging
            # or only log the first one as a sample, or log them only at the very end)
            pass

        if eval_metrics["loss"] <= prev_val_loss and epoch >= 4:
            prev_val_loss = eval_metrics["loss"]
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_rnn_model.pth"))

        val_loss = eval_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_rnn_model.pth"))
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                state_dict = torch.load(os.path.join(checkpoint_dir, "best_rnn_model.pth"))
                model.load_state_dict(state_dict)
                break

    logger.debug("Training complete.")

    # Save model as wandb artifact for reproducibility
    if use_wandb:
        # Create a detailed per-class performance table at the end
        columns = ["Class", "Precision", "Recall", "F1", "Val Accuracy"]
        data = []
        for i, class_name in index_to_label.items():
            data.append(
                [
                    class_name,
                    eval_metrics["precision"][i],
                    eval_metrics["recall"][i],
                    eval_metrics["f1"][i],
                    eval_metrics["class_accuracy"][i],
                ]
            )

        table = wandb.Table(columns=columns, data=data)
        wandb.log({"evaluation/per_class_table": table})

        # Also log confusion matrices at the END of training only
        for i, cm_image in enumerate(eval_metrics["cm_per_class"]):
            class_name = index_to_label[i]
            wandb.log(
                {
                    f"confusion_matrix/{class_name}": wandb.Image(
                        cm_image, caption=f"{class_name} - Final"
                    )
                }
            )

        model_artifact = wandb.Artifact(
            name=f"gavel-rnn-{num_classes}cls",
            type="model",
            description=f"GAVEL RNN classifier trained for {num_classes} classes",
            metadata={
                "num_classes": num_classes,
                "hidden_dim": model.hidden_dim,
                "num_rnn_layers": model.num_rnn_layers,
                "val_macro_f1": eval_metrics.get("macro_f1", 0.0),
                "val_accuracy": eval_metrics["accuracy"],
            },
        )
        best_model_path = os.path.join(checkpoint_dir, "best_rnn_model.pth")
        if os.path.exists(best_model_path):
            model_artifact.add_file(best_model_path)
            wandb.log_artifact(model_artifact)
        wandb.finish()

    return model
