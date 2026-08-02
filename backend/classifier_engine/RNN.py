# classifier_engine/RNN.py
# Adapted from the reference GAVEL RNN.py — wandb made optional for platform use.
import os
import re
import io
import time
import logging
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ignite.metrics import Accuracy, MultiLabelConfusionMatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logger = logging.getLogger(__name__)


def extract_indices(filename):
    match_full = re.search(r'batch_(\d+)_sequence_(\d+)_token_(\d+)_output\.pt', filename)
    if match_full:
        batch_num, sequence_num, token_num = map(int, match_full.groups())
        return batch_num, sequence_num, token_num

    match_simple = re.search(r'token_(\d+)\.pt', filename)
    if match_simple:
        token_num = int(match_simple.group(1))
        return (0, 0, token_num)

    return (9999, 9999, 9999)


class TokenRepresentationDataset(Dataset):
    def __init__(self, data_dir, label, config, num_classes):
        self.data_dir = data_dir
        self.label = label
        self.num_classes = num_classes
        self.sequence_length = config["RNN_sequence_length"]
        self.sequence_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, idx):
        sequence_path = os.path.join(self.data_dir, self.sequence_dirs[idx])
        token_files = [f for f in os.listdir(sequence_path) if f.endswith(".pt")]
        token_files.sort(key=extract_indices)
        token_representations = []

        for token_file in token_files:
            token_path = os.path.join(sequence_path, token_file)
            token_tensor = torch.load(token_path, weights_only=True)
            token_representations.append(token_tensor)

        seq_len = len(token_representations)
        if seq_len < self.sequence_length:
            pad_tensor = torch.zeros_like(token_representations[0])
            token_representations.extend([pad_tensor] * (self.sequence_length - seq_len))

        token_sequence = torch.stack(token_representations[:self.sequence_length])

        label_vector = torch.zeros(self.num_classes)
        label_vector[self.label] = 1

        return token_sequence, label_vector


class TopicRNN(nn.Module):
    def __init__(self, input_dim=1024, num_layers=16, hidden_dim=256, num_rnn_layers=3, num_topics=5, rnn_type="GRU", proj_dim=None):
        super(TopicRNN, self).__init__()

        self.input_dim = input_dim * num_layers
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_topics = num_topics
        self.rnn_type = rnn_type

        self.proj = nn.Linear(self.input_dim, proj_dim) if proj_dim is not None else None
        rnn_input_dim = proj_dim if proj_dim is not None else self.input_dim

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_dim * 2, num_topics)

    def forward(self, x):
        x = x.float()
        if self.proj is not None:
            x = self.proj(x)
        rnn_out, _ = self.rnn(x)
        final_hidden_state = rnn_out[:, -1, :]
        output = self.fc(final_hidden_state)
        return output


def evaluate_rnn_model(model, val_loader, num_classes, index_to_label):
    model.eval()
    from utils.device import get_torch_device
    device = get_torch_device()
    model.to(device)

    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    confusion_metric = MultiLabelConfusionMatrix(num_classes=num_classes)
    accuracy_metric = Accuracy(is_multilabel=True)

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

    confusion_matrices = confusion_metric.compute()
    tp = confusion_matrices[:, 1, 1].cpu().numpy()
    fp = confusion_matrices[:, 0, 1].cpu().numpy()
    fn = confusion_matrices[:, 1, 0].cpu().numpy()
    tn = confusion_matrices[:, 0, 0].cpu().numpy()
    class_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    total_accuracy = accuracy_metric.compute()

    cm_per_class_images = []
    for i in range(num_classes):
        cm = confusion_matrices[i].cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap="Blues", ax=ax, colorbar=True)
        cm_per_class_images.append(fig)
        plt.close(fig)

    avg_val_loss = total_loss / max(len(val_loader), 1)

    # Per-class precision, recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Evaluation complete. Overall Accuracy: {total_accuracy:.2f}, Avg Val Loss: {avg_val_loss:.4f}", flush=True)
    for i, acc in enumerate(class_accuracy):
        class_name = index_to_label.get(i, str(i))
        print(f"  - Class {i} ({class_name}) Acc: {acc:.2f} P: {precision[i]:.2f} R: {recall[i]:.2f} F1: {f1[i]:.2f}", flush=True)
    logger.info(f"Evaluation: Overall Accuracy={total_accuracy:.2f}, Avg Val Loss: {avg_val_loss:.4f}")

    return {
        "accuracy": total_accuracy,
        "loss": avg_val_loss,
        "tp": tp.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist(),
        "tn": tn.tolist(),
        "class_accuracy": class_accuracy.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "cm_per_class": cm_per_class_images,
    }


def _seed_everything(seed: int):
    """Seed Python/NumPy/torch (incl. CUDA) so each candidate fit is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_rnn_candidates(model_factory, *, rounds: int = 5, base_seed: int = 42,
                         progress_callback=None, **train_kwargs):
    """Train `rounds` independent candidate RNNs, each an EXACT-parity
    train_rnn_model fit (fresh seeded init + data order on the same cached
    features). Returns the list of trained models; the caller picks one by
    scoring them on out-of-domain (calibration) data — per-fit validation
    metrics saturate at ~1.0 and cannot distinguish candidates.

    `progress_callback(step, total_steps, metrics)` reports one CONTINUOUS
    counter across all rounds (step runs 1..rounds*epochs) so callers can
    surface a single smooth progress figure without exposing the per-round
    structure in user-facing output.

    `model_factory()` must return a FRESH TopicRNN on the training device."""
    rounds = max(1, int(rounds))
    epochs = train_kwargs["epochs"]
    total_steps = rounds * epochs
    candidates = []

    for r in range(rounds):
        _seed_everything(base_seed + r)
        model = model_factory()

        def _cb(epoch, _total, metrics, _r=r):
            if progress_callback:
                progress_callback(_r * epochs + epoch, total_steps, metrics)

        trained = train_rnn_model(model=model, progress_callback=_cb, **train_kwargs)
        logger.info(f"[candidates] fit {r + 1}/{rounds} (seed {base_seed + r}) complete")
        candidates.append(trained)

    return candidates


def train_rnn_model(
    model,
    labels_dict,
    train_loader,
    val_loader,
    epochs,
    train_class_counts,
    val_class_counts,
    checkpoint_dir,
    learning_rate=3e-4,
    patience=3,
    early_stopping=True,
    use_wandb=False,
    progress_callback=None,
):
    """
    Train the RNN model.

    Args:
        use_wandb: If True, logs metrics to wandb. Disabled by default for platform use.
        progress_callback: Optional callable(epoch, total_epochs, metrics) for status updates.
    """
    total_train_samples = len(train_loader.dataset)
    total_val_samples = len(val_loader.dataset)
    num_classes = len(labels_dict)
    train_class_distribution = dict(train_class_counts)
    val_class_distribution = dict(val_class_counts)

    if use_wandb:
        try:
            import wandb
            wandb.init(project="gavel-classifier", config={
                "epochs": epochs,
                "learning_rate": learning_rate,
                "train_samples": total_train_samples,
                "val_samples": total_val_samples,
                "num_classes": num_classes,
            })
            wandb.watch(model, log="all", log_freq=100)
        except ImportError:
            logger.warning("wandb not installed, skipping wandb logging")
            use_wandb = False

    print(f"Dataset Summary:", flush=True)
    print(f"- Total train samples: {total_train_samples}", flush=True)
    print(f"- Total validation samples: {total_val_samples}", flush=True)
    print(f"- Number of classes: {num_classes}", flush=True)
    print(f"- Train class distribution: {dict(train_class_counts)}", flush=True)
    print(f"- Validation class distribution: {dict(val_class_counts)}", flush=True)
    logger.info(f"Training: {total_train_samples} train, {total_val_samples} val, {num_classes} classes")

    from utils.device import get_torch_device
    device = get_torch_device()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # The reference trains at a CONSTANT lr — the reference has ReduceLROnPlateau
    # commented out. A scheduler here would halve the lr mid-run and diverge from
    # the reference's trained weights, so we omit it to keep parity.
    criterion = nn.BCEWithLogitsLoss()

    accuracy_metric = Accuracy(is_multilabel=True)
    confusion_matrix_metric = MultiLabelConfusionMatrix(num_classes=num_classes)

    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    global_step = 0
    prev_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        accuracy_metric.reset()
        confusion_matrix_metric.reset()

        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            train_outputs = model(batch_data)
            loss = criterion(train_outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probabilities = torch.sigmoid(train_outputs)
            predictions = (probabilities > 0.5).int()

            accuracy_metric.update((predictions, batch_labels.int()))
            confusion_matrix_metric.update((predictions, batch_labels.int()))

            global_step += 1
            if use_wandb:
                import wandb
                wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1}, step=global_step)

        avg_train_loss = total_loss / max(len(train_loader), 1)
        epoch_accuracy = accuracy_metric.compute()
        print(f"Epoch {epoch+1}/{epochs} completed. Overall Accuracy: {epoch_accuracy:.4f}, Avg Train Loss: {avg_train_loss:.4f}", flush=True)
        logger.info(f"Epoch {epoch+1}/{epochs} — Accuracy: {epoch_accuracy:.4f}, Loss: {avg_train_loss:.4f}")

        index_to_label = {v: k for k, v in labels_dict.items()}
        print("Evaluating on validation data...", flush=True)
        eval_metrics = evaluate_rnn_model(model, val_loader, num_classes, index_to_label)

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_accuracy": eval_metrics["accuracy"],
                "val_loss": eval_metrics["loss"],
            }, step=global_step)

        # No LR scheduler — constant lr, matching the reference.
        current_lr = optimizer.param_groups[0]['lr']

        if progress_callback:
            try:
                progress_callback(epoch + 1, epochs, {
                    "train_loss": avg_train_loss,
                    "val_loss": eval_metrics["loss"],
                    "val_accuracy": eval_metrics["accuracy"],
                    "learning_rate": current_lr,
                    "per_class": {
                        "precision": eval_metrics.get("precision", []),
                        "recall": eval_metrics.get("recall", []),
                        "f1": eval_metrics.get("f1", []),
                        "class_accuracy": eval_metrics.get("class_accuracy", []),
                    },
                })
            except Exception:
                pass

        checkpoint_path = os.path.join(checkpoint_dir, "best_rnn_model.pth")
        if eval_metrics["loss"] <= prev_val_loss and epoch >= 4:
            prev_val_loss = eval_metrics["loss"]
            torch.save(model.state_dict(), checkpoint_path)

        val_loss = eval_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs.")
                if os.path.exists(checkpoint_path):
                    state_dict = torch.load(checkpoint_path, weights_only=True)
                    model.load_state_dict(state_dict)
                break

    print("Training complete.", flush=True)
    logger.info("Training complete.")
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    # Parity with the reference (gavel/models/rnn.py): return the model
    # AS-IS. When all epochs run, that is the LAST-epoch weights; the best-val
    # checkpoint is reloaded ONLY inside the early-stopping branch above. We do NOT
    # restore the best checkpoint here — doing so was a divergence that turned tiny
    # epoch-level val-loss noise into discrete run-to-run weight swaps.
    return model
