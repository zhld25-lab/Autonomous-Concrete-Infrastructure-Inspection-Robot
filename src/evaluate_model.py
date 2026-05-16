"""Evaluation utilities for the crack detection model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch


def predict_on_loader(model, dataloader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict labels and crack probabilities for a DataLoader."""

    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)


def calculate_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, object]:
    """Calculate standard binary classification metrics."""

    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def save_classification_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    output_path: str | Path,
    target_names: Iterable[str] = ("no_crack", "crack"),
) -> Path:
    """Save a text classification report."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_true, y_pred, target_names=list(target_names), zero_division=0)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Iterable[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Plot and save a confusion matrix."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    class_names = list(class_names)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_prediction_examples(
    model,
    dataframe,
    transform,
    device: torch.device,
    output_path: str | Path,
    num_images: int = 8,
    random_seed: int = 42,
) -> Path:
    """Plot test images with true label, predicted label, and crack probability."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_df = dataframe.sample(n=min(num_images, len(dataframe)), random_state=random_seed).reset_index(drop=True)
    columns = min(4, len(sample_df))
    rows = int(np.ceil(len(sample_df) / max(columns, 1)))

    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.array(axes).reshape(-1)

    model.eval()
    with torch.no_grad():
        for index, (_, row) in enumerate(sample_df.iterrows()):
            image = Image.open(row["filepath"]).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probability = torch.softmax(logits, dim=1)[0, 1].item()
            prediction = 1 if probability >= 0.5 else 0

            axes[index].imshow(image)
            axes[index].axis("off")
            axes[index].set_title(
                f"True: {row['class_name']}\n"
                f"Pred: {'crack' if prediction else 'no_crack'}\n"
                f"Crack prob: {probability:.2f}"
            )

    for index in range(len(sample_df), len(axes)):
        axes[index].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

