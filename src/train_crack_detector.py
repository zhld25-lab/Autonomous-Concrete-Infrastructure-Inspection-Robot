"""Training utilities for the concrete crack detector."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class ConcreteCrackDataset(Dataset):
    """PyTorch Dataset that reads image paths from a DataFrame."""

    def __init__(self, dataframe, transform=None, path_column: str = "filepath", label_column: str = "label"):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.path_column = path_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image = Image.open(row[self.path_column]).convert("RGB")
        label = int(row[self.label_column])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class SimpleCrackCNN(nn.Module):
    """Small CNN designed to train quickly on a laptop."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(model_name: str = "simple_cnn", num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """Build either a small CNN or a ResNet18 classifier."""

    model_name = model_name.lower()

    if model_name == "simple_cnn":
        return SimpleCrackCNN(num_classes=num_classes)

    if model_name == "resnet18":
        from torchvision import models

        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=pretrained)

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError("model_name must be either 'simple_cnn' or 'resnet18'.")


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    optimizer=None,
) -> Dict[str, float]:
    """Run one training or evaluation epoch."""

    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_labels: List[int] = []
    all_predictions: List[int] = []

    with torch.set_grad_enabled(is_training):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = torch.argmax(logits, dim=1)
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_predictions.extend(predictions.detach().cpu().numpy().tolist())

    average_loss = total_loss / max(len(dataloader.dataset), 1)
    accuracy = float(np.mean(np.array(all_labels) == np.array(all_predictions)))
    f1 = float(f1_score(all_labels, all_predictions, zero_division=0))

    return {"loss": average_loss, "accuracy": accuracy, "f1": f1}


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device: torch.device,
    epochs: int = 3,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """Train a model and keep the best validation-F1 weights."""

    history: List[Dict[str, float]] = []
    best_state = deepcopy(model.state_dict())
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_one_epoch(model, val_loader, criterion, device, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = deepcopy(model.state_dict())

        tqdm.write(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_f1={row['val_f1']:.4f}"
        )

    model.load_state_dict(best_state)
    return model, history


def save_model(
    model: nn.Module,
    model_path: str | Path,
    image_size: Iterable[int] = (128, 128),
    model_name: str = "simple_cnn",
    class_names: Iterable[str] = ("no_crack", "crack"),
    extra_metadata: Dict | None = None,
) -> Path:
    """Save model weights plus small metadata needed for inference."""

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "image_size": tuple(image_size),
        "class_names": tuple(class_names),
    }

    if extra_metadata:
        checkpoint.update(extra_metadata)

    torch.save(checkpoint, model_path)
    return model_path

