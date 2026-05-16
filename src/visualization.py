"""Visualization helpers for the project notebooks and Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def _prepare_output_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_sample_images(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    num_per_class: int = 4,
    random_seed: int = 42,
) -> Path:
    """Plot a few crack and no-crack images."""

    output_path = _prepare_output_path(output_path)
    samples = []

    for class_name in ["crack", "no_crack"]:
        class_df = dataframe[dataframe["class_name"] == class_name]
        samples.append(class_df.sample(n=min(num_per_class, len(class_df)), random_state=random_seed))

    sample_df = pd.concat(samples, ignore_index=True)
    columns = num_per_class
    rows = 2
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 7))
    axes = np.array(axes).reshape(rows, columns)

    for row_index, class_name in enumerate(["crack", "no_crack"]):
        class_samples = sample_df[sample_df["class_name"] == class_name].reset_index(drop=True)
        for col_index in range(columns):
            ax = axes[row_index, col_index]
            ax.axis("off")
            if col_index < len(class_samples):
                image = Image.open(class_samples.loc[col_index, "filepath"]).convert("RGB")
                ax.imshow(image)
                ax.set_title(class_name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_training_history(history: Iterable[dict], output_path: str | Path) -> Path:
    """Plot loss, accuracy, and F1-score over epochs."""

    output_path = _prepare_output_path(output_path)
    history_df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], marker="o", label="Train")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], marker="o", label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history_df["epoch"], history_df["train_f1"], marker="o", label="Train")
    axes[2].plot(history_df["epoch"], history_df["val_f1"], marker="o", label="Validation")
    axes[2].set_title("F1-score")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_crack_risk_map(risk_map, output_path: str | Path, title: str = "Simulated Crack Risk Map") -> Path:
    """Plot a crack-risk heatmap."""

    output_path = _prepare_output_path(output_path)
    risk_map = np.asarray(risk_map)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(risk_map, cmap="YlOrRd", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Crack risk")
    ax.set_title(title)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")

    for row in range(risk_map.shape[0]):
        for col in range(risk_map.shape[1]):
            ax.text(col, row, f"{risk_map[row, col]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_inspection_route(
    risk_map,
    route: List[Tuple[int, int]],
    output_path: str | Path,
    inspected_cells=None,
    title: str = "Learned Inspection Route",
) -> Path:
    """Plot the learned route on top of the risk map."""

    output_path = _prepare_output_path(output_path)
    risk_map = np.asarray(risk_map)
    route_array = np.array(route)

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(risk_map, cmap="YlOrRd", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Crack risk")

    ax.plot(route_array[:, 1], route_array[:, 0], color="navy", linewidth=2, marker="o", markersize=4)
    ax.scatter(route_array[0, 1], route_array[0, 0], s=120, color="limegreen", edgecolor="black", label="Start")
    ax.scatter(route_array[-1, 1], route_array[-1, 0], s=120, color="dodgerblue", edgecolor="black", label="End")

    if inspected_cells is not None:
        inspected_cells = np.asarray(inspected_cells)
        inspected_positions = np.argwhere(inspected_cells)
        if len(inspected_positions) > 0:
            ax.scatter(
                inspected_positions[:, 1],
                inspected_positions[:, 0],
                marker="s",
                s=220,
                facecolors="none",
                edgecolors="black",
                linewidths=2,
                label="Inspected",
            )

    ax.set_xticks(range(risk_map.shape[1]))
    ax.set_yticks(range(risk_map.shape[0]))
    ax.set_title(title)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.grid(color="white", linewidth=1.2)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

