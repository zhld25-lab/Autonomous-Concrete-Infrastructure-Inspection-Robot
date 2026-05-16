"""Dataset loading utilities for concrete crack image classification.

The project supports both common layouts for the Concrete Crack Images for
Classification dataset:

data/raw/concrete_crack/Positive and data/raw/concrete_crack/Negative
data/raw/concrete_crack/crack and data/raw/concrete_crack/no_crack
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def detect_dataset_folders(data_dir: str | Path) -> Dict[str, Path]:
    """Detect crack and no-crack folders using supported folder names.

    Args:
        data_dir: Path to data/raw/concrete_crack.

    Returns:
        Dictionary with keys "crack" and "no_crack" mapped to folder paths.

    Raises:
        FileNotFoundError: If data_dir does not exist.
        ValueError: If a supported pair of class folders cannot be found.
    """

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "Download the dataset and place it under data/raw/concrete_crack/."
        )

    folders = {folder.name.lower(): folder for folder in data_dir.iterdir() if folder.is_dir()}

    crack_candidates = ["positive", "crack"]
    no_crack_candidates = ["negative", "no_crack", "no-crack", "nocrack"]

    crack_folder = next((folders[name] for name in crack_candidates if name in folders), None)
    no_crack_folder = next((folders[name] for name in no_crack_candidates if name in folders), None)

    if crack_folder is None or no_crack_folder is None:
        available = ", ".join(sorted(folders)) or "no class folders found"
        raise ValueError(
            "Could not detect supported dataset structure. Expected either "
            "Positive/Negative or crack/no_crack under "
            f"{data_dir}. Available folders: {available}"
        )

    return {"crack": crack_folder, "no_crack": no_crack_folder}


def list_image_files(folder: str | Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[Path]:
    """Return all image files from a folder, including nested image files."""

    folder = Path(folder)
    normalized_extensions = {extension.lower() for extension in extensions}
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in normalized_extensions
    )


def load_image_paths(
    data_dir: str | Path,
    use_subset: bool = False,
    samples_per_class: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load image paths and labels into a pandas DataFrame.

    Label convention:
        no_crack = 0
        crack = 1
    """

    class_folders = detect_dataset_folders(data_dir)
    rows = []

    for class_name, label in [("no_crack", 0), ("crack", 1)]:
        image_paths = list_image_files(class_folders[class_name])
        class_df = pd.DataFrame(
            {
                "filepath": [str(path) for path in image_paths],
                "label": label,
                "class_name": class_name,
            }
        )

        if use_subset and samples_per_class is not None and len(class_df) > samples_per_class:
            class_df = class_df.sample(n=samples_per_class, random_state=random_seed)

        rows.append(class_df)

    df = pd.concat(rows, ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


def create_train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train, validation, and test splits."""

    if "label" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'label' column.")

    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1.")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_seed,
    )

    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def summarize_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Create a class-count summary table."""

    summary = (
        df.groupby(["class_name", "label"])
        .size()
        .reset_index(name="count")
        .sort_values("label")
        .reset_index(drop=True)
    )
    return summary

