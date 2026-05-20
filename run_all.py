"""Run the full concrete inspection prototype from PyCharm or terminal.

This script is designed for beginners who want one clear entry point instead
of running notebook cells one by one. It runs:

1. Dataset loading and summary
2. Crack detector training and evaluation
3. RL inspection planner simulation
4. Output/model file generation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data_loader import create_train_val_test_split, load_image_paths, summarize_class_distribution
from src.evaluate_model import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_prediction_examples,
    predict_on_loader,
    save_classification_report,
)
from src.q_learning_agent import QLearningAgent
from src.rl_environment import InspectionEnvironment
from src.train_crack_detector import (
    ConcreteCrackDataset,
    build_model,
    count_trainable_parameters,
    save_model,
    train_model,
)
from src.visualization import (
    plot_crack_risk_map,
    plot_inspection_route,
    plot_sample_images,
    plot_training_history,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "concrete_crack"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    """Parse beginner-friendly command-line options."""

    parser = argparse.ArgumentParser(description="Run the full concrete inspection prototype.")
    parser.add_argument("--image-size", type=int, default=128, help="Image resize height and width.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of CNN training epochs.")
    parser.add_argument("--samples-per-class", type=int, default=1000, help="Subset size per class.")
    parser.add_argument("--use-full-dataset", action="store_true", help="Use all images instead of subset mode.")
    parser.add_argument("--grid-size", type=int, default=6, help="Grid size for RL simulation.")
    parser.add_argument("--rl-episodes", type=int, default=1000, help="Number of Q-learning episodes.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    """Make the run more reproducible."""

    np.random.seed(seed)
    torch.manual_seed(seed)


def run_crack_detection(args: argparse.Namespace) -> None:
    """Train and evaluate the vision-based crack detector."""

    print("\n=== Crack Detection Pipeline ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    use_subset = not args.use_full_dataset
    image_size = (args.image_size, args.image_size)

    print(f"Dataset directory: {DATA_DIR}")
    print(f"Subset mode: {use_subset}")
    print(f"Samples per class: {args.samples_per_class if use_subset else 'all'}")

    df = load_image_paths(
        DATA_DIR,
        use_subset=use_subset,
        samples_per_class=args.samples_per_class,
        random_seed=args.random_seed,
    )

    class_distribution = summarize_class_distribution(df)
    class_distribution_path = OUTPUT_DIR / "class_distribution.csv"
    class_distribution.to_csv(class_distribution_path, index=False)

    print(f"Images used: {len(df)}")
    print(class_distribution.to_string(index=False))
    print(f"Saved class distribution: {class_distribution_path}")

    sample_path = plot_sample_images(
        dataframe=df,
        output_path=OUTPUT_DIR / "sample_images.png",
        num_per_class=4,
        random_seed=args.random_seed,
    )
    print(f"Saved sample image grid: {sample_path}")

    train_df, val_df, test_df = create_train_val_test_split(
        df,
        val_size=0.15,
        test_size=0.15,
        random_seed=args.random_seed,
    )
    print(f"Train/val/test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_loader = DataLoader(
        ConcreteCrackDataset(train_df, transform=train_transform),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ConcreteCrackDataset(val_df, transform=eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        ConcreteCrackDataset(test_df, transform=eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name="simple_cnn", num_classes=2, pretrained=False).to(device)
    print(f"Training device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
    )

    history_csv = OUTPUT_DIR / "training_history.csv"
    pd.DataFrame(history).to_csv(history_csv, index=False)
    history_png = plot_training_history(history, OUTPUT_DIR / "training_history.png")
    print(f"Saved training history CSV: {history_csv}")
    print(f"Saved training history plot: {history_png}")

    y_true, y_pred, _ = predict_on_loader(model, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred)
    report_path = save_classification_report(
        y_true,
        y_pred,
        OUTPUT_DIR / "classification_report.txt",
        target_names=("no_crack", "crack"),
    )
    confusion_path = plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=("no_crack", "crack"),
        output_path=OUTPUT_DIR / "confusion_matrix.png",
    )
    predictions_path = plot_prediction_examples(
        model=model,
        dataframe=test_df,
        transform=eval_transform,
        device=device,
        output_path=OUTPUT_DIR / "prediction_examples.png",
        num_images=8,
        random_seed=args.random_seed,
    )

    print("\nFinal test metrics")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Saved classification report: {report_path}")
    print(f"Saved confusion matrix: {confusion_path}")
    print(f"Saved prediction examples: {predictions_path}")

    model_path = save_model(
        model=model,
        model_path=MODEL_DIR / "crack_detector.pt",
        image_size=image_size,
        model_name="simple_cnn",
        class_names=("no_crack", "crack"),
        extra_metadata={"source": "run_all.py"},
    )
    print(f"Saved trained model: {model_path}")


def create_simulated_risk_map(grid_size: int, random_seed: int) -> np.ndarray:
    """Create the same kind of crack-risk map used in the RL notebook."""

    rng = np.random.default_rng(random_seed)
    risk_map = rng.uniform(0.05, 0.45, size=(grid_size, grid_size))
    hotspot_count = max(3, grid_size // 2)
    hotspot_indices = rng.choice(grid_size * grid_size, size=hotspot_count, replace=False)

    for index in hotspot_indices:
        row, col = divmod(int(index), grid_size)
        risk_map[row, col] = rng.uniform(0.75, 1.0)

    return risk_map


def run_rl_planner(args: argparse.Namespace) -> None:
    """Train the Q-learning inspection planner and save visual outputs."""

    print("\n=== RL Inspection Planner ===")
    risk_map = create_simulated_risk_map(args.grid_size, args.random_seed)

    risk_map_path = plot_crack_risk_map(
        risk_map=risk_map,
        output_path=OUTPUT_DIR / "crack_risk_map.png",
        title="Simulated Crack Risk Map",
    )
    print(f"Saved crack risk map: {risk_map_path}")

    environment = InspectionEnvironment(risk_map=risk_map, high_risk_threshold=0.65)
    agent = QLearningAgent(
        num_actions=environment.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.30,
        epsilon_decay=0.995,
        min_epsilon=0.02,
        random_seed=args.random_seed,
    )

    progress_interval = max(1, args.rl_episodes // 10)
    rewards = agent.train(
        environment=environment,
        num_episodes=args.rl_episodes,
        progress_interval=progress_interval,
    )

    reward_series = pd.Series(rewards)
    rolling_rewards = reward_series.rolling(window=50, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(reward_series.values, alpha=0.35, label="Episode reward")
    ax.plot(rolling_rewards.values, linewidth=2, label="50-episode rolling average")
    ax.set_title("Q-learning Training Rewards")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend()
    fig.tight_layout()
    rewards_path = OUTPUT_DIR / "rl_training_rewards.png"
    fig.savefig(rewards_path, dpi=150)
    plt.close(fig)
    print(f"Saved RL training rewards: {rewards_path}")

    route_environment = InspectionEnvironment(risk_map=risk_map, high_risk_threshold=0.65)
    route, actions = agent.generate_route(route_environment)
    route_path = plot_inspection_route(
        risk_map=risk_map,
        route=route,
        output_path=OUTPUT_DIR / "inspection_route.png",
        inspected_cells=route_environment.inspected,
        title="Learned Inspection Route",
    )

    print(f"Generated route steps: {len(actions)}")
    print(f"First actions: {actions[:10]}")
    print(f"Saved inspection route: {route_path}")


def main() -> None:
    """Run all project stages."""

    args = parse_args()
    set_random_seeds(args.random_seed)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Place the dataset under data/raw/concrete_crack/ before running."
        )

    print("Autonomous Concrete Infrastructure Inspection Robot")
    print(f"Project root: {PROJECT_ROOT}")

    run_crack_detection(args)
    run_rl_planner(args)

    print("\n=== Done ===")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(f"Model saved in: {MODEL_DIR / 'crack_detector.pt'}")


if __name__ == "__main__":
    main()

