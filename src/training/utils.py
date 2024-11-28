import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class MetricsTracker:
    def __init__(self, metrics_dir: Path, experiment_name: str = None):
        self.metrics_dir = metrics_dir
        self.experiment_name = experiment_name or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.metrics = {
            "train": {"loss": [], "accuracy": []},
            "val": {"loss": [], "accuracy": []},
        }

    def update(self, phase, loss, accuracy, epoch):
        """Update metrics for the current epoch."""
        self.metrics[phase]["loss"].append(loss)
        self.metrics[phase]["accuracy"].append(accuracy)

    def save_metrics(self):
        """Save metrics to CSV and JSON files."""
        # Save as CSV
        df = pd.DataFrame(
            {
                "epoch": range(1, len(self.metrics["train"]["loss"]) + 1),
                "train_loss": self.metrics["train"]["loss"],
                "train_accuracy": self.metrics["train"]["accuracy"],
                "val_loss": self.metrics["val"]["loss"],
                "val_accuracy": self.metrics["val"]["accuracy"],
            }
        )

        csv_path = self.metrics_dir / f"{self.experiment_name}_metrics.csv"
        df.to_csv(csv_path, index=False)

        # Save as JSON
        json_path = self.metrics_dir / f"{self.experiment_name}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def plot_metrics(self, plots_dir: Path):
        """Plot training and validation metrics."""
        epochs = range(1, len(self.metrics["train"]["loss"]) + 1)

        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.metrics["train"]["loss"], "b-", label="Training Loss")
        plt.plot(epochs, self.metrics["val"]["loss"], "r-", label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(
            epochs, self.metrics["train"]["accuracy"], "b-", label="Training Accuracy"
        )
        plt.plot(
            epochs, self.metrics["val"]["accuracy"], "r-", label="Validation Accuracy"
        )
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(plots_dir / f"{self.experiment_name}_metrics.png")
        plt.close()
