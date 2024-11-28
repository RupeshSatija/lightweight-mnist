import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.config import Config
from .utils import MetricsTracker


class Trainer:
    def __init__(self, model: nn.Module, criterion, optimizer, device: str):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics_tracker = MetricsTracker(Config.METRICS_DIR)

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {
                    "loss": running_loss / len(train_loader),
                    "acc": 100.0 * correct / total,
                }
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        self.metrics_tracker.update("train", epoch_loss, epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def validate(self, val_loader: DataLoader, epoch: int):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total
        self.metrics_tracker.update("val", epoch_loss, epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader, epoch)

            print(f"\nEpoch {epoch}:")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    self.model.state_dict(), Config.CHECKPOINTS_DIR / "best_model.pth"
                )

            # Save metrics and plots
            self.metrics_tracker.save_metrics()
            self.metrics_tracker.plot_metrics(Config.PLOTS_DIR)
