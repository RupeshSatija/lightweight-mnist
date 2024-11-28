from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config.config import TrainingConfig
from src.models.mnist_cnn import MnistCNN


class Trainer:
    def __init__(self, model: MnistCNN, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=938,
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = correct / len(test_loader.dataset)
        return test_loss, accuracy
