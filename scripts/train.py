import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.dataset import get_mnist_dataset
from src.models.mnist_cnn import MnistCNN
from src.training.trainer import Trainer


def main():
    # Setup directories
    Config.setup_directories()

    # Set device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    # Data loading using our custom dataset implementation
    train_dataset = get_mnist_dataset(root="data", train=True, augment=True)
    val_dataset = get_mnist_dataset(root="data", train=False, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

    # Model, criterion, optimizer
    model = MnistCNN(Config.get_instance()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Add parameter count check
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable parameters: {count_parameters(model)}")

    # Training
    trainer = Trainer(model, criterion, optimizer, device)
    trainer.train(train_loader, val_loader, Config.NUM_EPOCHS)

    # Save augmentation samples
    from src.data.dataset import visualize_augmentations

    visualize_augmentations()


if __name__ == "__main__":
    main()
