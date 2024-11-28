import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.dataset import get_mnist_dataset
from src.models.mnist_cnn import MnistCNN
from src.training.trainer import Trainer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_size():
    """Test if model has less than 25000 parameters"""
    config = Config.get_instance()
    model = MnistCNN(config)
    param_count = count_parameters(model)
    assert (
        param_count < 25000
    ), f"Model has {param_count} parameters, should be less than 25000"


def test_model_accuracy():
    """Test if model achieves >95% accuracy in 1 epoch"""
    # Setup
    config = Config.get_instance()
    model = MnistCNN(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    trainer = Trainer(model, criterion, optimizer, device)

    # Get data
    train_dataset = get_mnist_dataset(root="./data", train=True)
    test_dataset = get_mnist_dataset(root="./data", train=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    # Train for 1 epoch
    trainer.train(train_loader, test_loader, num_epochs=1)

    # Evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be >95%"
