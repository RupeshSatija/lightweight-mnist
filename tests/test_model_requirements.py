import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.dataset import get_mnist_dataset
from src.models.mnist_cnn import MnistCNN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_size():
    """Test if model has less than 25000 parameters"""
    config = Config.get_instance()
    model = MnistCNN(config)
    param_count = count_parameters(model)
    print(f"Model has {param_count} parameters")
    assert (
        param_count < 25000
    ), f"Model has {param_count} parameters, should be less than 25000"


def test_model_accuracy():
    """Test if model achieves >95% accuracy in 1 epoch"""
    # Setup
    config = Config.get_instance()
    model = MnistCNN(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Get data
    train_dataset = get_mnist_dataset(root="./data", train=True)
    test_dataset = get_mnist_dataset(root="./data", train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}"
            )

    # Evaluate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Final accuracy: {accuracy}%")
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be >95%"
