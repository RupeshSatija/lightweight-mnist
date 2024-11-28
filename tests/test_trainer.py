import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config.config import Config
from src.models.mnist_cnn import MnistCNN
from src.training.trainer import Trainer


@pytest.fixture
def config():
    return Config.get_instance()


@pytest.fixture
def model(config):
    return MnistCNN(config)


@pytest.fixture
def trainer(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    return Trainer(model, criterion, optimizer, device)


@pytest.fixture
def train_loader():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)


@pytest.fixture
def test_loader():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("data", train=False, transform=transform)
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE)


def test_trainer_initialization(trainer):
    """Test if trainer initializes correctly"""
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.criterion is not None


def test_training_step(trainer, train_loader):
    """Test if a single training step works"""
    images, labels = next(iter(train_loader))

    images = images.to(trainer.device)
    labels = labels.to(trainer.device)

    outputs = trainer.model(images)
    assert outputs.shape == (images.shape[0], Config.NUM_CLASSES)


def test_full_training_loop(trainer, train_loader, test_loader):
    """Test if a full training loop can run without errors"""
    trainer.train(train_loader, test_loader, num_epochs=1)
