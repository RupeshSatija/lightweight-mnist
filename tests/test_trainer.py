import pytest
from torch.utils.data import DataLoader

from src.config.config import ModelConfig, TrainingConfig
from src.data.dataset import get_mnist_dataset
from src.models.mnist_cnn import MnistCNN
from src.training.trainer import Trainer


@pytest.fixture
def model_config():
    return ModelConfig(
        input_channels=1,
        hidden_channels=32,
        num_classes=10,
    )


@pytest.fixture
def training_config():
    return TrainingConfig(
        learning_rate=0.01,
        batch_size=32,
        epochs=1,
    )


@pytest.fixture
def model(model_config):
    return MnistCNN(model_config)


@pytest.fixture
def trainer(model, training_config):
    return Trainer(model, training_config)


@pytest.fixture
def train_loader():
    dataset = get_mnist_dataset(root="./data", train=True)
    return DataLoader(dataset, batch_size=32, shuffle=True)


@pytest.fixture
def test_loader():
    dataset = get_mnist_dataset(root="./data", train=False)
    return DataLoader(dataset, batch_size=32, shuffle=False)


def test_trainer_initialization(trainer):
    """Test if trainer initializes correctly"""
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.criterion is not None


def test_training_step(trainer, train_loader):
    """Test if a single training step works"""
    # Get a single batch
    images, labels = next(iter(train_loader))

    # Move to same device as model
    images = images.to(trainer.device)
    labels = labels.to(trainer.device)

    # Forward pass
    outputs = trainer.model(images)

    # Check output shape
    assert outputs.shape == (images.shape[0], 10)  # 10 classes for MNIST


def test_evaluation(trainer, test_loader):
    """Test if evaluation works"""
    loss, accuracy = trainer.evaluate(test_loader)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1  # Accuracy should be between 0 and 1


def test_full_training_loop(trainer, train_loader, test_loader):
    """Test if a full training loop can run without errors"""
    # Run one epoch
    train_loss = trainer.train_epoch(train_loader)
    test_loss, accuracy = trainer.evaluate(test_loader)

    assert isinstance(train_loss, float)
    assert isinstance(test_loss, float)
    assert isinstance(accuracy, float)
