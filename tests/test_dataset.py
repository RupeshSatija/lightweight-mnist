import torch
from torch.utils.data import DataLoader

from src.data.dataset import get_mnist_dataset


def test_mnist_dataset_shapes():
    """Test if MNIST dataset returns correct shapes"""
    train_dataset = get_mnist_dataset(root="./data", train=True)
    test_dataset = get_mnist_dataset(root="./data", train=False)

    # Get first item
    train_img, train_label = train_dataset[0]
    test_img, test_label = test_dataset[0]

    # Check shapes
    assert train_img.shape == (1, 28, 28)  # MNIST images are 28x28 with 1 channel
    assert test_img.shape == (1, 28, 28)
    assert isinstance(train_label, torch.Tensor)
    assert isinstance(test_label, torch.Tensor)


def test_mnist_dataloader():
    """Test if DataLoader works correctly with the dataset"""
    batch_size = 32
    train_dataset = get_mnist_dataset(root="./data", train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Get first batch
    images, labels = next(iter(train_loader))

    # Check batch shapes
    assert images.shape == (batch_size, 1, 28, 28)
    assert labels.shape == (batch_size,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long
