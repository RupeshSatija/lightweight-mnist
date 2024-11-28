import torch
from torchvision import datasets, transforms


def get_mnist_dataset(root: str, train: bool = True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    target_transform = transforms.Lambda(lambda y: torch.tensor(y))

    return datasets.MNIST(
        root=root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
