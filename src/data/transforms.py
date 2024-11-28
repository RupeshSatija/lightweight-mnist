import torch
from torchvision import transforms


def get_mnist_transforms(train=True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    target_transform = transforms.Compose(
        [transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))]
    )

    if train:
        return transform, target_transform
    else:
        return transform, None  # For test data, we might not need target transform
