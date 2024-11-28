import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


def get_mnist_dataset(root: str, train: bool = True, augment: bool = True):
    # Base transforms
    base_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Augmentation transforms for training
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    target_transform = transforms.Lambda(lambda y: torch.tensor(y))

    return datasets.MNIST(
        root=root,
        train=train,
        transform=train_transform if (train and augment) else base_transform,
        target_transform=target_transform,
        download=True,
    )


def visualize_augmentations(num_samples: int = 5):
    """Visualize original and augmented images side by side"""
    # Get datasets with and without augmentation
    dataset_orig = get_mnist_dataset("./data", train=True, augment=False)
    dataset_aug = get_mnist_dataset("./data", train=True, augment=True)

    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 2 * num_samples))
    fig.suptitle("Original vs Augmented Images")

    for i in range(num_samples):
        # Get the same image from both datasets
        img_orig, label = dataset_orig[i]
        img_aug, _ = dataset_aug[i]

        # Convert tensors to numpy arrays and denormalize
        img_orig = img_orig.numpy()[0]  # Remove channel dimension
        img_aug = img_aug.numpy()[0]

        # Plot original
        axes[i, 0].imshow(img_orig, cmap="gray")
        axes[i, 0].set_title(f"Original (Label: {label})")
        axes[i, 0].axis("off")

        # Plot augmented
        axes[i, 1].imshow(img_aug, cmap="gray")
        axes[i, 1].set_title("Augmented")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/augmentation_samples.png")
    plt.close()
