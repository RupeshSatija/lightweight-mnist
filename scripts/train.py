import os
import sys

from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config.config import ModelConfig, TrainingConfig, get_config
from src.data.dataset import get_mnist_dataset
from src.models.mnist_cnn import MnistCNN
from src.training.trainer import Trainer


def main():
    config_dict = get_config()
    model_config = ModelConfig(
        **{k: config_dict[k] for k in ModelConfig.__annotations__}
    )
    training_config = TrainingConfig(
        **{k: config_dict[k] for k in TrainingConfig.__annotations__}
    )

    model = MnistCNN(model_config)
    print(f"Total trainable parameters: {model.count_parameters():,}")

    train_dataset = get_mnist_dataset(root="./data", train=True)
    test_dataset = get_mnist_dataset(root="./data", train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=training_config.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=training_config.batch_size, shuffle=False
    )

    trainer = Trainer(model, training_config)

    for epoch in range(1, training_config.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        test_loss, accuracy = trainer.evaluate(test_loader)
        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
