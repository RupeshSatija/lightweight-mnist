import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Efficient architecture with fewer parameters
        self.conv1 = nn.Conv2d(
            1, 8, kernel_size=3, padding=1
        )  # 8 filters instead of more
        self.conv2 = nn.Conv2d(
            8, 16, kernel_size=3, padding=1
        )  # 16 filters to keep it light
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(16 * 7 * 7, 10)  # Reduced final layer size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc(x)
        return x
