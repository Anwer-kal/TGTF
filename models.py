"""Model definitions for CIFAR-10 and FEMNIST."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for 32x32 CIFAR-10 images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = resnet18(weights=None, num_classes=num_classes)
        # Adapt for 32x32: replace 7x7 stride-2 conv with 3x3 stride-1
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)


class FEMNISTCNN(nn.Module):
    """2-conv CNN for FEMNIST (1x28x28 grayscale)."""

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def get_model(dataset: str, num_classes: int):
    if dataset.lower() == "cifar10":
        return ResNet18CIFAR(num_classes=num_classes)
    elif dataset.lower() == "femnist":
        return FEMNISTCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor):
    """Load a flat parameter vector into a model."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view_as(p.data))
        offset += numel
