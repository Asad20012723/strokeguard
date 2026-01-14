"""
PyTorch model definitions for stroke risk prediction.
Matches the architecture used during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageModel(nn.Module):
    """CNN-based image feature extractor for facial expression analysis."""

    def __init__(self, output_size: int):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(output_size, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class TabularModel(nn.Module):
    """Neural network for processing tabular health data."""

    def __init__(self, input_dim: int):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class MultimodalModel(nn.Module):
    """
    Multimodal model combining image and tabular features for stroke risk prediction.

    This architecture matches the trained model weights.
    """

    def __init__(self, input_dim: int = 9):
        super(MultimodalModel, self).__init__()
        # Image dimensions: 640x480 -> after 2 pooling: 160x120
        # Output size: 32 channels * 160 * 120 = 614400
        output_size = 32 * 120 * 160
        self.image_model = ImageModel(output_size=output_size)
        self.tabular_model = TabularModel(input_dim)
        self.fc1 = nn.Linear(128 + 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, images: torch.Tensor, tabular_data: torch.Tensor) -> torch.Tensor:
        image_features = self.image_model(images)
        tabular_features = self.tabular_model(tabular_data)
        combined = torch.cat((image_features, tabular_features), dim=1)
        x = F.relu(self.fc1(combined))
        x = torch.sigmoid(self.fc2(x))
        return x


def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to the .pth file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded PyTorch model in eval mode
    """
    model = MultimodalModel(input_dim=9)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Model weights loaded successfully from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load weights from {model_path}: {e}")
        print("Using randomly initialized weights")

    model.to(device)
    model.eval()

    return model
