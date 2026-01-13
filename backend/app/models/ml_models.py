"""
PyTorch model definitions for stroke risk prediction.
Contains both CNN-based and Transformer-based architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageModel(nn.Module):
    """CNN-based image feature extractor for facial expression analysis."""

    def __init__(self, input_channels: int = 12, output_size: int = 128):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.3)

        # Calculate flattened size after conv layers
        # Input: 224x224 -> pool -> 112x112 -> pool -> 56x56 -> pool -> 28x28
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class TabularModel(nn.Module):
    """Neural network for processing tabular health data."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super(TabularModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return x


class MultimodalModel(nn.Module):
    """
    Multimodal model combining image and tabular features for stroke risk prediction.

    Architecture:
    - ImageModel: Processes 4 stacked facial expression images (12 channels = 4 images x 3 RGB)
    - TabularModel: Processes health metrics (age, BP, glucose, etc.)
    - Fusion layer: Concatenates features and produces risk score
    """

    def __init__(self, tabular_input_dim: int = 8):
        super(MultimodalModel, self).__init__()
        self.image_model = ImageModel(input_channels=12, output_size=128)
        self.tabular_model = TabularModel(input_dim=tabular_input_dim, output_dim=32)

        # Fusion layers
        self.fusion_fc1 = nn.Linear(128 + 32, 64)
        self.fusion_fc2 = nn.Linear(64, 32)
        self.classifier = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(
        self, images: torch.Tensor, tabular_data: torch.Tensor
    ) -> torch.Tensor:
        # Extract features from both modalities
        image_features = self.image_model(images)
        tabular_features = self.tabular_model(tabular_data)

        # Concatenate features
        combined = torch.cat((image_features, tabular_features), dim=1)

        # Fusion and classification
        x = F.relu(self.fusion_fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fusion_fc2(x))
        x = self.classifier(x)

        return x


class MultimodalModelLegacy(nn.Module):
    """
    Legacy model architecture for backward compatibility with existing weights.
    Uses larger image dimensions (640x480).
    """

    def __init__(self, tabular_input_dim: int = 8):
        super(MultimodalModelLegacy, self).__init__()

        # Image model for 640x480 images
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 2 pooling: 640/4=160, 480/4=120 -> 32 * 160 * 120
        self.image_fc = nn.Linear(32 * 160 * 120, 128)

        # Tabular model
        self.tab_fc1 = nn.Linear(tabular_input_dim, 64)
        self.tab_fc2 = nn.Linear(64, 32)
        self.tab_fc3 = nn.Linear(32, 16)

        # Fusion
        self.fusion_fc1 = nn.Linear(128 + 16, 64)
        self.fusion_fc2 = nn.Linear(64, 1)

    def forward(
        self, images: torch.Tensor, tabular_data: torch.Tensor
    ) -> torch.Tensor:
        # Image branch
        x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        image_features = F.relu(self.image_fc(x))

        # Tabular branch
        t = F.relu(self.tab_fc1(tabular_data))
        t = F.relu(self.tab_fc2(t))
        tabular_features = F.relu(self.tab_fc3(t))

        # Fusion
        combined = torch.cat((image_features, tabular_features), dim=1)
        x = F.relu(self.fusion_fc1(combined))
        x = torch.sigmoid(self.fusion_fc2(x))

        return x


def load_model(
    model_path: str, device: str = "cpu", legacy: bool = True
) -> nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to the .pth file
        device: Device to load model on ('cpu' or 'cuda')
        legacy: Whether to use legacy architecture (for existing weights)

    Returns:
        Loaded PyTorch model in eval mode
    """
    if legacy:
        model = MultimodalModelLegacy(tabular_input_dim=8)
    else:
        model = MultimodalModel(tabular_input_dim=8)

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Could not load weights from {model_path}: {e}")
        print("Using randomly initialized weights")

    model.to(device)
    model.eval()

    return model
