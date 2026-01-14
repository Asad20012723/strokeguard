"""
PyTorch model definitions for stroke risk prediction.
Includes both Teacher and Student models for Knowledge Distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ------------------------------ Image Model with Transfer Learning ------------------------------ #
class ImageModelTL(nn.Module):
    """ResNet50-based image feature extractor for Teacher model."""

    def __init__(self):
        super(ImageModelTL, self).__init__()
        try:
            import torchvision.models as models
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

            # Freeze early layers
            for param in base_model.parameters():
                param.requires_grad = False

            # Modify last layer to output 128 features
            base_model.fc = nn.Linear(base_model.fc.in_features, 128)
            self.model = base_model
            self.has_resnet = True
        except Exception:
            # Fallback if torchvision not available
            self.has_resnet = False
            self.fc = nn.Linear(3 * 224 * 224, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected shape: [batch, 4, 3, 224, 224] or [batch, 12, H, W]
        if x.dim() == 5:
            batch_size, num_images, channels, height, width = x.shape
            x = x.view(batch_size * num_images, channels, height, width)

            if self.has_resnet:
                features = self.model(x)  # [batch * 4, 128]
            else:
                x = x.view(x.size(0), -1)
                features = F.relu(self.fc(x))

            features = features.view(batch_size, num_images, -1)  # [batch, 4, 128]
            return torch.mean(features, dim=1)  # Aggregate: [batch, 128]
        else:
            # Handle [batch, 12, H, W] format (stacked images)
            batch_size = x.size(0)
            # Reshape to 4 images of 3 channels each
            x = x.view(batch_size, 4, 3, x.size(2), x.size(3))
            return self.forward(x)


# ------------------------------ Lightweight Image Model for Student ------------------------------ #
class ImageModel(nn.Module):
    """Lightweight CNN-based image feature extractor for Student model."""

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


# ------------------------------ Tabular Data Model ------------------------------ #
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


# ------------------------------ Multimodal Teacher Model ------------------------------ #
class MultimodalTeacher(nn.Module):
    """
    Teacher model using ResNet50 for images + FC for tabular data.
    Used to train the lighter Student model via Knowledge Distillation.
    """

    def __init__(self, tabular_dim: int = 9):
        super(MultimodalTeacher, self).__init__()
        self.image_model = ImageModelTL()
        self.tabular_model = TabularModel(tabular_dim)
        self.fc1 = nn.Linear(128 + 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, images: torch.Tensor, tabular_data: torch.Tensor) -> torch.Tensor:
        img_features = self.image_model(images)
        tab_features = self.tabular_model(tabular_data)
        fused = torch.cat((img_features, tab_features), dim=1)
        x = F.relu(self.fc1(fused))
        x = torch.sigmoid(self.fc2(x))
        return x

    def get_features(self, images: torch.Tensor, tabular_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract intermediate features for distillation."""
        img_features = self.image_model(images)
        tab_features = self.tabular_model(tabular_data)
        return img_features, tab_features


# ------------------------------ Multimodal Student Model ------------------------------ #
class MultimodalStudent(nn.Module):
    """
    Lightweight Student model trained via Knowledge Distillation.
    Uses simpler CNN for images + FC for tabular data.
    """

    def __init__(self, tabular_dim: int = 9, image_size: Tuple[int, int] = (224, 224)):
        super(MultimodalStudent, self).__init__()

        # Lightweight image feature extractor
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size after convolutions
        # After 2 pooling layers: H/4 x W/4
        h_out = image_size[0] // 4
        w_out = image_size[1] // 4
        self.fc_img = nn.Linear(32 * h_out * w_out, 128)

        # Tabular feature extractor
        self.tabular_model = TabularModel(tabular_dim)

        # Fusion layers
        self.fc1 = nn.Linear(128 + 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, images: torch.Tensor, tabular_data: torch.Tensor) -> torch.Tensor:
        # Image features
        x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        img_features = F.relu(self.fc_img(x))

        # Tabular features
        tab_features = self.tabular_model(tabular_data)

        # Fusion
        fused = torch.cat((img_features, tab_features), dim=1)
        x = F.relu(self.fc1(fused))
        x = torch.sigmoid(self.fc2(x))
        return x

    def get_features(self, images: torch.Tensor, tabular_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract intermediate features for analysis."""
        x = self.pool(F.relu(self.conv1(images)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        img_features = F.relu(self.fc_img(x))
        tab_features = self.tabular_model(tabular_data)
        return img_features, tab_features


# ------------------------------ Original Multimodal Model (Backwards Compatibility) ------------------------------ #
class MultimodalModel(nn.Module):
    """
    Original multimodal model for backwards compatibility with existing weights.
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


# ------------------------------ Knowledge Distillation Loss ------------------------------ #
def distillation_loss(
    student_outputs: torch.Tensor,
    teacher_outputs: torch.Tensor,
    hard_labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 3.0
) -> torch.Tensor:
    """
    Combined loss for Knowledge Distillation.

    Args:
        student_outputs: Student model predictions
        teacher_outputs: Teacher model predictions (soft targets)
        hard_labels: Ground truth labels
        alpha: Weight for soft loss (1-alpha for hard loss)
        temperature: Temperature for softening distributions

    Returns:
        Combined distillation loss
    """
    # Soft loss (KL Divergence with temperature)
    student_soft = F.log_softmax(student_outputs / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_outputs / temperature, dim=-1)
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

    # Hard loss (Binary Cross Entropy)
    hard_loss = F.binary_cross_entropy(student_outputs, hard_labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


# ------------------------------ Model Loading Functions ------------------------------ #
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


def load_student_model(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a trained Student model from checkpoint.

    Args:
        model_path: Path to the .pth file
        device: Device to load model on

    Returns:
        Loaded Student model in eval mode
    """
    model = MultimodalStudent(tabular_dim=9)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Student model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load student weights from {model_path}: {e}")
        print("Using randomly initialized weights")

    model.to(device)
    model.eval()

    return model


def load_teacher_model(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a trained Teacher model from checkpoint.

    Args:
        model_path: Path to the .pth file
        device: Device to load model on

    Returns:
        Loaded Teacher model in eval mode
    """
    model = MultimodalTeacher(tabular_dim=9)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Teacher model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load teacher weights from {model_path}: {e}")
        print("Using randomly initialized weights")

    model.to(device)
    model.eval()

    return model
