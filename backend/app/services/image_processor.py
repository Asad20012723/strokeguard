"""
Image preprocessing service for stroke risk prediction.
Handles image decoding, background removal, denoising, and sharpening.
"""

import base64
from io import BytesIO
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import convolve


class ImagePreprocessor:
    """
    Preprocesses facial expression images for the stroke prediction model.

    Pipeline:
    1. Decode base64 image
    2. Resize to target size
    3. Apply background removal (optional)
    4. Apply Gaussian denoising
    5. Apply sharpening filter
    6. Normalize and convert to tensor
    """

    def __init__(
        self,
        target_size: tuple = (224, 224),
        use_background_removal: bool = False,
        device: str = "cpu",
    ):
        self.target_size = target_size
        self.use_background_removal = use_background_removal
        self.device = device

        # Sharpening kernel
        self.sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # Background removal model (lazy loading)
        self._bg_remover = None

    @property
    def bg_remover(self):
        """Lazy load background removal model."""
        if self._bg_remover is None and self.use_background_removal:
            try:
                from rembg import remove

                self._bg_remover = remove
            except ImportError:
                print("rembg not installed, skipping background removal")
                self.use_background_removal = False
        return self._bg_remover

    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """
        Decode a base64 encoded image string to numpy array.

        Args:
            base64_string: Base64 encoded image (without data URI prefix)

        Returns:
            RGB numpy array of shape (H, W, 3)
        """
        # Handle data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return np.array(image)

    def remove_background(self, image_np: np.ndarray) -> np.ndarray:
        """
        Remove background from image using rembg.

        Args:
            image_np: RGB image as numpy array

        Returns:
            Image with background removed (black background)
        """
        if not self.use_background_removal or self.bg_remover is None:
            return image_np

        # Convert to PIL, process, convert back
        image_pil = Image.fromarray(image_np)
        result = self.bg_remover(image_pil)

        # Convert RGBA to RGB with black background
        if result.mode == "RGBA":
            background = Image.new("RGB", result.size, (0, 0, 0))
            background.paste(result, mask=result.split()[3])
            result = background

        return np.array(result)

    def apply_gaussian_denoise(
        self, image: np.ndarray, sigma: float = 1.0
    ) -> np.ndarray:
        """
        Apply Gaussian denoising to image.

        Args:
            image: Input image as numpy array (0-1 range)
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Denoised image
        """
        from scipy.ndimage import gaussian_filter

        if image.ndim == 3:
            denoised = np.zeros_like(image)
            for i in range(3):
                denoised[:, :, i] = gaussian_filter(image[:, :, i], sigma=sigma)
            return denoised
        return gaussian_filter(image, sigma=sigma)

    def apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter to enhance edges.

        Args:
            image: Input image as numpy array (0-1 range)

        Returns:
            Sharpened image clipped to [0, 1]
        """
        if image.ndim == 3:
            sharpened = np.zeros_like(image)
            for i in range(3):
                sharpened[:, :, i] = convolve(image[:, :, i], self.sharpen_kernel)
            return np.clip(sharpened, 0, 1)
        return np.clip(convolve(image, self.sharpen_kernel), 0, 1)

    def preprocess_single_image(self, base64_image: str) -> np.ndarray:
        """
        Full preprocessing pipeline for a single image.

        Args:
            base64_image: Base64 encoded image string

        Returns:
            Preprocessed image as numpy array, shape (3, H, W), range [0, 1]
        """
        # Decode
        img_np = self.decode_base64_image(base64_image)

        # Resize
        img_np = cv2.resize(img_np, self.target_size)

        # Background removal (optional)
        if self.use_background_removal:
            img_np = self.remove_background(img_np)

        # Normalize to 0-1
        img_np = img_np.astype(np.float32) / 255.0

        # Denoise
        img_np = self.apply_gaussian_denoise(img_np, sigma=1.0)

        # Sharpen
        img_np = self.apply_sharpening(img_np)

        # Convert to CHW format (channels first)
        img_np = img_np.transpose(2, 0, 1)

        return img_np

    def preprocess_all_images(
        self, images: Dict[str, str], expression_order: List[str] = None
    ) -> torch.Tensor:
        """
        Process all 4 expression images and combine into a single tensor.

        Args:
            images: Dictionary mapping expression names to base64 strings
            expression_order: Order of expressions (default: kiss, normal, spread, open)

        Returns:
            Tensor of shape (1, 12, H, W) - batch of 1, 12 channels (4 images x 3 RGB)
        """
        if expression_order is None:
            expression_order = ["kiss", "normal", "spread", "open"]

        processed = []
        for expr in expression_order:
            expr_lower = expr.lower()
            if expr_lower not in images:
                raise ValueError(f"Missing image for expression: {expr}")

            img = self.preprocess_single_image(images[expr_lower])
            processed.append(img)

        # Stack images: (4, 3, H, W) -> (12, H, W)
        stacked = np.concatenate(processed, axis=0)

        # Add batch dimension: (1, 12, H, W)
        tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)

        return tensor.to(self.device)

    def validate_images(self, images: Dict[str, str]) -> List[str]:
        """
        Validate that all required images are present and valid.

        Args:
            images: Dictionary of expression names to base64 strings

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        required = ["kiss", "normal", "spread", "open"]

        for expr in required:
            if expr not in images:
                errors.append(f"Missing required image: {expr}")
            elif not images[expr]:
                errors.append(f"Empty image data for: {expr}")
            else:
                try:
                    self.decode_base64_image(images[expr])
                except Exception as e:
                    errors.append(f"Invalid image format for {expr}: {str(e)}")

        return errors
