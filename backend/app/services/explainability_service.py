"""
Explainability service for model interpretations using SHAP and LIME.
Provides local explanations for both image and tabular features.
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
import base64
from io import BytesIO

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ExplainabilityService:
    """
    Service for generating model explanations using SHAP and LIME.

    Provides:
    - SHAP values for tabular features
    - SHAP DeepExplainer for image features
    - LIME explanations for tabular data
    - LIME image segmentation explanations
    """

    def __init__(self):
        self.tabular_feature_names = [
            "Age", "Gender", "Systolic BP", "Diastolic BP",
            "Glucose", "BMI", "Cholesterol", "Former Smoker", "Current Smoker"
        ]
        self.shap_explainer = None
        self.lime_tabular_explainer = None
        self.lime_image_explainer = None

        # Background data for SHAP (representative samples)
        self._background_data = self._create_background_data()

    def _create_background_data(self) -> np.ndarray:
        """Create representative background data for SHAP."""
        # Representative population data for stroke risk factors
        np.random.seed(42)
        n_samples = 100

        background = np.zeros((n_samples, 9))

        # Age: 30-80, centered around 55
        background[:, 0] = np.random.normal(55, 15, n_samples).clip(30, 80)

        # Gender: 0/1
        background[:, 1] = np.random.binomial(1, 0.5, n_samples)

        # Systolic BP: 100-180
        background[:, 2] = np.random.normal(130, 20, n_samples).clip(100, 180)

        # Diastolic BP: 60-110
        background[:, 3] = np.random.normal(80, 12, n_samples).clip(60, 110)

        # Glucose: 70-200
        background[:, 4] = np.random.normal(110, 30, n_samples).clip(70, 200)

        # BMI: 18-40
        background[:, 5] = np.random.normal(26, 5, n_samples).clip(18, 40)

        # Cholesterol: 150-280
        background[:, 6] = np.random.normal(200, 40, n_samples).clip(150, 280)

        # Smoking former: 0/1
        background[:, 7] = np.random.binomial(1, 0.2, n_samples)

        # Smoking current: 0/1
        background[:, 8] = np.random.binomial(1, 0.15, n_samples)

        return background

    def explain_tabular_shap(
        self,
        model: torch.nn.Module,
        tabular_data: np.ndarray,
        images: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for tabular features.

        Args:
            model: PyTorch model
            tabular_data: Input tabular features (1, 9)
            images: Optional image tensor for multimodal model

        Returns:
            Dictionary with SHAP values and feature attributions
        """
        if not SHAP_AVAILABLE:
            return self._fallback_tabular_explanation(tabular_data)

        try:
            # Create prediction function for SHAP
            def predict_fn(x):
                x_tensor = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    if images is not None:
                        # Multimodal prediction
                        batch_size = x.shape[0]
                        img_batch = images.repeat(batch_size, 1, 1, 1)
                        output = model(img_batch, x_tensor)
                    else:
                        # Tabular only - use dummy images
                        dummy_images = torch.zeros(x.shape[0], 12, 480, 640)
                        output = model(dummy_images, x_tensor)
                return output.numpy()

            # Use KernelExplainer for model-agnostic explanations
            explainer = shap.KernelExplainer(predict_fn, self._background_data[:50])

            # Calculate SHAP values
            shap_values = explainer.shap_values(tabular_data)

            # Format results
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            feature_importance = {}
            for i, name in enumerate(self.tabular_feature_names):
                feature_importance[name] = {
                    "shap_value": float(shap_values[0][i]) if len(shap_values.shape) > 1 else float(shap_values[i]),
                    "feature_value": float(tabular_data[0][i]) if len(tabular_data.shape) > 1 else float(tabular_data[i]),
                    "direction": "increases risk" if (shap_values[0][i] if len(shap_values.shape) > 1 else shap_values[i]) > 0 else "decreases risk"
                }

            # Sort by absolute SHAP value
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]["shap_value"]),
                reverse=True
            )

            return {
                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                "feature_importance": dict(sorted_features),
                "base_value": float(explainer.expected_value) if hasattr(explainer.expected_value, '__float__') else 0.5,
                "method": "SHAP KernelExplainer"
            }

        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return self._fallback_tabular_explanation(tabular_data)

    def explain_tabular_lime(
        self,
        model: torch.nn.Module,
        tabular_data: np.ndarray,
        images: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations for tabular features.

        Args:
            model: PyTorch model
            tabular_data: Input tabular features
            images: Optional image tensor

        Returns:
            Dictionary with LIME feature weights and explanations
        """
        if not LIME_AVAILABLE:
            return self._fallback_tabular_explanation(tabular_data)

        try:
            # Create prediction function
            def predict_fn(x):
                x_tensor = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    if images is not None:
                        batch_size = x.shape[0]
                        img_batch = images.repeat(batch_size, 1, 1, 1)
                        output = model(img_batch, x_tensor)
                    else:
                        dummy_images = torch.zeros(x.shape[0], 12, 480, 640)
                        output = model(dummy_images, x_tensor)

                # Return probabilities for both classes
                probs = output.numpy()
                return np.column_stack([1 - probs, probs])

            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=self._background_data,
                feature_names=self.tabular_feature_names,
                class_names=["Low Risk", "High Risk"],
                mode="classification"
            )

            # Generate explanation
            instance = tabular_data[0] if len(tabular_data.shape) > 1 else tabular_data
            explanation = explainer.explain_instance(
                instance,
                predict_fn,
                num_features=9,
                top_labels=1
            )

            # Extract feature weights
            feature_weights = {}
            exp_list = explanation.as_list(label=1)

            for feature_desc, weight in exp_list:
                # Parse feature name from description
                for fname in self.tabular_feature_names:
                    if fname.lower() in feature_desc.lower():
                        feature_weights[fname] = {
                            "weight": float(weight),
                            "description": feature_desc,
                            "direction": "increases risk" if weight > 0 else "decreases risk"
                        }
                        break

            return {
                "feature_weights": feature_weights,
                "prediction_local": explanation.local_pred[0] if hasattr(explanation, 'local_pred') else None,
                "intercept": float(explanation.intercept[1]) if hasattr(explanation, 'intercept') else 0,
                "method": "LIME TabularExplainer"
            }

        except Exception as e:
            print(f"LIME explanation failed: {e}")
            return self._fallback_tabular_explanation(tabular_data)

    def explain_image_lime(
        self,
        model: torch.nn.Module,
        image: np.ndarray,
        tabular_data: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Generate LIME image segmentation explanations.

        Args:
            model: PyTorch model
            image: Input image as numpy array (H, W, C)
            tabular_data: Tabular features tensor

        Returns:
            Dictionary with image segments and their attributions
        """
        if not LIME_AVAILABLE:
            return {"error": "LIME not available", "segments": []}

        try:
            from skimage.segmentation import quickshift

            # Prediction function for single image
            def predict_fn(images_batch):
                results = []
                for img in images_batch:
                    # Convert to tensor format
                    img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

                    # Stack 4 copies to match expected input (4 expressions)
                    img_stacked = img_tensor.repeat(1, 4, 1, 1)  # (1, 12, H, W)

                    with torch.no_grad():
                        output = model(img_stacked, tabular_data)

                    prob = output.item()
                    results.append([1 - prob, prob])

                return np.array(results)

            # Create LIME image explainer
            explainer = lime_image.LimeImageExplainer()

            # Generate explanation
            explanation = explainer.explain_instance(
                image,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=100
            )

            # Get image and mask
            temp, mask = explanation.get_image_and_mask(
                label=1,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )

            # Convert to base64 for transmission
            from PIL import Image
            import io

            # Create heatmap overlay
            heatmap = self._create_heatmap(image, mask)
            buffered = io.BytesIO()
            Image.fromarray(heatmap).save(buffered, format="PNG")
            heatmap_b64 = base64.b64encode(buffered.getvalue()).decode()

            return {
                "heatmap_base64": heatmap_b64,
                "top_segments": explanation.top_labels,
                "num_segments": len(np.unique(mask)),
                "method": "LIME ImageExplainer"
            }

        except Exception as e:
            print(f"LIME image explanation failed: {e}")
            return {"error": str(e), "segments": []}

    def _create_heatmap(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create a heatmap overlay on the image."""
        import cv2

        # Normalize mask to 0-255
        if mask.max() > 0:
            mask_normalized = (mask * 255 / mask.max()).astype(np.uint8)
        else:
            mask_normalized = np.zeros_like(mask, dtype=np.uint8)

        # Apply colormap
        heatmap = cv2.applyColorMap(mask_normalized, cv2.COLORMAP_JET)

        # Blend with original image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        blended = cv2.addWeighted(image.astype(np.uint8), 0.6, heatmap, 0.4, 0)

        return blended

    def _fallback_tabular_explanation(self, tabular_data: np.ndarray) -> Dict[str, Any]:
        """
        Fallback explanation using coefficient-based feature importance.
        Used when SHAP/LIME are not available.
        """
        # Pre-defined importance weights based on clinical literature
        importance_weights = {
            "Age": 0.15,
            "Gender": 0.05,
            "Systolic BP": 0.20,
            "Diastolic BP": 0.10,
            "Glucose": 0.12,
            "BMI": 0.10,
            "Cholesterol": 0.08,
            "Former Smoker": 0.08,
            "Current Smoker": 0.12
        }

        # Reference values (population means)
        reference_values = [55, 0.5, 130, 80, 110, 26, 200, 0.2, 0.15]

        instance = tabular_data[0] if len(tabular_data.shape) > 1 else tabular_data

        feature_importance = {}
        for i, name in enumerate(self.tabular_feature_names):
            deviation = (instance[i] - reference_values[i]) / (reference_values[i] + 1e-6)
            importance = deviation * importance_weights[name]

            feature_importance[name] = {
                "importance_score": float(importance),
                "feature_value": float(instance[i]),
                "reference_value": reference_values[i],
                "direction": "increases risk" if importance > 0 else "decreases risk"
            }

        # Sort by absolute importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]["importance_score"]),
            reverse=True
        )

        return {
            "feature_importance": dict(sorted_features),
            "method": "Coefficient-based (fallback)"
        }

    def generate_clinical_summary(
        self,
        shap_results: Dict[str, Any],
        lime_results: Dict[str, Any],
        risk_score: float
    ) -> List[str]:
        """
        Generate a clinical summary from explainability results.

        Args:
            shap_results: SHAP explanation results
            lime_results: LIME explanation results
            risk_score: Model prediction risk score

        Returns:
            List of clinical interpretation strings
        """
        summary = []

        # Get top contributing factors from SHAP
        if "feature_importance" in shap_results:
            top_factors = list(shap_results["feature_importance"].items())[:3]
            for name, data in top_factors:
                if abs(data.get("shap_value", data.get("importance_score", 0))) > 0.01:
                    direction = data.get("direction", "affects")
                    summary.append(f"{name} {direction} (value: {data.get('feature_value', 'N/A')})")

        # Add LIME insights if available
        if "feature_weights" in lime_results:
            lime_factors = list(lime_results["feature_weights"].items())[:2]
            for name, data in lime_factors:
                if abs(data.get("weight", 0)) > 0.01:
                    summary.append(f"LIME analysis: {data.get('description', name)}")

        # Risk level interpretation
        if risk_score < 0.3:
            summary.append("Overall model interpretation suggests low immediate concern.")
        elif risk_score < 0.7:
            summary.append("Model indicates moderate risk - clinical evaluation recommended.")
        else:
            summary.append("High risk indicators detected - urgent clinical assessment advised.")

        return summary


# Global instance
explainability_service = ExplainabilityService()
