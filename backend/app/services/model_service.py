"""
Model service for loading and running stroke risk inference.
"""

import time
from typing import Any, Dict, List

import numpy as np
import torch

from app.models.ml_models import load_model
from app.models.schemas import ContributingFactor, RiskLevel


class ModelService:
    """
    Service for stroke risk prediction model inference.

    Handles:
    - Model loading and caching
    - Tabular data normalization
    - Running inference
    - Post-processing results
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None

        # Feature definitions
        self.tabular_features = [
            "age",
            "gender",
            "systolic",
            "diastolic",
            "glucose",
            "bmi",
            "cholesterol",
            "smoking",
        ]

        # Normalization parameters (from training data)
        self.scaler_params = {
            "age": {"mean": 55.0, "std": 15.0},
            "systolic": {"mean": 130.0, "std": 20.0},
            "diastolic": {"mean": 80.0, "std": 12.0},
            "glucose": {"mean": 110.0, "std": 30.0},
            "bmi": {"mean": 26.0, "std": 5.0},
            "cholesterol": {"mean": 200.0, "std": 40.0},
        }

        # Risk factor thresholds for analysis
        self.risk_thresholds = {
            "age": {"threshold": 65, "label": "Age", "severity": "low"},
            "systolic": {
                "threshold": 140,
                "label": "High Blood Pressure (Systolic)",
                "severity": "high",
            },
            "diastolic": {
                "threshold": 90,
                "label": "High Blood Pressure (Diastolic)",
                "severity": "moderate",
            },
            "glucose": {
                "threshold": 140,
                "label": "High Blood Sugar",
                "severity": "moderate",
            },
            "bmi": {"threshold": 30, "label": "Obesity", "severity": "moderate"},
            "cholesterol": {
                "threshold": 240,
                "label": "High Cholesterol",
                "severity": "moderate",
            },
            "smoking": {"threshold": 1, "label": "Smoking", "severity": "high"},
        }

    def load(self) -> None:
        """Load the model from disk."""
        self.model = load_model(self.model_path, self.device, legacy=True)

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def prepare_tabular_data(self, health_data: Dict[str, Any]) -> torch.Tensor:
        """
        Prepare and normalize tabular health data for model input.

        Args:
            health_data: Dictionary with health metrics

        Returns:
            Normalized tensor of shape (1, num_features)
        """
        features = []

        # Process numerical features with normalization
        for feat in ["age", "systolic", "diastolic", "glucose", "bmi", "cholesterol"]:
            value = health_data.get(feat, 0)
            if feat in self.scaler_params:
                normalized = (value - self.scaler_params[feat]["mean"]) / self.scaler_params[feat]["std"]
            else:
                normalized = float(value)
            features.append(normalized)

        # Gender: encode as 0/1 (female=0, male=1)
        gender = health_data.get("gender", "male")
        if isinstance(gender, str):
            gender_encoded = 1 if gender.lower() == "male" else 0
        else:
            gender_encoded = int(gender)
        features.append(float(gender_encoded))

        # Smoking status: 0, 1, or 2
        smoking = health_data.get("smoking", 0)
        if hasattr(smoking, "value"):
            smoking = smoking.value
        features.append(float(smoking))

        tensor = torch.tensor([features], dtype=torch.float32)
        return tensor.to(self.device)

    def predict(
        self, images: torch.Tensor, tabular: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Run stroke risk prediction.

        Args:
            images: Preprocessed image tensor (1, 12, H, W)
            tabular: Normalized tabular data tensor (1, 8)

        Returns:
            Dictionary with risk_score, risk_level, confidence, processing_time_ms
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        images = images.to(self.device)
        tabular = tabular.to(self.device)

        with torch.no_grad():
            logits = self.model(images, tabular)

            # Handle different output formats
            if logits.dim() > 1:
                logits = logits.squeeze()

            # Apply sigmoid if not already applied
            if logits.min() < 0 or logits.max() > 1:
                probability = torch.sigmoid(logits).item()
            else:
                probability = logits.item()

        processing_time = (time.time() - start_time) * 1000

        # Determine risk level
        if probability < 0.3:
            risk_level = RiskLevel.low
        elif probability < 0.7:
            risk_level = RiskLevel.moderate
        else:
            risk_level = RiskLevel.high

        # Calculate confidence (higher when further from 0.5)
        confidence = abs(0.5 - probability) * 2

        return {
            "risk_score": probability,
            "risk_level": risk_level,
            "confidence": confidence,
            "processing_time_ms": processing_time,
        }

    def analyze_risk_factors(
        self, health_data: Dict[str, Any]
    ) -> List[ContributingFactor]:
        """
        Analyze which health factors contribute to stroke risk.

        Args:
            health_data: Dictionary with health metrics

        Returns:
            List of contributing factors exceeding thresholds
        """
        factors = []

        for key, config in self.risk_thresholds.items():
            value = health_data.get(key)

            # Handle enum values
            if hasattr(value, "value"):
                value = value.value

            if value is None:
                continue

            if float(value) >= config["threshold"]:
                factors.append(
                    ContributingFactor(
                        factor=config["label"],
                        value=float(value),
                        threshold=float(config["threshold"]),
                        severity=config["severity"],
                    )
                )

        # Sort by severity
        severity_order = {"high": 0, "moderate": 1, "low": 2}
        factors.sort(key=lambda x: severity_order.get(x.severity, 3))

        return factors

    def generate_recommendations(
        self, health_data: Dict[str, Any], risk_level: RiskLevel
    ) -> List[str]:
        """
        Generate health recommendations based on risk factors.

        Args:
            health_data: Dictionary with health metrics
            risk_level: Predicted risk level

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Risk level-based urgent recommendations
        if risk_level == RiskLevel.high:
            recommendations.append(
                "Please consult a healthcare professional as soon as possible"
            )
        elif risk_level == RiskLevel.moderate:
            recommendations.append(
                "Schedule an appointment with your healthcare provider for a comprehensive evaluation"
            )

        # Factor-specific recommendations
        systolic = health_data.get("systolic", 0)
        if systolic > 140:
            recommendations.append(
                "Monitor and manage blood pressure regularly with guidance from your doctor"
            )

        bmi = health_data.get("bmi", 0)
        if bmi > 30:
            recommendations.append(
                "Consider lifestyle modifications including diet and exercise for weight management"
            )

        smoking = health_data.get("smoking", 0)
        if hasattr(smoking, "value"):
            smoking = smoking.value
        if smoking > 0:
            recommendations.append(
                "Smoking cessation can significantly reduce your stroke risk - consider a cessation program"
            )

        cholesterol = health_data.get("cholesterol", 0)
        if cholesterol > 240:
            recommendations.append(
                "Discuss cholesterol management options with your healthcare provider"
            )

        glucose = health_data.get("glucose", 0)
        if glucose > 140:
            recommendations.append(
                "Monitor blood sugar levels and consider dietary adjustments"
            )

        # General recommendations
        if risk_level != RiskLevel.low:
            recommendations.append(
                "Maintain a heart-healthy diet rich in fruits, vegetables, and whole grains"
            )
            recommendations.append(
                "Aim for at least 150 minutes of moderate aerobic activity per week"
            )

        return recommendations
