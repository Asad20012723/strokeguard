"""
Pydantic schemas for API request/response validation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenderEnum(str, Enum):
    male = "male"
    female = "female"


class SmokingStatusEnum(int, Enum):
    never = 0
    former = 1
    current = 2


class RiskLevel(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"


class HealthData(BaseModel):
    """Health metrics input data."""

    age: int = Field(..., ge=18, le=120, description="Patient age in years")
    gender: GenderEnum = Field(..., description="Patient gender")
    systolic: int = Field(
        ..., ge=70, le=250, description="Systolic blood pressure (mmHg)"
    )
    diastolic: int = Field(
        ..., ge=40, le=150, description="Diastolic blood pressure (mmHg)"
    )
    glucose: int = Field(..., ge=50, le=500, description="Blood glucose (mg/dL)")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    cholesterol: int = Field(
        ..., ge=100, le=400, description="Total cholesterol (mg/dL)"
    )
    smoking: SmokingStatusEnum = Field(
        ..., description="Smoking status: 0=never, 1=former, 2=current"
    )

    @field_validator("diastolic")
    @classmethod
    def diastolic_less_than_systolic(cls, v: int, info) -> int:
        if "systolic" in info.data and v >= info.data["systolic"]:
            raise ValueError("Diastolic BP must be less than systolic BP")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 55,
                "gender": "male",
                "systolic": 130,
                "diastolic": 85,
                "glucose": 110,
                "bmi": 26.5,
                "cholesterol": 210,
                "smoking": 0,
            }
        }


class ImageData(BaseModel):
    """Facial expression images (base64 encoded)."""

    kiss: str = Field(..., description="Kiss expression image (base64)")
    normal: str = Field(..., description="Normal expression image (base64)")
    spread: str = Field(..., description="Spread expression image (base64)")
    open: str = Field(..., description="Open mouth expression image (base64)")


class PredictionRequest(BaseModel):
    """Request payload for stroke risk prediction."""

    health_data: HealthData
    images: ImageData

    class Config:
        json_schema_extra = {
            "example": {
                "health_data": {
                    "age": 55,
                    "gender": "male",
                    "systolic": 130,
                    "diastolic": 85,
                    "glucose": 110,
                    "bmi": 26.5,
                    "cholesterol": 210,
                    "smoking": 0,
                },
                "images": {
                    "kiss": "base64_encoded_string...",
                    "normal": "base64_encoded_string...",
                    "spread": "base64_encoded_string...",
                    "open": "base64_encoded_string...",
                },
            }
        }


class ContributingFactor(BaseModel):
    """A factor contributing to stroke risk."""

    factor: str = Field(..., description="Name of the risk factor")
    value: float = Field(..., description="Patient's value for this factor")
    threshold: float = Field(..., description="Risk threshold for this factor")
    severity: str = Field(..., description="Severity level: low, moderate, high")


class PredictionResponse(BaseModel):
    """Response payload with stroke risk assessment."""

    risk_score: float = Field(
        ..., ge=0, le=1, description="Stroke risk probability (0-1)"
    )
    risk_level: RiskLevel = Field(..., description="Risk category")
    confidence: float = Field(
        ..., ge=0, le=1, description="Model confidence in prediction"
    )
    contributing_factors: List[ContributingFactor] = Field(
        default_factory=list, description="Factors contributing to risk"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Health recommendations"
    )
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 0.35,
                "risk_level": "moderate",
                "confidence": 0.78,
                "contributing_factors": [
                    {
                        "factor": "High Blood Pressure",
                        "value": 145,
                        "threshold": 140,
                        "severity": "moderate",
                    }
                ],
                "recommendations": [
                    "Monitor and manage blood pressure with your healthcare provider",
                    "Schedule regular check-ups",
                ],
                "processing_time_ms": 1250.5,
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code if applicable")


# ======================= Dual-Model System Schemas =======================

class StatisticalPredictionRequest(BaseModel):
    """Request for statistical (tabular-only) prediction."""

    health_data: HealthData
    include_statistics: bool = Field(
        default=True,
        description="Include detailed statistics (OR, p-values, CI)"
    )


class FeatureContribution(BaseModel):
    """Detailed feature contribution from statistical model."""

    value: float = Field(..., description="Feature value")
    coefficient: float = Field(..., description="Model coefficient")
    contribution: float = Field(..., description="Contribution to log-odds")
    odds_ratio: float = Field(..., description="Odds ratio for this feature")
    p_value: float = Field(..., description="Statistical significance")
    ci_95: List[float] = Field(..., description="95% confidence interval [lower, upper]")
    significant: bool = Field(..., description="Whether p < 0.05")


class StatisticalPredictionResponse(BaseModel):
    """Response from statistical model with full statistics."""

    risk_score: float = Field(..., ge=0, le=1, description="Risk probability")
    risk_level: RiskLevel = Field(..., description="Risk category")
    odds_ratios: Dict[str, float] = Field(..., description="Odds ratios per feature")
    p_values: Dict[str, float] = Field(..., description="P-values per feature")
    std_errors: Dict[str, float] = Field(..., description="Standard errors per feature")
    ci_lower: Dict[str, float] = Field(..., description="Lower 95% CI per feature")
    ci_upper: Dict[str, float] = Field(..., description="Upper 95% CI per feature")
    feature_contributions: Dict[str, FeatureContribution] = Field(
        ..., description="Detailed feature contributions"
    )
    model_statistics: Dict[str, float] = Field(
        ..., description="Model fit statistics (pseudo R2, AIC, etc.)"
    )
    interpretation: List[str] = Field(
        default_factory=list, description="Clinical interpretation"
    )
    processing_time_ms: float = Field(..., description="Processing time in ms")


class MultimodalPredictionRequest(BaseModel):
    """Request for multimodal prediction with images."""

    health_data: HealthData
    images: ImageData
    include_explainability: bool = Field(
        default=True,
        description="Include SHAP/LIME explanations"
    )


class ExplainabilityResults(BaseModel):
    """SHAP and LIME explanation results."""

    shap_values: Optional[Dict[str, Any]] = Field(
        None, description="SHAP feature attributions"
    )
    lime_tabular: Optional[Dict[str, Any]] = Field(
        None, description="LIME tabular explanations"
    )
    lime_image: Optional[Dict[str, Any]] = Field(
        None, description="LIME image segmentation"
    )
    clinical_summary: List[str] = Field(
        default_factory=list, description="Clinical interpretation summary"
    )


class MultimodalPredictionResponse(BaseModel):
    """Response from multimodal model with explainability."""

    risk_score: float = Field(..., ge=0, le=1, description="Risk probability")
    risk_level: RiskLevel = Field(..., description="Risk category")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    contributing_factors: List[ContributingFactor] = Field(
        default_factory=list, description="Risk factors"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Health recommendations"
    )
    explainability: Optional[ExplainabilityResults] = Field(
        None, description="SHAP/LIME explanations"
    )
    processing_time_ms: float = Field(..., description="Processing time in ms")


class DualModelPredictionRequest(BaseModel):
    """Request for dual-model prediction (optional images)."""

    health_data: HealthData
    images: Optional[ImageData] = Field(
        None, description="Optional facial expression images"
    )
    include_statistics: bool = Field(
        default=True, description="Include statistical analysis"
    )
    include_explainability: bool = Field(
        default=True, description="Include SHAP/LIME explanations"
    )


class AIInterpretationSection(BaseModel):
    """A section of the AI interpretation."""

    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")


class AIInterpretation(BaseModel):
    """AI-generated medical interpretation from n8n."""

    risk_score: float = Field(..., description="Risk score")
    risk_level: str = Field(..., description="Risk level")
    sections: List[AIInterpretationSection] = Field(
        default_factory=list, description="Interpretation sections"
    )
    generated_by: str = Field(..., description="Source: n8n_ai_agent or rule_based_system")
    timestamp: str = Field(..., description="Generation timestamp")


class DualModelPredictionResponse(BaseModel):
    """Combined response from both statistical and multimodal models."""

    patient_id: str = Field(..., description="Generated patient session ID")
    image_provided: bool = Field(..., description="Whether images were provided")

    # Statistical model results (always available)
    statistical_results: StatisticalPredictionResponse = Field(
        ..., description="Logistic regression results"
    )

    # Multimodal results (only if images provided)
    multimodal_results: Optional[MultimodalPredictionResponse] = Field(
        None, description="Multimodal model results (if images provided)"
    )

    # Combined risk assessment
    combined_risk_score: float = Field(
        ..., ge=0, le=1, description="Combined risk score"
    )
    combined_risk_level: RiskLevel = Field(
        ..., description="Combined risk category"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Combined recommendations"
    )

    # AI Expert Interpretation (from n8n or rule-based fallback)
    ai_interpretation: Optional[AIInterpretation] = Field(
        None, description="AI-generated medical interpretation"
    )
    interpretation_source: str = Field(
        default="none", description="Source: n8n_ai_agent, rule_based_fallback, or none"
    )
    fallback_used: bool = Field(
        default=False, description="Whether rule-based fallback was used"
    )
    fallback_reason: Optional[str] = Field(
        None, description="Reason for fallback if used"
    )

    total_processing_time_ms: float = Field(
        ..., description="Total processing time"
    )


class GenerateReportRequest(BaseModel):
    """Request to generate medical report via n8n."""

    patient_id: str = Field(..., description="Patient session ID")
    health_data: HealthData
    images: Optional[ImageData] = Field(None, description="Optional images")
    report_format: str = Field(
        default="pdf",
        description="Report format: pdf, html, json"
    )


class GenerateReportResponse(BaseModel):
    """Response from report generation."""

    success: bool = Field(..., description="Whether report was generated")
    report_url: Optional[str] = Field(None, description="URL to download report")
    report_content: Optional[str] = Field(None, description="Report content (if inline)")
    error: Optional[str] = Field(None, description="Error message if failed")
