"""
Pydantic schemas for API request/response validation.
"""

from enum import Enum
from typing import Dict, List, Optional

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
