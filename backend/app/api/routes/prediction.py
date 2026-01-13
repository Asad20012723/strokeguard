"""
Stroke risk prediction endpoints.
"""

import time
from typing import Dict

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
)

router = APIRouter()


@router.post(
    "/",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def predict_stroke_risk(
    request: Request,
    data: PredictionRequest,
) -> PredictionResponse:
    """
    Predict stroke risk from health data and facial expression images.

    This endpoint accepts:
    - Health metrics (age, blood pressure, glucose, BMI, cholesterol, smoking status)
    - Four facial expression images (kiss, normal, spread, open) as base64 strings

    Returns a stroke risk assessment with:
    - Risk score (0-1 probability)
    - Risk level (low/moderate/high)
    - Contributing factors
    - Health recommendations
    """
    from app.main import model_service, image_processor

    # Check if services are available
    if model_service is None or not model_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model service not available. Please try again later.",
        )

    if image_processor is None:
        raise HTTPException(
            status_code=503,
            detail="Image processor not available. Please try again later.",
        )

    total_start = time.time()

    try:
        # Validate images
        images_dict = {
            "kiss": data.images.kiss,
            "normal": data.images.normal,
            "spread": data.images.spread,
            "open": data.images.open,
        }

        validation_errors = image_processor.validate_images(images_dict)
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail=f"Image validation failed: {'; '.join(validation_errors)}",
            )

        # Process images
        image_tensor = image_processor.preprocess_all_images(images_dict)

        # Prepare tabular data
        health_dict = data.health_data.model_dump()
        tabular_tensor = model_service.prepare_tabular_data(health_dict)

        # Run inference
        result = model_service.predict(image_tensor, tabular_tensor)

        # Analyze risk factors
        contributing_factors = model_service.analyze_risk_factors(health_dict)

        # Generate recommendations
        recommendations = model_service.generate_recommendations(
            health_dict, result["risk_level"]
        )

        total_time = (time.time() - total_start) * 1000

        return PredictionResponse(
            risk_score=result["risk_score"],
            risk_level=result["risk_level"],
            confidence=result["confidence"],
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            processing_time_ms=total_time,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )


@router.post("/validate")
async def validate_input(data: PredictionRequest) -> Dict[str, bool]:
    """
    Validate prediction input without running inference.
    Useful for client-side validation feedback.
    """
    from app.main import image_processor

    if image_processor is None:
        return {"valid": False, "error": "Image processor not available"}

    images_dict = {
        "kiss": data.images.kiss,
        "normal": data.images.normal,
        "spread": data.images.spread,
        "open": data.images.open,
    }

    errors = image_processor.validate_images(images_dict)

    if errors:
        return {"valid": False, "errors": errors}

    return {"valid": True}
