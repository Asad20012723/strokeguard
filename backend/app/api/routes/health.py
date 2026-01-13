"""
Health check endpoints.
"""

from fastapi import APIRouter, Depends

from app.models.schemas import HealthCheckResponse

router = APIRouter()


@router.get("/", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for monitoring and load balancers.
    """
    from app.main import model_service

    return HealthCheckResponse(
        status="healthy",
        model_loaded=model_service is not None and model_service.is_loaded(),
        version="1.0.0",
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Readiness check - returns 200 only when model is loaded.
    """
    from app.main import model_service

    if model_service is None or not model_service.is_loaded():
        return {"status": "not_ready", "reason": "Model not loaded"}

    return {"status": "ready"}
