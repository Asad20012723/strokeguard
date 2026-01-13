"""
FastAPI main application for Stroke Monitoring System.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import health, prediction
from app.services.model_service import ModelService
from app.services.image_processor import ImagePreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global service instances
model_service: Optional[ModelService] = None
image_processor: Optional[ImagePreprocessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads ML models on startup and cleans up on shutdown.
    """
    global model_service, image_processor

    settings = get_settings()

    logger.info("Starting Stroke Monitoring API...")
    logger.info(f"Model path: {settings.MODEL_PATH}")
    logger.info(f"Device: {settings.MODEL_DEVICE}")

    # Initialize image processor
    logger.info("Initializing image processor...")
    image_processor = ImagePreprocessor(
        target_size=(640, 480),  # Match training dimensions
        use_background_removal=False,  # Disable for faster inference
        device=settings.MODEL_DEVICE,
    )

    # Initialize and load model service
    logger.info("Loading ML model...")
    model_service = ModelService(
        model_path=settings.MODEL_PATH,
        device=settings.MODEL_DEVICE,
    )

    try:
        model_service.load()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model weights: {e}")
        logger.warning("API will run with random weights (for testing)")

    yield

    # Cleanup
    logger.info("Shutting down...")
    model_service = None
    image_processor = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="""
        Stroke Monitoring System API

        This API provides stroke risk prediction using multimodal deep learning,
        combining facial expression analysis with health metrics.

        ## Features
        - Analyze 4 facial expressions (kiss, normal, spread, open)
        - Process health data (age, BP, glucose, BMI, cholesterol, smoking)
        - Return risk score with contributing factors and recommendations

        ## Usage
        1. Collect health metrics from the patient
        2. Capture 4 facial expression images
        3. Submit to /api/v1/predict endpoint
        4. Receive risk assessment with recommendations
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(
        health.router,
        prefix="/health",
        tags=["Health Check"],
    )
    app.include_router(
        prediction.router,
        prefix=f"{settings.API_V1_PREFIX}/predict",
        tags=["Prediction"],
    )

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": settings.PROJECT_NAME,
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
