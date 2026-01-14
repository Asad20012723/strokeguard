"""
Stroke risk prediction endpoints.
Supports dual-model inference (statistical + multimodal).
"""

import time
from typing import Any, Dict, Optional
import numpy as np

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks

from app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
    StatisticalPredictionRequest,
    StatisticalPredictionResponse,
    DualModelPredictionRequest,
    DualModelPredictionResponse,
    MultimodalPredictionResponse,
    ExplainabilityResults,
    FeatureContribution,
    GenerateReportRequest,
    GenerateReportResponse,
    RiskLevel,
    AIInterpretation,
    AIInterpretationSection,
)

router = APIRouter()


# ======================= Original Endpoint (Backwards Compatible) =======================

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
    (Original endpoint - backwards compatible)
    """
    from app.main import model_service, image_processor

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

        image_tensor = image_processor.preprocess_all_images(images_dict)
        health_dict = data.health_data.model_dump()
        tabular_tensor = model_service.prepare_tabular_data(health_dict)

        result = model_service.predict(image_tensor, tabular_tensor)
        contributing_factors = model_service.analyze_risk_factors(health_dict)
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


# ======================= Statistical Model Endpoint =======================

@router.post(
    "/statistical",
    response_model=StatisticalPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
async def predict_statistical(
    data: StatisticalPredictionRequest,
) -> StatisticalPredictionResponse:
    """
    Predict stroke risk using logistic regression (tabular data only).

    Returns detailed statistics including:
    - Odds ratios for each feature
    - P-values for statistical significance
    - 95% confidence intervals
    - Feature contributions to risk
    """
    from app.services.statistical_model import statistical_model_service

    total_start = time.time()

    try:
        health_dict = data.health_data.model_dump()
        result = statistical_model_service.predict(health_dict)

        total_time = (time.time() - total_start) * 1000

        # Convert feature contributions to schema format
        feature_contributions = {}
        for name, contrib in result.get("feature_contributions", {}).items():
            feature_contributions[name] = FeatureContribution(
                value=contrib["value"],
                coefficient=contrib["coefficient"],
                contribution=contrib["contribution"],
                odds_ratio=contrib["odds_ratio"],
                p_value=contrib["p_value"],
                ci_95=contrib["ci_95"],
                significant=contrib["significant"]
            )

        return StatisticalPredictionResponse(
            risk_score=result["risk_score"],
            risk_level=RiskLevel(result["risk_level"]),
            odds_ratios=result["odds_ratios"],
            p_values=result["p_values"],
            std_errors=result["std_errors"],
            ci_lower=result["ci_lower"],
            ci_upper=result["ci_upper"],
            feature_contributions=feature_contributions,
            model_statistics=result["model_statistics"],
            interpretation=result.get("interpretation", []),
            processing_time_ms=total_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Statistical prediction failed: {str(e)}"
        )


# ======================= Dual Model Endpoint =======================

@router.post(
    "/dual",
    response_model=DualModelPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def predict_dual_model(
    data: DualModelPredictionRequest,
    background_tasks: BackgroundTasks,
) -> DualModelPredictionResponse:
    """
    Dual-model stroke risk prediction.

    This endpoint:
    1. Always runs statistical (logistic regression) analysis
    2. Optionally runs multimodal analysis if images are provided
    3. Combines results for comprehensive risk assessment
    4. Includes SHAP/LIME explainability when requested
    """
    from app.main import model_service, image_processor
    from app.services.statistical_model import statistical_model_service
    from app.services.explainability_service import explainability_service
    from app.services.n8n_service import n8n_service

    total_start = time.time()
    health_dict = data.health_data.model_dump()
    image_provided = data.images is not None

    # Generate patient ID for this session
    patient_id = n8n_service.create_patient_id()

    try:
        # ===== Statistical Model (Always runs) =====
        stat_start = time.time()
        stat_result = statistical_model_service.predict(health_dict)
        stat_time = (time.time() - stat_start) * 1000

        # Convert to response format
        feature_contributions = {}
        for name, contrib in stat_result.get("feature_contributions", {}).items():
            feature_contributions[name] = FeatureContribution(
                value=contrib["value"],
                coefficient=contrib["coefficient"],
                contribution=contrib["contribution"],
                odds_ratio=contrib["odds_ratio"],
                p_value=contrib["p_value"],
                ci_95=contrib["ci_95"],
                significant=contrib["significant"]
            )

        statistical_response = StatisticalPredictionResponse(
            risk_score=stat_result["risk_score"],
            risk_level=RiskLevel(stat_result["risk_level"]),
            odds_ratios=stat_result["odds_ratios"],
            p_values=stat_result["p_values"],
            std_errors=stat_result["std_errors"],
            ci_lower=stat_result["ci_lower"],
            ci_upper=stat_result["ci_upper"],
            feature_contributions=feature_contributions,
            model_statistics=stat_result["model_statistics"],
            interpretation=stat_result.get("interpretation", []),
            processing_time_ms=stat_time,
        )

        # ===== Multimodal Model (If images provided) =====
        multimodal_response = None
        multimodal_risk_score = None

        if image_provided and model_service is not None and model_service.is_loaded():
            if image_processor is None:
                raise HTTPException(
                    status_code=503,
                    detail="Image processor not available.",
                )

            mm_start = time.time()

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

            image_tensor = image_processor.preprocess_all_images(images_dict)
            tabular_tensor = model_service.prepare_tabular_data(health_dict)

            result = model_service.predict(image_tensor, tabular_tensor)
            contributing_factors = model_service.analyze_risk_factors(health_dict)
            recommendations = model_service.generate_recommendations(
                health_dict, result["risk_level"]
            )

            multimodal_risk_score = result["risk_score"]

            # Generate explainability if requested
            explainability = None
            if data.include_explainability:
                tabular_np = tabular_tensor.cpu().numpy()

                shap_results = explainability_service.explain_tabular_shap(
                    model_service.model, tabular_np, image_tensor
                )
                lime_results = explainability_service.explain_tabular_lime(
                    model_service.model, tabular_np, image_tensor
                )

                clinical_summary = explainability_service.generate_clinical_summary(
                    shap_results, lime_results, result["risk_score"]
                )

                explainability = ExplainabilityResults(
                    shap_values=shap_results,
                    lime_tabular=lime_results,
                    lime_image=None,  # Can be enabled for detailed image analysis
                    clinical_summary=clinical_summary,
                )

            mm_time = (time.time() - mm_start) * 1000

            multimodal_response = MultimodalPredictionResponse(
                risk_score=result["risk_score"],
                risk_level=result["risk_level"],
                confidence=result["confidence"],
                contributing_factors=contributing_factors,
                recommendations=recommendations,
                explainability=explainability,
                processing_time_ms=mm_time,
            )

        # ===== Combine Results =====
        if multimodal_risk_score is not None:
            # Weighted average: 40% statistical, 60% multimodal when images available
            combined_risk_score = 0.4 * stat_result["risk_score"] + 0.6 * multimodal_risk_score
        else:
            combined_risk_score = stat_result["risk_score"]

        # Determine combined risk level
        if combined_risk_score < 0.3:
            combined_risk_level = RiskLevel.low
        elif combined_risk_score < 0.7:
            combined_risk_level = RiskLevel.moderate
        else:
            combined_risk_level = RiskLevel.high

        # Combine recommendations
        all_recommendations = stat_result.get("interpretation", [])
        if multimodal_response:
            all_recommendations.extend(multimodal_response.recommendations)
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        # ===== n8n AI Expert Interpretation =====
        ai_interpretation = None
        interpretation_source = "none"
        fallback_used = False
        fallback_reason = None

        try:
            # Prepare multimodal results dict for n8n
            multimodal_dict = None
            if multimodal_response:
                multimodal_dict = {
                    "risk_score": multimodal_response.risk_score,
                    "risk_level": multimodal_response.risk_level.value if hasattr(multimodal_response.risk_level, 'value') else str(multimodal_response.risk_level),
                    "confidence": multimodal_response.confidence,
                }

            # Get SHAP/LIME explanations if available
            shap_data = None
            lime_data = None
            if multimodal_response and multimodal_response.explainability:
                shap_data = multimodal_response.explainability.shap_values
                lime_data = multimodal_response.explainability.lime_tabular

            # Send to n8n for AI interpretation
            n8n_response = await n8n_service.send_for_interpretation(
                patient_id=patient_id,
                clinical_data=health_dict,
                logistic_results=stat_result,
                multimodal_results=multimodal_dict,
                shap_explanations=shap_data,
                lime_explanations=lime_data,
                image_provided=image_provided
            )

            if n8n_response.get("success"):
                interpretation_data = n8n_response.get("interpretation", {})
                interpretation_source = n8n_response.get("source", "unknown")
                fallback_used = n8n_response.get("fallback_used", False)
                fallback_reason = n8n_response.get("fallback_reason")

                # Convert sections to schema format
                sections = []
                for section in interpretation_data.get("sections", []):
                    sections.append(AIInterpretationSection(
                        title=section.get("title", ""),
                        content=section.get("content", "")
                    ))

                ai_interpretation = AIInterpretation(
                    risk_score=interpretation_data.get("risk_score", combined_risk_score),
                    risk_level=interpretation_data.get("risk_level", combined_risk_level.value),
                    sections=sections,
                    generated_by=interpretation_data.get("generated_by", interpretation_source),
                    timestamp=interpretation_data.get("timestamp", "")
                )

        except Exception as e:
            # If n8n call fails, continue without AI interpretation
            print(f"n8n interpretation error (non-blocking): {e}")
            interpretation_source = "error"
            fallback_reason = str(e)

        total_time = (time.time() - total_start) * 1000

        return DualModelPredictionResponse(
            patient_id=patient_id,
            image_provided=image_provided,
            statistical_results=statistical_response,
            multimodal_results=multimodal_response,
            combined_risk_score=combined_risk_score,
            combined_risk_level=combined_risk_level,
            recommendations=unique_recommendations,
            ai_interpretation=ai_interpretation,
            interpretation_source=interpretation_source,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            total_processing_time_ms=total_time,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Dual-model prediction failed: {str(e)}"
        )


# ======================= Report Generation Endpoint =======================

@router.post(
    "/generate-report",
    response_model=GenerateReportResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Report generation failed"},
    },
)
async def generate_medical_report(
    data: GenerateReportRequest,
) -> GenerateReportResponse:
    """
    Generate a medical report via n8n RAG agent.

    Sends all prediction data to n8n for AI-powered report synthesis.
    """
    from app.main import model_service, image_processor
    from app.services.statistical_model import statistical_model_service
    from app.services.explainability_service import explainability_service
    from app.services.n8n_service import n8n_service

    try:
        health_dict = data.health_data.model_dump()
        image_provided = data.images is not None

        # Run statistical prediction
        stat_result = statistical_model_service.predict(health_dict)

        # Run multimodal if images provided
        multimodal_result = None
        shap_explanations = None
        lime_explanations = None

        if image_provided and model_service is not None and model_service.is_loaded():
            images_dict = {
                "kiss": data.images.kiss,
                "normal": data.images.normal,
                "spread": data.images.spread,
                "open": data.images.open,
            }

            image_tensor = image_processor.preprocess_all_images(images_dict)
            tabular_tensor = model_service.prepare_tabular_data(health_dict)

            multimodal_result = model_service.predict(image_tensor, tabular_tensor)
            tabular_np = tabular_tensor.cpu().numpy()

            shap_explanations = explainability_service.explain_tabular_shap(
                model_service.model, tabular_np, image_tensor
            )
            lime_explanations = explainability_service.explain_tabular_lime(
                model_service.model, tabular_np, image_tensor
            )

        # Send to n8n for report generation
        response = await n8n_service.generate_medical_report(
            patient_id=data.patient_id,
            clinical_data=health_dict,
            logistic_results=stat_result,
            multimodal_results=multimodal_result,
            shap_explanations=shap_explanations,
            lime_explanations=lime_explanations,
            image_provided=image_provided,
            report_format=data.report_format,
        )

        if response.get("success"):
            return GenerateReportResponse(
                success=True,
                report_url=response.get("data", {}).get("report_url"),
                report_content=response.get("data", {}).get("report_content"),
                error=None,
            )
        else:
            return GenerateReportResponse(
                success=False,
                report_url=None,
                report_content=None,
                error=response.get("error", "Report generation failed"),
            )

    except Exception as e:
        return GenerateReportResponse(
            success=False,
            report_url=None,
            report_content=None,
            error=str(e),
        )


# ======================= Validation Endpoint =======================

@router.post("/validate")
async def validate_input(data: PredictionRequest) -> Dict[str, Any]:
    """
    Validate prediction input without running inference.
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
