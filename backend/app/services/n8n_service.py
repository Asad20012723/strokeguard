"""
n8n Webhook integration service for AI expert interpretation.
Sends clinical data to n8n RAG workflow for medical report generation.
"""

import httpx
import asyncio
from typing import Any, Dict, List, Optional
import os
import json
from datetime import datetime
import uuid


class N8NService:
    """
    Service for integrating with n8n workflows via webhooks.

    Features:
    - Send prediction results to n8n for RAG-based interpretation
    - Trigger medical report generation
    - Async webhook calls with retry logic
    """

    def __init__(self):
        self.webhook_url = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/stroke-analysis")
        self.report_webhook_url = os.getenv("N8N_REPORT_WEBHOOK_URL", "http://localhost:5678/webhook/generate-report")
        self.timeout = float(os.getenv("N8N_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("N8N_MAX_RETRIES", "3"))

    async def send_prediction_data(
        self,
        patient_id: str,
        clinical_data: Dict[str, Any],
        logistic_results: Optional[Dict[str, Any]],
        multimodal_results: Optional[Dict[str, Any]],
        image_provided: bool
    ) -> Dict[str, Any]:
        """
        Send prediction data to n8n webhook for AI interpretation.

        Args:
            patient_id: Unique patient identifier
            clinical_data: Raw health data input
            logistic_results: Results from statistical model
            multimodal_results: Results from multimodal model
            image_provided: Whether images were provided

        Returns:
            Response from n8n workflow
        """
        payload = {
            "patient_id": patient_id,
            "timestamp": datetime.utcnow().isoformat(),
            "clinical_data": self._format_clinical_data(clinical_data),
            "logistic_results": logistic_results or {},
            "multimodal_results": multimodal_results or {},
            "image_provided": image_provided,
            "request_type": "prediction_analysis"
        }

        return await self._send_webhook(self.webhook_url, payload)

    async def generate_medical_report(
        self,
        patient_id: str,
        clinical_data: Dict[str, Any],
        logistic_results: Optional[Dict[str, Any]],
        multimodal_results: Optional[Dict[str, Any]],
        shap_explanations: Optional[Dict[str, Any]],
        lime_explanations: Optional[Dict[str, Any]],
        image_provided: bool,
        report_format: str = "pdf"
    ) -> Dict[str, Any]:
        """
        Request medical report generation from n8n RAG agent.

        Args:
            patient_id: Unique patient identifier
            clinical_data: Raw health data
            logistic_results: Statistical model results
            multimodal_results: Multimodal model results
            shap_explanations: SHAP feature attributions
            lime_explanations: LIME explanations
            image_provided: Whether images were analyzed
            report_format: Output format (pdf, html, json)

        Returns:
            Response containing report URL or content
        """
        payload = {
            "patient_id": patient_id,
            "timestamp": datetime.utcnow().isoformat(),
            "request_type": "generate_report",
            "report_format": report_format,
            "data": {
                "clinical_data": self._format_clinical_data(clinical_data),
                "logistic_analysis": {
                    "enabled": logistic_results is not None,
                    "results": logistic_results or {}
                },
                "multimodal_analysis": {
                    "enabled": multimodal_results is not None,
                    "results": multimodal_results or {},
                    "image_analysis_performed": image_provided
                },
                "explainability": {
                    "shap": shap_explanations or {},
                    "lime": lime_explanations or {}
                }
            },
            "report_sections": [
                "executive_summary",
                "risk_assessment",
                "statistical_analysis",
                "contributing_factors",
                "recommendations",
                "methodology_notes"
            ]
        }

        return await self._send_webhook(self.report_webhook_url, payload)

    def _format_clinical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format clinical data for n8n consumption."""
        # Convert enum values to strings
        formatted = {}
        for key, value in data.items():
            if hasattr(value, "value"):
                formatted[key] = value.value
            elif hasattr(value, "dict"):
                formatted[key] = value.dict()
            else:
                formatted[key] = value

        # Add human-readable labels
        labels = {
            "age": "Age (years)",
            "gender": "Gender",
            "systolic": "Systolic Blood Pressure (mmHg)",
            "diastolic": "Diastolic Blood Pressure (mmHg)",
            "glucose": "Blood Glucose (mg/dL)",
            "bmi": "Body Mass Index",
            "cholesterol": "Total Cholesterol (mg/dL)",
            "smoking": "Smoking Status"
        }

        smoking_labels = {0: "Never Smoked", 1: "Former Smoker", 2: "Current Smoker"}

        formatted_with_labels = {}
        for key, value in formatted.items():
            label = labels.get(key, key)
            display_value = smoking_labels.get(value, value) if key == "smoking" else value
            formatted_with_labels[key] = {
                "value": value,
                "label": label,
                "display_value": display_value
            }

        return formatted_with_labels

    async def _send_webhook(
        self,
        url: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send webhook request with retry logic.

        Args:
            url: Webhook URL
            payload: JSON payload

        Returns:
            Response from webhook
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "X-Request-ID": str(uuid.uuid4())
                        }
                    )

                    if response.status_code == 200:
                        return {
                            "success": True,
                            "data": response.json() if response.content else {},
                            "status_code": response.status_code
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status_code}",
                            "status_code": response.status_code,
                            "response_text": response.text
                        }

            except httpx.TimeoutException as e:
                last_error = f"Timeout after {self.timeout}s"
                print(f"n8n webhook timeout (attempt {attempt + 1}): {e}")

            except httpx.ConnectError as e:
                last_error = f"Connection failed: {str(e)}"
                print(f"n8n webhook connection error (attempt {attempt + 1}): {e}")

            except Exception as e:
                last_error = str(e)
                print(f"n8n webhook error (attempt {attempt + 1}): {e}")

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        return {
            "success": False,
            "error": last_error or "Unknown error",
            "retries_exhausted": True
        }

    def create_patient_id(self) -> str:
        """Generate a unique patient identifier for the session."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique_part = uuid.uuid4().hex[:8]
        return f"PT-{timestamp}-{unique_part}"


# Global instance
n8n_service = N8NService()
