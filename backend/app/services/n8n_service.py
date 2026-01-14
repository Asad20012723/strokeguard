"""
n8n Webhook integration service for AI expert interpretation.
Sends clinical data to n8n RAG workflow for medical report generation.
Includes rule-based fallback for error scenarios.
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
    - Rule-based fallback when n8n is unavailable
    """

    def __init__(self):
        # Primary n8n webhook endpoint
        self.webhook_url = os.getenv(
            "N8N_WEBHOOK_URL",
            "https://n8n.analytiqe.com/webhook/22d3939a-f398-44e6-913a-5326daeeb97c"
        )
        self.timeout = float(os.getenv("N8N_TIMEOUT", "60"))
        self.max_retries = int(os.getenv("N8N_MAX_RETRIES", "2"))

    async def send_for_interpretation(
        self,
        patient_id: str,
        clinical_data: Dict[str, Any],
        logistic_results: Optional[Dict[str, Any]],
        multimodal_results: Optional[Dict[str, Any]],
        shap_explanations: Optional[Dict[str, Any]] = None,
        lime_explanations: Optional[Dict[str, Any]] = None,
        image_provided: bool = False
    ) -> Dict[str, Any]:
        """
        Send prediction data to n8n webhook for AI interpretation.

        Args:
            patient_id: Unique patient identifier
            clinical_data: Raw health data input
            logistic_results: Results from statistical model
            multimodal_results: Results from multimodal model
            shap_explanations: SHAP feature attributions
            lime_explanations: LIME explanations
            image_provided: Whether images were provided

        Returns:
            AI interpretation response or rule-based fallback
        """
        payload = {
            "request_type": "medical_interpretation",
            "patient_id": patient_id,
            "timestamp": datetime.utcnow().isoformat(),
            "clinical_data": self._format_clinical_data(clinical_data),
            "analysis_results": {
                "statistical": {
                    "enabled": logistic_results is not None,
                    "risk_score": logistic_results.get("risk_score") if logistic_results else None,
                    "risk_level": logistic_results.get("risk_level") if logistic_results else None,
                    "odds_ratios": logistic_results.get("odds_ratios") if logistic_results else {},
                    "p_values": logistic_results.get("p_values") if logistic_results else {},
                    "significant_factors": self._get_significant_factors(logistic_results) if logistic_results else []
                },
                "multimodal": {
                    "enabled": multimodal_results is not None,
                    "risk_score": multimodal_results.get("risk_score") if multimodal_results else None,
                    "risk_level": multimodal_results.get("risk_level") if multimodal_results else None,
                    "confidence": multimodal_results.get("confidence") if multimodal_results else None,
                    "image_analysis_performed": image_provided
                }
            },
            "explainability": {
                "shap": shap_explanations or {},
                "lime": lime_explanations or {}
            }
        }

        # Try to get AI interpretation from n8n
        response = await self._send_webhook(payload)

        if response.get("success"):
            return {
                "success": True,
                "source": "n8n_ai_agent",
                "interpretation": response.get("data", {}),
                "fallback_used": False
            }
        else:
            # Fallback to rule-based interpretation
            fallback_interpretation = self._generate_rule_based_interpretation(
                clinical_data, logistic_results, multimodal_results, image_provided
            )
            return {
                "success": True,
                "source": "rule_based_fallback",
                "interpretation": fallback_interpretation,
                "fallback_used": True,
                "fallback_reason": response.get("error", "n8n service unavailable")
            }

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
            Response containing report URL/content or fallback report
        """
        payload = {
            "request_type": "generate_report",
            "patient_id": patient_id,
            "timestamp": datetime.utcnow().isoformat(),
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

        response = await self._send_webhook(payload)

        if response.get("success"):
            return {
                "success": True,
                "source": "n8n_ai_agent",
                "data": response.get("data", {}),
                "fallback_used": False
            }
        else:
            # Generate rule-based report as fallback
            fallback_report = self._generate_rule_based_report(
                patient_id, clinical_data, logistic_results, multimodal_results, image_provided
            )
            return {
                "success": True,
                "source": "rule_based_fallback",
                "data": {"report_content": fallback_report},
                "fallback_used": True,
                "fallback_reason": response.get("error", "n8n service unavailable")
            }

    def _format_clinical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format clinical data for n8n consumption."""
        formatted = {}
        for key, value in data.items():
            if hasattr(value, "value"):
                formatted[key] = value.value
            elif hasattr(value, "dict"):
                formatted[key] = value.dict()
            else:
                formatted[key] = value

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

    def _get_significant_factors(self, logistic_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract statistically significant factors from logistic regression results."""
        significant = []
        p_values = logistic_results.get("p_values", {})
        odds_ratios = logistic_results.get("odds_ratios", {})
        feature_contributions = logistic_results.get("feature_contributions", {})

        for feature, p_val in p_values.items():
            if p_val < 0.05:
                contribution = feature_contributions.get(feature, {})
                significant.append({
                    "feature": feature,
                    "odds_ratio": odds_ratios.get(feature, 1.0),
                    "p_value": p_val,
                    "direction": "increases_risk" if odds_ratios.get(feature, 1.0) > 1 else "decreases_risk",
                    "value": contribution.get("value") if contribution else None
                })

        return sorted(significant, key=lambda x: x["p_value"])

    async def _send_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send POST request to n8n webhook and await response.

        Args:
            payload: JSON payload to send

        Returns:
            Response from webhook or error dict
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.webhook_url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                            "X-Request-ID": str(uuid.uuid4()),
                            "X-Patient-ID": payload.get("patient_id", "unknown")
                        }
                    )

                    if response.status_code == 200:
                        try:
                            data = response.json()
                            return {
                                "success": True,
                                "data": data,
                                "status_code": response.status_code
                            }
                        except json.JSONDecodeError:
                            # Response is not JSON, treat text as interpretation
                            return {
                                "success": True,
                                "data": {"interpretation": response.text},
                                "status_code": response.status_code
                            }
                    elif response.status_code >= 500:
                        # Server error - will trigger fallback
                        last_error = f"n8n server error: HTTP {response.status_code}"
                        print(f"n8n webhook server error (attempt {attempt + 1}): {last_error}")
                    else:
                        # Client error
                        return {
                            "success": False,
                            "error": f"HTTP {response.status_code}: {response.text[:200]}",
                            "status_code": response.status_code
                        }

            except httpx.TimeoutException as e:
                last_error = f"Request timeout after {self.timeout}s"
                print(f"n8n webhook timeout (attempt {attempt + 1}): {e}")

            except httpx.ConnectError as e:
                last_error = f"Connection failed to n8n.analytiqe.com: {str(e)}"
                print(f"n8n webhook connection error (attempt {attempt + 1}): {e}")

            except Exception as e:
                last_error = str(e)
                print(f"n8n webhook error (attempt {attempt + 1}): {e}")

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        return {
            "success": False,
            "error": last_error or "Unknown error after retries",
            "retries_exhausted": True
        }

    def _generate_rule_based_interpretation(
        self,
        clinical_data: Dict[str, Any],
        logistic_results: Optional[Dict[str, Any]],
        multimodal_results: Optional[Dict[str, Any]],
        image_provided: bool
    ) -> Dict[str, Any]:
        """
        Generate rule-based medical interpretation as fallback.
        Used when n8n is unavailable.
        """
        interpretation_sections = []

        # Determine overall risk
        if logistic_results:
            risk_score = logistic_results.get("risk_score", 0)
            risk_level = logistic_results.get("risk_level", "unknown")
        elif multimodal_results:
            risk_score = multimodal_results.get("risk_score", 0)
            risk_level = multimodal_results.get("risk_level", "unknown")
        else:
            risk_score = 0
            risk_level = "unknown"

        # Executive Summary
        if risk_level == "high":
            summary = (
                "This patient presents with elevated stroke risk factors requiring immediate "
                "clinical attention. The assessment indicates significant modifiable risk factors "
                "that should be addressed through lifestyle modifications and potential pharmacological intervention."
            )
        elif risk_level == "moderate":
            summary = (
                "This patient shows moderate stroke risk with several contributing factors identified. "
                "Preventive measures and lifestyle modifications are recommended, along with regular monitoring "
                "of key health indicators."
            )
        else:
            summary = (
                "This patient's stroke risk assessment indicates favorable results with low immediate concern. "
                "Continuation of healthy lifestyle practices is recommended with routine health monitoring."
            )

        interpretation_sections.append({
            "title": "Executive Summary",
            "content": summary
        })

        # Risk Factor Analysis
        risk_factors = []
        if logistic_results and "feature_contributions" in logistic_results:
            for feature, data in logistic_results["feature_contributions"].items():
                if data.get("significant", False):
                    or_val = data.get("odds_ratio", 1)
                    p_val = data.get("p_value", 1)

                    if or_val > 1.5:
                        risk_factors.append(
                            f"- {self._get_readable_name(feature)}: Significantly elevated "
                            f"(OR={or_val:.2f}, p={p_val:.4f}), contributing to increased stroke risk."
                        )
                    elif or_val < 0.7:
                        risk_factors.append(
                            f"- {self._get_readable_name(feature)}: Protective factor identified "
                            f"(OR={or_val:.2f}, p={p_val:.4f})."
                        )

        if risk_factors:
            interpretation_sections.append({
                "title": "Key Risk Factor Analysis",
                "content": "\n".join(risk_factors)
            })

        # Clinical Recommendations
        recommendations = self._generate_clinical_recommendations(clinical_data, risk_level)
        interpretation_sections.append({
            "title": "Clinical Recommendations",
            "content": "\n".join([f"- {r}" for r in recommendations])
        })

        # Methodology Note
        method_note = (
            "This interpretation was generated using a rule-based clinical decision support system. "
            "Results are based on logistic regression analysis of patient health metrics"
        )
        if image_provided:
            method_note += " combined with multimodal facial expression analysis"
        method_note += ". This assessment should be used as a screening tool and does not replace clinical judgment."

        interpretation_sections.append({
            "title": "Methodology Note",
            "content": method_note
        })

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "sections": interpretation_sections,
            "generated_by": "rule_based_system",
            "timestamp": datetime.utcnow().isoformat()
        }

    def _generate_rule_based_report(
        self,
        patient_id: str,
        clinical_data: Dict[str, Any],
        logistic_results: Optional[Dict[str, Any]],
        multimodal_results: Optional[Dict[str, Any]],
        image_provided: bool
    ) -> str:
        """Generate a text-based medical report as fallback."""
        interpretation = self._generate_rule_based_interpretation(
            clinical_data, logistic_results, multimodal_results, image_provided
        )

        report_lines = [
            "=" * 60,
            "STROKE RISK ASSESSMENT REPORT",
            "=" * 60,
            "",
            f"Patient ID: {patient_id}",
            f"Assessment Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"Risk Level: {interpretation['risk_level'].upper()}",
            f"Risk Score: {interpretation['risk_score']:.1%}",
            "",
            "-" * 60,
        ]

        for section in interpretation.get("sections", []):
            report_lines.extend([
                "",
                section["title"].upper(),
                "-" * len(section["title"]),
                section["content"],
                ""
            ])

        report_lines.extend([
            "-" * 60,
            "",
            "DISCLAIMER",
            "This report is generated by an automated clinical decision support system.",
            "It is intended for use by healthcare professionals and should not be used",
            "as the sole basis for clinical decisions. Always correlate with clinical",
            "findings and other diagnostic information.",
            "",
            "=" * 60
        ])

        return "\n".join(report_lines)

    def _generate_clinical_recommendations(
        self,
        clinical_data: Dict[str, Any],
        risk_level: str
    ) -> List[str]:
        """Generate clinical recommendations based on health data."""
        recommendations = []

        # Risk level-based recommendations
        if risk_level == "high":
            recommendations.append(
                "Urgent referral to cardiology/neurology for comprehensive stroke risk evaluation"
            )
            recommendations.append(
                "Consider initiation of antiplatelet therapy if not contraindicated"
            )
        elif risk_level == "moderate":
            recommendations.append(
                "Schedule follow-up appointment within 2-4 weeks for risk factor management"
            )

        # Blood pressure
        systolic = clinical_data.get("systolic", 0)
        if isinstance(systolic, dict):
            systolic = systolic.get("value", 0)
        if systolic > 140:
            recommendations.append(
                "Blood pressure optimization recommended - consider antihypertensive therapy "
                "with target BP < 130/80 mmHg"
            )

        # Glucose
        glucose = clinical_data.get("glucose", 0)
        if isinstance(glucose, dict):
            glucose = glucose.get("value", 0)
        if glucose > 140:
            recommendations.append(
                "Glycemic control assessment needed - consider HbA1c testing and "
                "diabetes screening if not previously diagnosed"
            )

        # BMI
        bmi = clinical_data.get("bmi", 0)
        if isinstance(bmi, dict):
            bmi = bmi.get("value", 0)
        if bmi > 30:
            recommendations.append(
                "Weight management program recommended - dietary consultation and "
                "structured exercise program"
            )

        # Cholesterol
        cholesterol = clinical_data.get("cholesterol", 0)
        if isinstance(cholesterol, dict):
            cholesterol = cholesterol.get("value", 0)
        if cholesterol > 240:
            recommendations.append(
                "Lipid panel review recommended - consider statin therapy for "
                "cardiovascular risk reduction"
            )

        # Smoking
        smoking = clinical_data.get("smoking", 0)
        if isinstance(smoking, dict):
            smoking = smoking.get("value", 0)
        if smoking == 2:
            recommendations.append(
                "Smoking cessation is critical - refer to tobacco cessation program, "
                "consider pharmacological support (varenicline/bupropion)"
            )
        elif smoking == 1:
            recommendations.append(
                "Continue smoking abstinence - cardiovascular benefits increase with "
                "continued cessation"
            )

        # General recommendations
        recommendations.append(
            "Lifestyle modifications: Mediterranean diet, regular aerobic exercise "
            "(150 min/week moderate intensity), stress management"
        )

        return recommendations

    def _get_readable_name(self, feature: str) -> str:
        """Convert feature name to readable format."""
        names = {
            "age": "Age",
            "gender": "Gender",
            "systolic": "Systolic Blood Pressure",
            "diastolic": "Diastolic Blood Pressure",
            "glucose": "Blood Glucose",
            "bmi": "Body Mass Index",
            "cholesterol": "Total Cholesterol",
            "smoking_former": "Former Smoking Status",
            "smoking_current": "Current Smoking Status"
        }
        return names.get(feature, feature.replace("_", " ").title())

    def create_patient_id(self) -> str:
        """Generate a unique patient identifier for the session."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique_part = uuid.uuid4().hex[:8]
        return f"PT-{timestamp}-{unique_part}"


# Global instance
n8n_service = N8NService()
