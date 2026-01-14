const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// ======================= Base Types =======================

export interface HealthData {
  age: number
  gender: 'male' | 'female'
  systolic: number
  diastolic: number
  glucose: number
  bmi: number
  cholesterol: number
  smoking: number
}

export interface ImageData {
  kiss: string
  normal: string
  spread: string
  open: string
}

export interface ContributingFactor {
  factor: string
  value: number
  threshold: number
  severity: 'low' | 'moderate' | 'high'
}

export interface ApiError {
  detail: string
  error_code?: string
}

// ======================= Original Prediction Types =======================

export interface PredictionRequest {
  health_data: HealthData
  images: ImageData
}

export interface PredictionResponse {
  risk_score: number
  risk_level: 'low' | 'moderate' | 'high'
  confidence: number
  contributing_factors: ContributingFactor[]
  recommendations: string[]
  processing_time_ms: number
}

// ======================= Statistical Model Types =======================

export interface FeatureContribution {
  value: number
  coefficient: number
  contribution: number
  odds_ratio: number
  p_value: number
  ci_95: [number, number]
  significant: boolean
}

export interface StatisticalPredictionRequest {
  health_data: HealthData
  include_statistics?: boolean
}

export interface StatisticalPredictionResponse {
  risk_score: number
  risk_level: 'low' | 'moderate' | 'high'
  odds_ratios: Record<string, number>
  p_values: Record<string, number>
  std_errors: Record<string, number>
  ci_lower: Record<string, number>
  ci_upper: Record<string, number>
  feature_contributions: Record<string, FeatureContribution>
  model_statistics: {
    pseudo_r2: number
    aic: number
    log_likelihood: number
    n_features: number
  }
  interpretation: string[]
  processing_time_ms: number
}

// ======================= Dual Model Types =======================

export interface ExplainabilityResults {
  shap_values: Record<string, unknown> | null
  lime_tabular: Record<string, unknown> | null
  lime_image: Record<string, unknown> | null
  clinical_summary: string[]
}

export interface MultimodalPredictionResponse {
  risk_score: number
  risk_level: 'low' | 'moderate' | 'high'
  confidence: number
  contributing_factors: ContributingFactor[]
  recommendations: string[]
  explainability: ExplainabilityResults | null
  processing_time_ms: number
}

export interface DualModelPredictionRequest {
  health_data: HealthData
  images?: ImageData | null
  include_statistics?: boolean
  include_explainability?: boolean
}

export interface DualModelPredictionResponse {
  patient_id: string
  image_provided: boolean
  statistical_results: StatisticalPredictionResponse
  multimodal_results: MultimodalPredictionResponse | null
  combined_risk_score: number
  combined_risk_level: 'low' | 'moderate' | 'high'
  recommendations: string[]
  total_processing_time_ms: number
}

// ======================= Report Generation Types =======================

export interface GenerateReportRequest {
  patient_id: string
  health_data: HealthData
  images?: ImageData | null
  report_format?: 'pdf' | 'html' | 'json'
}

export interface GenerateReportResponse {
  success: boolean
  report_url: string | null
  report_content: string | null
  error: string | null
}

// ======================= API Functions =======================

/**
 * Original prediction endpoint (requires images)
 */
export async function predictStrokeRisk(
  data: PredictionRequest
): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/predict/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: 'Unknown error occurred',
    }))
    throw new Error(error.detail || `Request failed with status ${response.status}`)
  }

  return response.json()
}

/**
 * Statistical-only prediction (no images required)
 */
export async function predictStatistical(
  data: StatisticalPredictionRequest
): Promise<StatisticalPredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/predict/statistical`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: 'Unknown error occurred',
    }))
    throw new Error(error.detail || `Request failed with status ${response.status}`)
  }

  return response.json()
}

/**
 * Dual-model prediction (images optional)
 */
export async function predictDualModel(
  data: DualModelPredictionRequest
): Promise<DualModelPredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/predict/dual`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: 'Unknown error occurred',
    }))
    throw new Error(error.detail || `Request failed with status ${response.status}`)
  }

  return response.json()
}

/**
 * Generate medical report via n8n
 */
export async function generateReport(
  data: GenerateReportRequest
): Promise<GenerateReportResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/predict/generate-report`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({
      detail: 'Unknown error occurred',
    }))
    throw new Error(error.detail || `Request failed with status ${response.status}`)
  }

  return response.json()
}

/**
 * Check API health
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}

/**
 * Validate prediction input
 */
export async function validateInput(data: PredictionRequest): Promise<{
  valid: boolean
  errors?: string[]
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/predict/validate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    })
    return response.json()
  } catch {
    return { valid: false, errors: ['Failed to validate input'] }
  }
}
