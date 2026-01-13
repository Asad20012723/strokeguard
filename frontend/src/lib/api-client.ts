const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

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

export interface PredictionRequest {
  health_data: HealthData
  images: ImageData
}

export interface ContributingFactor {
  factor: string
  value: number
  threshold: number
  severity: 'low' | 'moderate' | 'high'
}

export interface PredictionResponse {
  risk_score: number
  risk_level: 'low' | 'moderate' | 'high'
  confidence: number
  contributing_factors: ContributingFactor[]
  recommendations: string[]
  processing_time_ms: number
}

export interface ApiError {
  detail: string
  error_code?: string
}

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

export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}

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
