export type RiskLevel = 'low' | 'moderate' | 'high'
export type Severity = 'low' | 'moderate' | 'high'
export type Gender = 'male' | 'female'
export type SmokingStatus = 0 | 1 | 2

export interface HealthFormData {
  age: number
  gender: Gender
  systolic: number
  diastolic: number
  glucose: number
  bmi: number
  cholesterol: number
  smoking: SmokingStatus
}

export interface ImageFormData {
  kiss: string | null
  normal: string | null
  spread: string | null
  open: string | null
}

export interface AssessmentData {
  healthData: HealthFormData | null
  images: ImageFormData
}

export interface ContributingFactor {
  factor: string
  value: number
  threshold: number
  severity: Severity
}

export interface AssessmentResult {
  risk_score: number
  risk_level: RiskLevel
  confidence: number
  contributing_factors: ContributingFactor[]
  recommendations: string[]
  processing_time_ms: number
}

export const EXPRESSIONS = [
  {
    key: 'normal' as const,
    title: 'Normal',
    instruction: 'Keep your face relaxed and look straight at the camera',
  },
  {
    key: 'kiss' as const,
    title: 'Kiss',
    instruction: 'Purse your lips as if you are going to kiss',
  },
  {
    key: 'spread' as const,
    title: 'Spread',
    instruction: 'Spread your lips wide showing your teeth (like saying "cheese")',
  },
  {
    key: 'open' as const,
    title: 'Open',
    instruction: 'Open your mouth wide as if yawning',
  },
] as const

export type ExpressionKey = typeof EXPRESSIONS[number]['key']
