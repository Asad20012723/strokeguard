'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { RiskGauge } from '@/components/results/RiskGauge'
import { RiskFactors } from '@/components/results/RiskFactors'
import { Recommendations } from '@/components/results/Recommendations'
import { ClinicalEvidence } from '@/components/results/ClinicalEvidence'
import { GenerateReportButton } from '@/components/results/GenerateReportButton'
import { ArrowLeft, Download, Share2, Clock, Calculator, Camera, Bot, AlertTriangle } from 'lucide-react'
import type { DualModelPredictionResponse, HealthData, ImageData, AIInterpretation } from '@/lib/api-client'

interface ExtendedResult extends Omit<DualModelPredictionResponse, 'ai_interpretation' | 'interpretation_source' | 'fallback_used' | 'fallback_reason'> {
  // Legacy fields for backwards compatibility
  risk_score?: number
  risk_level?: 'low' | 'moderate' | 'high'
  confidence?: number
  contributing_factors?: Array<{
    factor: string
    value: number
    threshold: number
    severity: 'low' | 'moderate' | 'high'
  }>
  processing_time_ms?: number
  // AI interpretation fields (may be undefined in legacy responses)
  ai_interpretation?: AIInterpretation | null
  interpretation_source?: string
  fallback_used?: boolean
  fallback_reason?: string | null
}

export default function ResultsPage() {
  const router = useRouter()
  const [result, setResult] = useState<ExtendedResult | null>(null)
  const [assessmentMode, setAssessmentMode] = useState<'statistical' | 'multimodal'>('multimodal')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const storedResult = sessionStorage.getItem('assessmentResult')
    const storedMode = sessionStorage.getItem('assessmentMode')

    if (storedResult) {
      setResult(JSON.parse(storedResult))
    }
    if (storedMode) {
      setAssessmentMode(storedMode as 'statistical' | 'multimodal')
    }
    setLoading(false)
  }, [])

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-slate-200 rounded w-1/3"></div>
          <div className="h-64 bg-slate-200 rounded"></div>
          <div className="h-32 bg-slate-200 rounded"></div>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="container mx-auto px-4 py-12 max-w-4xl text-center">
        <h1 className="text-2xl font-bold mb-4">No Results Found</h1>
        <p className="text-muted-foreground mb-6">
          It looks like you haven't completed an assessment yet.
        </p>
        <Link href="/assessment">
          <Button>Start Assessment</Button>
        </Link>
      </div>
    )
  }

  // Handle both new dual-model format and legacy format
  const isDualModel = 'statistical_results' in result
  const riskScore = isDualModel ? result.combined_risk_score : (result.risk_score || 0)
  const riskLevel = isDualModel ? result.combined_risk_level : (result.risk_level || 'low')
  const processingTime = isDualModel ? result.total_processing_time_ms : (result.processing_time_ms || 0)
  const recommendations = isDualModel ? result.recommendations : (result.recommendations || [])
  const confidence = isDualModel
    ? (result.multimodal_results?.confidence || 0.7)
    : (result.confidence || 0.7)
  const contributingFactors = isDualModel
    ? (result.multimodal_results?.contributing_factors || [])
    : (result.contributing_factors || [])

  // Get health data for report generation (from session if available)
  const healthData: HealthData = {
    age: 55,
    gender: 'male',
    systolic: 130,
    diastolic: 80,
    glucose: 110,
    bmi: 26,
    cholesterol: 200,
    smoking: 0,
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <Link
            href="/assessment"
            className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1 mb-2"
          >
            <ArrowLeft className="w-4 h-4" />
            New Assessment
          </Link>
          <h1 className="text-3xl font-bold">Your Assessment Results</h1>
          {isDualModel && (
            <div className="flex items-center gap-2 mt-2 flex-wrap">
              {result.image_provided ? (
                <>
                  <Camera className="w-4 h-4 text-green-600" />
                  <Badge variant="default">Full Multimodal Analysis</Badge>
                </>
              ) : (
                <>
                  <Calculator className="w-4 h-4 text-blue-600" />
                  <Badge variant="secondary">Statistical Analysis</Badge>
                </>
              )}
              {result.ai_interpretation && (
                <>
                  <Bot className="w-4 h-4 text-purple-600" />
                  <Badge variant={result.fallback_used ? "outline" : "default"} className={result.fallback_used ? "" : "bg-purple-600"}>
                    {result.fallback_used ? "Rule-Based Interpretation" : "AI Expert Interpretation"}
                  </Badge>
                </>
              )}
              <span className="text-sm text-muted-foreground">
                Patient ID: {result.patient_id}
              </span>
            </div>
          )}
        </div>
        <div className="flex gap-2">
          {isDualModel && (
            <GenerateReportButton
              patientId={result.patient_id}
              healthData={healthData}
              images={result.image_provided ? undefined : null}
            />
          )}
          <Button variant="outline" size="sm">
            <Share2 className="w-4 h-4 mr-2" />
            Share
          </Button>
        </div>
      </div>

      {/* Risk Score Overview */}
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row items-center gap-8">
            <RiskGauge score={riskScore} riskLevel={riskLevel} />
            <div className="flex-1 text-center md:text-left">
              <h2 className="text-2xl font-bold mb-2">
                {riskLevel === 'low' && 'Good News!'}
                {riskLevel === 'moderate' && 'Attention Recommended'}
                {riskLevel === 'high' && 'Action Required'}
              </h2>
              <p className="text-muted-foreground mb-4">
                {riskLevel === 'low' &&
                  'Your stroke risk assessment shows favorable results. Continue maintaining your healthy lifestyle.'}
                {riskLevel === 'moderate' &&
                  'Your assessment indicates some risk factors that could be improved. Consider consulting with a healthcare provider.'}
                {riskLevel === 'high' &&
                  'Your assessment shows elevated risk factors. We strongly recommend consulting with a healthcare professional soon.'}
              </p>
              <div className="flex items-center gap-4 text-sm text-muted-foreground justify-center md:justify-start flex-wrap">
                <div className="flex items-center gap-1">
                  <span className="font-medium">Confidence:</span>
                  <span>{Math.round(confidence * 100)}%</span>
                </div>
                <div className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  <span>{Math.round(processingTime)}ms</span>
                </div>
                {isDualModel && (
                  <>
                    <div className="flex items-center gap-1">
                      <span className="font-medium">Statistical:</span>
                      <span>{Math.round(result.statistical_results.risk_score * 100)}%</span>
                    </div>
                    {result.multimodal_results && (
                      <div className="flex items-center gap-1">
                        <span className="font-medium">Multimodal:</span>
                        <span>{Math.round(result.multimodal_results.risk_score * 100)}%</span>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tabbed Content for Dual Model */}
      {isDualModel ? (
        <Tabs defaultValue="summary" className="space-y-6">
          <TabsList className={`grid w-full ${result.ai_interpretation ? 'grid-cols-4' : 'grid-cols-3'}`}>
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="clinical">Clinical Evidence</TabsTrigger>
            {result.ai_interpretation && (
              <TabsTrigger value="ai-expert">AI Expert</TabsTrigger>
            )}
            {result.multimodal_results && (
              <TabsTrigger value="multimodal">Image Analysis</TabsTrigger>
            )}
          </TabsList>

          {/* Summary Tab */}
          <TabsContent value="summary" className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              {/* Risk Factors */}
              <Card>
                <CardHeader>
                  <CardTitle>Contributing Factors</CardTitle>
                  <CardDescription>
                    Health metrics that may contribute to your risk level
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <RiskFactors factors={contributingFactors} />
                </CardContent>
              </Card>

              {/* Recommendations */}
              <Card>
                <CardHeader>
                  <CardTitle>Recommendations</CardTitle>
                  <CardDescription>
                    Personalized suggestions based on your assessment
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Recommendations recommendations={recommendations} />
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Clinical Evidence Tab */}
          <TabsContent value="clinical">
            <ClinicalEvidence
              statisticalResults={result.statistical_results}
              explainability={result.multimodal_results?.explainability}
              showMultimodal={result.image_provided}
            />
          </TabsContent>

          {/* AI Expert Interpretation Tab */}
          {result.ai_interpretation && (
            <TabsContent value="ai-expert" className="space-y-6">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Bot className="w-5 h-5 text-purple-600" />
                      <CardTitle>AI Expert Interpretation</CardTitle>
                    </div>
                    <div className="flex items-center gap-2">
                      {result.fallback_used ? (
                        <Badge variant="secondary" className="flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          Rule-Based Fallback
                        </Badge>
                      ) : (
                        <Badge variant="default" className="bg-purple-600">
                          AI Generated
                        </Badge>
                      )}
                    </div>
                  </div>
                  <CardDescription>
                    {result.fallback_used
                      ? `Clinical interpretation generated using rule-based system. ${result.fallback_reason || ''}`
                      : `Clinical interpretation generated by ${result.ai_interpretation.generated_by}`}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {result.ai_interpretation.sections.map((section, index) => (
                    <div key={index} className="border-b pb-4 last:border-b-0 last:pb-0">
                      <h4 className="font-semibold text-lg mb-2">{section.title}</h4>
                      <div className="text-sm text-muted-foreground whitespace-pre-wrap">
                        {section.content}
                      </div>
                    </div>
                  ))}
                  {result.ai_interpretation.timestamp && (
                    <div className="text-xs text-muted-foreground text-right">
                      Generated: {new Date(result.ai_interpretation.timestamp).toLocaleString()}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          )}

          {/* Multimodal Analysis Tab */}
          {result.multimodal_results && (
            <TabsContent value="multimodal" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Multimodal Analysis Results</CardTitle>
                  <CardDescription>
                    Combined analysis of facial expressions and health data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium mb-2">Risk Metrics</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Risk Score:</span>
                          <span className="font-medium">
                            {Math.round(result.multimodal_results.risk_score * 100)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Confidence:</span>
                          <span className="font-medium">
                            {Math.round(result.multimodal_results.confidence * 100)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Processing Time:</span>
                          <span className="font-medium">
                            {Math.round(result.multimodal_results.processing_time_ms)}ms
                          </span>
                        </div>
                      </div>
                    </div>

                    {result.multimodal_results.explainability?.clinical_summary && (
                      <div>
                        <h4 className="font-medium mb-2">AI Clinical Summary</h4>
                        <ul className="space-y-1 text-sm">
                          {result.multimodal_results.explainability.clinical_summary.map(
                            (item, index) => (
                              <li key={index} className="flex items-start gap-2">
                                <span className="text-primary">-</span>
                                <span>{item}</span>
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          )}
        </Tabs>
      ) : (
        // Legacy display for non-dual-model results
        <div className="grid md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Contributing Factors</CardTitle>
              <CardDescription>
                Health metrics that may contribute to your risk level
              </CardDescription>
            </CardHeader>
            <CardContent>
              <RiskFactors factors={contributingFactors} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recommendations</CardTitle>
              <CardDescription>
                Personalized suggestions based on your assessment
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Recommendations recommendations={recommendations} />
            </CardContent>
          </Card>
        </div>
      )}

      {/* Disclaimer */}
      <div className="mt-8 bg-slate-50 dark:bg-slate-900 border rounded-lg p-4 text-sm text-slate-600 dark:text-slate-400">
        <p className="font-medium mb-1">Medical Disclaimer</p>
        <p>
          This assessment is provided for educational and informational purposes only and is not
          intended as medical advice. It should not be used to diagnose, treat, cure, or prevent
          any disease or health condition. Always consult with qualified healthcare professionals
          for medical advice, diagnosis, and treatment. If you are experiencing symptoms of a
          stroke or any medical emergency, call emergency services immediately.
        </p>
      </div>
    </div>
  )
}
