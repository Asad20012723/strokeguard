'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { RiskGauge } from '@/components/results/RiskGauge'
import { RiskFactors } from '@/components/results/RiskFactors'
import { Recommendations } from '@/components/results/Recommendations'
import { ArrowLeft, Download, Share2, Clock } from 'lucide-react'
import type { AssessmentResult } from '@/types'

export default function ResultsPage() {
  const router = useRouter()
  const [result, setResult] = useState<AssessmentResult | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const storedResult = sessionStorage.getItem('assessmentResult')
    if (storedResult) {
      setResult(JSON.parse(storedResult))
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
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Share2 className="w-4 h-4 mr-2" />
            Share
          </Button>
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Risk Score */}
        <Card className="md:col-span-2">
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row items-center gap-8">
              <RiskGauge score={result.risk_score} riskLevel={result.risk_level} />
              <div className="flex-1 text-center md:text-left">
                <h2 className="text-2xl font-bold mb-2">
                  {result.risk_level === 'low' && 'Good News!'}
                  {result.risk_level === 'moderate' && 'Attention Recommended'}
                  {result.risk_level === 'high' && 'Action Required'}
                </h2>
                <p className="text-muted-foreground mb-4">
                  {result.risk_level === 'low' &&
                    'Your stroke risk assessment shows favorable results. Continue maintaining your healthy lifestyle.'}
                  {result.risk_level === 'moderate' &&
                    'Your assessment indicates some risk factors that could be improved. Consider consulting with a healthcare provider.'}
                  {result.risk_level === 'high' &&
                    'Your assessment shows elevated risk factors. We strongly recommend consulting with a healthcare professional soon.'}
                </p>
                <div className="flex items-center gap-4 text-sm text-muted-foreground justify-center md:justify-start">
                  <div className="flex items-center gap-1">
                    <span className="font-medium">Confidence:</span>
                    <span>{Math.round(result.confidence * 100)}%</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-4 h-4" />
                    <span>{Math.round(result.processing_time_ms)}ms</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Risk Factors */}
        <Card>
          <CardHeader>
            <CardTitle>Contributing Factors</CardTitle>
            <CardDescription>
              Health metrics that may contribute to your risk level
            </CardDescription>
          </CardHeader>
          <CardContent>
            <RiskFactors factors={result.contributing_factors} />
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
            <Recommendations recommendations={result.recommendations} />
          </CardContent>
        </Card>
      </div>

      {/* Disclaimer */}
      <div className="mt-8 bg-slate-50 border rounded-lg p-4 text-sm text-slate-600">
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
