'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Loader2, AlertCircle, Camera, Calculator } from 'lucide-react'
import { EXPRESSIONS, type AssessmentData } from '@/types'
import { predictDualModel } from '@/lib/api-client'

interface ReviewSubmitProps {
  formData: AssessmentData
  onBack: () => void
  skipImages?: boolean
}

export function ReviewSubmit({ formData, onBack, skipImages = false }: ReviewSubmitProps) {
  const router = useRouter()
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async () => {
    if (!formData.healthData) {
      setError('Health data is missing')
      return
    }

    // Only check for images if we're not skipping them
    if (!skipImages) {
      const missingImages = EXPRESSIONS.filter(e => !formData.images[e.key])
      if (missingImages.length > 0) {
        setError(`Missing images: ${missingImages.map(e => e.title).join(', ')}`)
        return
      }
    }

    setIsSubmitting(true)
    setError(null)

    try {
      // Use dual-model endpoint which handles both cases
      const result = await predictDualModel({
        health_data: formData.healthData,
        images: skipImages ? null : {
          kiss: formData.images.kiss!,
          normal: formData.images.normal!,
          spread: formData.images.spread!,
          open: formData.images.open!,
        },
        include_statistics: true,
        include_explainability: !skipImages,
      })

      // Store result in sessionStorage and navigate to results page
      sessionStorage.setItem('assessmentResult', JSON.stringify(result))
      sessionStorage.setItem('assessmentMode', skipImages ? 'statistical' : 'multimodal')
      router.push('/results')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during assessment')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Assessment Mode Indicator */}
      <div className="flex items-center justify-center gap-2 p-3 rounded-lg bg-muted">
        {skipImages ? (
          <>
            <Calculator className="w-5 h-5 text-blue-600" />
            <span className="font-medium">Statistical Analysis Mode</span>
            <Badge variant="secondary">Tabular Data Only</Badge>
          </>
        ) : (
          <>
            <Camera className="w-5 h-5 text-green-600" />
            <span className="font-medium">Full Multimodal Analysis</span>
            <Badge variant="default">Images + Data</Badge>
          </>
        )}
      </div>

      {/* Health Data Review */}
      <Card>
        <CardHeader>
          <CardTitle>Health Information</CardTitle>
          <CardDescription>Review your entered health data</CardDescription>
        </CardHeader>
        <CardContent>
          {formData.healthData ? (
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Age:</span>{' '}
                <span className="font-medium">{formData.healthData.age} years</span>
              </div>
              <div>
                <span className="text-muted-foreground">Gender:</span>{' '}
                <span className="font-medium capitalize">{formData.healthData.gender}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Blood Pressure:</span>{' '}
                <span className="font-medium">
                  {formData.healthData.systolic}/{formData.healthData.diastolic} mmHg
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Blood Glucose:</span>{' '}
                <span className="font-medium">{formData.healthData.glucose} mg/dL</span>
              </div>
              <div>
                <span className="text-muted-foreground">BMI:</span>{' '}
                <span className="font-medium">{formData.healthData.bmi}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Cholesterol:</span>{' '}
                <span className="font-medium">{formData.healthData.cholesterol} mg/dL</span>
              </div>
              <div className="col-span-2">
                <span className="text-muted-foreground">Smoking Status:</span>{' '}
                <span className="font-medium">
                  {formData.healthData.smoking === 0
                    ? 'Never Smoked'
                    : formData.healthData.smoking === 1
                    ? 'Former Smoker'
                    : 'Current Smoker'}
                </span>
              </div>
            </div>
          ) : (
            <p className="text-destructive">Health data not provided</p>
          )}
        </CardContent>
      </Card>

      {/* Images Review - Only show if not skipping */}
      {!skipImages && (
        <Card>
          <CardHeader>
            <CardTitle>Captured Images</CardTitle>
            <CardDescription>Review your facial expression images</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-2">
              {EXPRESSIONS.map(expr => (
                <div key={expr.key} className="text-center">
                  <div className="aspect-square rounded-lg overflow-hidden bg-muted mb-1">
                    {formData.images[expr.key] ? (
                      <img
                        src={`data:image/jpeg;base64,${formData.images[expr.key]}`}
                        alt={expr.title}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-destructive">
                        <AlertCircle className="w-6 h-6" />
                      </div>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground">{expr.title}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Skip Images Notice */}
      {skipImages && (
        <Card className="border-blue-200 bg-blue-50 dark:bg-blue-950 dark:border-blue-800">
          <CardHeader>
            <CardTitle className="text-blue-800 dark:text-blue-200">Statistical Analysis Only</CardTitle>
            <CardDescription className="text-blue-700 dark:text-blue-300">
              Image analysis has been skipped
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-blue-800 dark:text-blue-200">
              Your assessment will use a logistic regression model to calculate stroke risk
              based on your health data. Results will include:
            </p>
            <ul className="mt-2 text-sm text-blue-700 dark:text-blue-300 list-disc list-inside space-y-1">
              <li>Risk score and risk level</li>
              <li>Odds ratios for each risk factor</li>
              <li>P-values and 95% confidence intervals</li>
              <li>Statistical interpretation</li>
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-destructive">Error</p>
            <p className="text-sm text-destructive/80">{error}</p>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-lg p-4 text-sm text-amber-800 dark:text-amber-200">
        <p className="font-medium mb-1">Important Notice</p>
        <p>
          This assessment is for educational purposes only and should not be considered
          medical advice. Always consult with healthcare professionals for any health concerns.
        </p>
      </div>

      {/* Actions */}
      <div className="flex gap-4">
        <Button variant="outline" onClick={onBack} className="flex-1" disabled={isSubmitting}>
          Back
        </Button>
        <Button onClick={handleSubmit} className="flex-1" disabled={isSubmitting}>
          {isSubmitting ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              {skipImages ? 'Analyzing Data...' : 'Analyzing...'}
            </>
          ) : (
            'Submit Assessment'
          )}
        </Button>
      </div>
    </div>
  )
}
