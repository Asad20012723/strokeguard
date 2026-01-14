'use client'

import { useState } from 'react'
import { Progress } from '@/components/ui/progress'
import { HealthDataForm } from '@/components/forms/HealthDataForm'
import { ImageCapture } from '@/components/forms/ImageCapture'
import { ReviewSubmit } from '@/components/forms/ReviewSubmit'
import type { AssessmentData, HealthFormData, ImageFormData } from '@/types'

export default function AssessmentPage() {
  const [currentStep, setCurrentStep] = useState(1)
  const [skipImages, setSkipImages] = useState(false)
  const [formData, setFormData] = useState<AssessmentData>({
    healthData: null,
    images: { kiss: null, normal: null, spread: null, open: null },
  })

  // Dynamic steps based on whether images are skipped
  const STEPS = skipImages
    ? [
        { id: 1, title: 'Health Info', description: 'Enter your health metrics' },
        { id: 2, title: 'Review', description: 'Review and submit' },
      ]
    : [
        { id: 1, title: 'Health Info', description: 'Enter your health metrics' },
        { id: 2, title: 'Capture Images', description: 'Take 4 facial expression photos' },
        { id: 3, title: 'Review', description: 'Review and submit' },
      ]

  const progress = ((currentStep - 1) / (STEPS.length - 1)) * 100

  const handleHealthDataSubmit = (data: HealthFormData, shouldSkipImages: boolean) => {
    setFormData(prev => ({ ...prev, healthData: data }))
    setSkipImages(shouldSkipImages)

    if (shouldSkipImages) {
      // Clear images and go directly to review
      setFormData(prev => ({
        ...prev,
        healthData: data,
        images: { kiss: null, normal: null, spread: null, open: null },
      }))
      setCurrentStep(2)
    } else {
      setCurrentStep(2)
    }
  }

  const handleImagesSubmit = (images: ImageFormData) => {
    setFormData(prev => ({ ...prev, images }))
    setCurrentStep(3)
  }

  const handleBackFromReview = () => {
    if (skipImages) {
      setCurrentStep(1)
    } else {
      setCurrentStep(2)
    }
  }

  // Calculate actual step for display
  const displayStep = skipImages && currentStep === 2 ? 2 : currentStep
  const isReviewStep = skipImages ? currentStep === 2 : currentStep === 3
  const isImageStep = !skipImages && currentStep === 2

  return (
    <div className="container mx-auto px-4 py-8 max-w-2xl">
      {/* Progress Header */}
      <div className="mb-8">
        <Progress value={progress} className="mb-4" />
        <div className="flex justify-between">
          {STEPS.map((step, index) => {
            const stepNumber = index + 1
            const isCompleted = stepNumber < displayStep
            const isCurrent = stepNumber === displayStep

            return (
              <div
                key={step.id}
                className={`flex flex-col items-center text-center ${
                  isCurrent || isCompleted ? 'text-primary' : 'text-muted-foreground'
                }`}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium mb-2 transition-colors ${
                    isCompleted
                      ? 'bg-primary text-white'
                      : isCurrent
                      ? 'border-2 border-primary text-primary'
                      : 'border-2 border-muted text-muted-foreground'
                  }`}
                >
                  {isCompleted ? '✓' : stepNumber}
                </div>
                <span className="text-sm font-medium hidden sm:block">{step.title}</span>
                <span className="text-xs text-muted-foreground hidden sm:block">
                  {step.description}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Step Content */}
      {currentStep === 1 && (
        <HealthDataForm
          initialData={formData.healthData}
          onNext={handleHealthDataSubmit}
          showSkipOption={true}
        />
      )}

      {isImageStep && (
        <ImageCapture
          initialImages={formData.images}
          onNext={handleImagesSubmit}
          onBack={() => setCurrentStep(1)}
        />
      )}

      {isReviewStep && (
        <ReviewSubmit
          formData={formData}
          onBack={handleBackFromReview}
          skipImages={skipImages}
        />
      )}
    </div>
  )
}
