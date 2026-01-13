'use client'

import { useState } from 'react'
import { Progress } from '@/components/ui/progress'
import { HealthDataForm } from '@/components/forms/HealthDataForm'
import { ImageCapture } from '@/components/forms/ImageCapture'
import { ReviewSubmit } from '@/components/forms/ReviewSubmit'
import type { AssessmentData, HealthFormData, ImageFormData } from '@/types'

const STEPS = [
  { id: 1, title: 'Health Info', description: 'Enter your health metrics' },
  { id: 2, title: 'Capture Images', description: 'Take 4 facial expression photos' },
  { id: 3, title: 'Review', description: 'Review and submit' },
]

export default function AssessmentPage() {
  const [currentStep, setCurrentStep] = useState(1)
  const [formData, setFormData] = useState<AssessmentData>({
    healthData: null,
    images: { kiss: null, normal: null, spread: null, open: null },
  })

  const progress = ((currentStep - 1) / (STEPS.length - 1)) * 100

  const handleHealthDataSubmit = (data: HealthFormData) => {
    setFormData(prev => ({ ...prev, healthData: data }))
    setCurrentStep(2)
  }

  const handleImagesSubmit = (images: ImageFormData) => {
    setFormData(prev => ({ ...prev, images }))
    setCurrentStep(3)
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-2xl">
      {/* Progress Header */}
      <div className="mb-8">
        <Progress value={progress} className="mb-4" />
        <div className="flex justify-between">
          {STEPS.map(step => (
            <div
              key={step.id}
              className={`flex flex-col items-center text-center ${
                step.id <= currentStep ? 'text-primary' : 'text-muted-foreground'
              }`}
            >
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium mb-2 transition-colors ${
                  step.id < currentStep
                    ? 'bg-primary text-white'
                    : step.id === currentStep
                    ? 'border-2 border-primary text-primary'
                    : 'border-2 border-muted text-muted-foreground'
                }`}
              >
                {step.id < currentStep ? '✓' : step.id}
              </div>
              <span className="text-sm font-medium hidden sm:block">{step.title}</span>
              <span className="text-xs text-muted-foreground hidden sm:block">
                {step.description}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Step Content */}
      {currentStep === 1 && (
        <HealthDataForm
          initialData={formData.healthData}
          onNext={handleHealthDataSubmit}
        />
      )}

      {currentStep === 2 && (
        <ImageCapture
          initialImages={formData.images}
          onNext={handleImagesSubmit}
          onBack={() => setCurrentStep(1)}
        />
      )}

      {currentStep === 3 && (
        <ReviewSubmit formData={formData} onBack={() => setCurrentStep(2)} />
      )}
    </div>
  )
}
