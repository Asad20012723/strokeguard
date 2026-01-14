'use client'

import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NativeSelect } from '@/components/ui/native-select'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import type { HealthFormData } from '@/types'

const healthDataSchema = z.object({
  age: z.coerce.number().min(18, 'Must be at least 18').max(120, 'Must be at most 120'),
  gender: z.enum(['male', 'female'], { required_error: 'Please select a gender' }),
  systolic: z.coerce.number().min(70, 'Too low').max(250, 'Too high'),
  diastolic: z.coerce.number().min(40, 'Too low').max(150, 'Too high'),
  glucose: z.coerce.number().min(50, 'Too low').max(500, 'Too high'),
  bmi: z.coerce.number().min(10, 'Too low').max(60, 'Too high'),
  cholesterol: z.coerce.number().min(100, 'Too low').max(400, 'Too high'),
  smoking: z.coerce.number().min(0).max(2),
}).refine(data => data.diastolic < data.systolic, {
  message: 'Diastolic must be less than systolic',
  path: ['diastolic'],
})

interface HealthDataFormProps {
  initialData: HealthFormData | null
  onNext: (data: HealthFormData, skipImages: boolean) => void
  showSkipOption?: boolean
}

export function HealthDataForm({ initialData, onNext, showSkipOption = false }: HealthDataFormProps) {
  const [skipImages, setSkipImages] = useState(false)

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<HealthFormData>({
    resolver: zodResolver(healthDataSchema),
    defaultValues: initialData || {
      age: undefined,
      gender: undefined,
      systolic: undefined,
      diastolic: undefined,
      glucose: undefined,
      bmi: undefined,
      cholesterol: undefined,
      smoking: 0,
    },
  })

  const onSubmit = (data: HealthFormData) => {
    onNext(data, skipImages)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Health Information</CardTitle>
        <CardDescription>
          Please enter your health metrics. All fields are required for accurate assessment.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {/* Age and Gender */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="age">Age (years)</Label>
              <Input
                id="age"
                type="number"
                placeholder="e.g., 45"
                {...register('age')}
              />
              {errors.age && (
                <p className="text-sm text-destructive">{errors.age.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="gender">Gender</Label>
              <NativeSelect id="gender" {...register('gender')}>
                <option value="">Select gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </NativeSelect>
              {errors.gender && (
                <p className="text-sm text-destructive">{errors.gender.message}</p>
              )}
            </div>
          </div>

          {/* Blood Pressure */}
          <div className="space-y-2">
            <Label>Blood Pressure</Label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Input
                  type="number"
                  placeholder="Systolic (e.g., 120)"
                  {...register('systolic')}
                />
                <p className="text-xs text-muted-foreground mt-1">Systolic (mmHg)</p>
                {errors.systolic && (
                  <p className="text-sm text-destructive">{errors.systolic.message}</p>
                )}
              </div>
              <div>
                <Input
                  type="number"
                  placeholder="Diastolic (e.g., 80)"
                  {...register('diastolic')}
                />
                <p className="text-xs text-muted-foreground mt-1">Diastolic (mmHg)</p>
                {errors.diastolic && (
                  <p className="text-sm text-destructive">{errors.diastolic.message}</p>
                )}
              </div>
            </div>
          </div>

          {/* Glucose and BMI */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="glucose">Blood Glucose (mg/dL)</Label>
              <Input
                id="glucose"
                type="number"
                placeholder="e.g., 100"
                {...register('glucose')}
              />
              {errors.glucose && (
                <p className="text-sm text-destructive">{errors.glucose.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="bmi">BMI</Label>
              <Input
                id="bmi"
                type="number"
                step="0.1"
                placeholder="e.g., 24.5"
                {...register('bmi')}
              />
              {errors.bmi && (
                <p className="text-sm text-destructive">{errors.bmi.message}</p>
              )}
            </div>
          </div>

          {/* Cholesterol and Smoking */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="cholesterol">Total Cholesterol (mg/dL)</Label>
              <Input
                id="cholesterol"
                type="number"
                placeholder="e.g., 200"
                {...register('cholesterol')}
              />
              {errors.cholesterol && (
                <p className="text-sm text-destructive">{errors.cholesterol.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="smoking">Smoking Status</Label>
              <NativeSelect id="smoking" {...register('smoking')}>
                <option value="0">Never Smoked</option>
                <option value="1">Former Smoker</option>
                <option value="2">Current Smoker</option>
              </NativeSelect>
              {errors.smoking && (
                <p className="text-sm text-destructive">{errors.smoking.message}</p>
              )}
            </div>
          </div>

          {/* Skip Images Option */}
          {showSkipOption && (
            <div className="p-4 rounded-lg border bg-muted/50 space-y-3">
              <div className="flex items-center space-x-3">
                <Checkbox
                  id="skipImages"
                  checked={skipImages}
                  onCheckedChange={(checked) => setSkipImages(checked === true)}
                />
                <div className="space-y-1">
                  <Label
                    htmlFor="skipImages"
                    className="text-sm font-medium leading-none cursor-pointer"
                  >
                    Skip Image Upload
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Get a statistical risk assessment using only health data (Logistic Regression)
                  </p>
                </div>
              </div>

              {skipImages && (
                <div className="mt-2 p-3 rounded bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800">
                  <p className="text-sm text-blue-800 dark:text-blue-200">
                    <strong>Statistical Analysis Mode:</strong> You will receive a detailed
                    logistic regression analysis with odds ratios, p-values, and confidence
                    intervals. For a more comprehensive assessment including facial expression
                    analysis, uncheck this option.
                  </p>
                </div>
              )}
            </div>
          )}

          <Button type="submit" className="w-full">
            {skipImages ? 'Continue to Review' : 'Continue to Image Capture'}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
