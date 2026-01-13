'use client'

import { useState, useRef, useCallback } from 'react'
import Webcam from 'react-webcam'
import { useDropzone } from 'react-dropzone'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Camera, Upload, RotateCcw, Check, ChevronLeft, ChevronRight } from 'lucide-react'
import { EXPRESSIONS, type ExpressionKey, type ImageFormData } from '@/types'
import { cn } from '@/lib/utils'

interface ImageCaptureProps {
  initialImages: ImageFormData
  onNext: (images: ImageFormData) => void
  onBack: () => void
}

export function ImageCapture({ initialImages, onNext, onBack }: ImageCaptureProps) {
  const webcamRef = useRef<Webcam>(null)
  const [mode, setMode] = useState<'webcam' | 'upload'>('webcam')
  const [currentIndex, setCurrentIndex] = useState(0)
  const [images, setImages] = useState<ImageFormData>(initialImages)
  const [isCapturing, setIsCapturing] = useState(false)

  const currentExpression = EXPRESSIONS[currentIndex]
  const allCaptured = EXPRESSIONS.every(e => images[e.key])
  const capturedCount = EXPRESSIONS.filter(e => images[e.key]).length

  const capture = useCallback(() => {
    if (webcamRef.current) {
      setIsCapturing(true)
      const imageSrc = webcamRef.current.getScreenshot()

      if (imageSrc) {
        const base64 = imageSrc.split(',')[1]
        setImages(prev => ({
          ...prev,
          [currentExpression.key]: base64,
        }))

        setTimeout(() => {
          setIsCapturing(false)
          if (currentIndex < EXPRESSIONS.length - 1) {
            setCurrentIndex(prev => prev + 1)
          }
        }, 500)
      } else {
        setIsCapturing(false)
      }
    }
  }, [currentExpression.key, currentIndex])

  const handleUpload = useCallback((acceptedFiles: File[], expressionKey: ExpressionKey) => {
    const file = acceptedFiles[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        const base64 = result.split(',')[1]
        setImages(prev => ({
          ...prev,
          [expressionKey]: base64,
        }))
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const retake = useCallback((key: ExpressionKey) => {
    setImages(prev => ({ ...prev, [key]: null }))
    const index = EXPRESSIONS.findIndex(e => e.key === key)
    setCurrentIndex(index)
  }, [])

  const handleSubmit = () => {
    if (allCaptured) {
      onNext(images as { kiss: string; normal: string; spread: string; open: string })
    }
  }

  return (
    <div className="space-y-6">
      {/* Mode Toggle */}
      <div className="flex justify-center gap-2">
        <Button
          variant={mode === 'webcam' ? 'default' : 'outline'}
          onClick={() => setMode('webcam')}
        >
          <Camera className="w-4 h-4 mr-2" />
          Webcam
        </Button>
        <Button
          variant={mode === 'upload' ? 'default' : 'outline'}
          onClick={() => setMode('upload')}
        >
          <Upload className="w-4 h-4 mr-2" />
          Upload
        </Button>
      </div>

      {mode === 'webcam' ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <span className="bg-primary text-primary-foreground w-6 h-6 rounded-full text-sm flex items-center justify-center">
                {currentIndex + 1}
              </span>
              {currentExpression.title} Expression
            </CardTitle>
            <CardDescription>{currentExpression.instruction}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden mb-4">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                screenshotQuality={0.9}
                className="w-full h-full object-cover"
                videoConstraints={{
                  width: 640,
                  height: 480,
                  facingMode: 'user',
                }}
              />
              {isCapturing && (
                <div className="absolute inset-0 bg-white/50 animate-pulse" />
              )}
            </div>

            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => setCurrentIndex(prev => Math.max(0, prev - 1))}
                disabled={currentIndex === 0}
              >
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <Button onClick={capture} className="flex-1" disabled={isCapturing}>
                <Camera className="w-4 h-4 mr-2" />
                Capture {currentExpression.title}
              </Button>
              <Button
                variant="outline"
                onClick={() => setCurrentIndex(prev => Math.min(EXPRESSIONS.length - 1, prev + 1))}
                disabled={currentIndex === EXPRESSIONS.length - 1}
              >
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Upload Images</CardTitle>
            <CardDescription>
              Upload one image for each expression. Images should clearly show your face.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              {EXPRESSIONS.map(expr => (
                <UploadZone
                  key={expr.key}
                  expression={expr}
                  image={images[expr.key]}
                  onUpload={(files) => handleUpload(files, expr.key)}
                  onRetake={() => retake(expr.key)}
                />
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Captured Images Preview */}
      <div className="grid grid-cols-4 gap-2">
        {EXPRESSIONS.map((expr, index) => (
          <div
            key={expr.key}
            className={cn(
              'aspect-square rounded-lg border-2 overflow-hidden relative cursor-pointer transition-all',
              index === currentIndex && mode === 'webcam'
                ? 'ring-2 ring-primary border-primary'
                : 'border-muted',
              images[expr.key] ? 'border-green-500' : ''
            )}
            onClick={() => {
              if (!images[expr.key]) setCurrentIndex(index)
            }}
          >
            {images[expr.key] ? (
              <>
                <img
                  src={`data:image/jpeg;base64,${images[expr.key]}`}
                  alt={expr.title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-1 right-1 bg-green-500 rounded-full p-0.5">
                  <Check className="w-3 h-3 text-white" />
                </div>
                <button
                  className="absolute bottom-1 right-1 bg-white/80 rounded p-1 hover:bg-white"
                  onClick={(e) => {
                    e.stopPropagation()
                    retake(expr.key)
                  }}
                >
                  <RotateCcw className="w-3 h-3" />
                </button>
              </>
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-muted text-muted-foreground text-xs text-center p-1">
                {expr.title}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Progress */}
      <div className="text-center text-sm text-muted-foreground">
        {capturedCount} of {EXPRESSIONS.length} expressions captured
      </div>

      {/* Actions */}
      <div className="flex gap-4">
        <Button variant="outline" onClick={onBack} className="flex-1">
          Back
        </Button>
        <Button
          onClick={handleSubmit}
          disabled={!allCaptured}
          className="flex-1"
        >
          {allCaptured ? 'Continue to Review' : `Capture ${4 - capturedCount} more`}
        </Button>
      </div>
    </div>
  )
}

interface UploadZoneProps {
  expression: typeof EXPRESSIONS[number]
  image: string | null
  onUpload: (files: File[]) => void
  onRetake: () => void
}

function UploadZone({ expression, image, onUpload, onRetake }: UploadZoneProps) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: onUpload,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    maxFiles: 1,
  })

  if (image) {
    return (
      <div className="aspect-square rounded-lg border overflow-hidden relative">
        <img
          src={`data:image/jpeg;base64,${image}`}
          alt={expression.title}
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-x-0 bottom-0 bg-black/50 text-white text-center py-1 text-sm">
          {expression.title}
        </div>
        <button
          className="absolute top-2 right-2 bg-white rounded-full p-1.5 shadow hover:bg-gray-100"
          onClick={onRetake}
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>
    )
  }

  return (
    <div
      {...getRootProps()}
      className={cn(
        'aspect-square rounded-lg border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-colors',
        isDragActive ? 'border-primary bg-primary/5' : 'border-muted hover:border-primary/50'
      )}
    >
      <input {...getInputProps()} />
      <Upload className="w-8 h-8 text-muted-foreground mb-2" />
      <p className="text-sm font-medium">{expression.title}</p>
      <p className="text-xs text-muted-foreground text-center px-2">
        {expression.instruction}
      </p>
    </div>
  )
}
