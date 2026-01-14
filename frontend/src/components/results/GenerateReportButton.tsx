'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { generateReport, HealthData, ImageData } from '@/lib/api-client'
import { FileText, Download, Loader2, AlertCircle, CheckCircle } from 'lucide-react'

interface GenerateReportButtonProps {
  patientId: string
  healthData: HealthData
  images?: ImageData | null
}

export function GenerateReportButton({
  patientId,
  healthData,
  images,
}: GenerateReportButtonProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [format, setFormat] = useState<'pdf' | 'html' | 'json'>('pdf')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<{
    success: boolean
    url?: string
    content?: string
    error?: string
  } | null>(null)

  const handleGenerate = async () => {
    setIsLoading(true)
    setResult(null)

    try {
      const response = await generateReport({
        patient_id: patientId,
        health_data: healthData,
        images: images || undefined,
        report_format: format,
      })

      setResult({
        success: response.success,
        url: response.report_url || undefined,
        content: response.report_content || undefined,
        error: response.error || undefined,
      })
    } catch (error) {
      setResult({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to generate report',
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="gap-2">
          <FileText className="h-4 w-4" />
          Generate Medical Report
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Generate Medical Report</DialogTitle>
          <DialogDescription>
            Create a comprehensive medical report using AI-powered analysis.
            The report will be synthesized by our clinical AI assistant.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Format Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Report Format</label>
            <Select
              value={format}
              onValueChange={(value) => setFormat(value as 'pdf' | 'html' | 'json')}
              disabled={isLoading}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select format" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="pdf">PDF Document</SelectItem>
                <SelectItem value="html">HTML Report</SelectItem>
                <SelectItem value="json">JSON Data</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Patient Info */}
          <div className="text-sm text-muted-foreground p-3 rounded-lg bg-muted">
            <p>
              <strong>Patient ID:</strong> {patientId}
            </p>
            <p>
              <strong>Images Included:</strong> {images ? 'Yes' : 'No'}
            </p>
          </div>

          {/* Result Display */}
          {result && (
            <div
              className={`p-4 rounded-lg ${
                result.success
                  ? 'bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800'
                  : 'bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800'
              }`}
            >
              {result.success ? (
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-5 w-5 text-green-600 mt-0.5" />
                  <div>
                    <p className="font-medium text-green-800 dark:text-green-200">
                      Report Generated Successfully
                    </p>
                    {result.url && (
                      <a
                        href={result.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 mt-2 text-sm text-green-700 dark:text-green-300 hover:underline"
                      >
                        <Download className="h-4 w-4" />
                        Download Report
                      </a>
                    )}
                    {result.content && format === 'json' && (
                      <pre className="mt-2 p-2 bg-black/10 rounded text-xs overflow-auto max-h-40">
                        {result.content}
                      </pre>
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex items-start gap-2">
                  <AlertCircle className="h-5 w-5 text-red-600 mt-0.5" />
                  <div>
                    <p className="font-medium text-red-800 dark:text-red-200">
                      Report Generation Failed
                    </p>
                    <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                      {result.error}
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Generate Button */}
          <Button
            onClick={handleGenerate}
            disabled={isLoading}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating Report...
              </>
            ) : (
              <>
                <FileText className="mr-2 h-4 w-4" />
                Generate Report
              </>
            )}
          </Button>

          <p className="text-xs text-muted-foreground text-center">
            Reports are generated using RAG-based AI analysis and may take
            a few moments to complete.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
}
