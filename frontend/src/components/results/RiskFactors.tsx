'use client'

import { AlertTriangle, AlertCircle, Info } from 'lucide-react'
import type { ContributingFactor } from '@/types'

interface RiskFactorsProps {
  factors: ContributingFactor[]
}

export function RiskFactors({ factors }: RiskFactorsProps) {
  if (factors.length === 0) {
    return (
      <div className="text-center py-6 text-muted-foreground">
        <Info className="w-8 h-8 mx-auto mb-2" />
        <p>No significant risk factors identified</p>
      </div>
    )
  }

  const getIcon = (severity: string) => {
    switch (severity) {
      case 'high':
        return <AlertTriangle className="w-5 h-5 text-red-500" />
      case 'moderate':
        return <AlertCircle className="w-5 h-5 text-amber-500" />
      default:
        return <Info className="w-5 h-5 text-blue-500" />
    }
  }

  const getSeverityStyles = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'border-l-red-500 bg-red-50'
      case 'moderate':
        return 'border-l-amber-500 bg-amber-50'
      default:
        return 'border-l-blue-500 bg-blue-50'
    }
  }

  return (
    <div className="space-y-3">
      {factors.map((factor, index) => (
        <div
          key={index}
          className={`border-l-4 rounded-r-lg p-4 ${getSeverityStyles(factor.severity)}`}
        >
          <div className="flex items-start gap-3">
            {getIcon(factor.severity)}
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-slate-900">{factor.factor}</h4>
                <span className="text-sm font-medium">
                  {factor.value} / {factor.threshold}
                </span>
              </div>
              <p className="text-sm text-slate-600 mt-1">
                Your value exceeds the recommended threshold
              </p>
              {/* Progress bar showing how much over threshold */}
              <div className="mt-2 h-2 bg-white rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-500 ${
                    factor.severity === 'high'
                      ? 'bg-red-500'
                      : factor.severity === 'moderate'
                      ? 'bg-amber-500'
                      : 'bg-blue-500'
                  }`}
                  style={{
                    width: `${Math.min(100, (factor.value / factor.threshold) * 80)}%`,
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
