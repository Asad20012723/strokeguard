'use client'

import { CheckCircle2 } from 'lucide-react'

interface RecommendationsProps {
  recommendations: string[]
}

export function Recommendations({ recommendations }: RecommendationsProps) {
  if (recommendations.length === 0) {
    return (
      <div className="text-center py-6 text-muted-foreground">
        <p>No specific recommendations at this time. Keep up the healthy lifestyle!</p>
      </div>
    )
  }

  return (
    <ul className="space-y-3">
      {recommendations.map((recommendation, index) => (
        <li key={index} className="flex items-start gap-3">
          <CheckCircle2 className="w-5 h-5 text-green-500 shrink-0 mt-0.5" />
          <span className="text-slate-700">{recommendation}</span>
        </li>
      ))}
    </ul>
  )
}
