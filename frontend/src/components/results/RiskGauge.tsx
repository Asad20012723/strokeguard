'use client'

import { useEffect, useState } from 'react'
import type { RiskLevel } from '@/types'

interface RiskGaugeProps {
  score: number
  riskLevel: RiskLevel
}

export function RiskGauge({ score, riskLevel }: RiskGaugeProps) {
  const [animatedScore, setAnimatedScore] = useState(0)

  useEffect(() => {
    const duration = 1500
    const steps = 60
    const increment = score / steps
    let current = 0

    const timer = setInterval(() => {
      current += increment
      if (current >= score) {
        setAnimatedScore(score)
        clearInterval(timer)
      } else {
        setAnimatedScore(current)
      }
    }, duration / steps)

    return () => clearInterval(timer)
  }, [score])

  const percentage = Math.round(animatedScore * 100)
  const circumference = 2 * Math.PI * 90
  const strokeDashoffset = circumference - animatedScore * circumference

  const getColors = () => {
    switch (riskLevel) {
      case 'low':
        return {
          stroke: '#22c55e',
          bg: 'bg-green-50',
          text: 'text-green-700',
          label: 'Low Risk',
        }
      case 'moderate':
        return {
          stroke: '#f59e0b',
          bg: 'bg-amber-50',
          text: 'text-amber-700',
          label: 'Moderate Risk',
        }
      case 'high':
        return {
          stroke: '#ef4444',
          bg: 'bg-red-50',
          text: 'text-red-700',
          label: 'High Risk',
        }
    }
  }

  const colors = getColors()

  return (
    <div
      className={`relative flex flex-col items-center p-8 rounded-2xl ${colors.bg}`}
    >
      <svg className="transform -rotate-90" width="200" height="200">
        {/* Background circle */}
        <circle
          cx="100"
          cy="100"
          r="90"
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="12"
        />
        {/* Progress circle */}
        <circle
          cx="100"
          cy="100"
          r="90"
          fill="none"
          stroke={colors.stroke}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-300"
        />
      </svg>

      {/* Center text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-5xl font-bold ${colors.text}`}>{percentage}%</span>
        <span className={`text-lg font-medium ${colors.text} mt-1`}>
          {colors.label}
        </span>
      </div>
    </div>
  )
}
