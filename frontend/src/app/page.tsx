'use client'

import Link from 'next/link'
import { ArrowRight, Brain, HeartPulse, Shield, Camera } from 'lucide-react'

export default function HomePage() {
  return (
    <div className="container mx-auto px-4 py-12">
      {/* Hero Section */}
      <section className="text-center max-w-4xl mx-auto mb-16">
        <div className="inline-flex items-center gap-2 bg-blue-50 text-blue-700 px-4 py-2 rounded-full text-sm font-medium mb-6">
          <Brain className="w-4 h-4" />
          AI-Powered Health Assessment
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6">
          Understand Your{' '}
          <span className="gradient-text">Stroke Risk</span>
        </h1>
        <p className="text-xl text-slate-600 mb-8 max-w-2xl mx-auto">
          Our advanced AI analyzes facial expressions and health metrics to provide
          a comprehensive stroke risk assessment in minutes.
        </p>
        <Link
          href="/assessment"
          className="inline-flex items-center gap-2 bg-primary text-white px-8 py-4 rounded-lg font-semibold text-lg hover:bg-primary/90 transition-colors shadow-lg shadow-primary/25"
        >
          Start Your Assessment
          <ArrowRight className="w-5 h-5" />
        </Link>
      </section>

      {/* Features Section */}
      <section className="grid md:grid-cols-3 gap-8 mb-16">
        <div className="bg-white rounded-xl p-6 shadow-sm border">
          <div className="w-12 h-12 rounded-lg bg-blue-50 flex items-center justify-center mb-4">
            <Camera className="w-6 h-6 text-blue-600" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 mb-2">
            Facial Analysis
          </h3>
          <p className="text-slate-600">
            Capture 4 simple facial expressions that our AI analyzes for potential
            indicators of stroke risk.
          </p>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-sm border">
          <div className="w-12 h-12 rounded-lg bg-green-50 flex items-center justify-center mb-4">
            <HeartPulse className="w-6 h-6 text-green-600" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 mb-2">
            Health Metrics
          </h3>
          <p className="text-slate-600">
            Input your basic health data including blood pressure, BMI, and cholesterol
            for a comprehensive analysis.
          </p>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-sm border">
          <div className="w-12 h-12 rounded-lg bg-purple-50 flex items-center justify-center mb-4">
            <Shield className="w-6 h-6 text-purple-600" />
          </div>
          <h3 className="text-lg font-semibold text-slate-900 mb-2">
            Personalized Results
          </h3>
          <p className="text-slate-600">
            Receive detailed insights with risk factors, contributing elements, and
            actionable health recommendations.
          </p>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="bg-white rounded-2xl p-8 md:p-12 shadow-sm border">
        <h2 className="text-2xl font-bold text-slate-900 text-center mb-8">
          How It Works
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div className="text-center">
            <div className="w-12 h-12 rounded-full gradient-bg text-white font-bold text-xl flex items-center justify-center mx-auto mb-4">
              1
            </div>
            <h3 className="font-semibold text-slate-900 mb-2">Enter Health Data</h3>
            <p className="text-slate-600 text-sm">
              Fill in your basic health information including age, blood pressure,
              and other vital metrics.
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 rounded-full gradient-bg text-white font-bold text-xl flex items-center justify-center mx-auto mb-4">
              2
            </div>
            <h3 className="font-semibold text-slate-900 mb-2">Capture Expressions</h3>
            <p className="text-slate-600 text-sm">
              Use your webcam to capture 4 facial expressions or upload existing
              photos for analysis.
            </p>
          </div>
          <div className="text-center">
            <div className="w-12 h-12 rounded-full gradient-bg text-white font-bold text-xl flex items-center justify-center mx-auto mb-4">
              3
            </div>
            <h3 className="font-semibold text-slate-900 mb-2">Get Your Results</h3>
            <p className="text-slate-600 text-sm">
              Our AI analyzes your data and provides a comprehensive risk assessment
              with recommendations.
            </p>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="mt-12 text-center">
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 max-w-2xl mx-auto">
          <p className="text-amber-800 text-sm">
            <strong>Important:</strong> This tool provides educational insights only and is not a
            substitute for professional medical diagnosis. Always consult with healthcare
            providers for medical advice.
          </p>
        </div>
      </section>
    </div>
  )
}
