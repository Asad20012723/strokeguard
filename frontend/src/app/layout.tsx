import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Stroke Risk Assessment',
  description: 'AI-powered stroke risk assessment using facial analysis and health metrics',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
          {/* Header */}
          <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
            <div className="container mx-auto px-4 py-4 flex items-center justify-between">
              <a href="/" className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg gradient-bg flex items-center justify-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="w-5 h-5 text-white"
                  >
                    <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
                  </svg>
                </div>
                <span className="font-semibold text-lg">StrokeGuard</span>
              </a>
              <nav className="flex items-center gap-4">
                <a
                  href="/assessment"
                  className="text-sm font-medium text-slate-600 hover:text-slate-900 transition-colors"
                >
                  Start Assessment
                </a>
              </nav>
            </div>
          </header>

          {/* Main content */}
          <main>{children}</main>

          {/* Footer */}
          <footer className="border-t bg-white mt-auto">
            <div className="container mx-auto px-4 py-6 text-center text-sm text-slate-500">
              <p>
                This tool is for educational purposes only and should not replace professional medical advice.
              </p>
              <p className="mt-1">
                Always consult with healthcare professionals for medical decisions.
              </p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  )
}
