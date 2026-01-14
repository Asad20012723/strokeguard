'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { StatisticalPredictionResponse, ExplainabilityResults, FeatureContribution } from '@/lib/api-client'

interface ClinicalEvidenceProps {
  statisticalResults: StatisticalPredictionResponse
  explainability?: ExplainabilityResults | null
  showMultimodal?: boolean
}

export function ClinicalEvidence({
  statisticalResults,
  explainability,
  showMultimodal = false,
}: ClinicalEvidenceProps) {
  const featureLabels: Record<string, string> = {
    age: 'Age',
    gender: 'Gender (Male)',
    systolic: 'Systolic BP',
    diastolic: 'Diastolic BP',
    glucose: 'Blood Glucose',
    bmi: 'BMI',
    cholesterol: 'Cholesterol',
    smoking_former: 'Former Smoker',
    smoking_current: 'Current Smoker',
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Clinical Evidence
          <Badge variant="outline">Statistical Analysis</Badge>
        </CardTitle>
        <CardDescription>
          Detailed statistical analysis and model explanations
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="odds-ratios" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="odds-ratios">Odds Ratios</TabsTrigger>
            <TabsTrigger value="contributions">Feature Impact</TabsTrigger>
            <TabsTrigger value="interpretation">Interpretation</TabsTrigger>
          </TabsList>

          {/* Odds Ratios Tab */}
          <TabsContent value="odds-ratios" className="mt-4">
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature</TableHead>
                    <TableHead className="text-right">Odds Ratio</TableHead>
                    <TableHead className="text-right">95% CI</TableHead>
                    <TableHead className="text-right">P-Value</TableHead>
                    <TableHead className="text-center">Significance</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(statisticalResults.feature_contributions).map(
                    ([key, contrib]) => (
                      <TableRow key={key}>
                        <TableCell className="font-medium">
                          {featureLabels[key] || key}
                        </TableCell>
                        <TableCell className="text-right">
                          <span
                            className={
                              contrib.odds_ratio > 1
                                ? 'text-red-600'
                                : contrib.odds_ratio < 1
                                ? 'text-green-600'
                                : ''
                            }
                          >
                            {contrib.odds_ratio.toFixed(3)}
                          </span>
                        </TableCell>
                        <TableCell className="text-right text-muted-foreground">
                          [{contrib.ci_95[0].toFixed(3)} - {contrib.ci_95[1].toFixed(3)}]
                        </TableCell>
                        <TableCell className="text-right">
                          {contrib.p_value < 0.001
                            ? '<0.001'
                            : contrib.p_value.toFixed(4)}
                        </TableCell>
                        <TableCell className="text-center">
                          {contrib.significant ? (
                            <Badge variant="destructive">Significant</Badge>
                          ) : (
                            <Badge variant="secondary">NS</Badge>
                          )}
                        </TableCell>
                      </TableRow>
                    )
                  )}
                </TableBody>
              </Table>
            </div>

            {/* Model Statistics */}
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 rounded-lg bg-muted">
                <p className="text-xs text-muted-foreground">Pseudo R2</p>
                <p className="text-lg font-semibold">
                  {statisticalResults.model_statistics.pseudo_r2.toFixed(4)}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-muted">
                <p className="text-xs text-muted-foreground">AIC</p>
                <p className="text-lg font-semibold">
                  {statisticalResults.model_statistics.aic.toFixed(2)}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-muted">
                <p className="text-xs text-muted-foreground">Log-Likelihood</p>
                <p className="text-lg font-semibold">
                  {statisticalResults.model_statistics.log_likelihood.toFixed(4)}
                </p>
              </div>
              <div className="p-3 rounded-lg bg-muted">
                <p className="text-xs text-muted-foreground">Features</p>
                <p className="text-lg font-semibold">
                  {statisticalResults.model_statistics.n_features}
                </p>
              </div>
            </div>
          </TabsContent>

          {/* Feature Contributions Tab */}
          <TabsContent value="contributions" className="mt-4">
            <div className="space-y-4">
              {Object.entries(statisticalResults.feature_contributions)
                .sort((a, b) => Math.abs(b[1].contribution) - Math.abs(a[1].contribution))
                .map(([key, contrib]) => (
                  <div key={key} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">
                        {featureLabels[key] || key}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        Value: {contrib.value.toFixed(2)}
                      </span>
                    </div>
                    <div className="relative h-4 bg-muted rounded-full overflow-hidden">
                      <div
                        className={`absolute h-full ${
                          contrib.contribution > 0
                            ? 'bg-red-500 left-1/2'
                            : 'bg-green-500 right-1/2'
                        }`}
                        style={{
                          width: `${Math.min(Math.abs(contrib.contribution) * 50, 50)}%`,
                        }}
                      />
                      <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-foreground" />
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Decreases Risk</span>
                      <span>
                        Contribution: {contrib.contribution > 0 ? '+' : ''}
                        {contrib.contribution.toFixed(4)}
                      </span>
                      <span>Increases Risk</span>
                    </div>
                  </div>
                ))}
            </div>
          </TabsContent>

          {/* Interpretation Tab */}
          <TabsContent value="interpretation" className="mt-4">
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-muted">
                <h4 className="font-semibold mb-2">Statistical Interpretation</h4>
                <ul className="space-y-2">
                  {statisticalResults.interpretation.map((item, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className="text-primary">-</span>
                      <span className="text-sm">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {explainability?.clinical_summary && explainability.clinical_summary.length > 0 && (
                <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-950">
                  <h4 className="font-semibold mb-2 text-blue-800 dark:text-blue-200">
                    AI-Powered Clinical Summary
                  </h4>
                  <ul className="space-y-2">
                    {explainability.clinical_summary.map((item, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <span className="text-blue-600">-</span>
                        <span className="text-sm text-blue-800 dark:text-blue-200">
                          {item}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="p-4 rounded-lg border border-yellow-200 bg-yellow-50 dark:bg-yellow-950 dark:border-yellow-800">
                <h4 className="font-semibold mb-2 text-yellow-800 dark:text-yellow-200">
                  Clinical Disclaimer
                </h4>
                <p className="text-sm text-yellow-700 dark:text-yellow-300">
                  These results are for informational purposes only and should not replace
                  professional medical advice. Odds ratios indicate statistical associations,
                  not causation. Always consult with a qualified healthcare provider for
                  diagnosis and treatment decisions.
                </p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
