import { useState, memo, useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from './ui/card'
import type { PaperScores } from '@/types/paper'

interface AnalysisPanelProps {
  scores: PaperScores
  isLive: boolean
}

const getScoreColor = (score: number): string => {
  if (score >= 90) return 'text-green-600'
  if (score >= 75) return 'text-blue-600'
  if (score >= 60) return 'text-yellow-600'
  return 'text-red-600'
}

const getCategoryLabel = (category: string): string => {
  const labels: Record<string, string> = {
    novelty: 'Novelty',
    technical_quality: 'Technical Quality',
    clarity: 'Clarity',
    impact: 'Impact',
  }
  return labels[category] || category
}

export const AnalysisPanel = memo(function AnalysisPanel({ scores, isLive }: AnalysisPanelProps) {
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null)

  const categories = ['novelty', 'technical_quality', 'clarity', 'impact'] as const
  const categoryScores = useMemo(
    () => categories.map(cat => scores[cat]).filter(Boolean),
    [scores]
  )

  const hasScores = useMemo(
    () => scores.overall > 0 || categoryScores.some(s => s.score > 0),
    [scores.overall, categoryScores]
  )

  if (!hasScores) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Analysis Scores</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            {isLive ? 'Analyzing' : 'Pending'}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Analysis Scores</CardTitle>
        {isLive && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            <span>Live Updating</span>
          </div>
        )}
      </CardHeader>
      <CardContent>
        {/* Overall Score */}
        <div className="mb-6 text-center" role="region" aria-label="Overall score">
          <div
            className={`text-4xl font-bold ${getScoreColor(scores.overall)}`}
            aria-label={`Overall score: ${scores.overall} out of 100`}
          >
            Overall {scores.overall}
          </div>
        </div>

        {/* Category Scores Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4" role="list" aria-label="Category scores">
          {categoryScores.map((categoryScore) => (
            <div
              key={categoryScore.category}
              className="border rounded-lg p-4 hover:bg-accent transition-colors"
              role="listitem"
              aria-label={`${getCategoryLabel(categoryScore.category)}: ${categoryScore.score} out of 100`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="font-semibold">
                  {getCategoryLabel(categoryScore.category)}
                </div>
                <div className={`text-2xl font-bold ${getScoreColor(categoryScore.score)}`}>
                  {categoryScore.score}
                </div>
              </div>

              {/* Confidence Indicator */}
              {categoryScore.confidence > 0 && (
                <div className="mb-2">
                  <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                    <span>{Math.round(categoryScore.confidence * 100)}% Confidence</span>
                  </div>
                  <div
                    className="h-1 bg-secondary rounded-full overflow-hidden"
                    role="progressbar"
                    aria-valuenow={categoryScore.confidence * 100}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  >
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${categoryScore.confidence * 100}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Explanation */}
              {categoryScore.explanation && (
                <div className="mt-2">
                  <button
                    onClick={() =>
                      setExpandedCategory(
                        expandedCategory === categoryScore.category
                          ? null
                          : categoryScore.category
                      )
                    }
                    className="text-xs text-primary hover:underline"
                  >
                    {expandedCategory === categoryScore.category
                      ? 'Show Less'
                      : 'Show More'}
                  </button>
                  {expandedCategory === categoryScore.category && (
                    <p className="mt-2 text-sm text-muted-foreground">
                      {categoryScore.explanation}
                    </p>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
})
