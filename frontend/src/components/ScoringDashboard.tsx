import { useState, useEffect } from 'react'
import { useWebSocket } from '@/hooks/useWebSocket'
import { Card } from './ui/card'
import { Wifi, WifiOff, Loader2, CheckCircle } from 'lucide-react'

interface ScoreData {
  category: string
  score: number
  confidence: number
}

interface ProgressData {
  stage: string
  percentage: number
  description?: string
}

interface OverallScoreData {
  score: number
  breakdown: Record<string, number>
}

interface ScoreCardProps {
  title: string
  score: number | null
  confidence?: number
}

function ScoreCard({ title, score, confidence }: ScoreCardProps) {
  const getScoreColor = (score: number | null) => {
    if (score === null) return 'text-muted-foreground'
    if (score >= 90) return 'text-green-600 dark:text-green-400'
    if (score >= 75) return 'text-blue-600 dark:text-blue-400'
    if (score >= 60) return 'text-yellow-600 dark:text-yellow-400'
    return 'text-red-600 dark:text-red-400'
  }

  return (
    <Card className="p-6" role="article">
      <h3 className="text-sm font-medium text-muted-foreground mb-2">{title}</h3>
      <div className={`text-4xl font-bold ${getScoreColor(score)}`}>
        {score !== null ? score : (
          <span className="text-2xl text-muted-foreground">Analyzing...</span>
        )}
      </div>
      {confidence !== undefined && score !== null && (
        <p className="text-xs text-muted-foreground mt-2">
          {Math.round(confidence * 100)}% confidence
        </p>
      )}
    </Card>
  )
}

interface ScoringDashboardProps {
  paperId: string
}

export function ScoringDashboard({ paperId }: ScoringDashboardProps) {
  const wsUrl = `ws://localhost:8000/ws/papers/${paperId}`
  const { isConnected, lastMessage, error, readyState } = useWebSocket(wsUrl)

  const [scores, setScores] = useState<Record<string, ScoreData>>({})
  const [overallScore, setOverallScore] = useState<number | null>(null)
  const [progress, setProgress] = useState<ProgressData | null>(null)
  const [analysisComplete, setAnalysisComplete] = useState(false)

  useEffect(() => {
    if (!lastMessage) return

    const { type, data } = lastMessage as any

    switch (type) {
      case 'score_update':
        setScores(prev => ({
          ...prev,
          [data.category]: data,
        }))
        break

      case 'overall_score':
        setOverallScore(data.score)
        if (data.breakdown) {
          // Update individual scores from breakdown
          Object.entries(data.breakdown).forEach(([category, score]) => {
            setScores(prev => ({
              ...prev,
              [category]: {
                category,
                score: score as number,
                confidence: 1.0,
              },
            }))
          })
        }
        break

      case 'progress':
        setProgress({
          stage: data.stage,
          percentage: data.percentage,
        })
        break

      case 'stage_update':
        setProgress({
          stage: data.stage,
          percentage: progress?.percentage || 0,
          description: data.description,
        })
        break

      case 'analysis_complete':
        setAnalysisComplete(true)
        setProgress(null)
        break
    }
  }, [lastMessage])

  const getConnectionStatus = () => {
    if (error) {
      return (
        <div className="flex items-center gap-2 text-red-600 dark:text-red-400">
          <WifiOff className="h-4 w-4" />
          <span className="text-sm">Connection failed</span>
        </div>
      )
    }

    if (readyState === WebSocket.CONNECTING) {
      return (
        <div className="flex items-center gap-2 text-yellow-600 dark:text-yellow-400">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-sm">Connecting...</span>
        </div>
      )
    }

    if (isConnected) {
      return (
        <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
          <Wifi className="h-4 w-4" />
          <span className="text-sm">Connected</span>
        </div>
      )
    }

    return (
      <div className="flex items-center gap-2 text-muted-foreground">
        <WifiOff className="h-4 w-4" />
        <span className="text-sm">Disconnected</span>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-8 max-w-6xl">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2">Paper Analysis Dashboard</h1>
          <p className="text-lg text-muted-foreground">
            Real-time scoring for paper: <span className="font-mono">{paperId}</span>
          </p>
        </div>
        {getConnectionStatus()}
      </div>

      {/* Progress Indicator */}
      {progress && !analysisComplete && (
        <Card className="p-6 mb-6 bg-blue-50 dark:bg-blue-950">
          <div className="flex items-center gap-3 mb-3">
            <Loader2 className="h-5 w-5 text-blue-600 dark:text-blue-400 animate-spin" />
            <div className="flex-1">
              <h3 className="font-semibold text-blue-900 dark:text-blue-100">
                {progress.description || progress.stage.replace(/_/g, ' ')}
              </h3>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                {progress.percentage}% complete
              </p>
            </div>
          </div>
          <div className="w-full bg-blue-200 dark:bg-blue-900 rounded-full h-2">
            <div
              className="bg-blue-600 dark:bg-blue-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress.percentage}%` }}
            />
          </div>
        </Card>
      )}

      {/* Analysis Complete */}
      {analysisComplete && (
        <Card className="p-6 mb-6 bg-green-50 dark:bg-green-950">
          <div className="flex items-center gap-3">
            <CheckCircle className="h-6 w-6 text-green-600 dark:text-green-400" />
            <div>
              <h3 className="font-semibold text-green-900 dark:text-green-100">
                Analysis Complete
              </h3>
              <p className="text-sm text-green-800 dark:text-green-200">
                All scoring has been completed
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Overall Score */}
      <div className="mb-6">
        <ScoreCard
          title="Overall Score"
          score={overallScore}
        />
      </div>

      {/* Category Scores */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <ScoreCard
          title="Novelty"
          score={scores.novelty?.score ?? null}
          confidence={scores.novelty?.confidence}
        />
        <ScoreCard
          title="Technical Quality"
          score={scores.technical_quality?.score ?? null}
          confidence={scores.technical_quality?.confidence}
        />
        <ScoreCard
          title="Clarity"
          score={scores.clarity?.score ?? null}
          confidence={scores.clarity?.confidence}
        />
        <ScoreCard
          title="Impact"
          score={scores.impact?.score ?? null}
          confidence={scores.impact?.confidence}
        />
      </div>
    </div>
  )
}
