import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { papersApi } from '@/api/papers'
import { useWebSocket } from '@/hooks/useWebSocket'
import { AnalysisPanel } from '@/components/AnalysisPanel'
import { RecommendationsPanel } from '@/components/RecommendationsPanel'
import type {
  PaperData,
  PaperScores,
  Recommendation,
  CategoryScore,
} from '@/types/paper'

type TabType = 'overview' | 'scores' | 'recommendations'

export function PaperDetails() {
  const { paperId } = useParams<{ paperId: string }>()
  const [paperData, setPaperData] = useState<PaperData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [activeTab, setActiveTab] = useState<TabType>('overview')

  // Real-time scores state
  const [scores, setScores] = useState<PaperScores>({
    overall: 0,
    novelty: { category: 'novelty', score: 0, confidence: 0 },
    technical_quality: { category: 'technical_quality', score: 0, confidence: 0 },
    clarity: { category: 'clarity', score: 0, confidence: 0 },
    impact: { category: 'impact', score: 0, confidence: 0 },
  })

  const [recommendations, setRecommendations] = useState<Recommendation[]>([])

  // WebSocket connection
  const wsUrl = paperId
    ? `ws://localhost:8000/ws/papers/${paperId}`
    : ''
  const { isConnected, lastMessage, error: wsError, readyState } = useWebSocket(
    wsUrl,
    {
      reconnect: true,
      reconnectAttempts: 5,
      reconnectInterval: 3000,
    }
  )

  // Fetch paper data
  useEffect(() => {
    if (!paperId) return

    const fetchPaper = async () => {
      try {
        setLoading(true)
        const data = await papersApi.getPaperById(paperId)
        setPaperData(data)

        // Initialize scores if available
        if (data.scores) {
          setScores(data.scores)
        }

        // Initialize recommendations if available
        if (data.recommendations) {
          setRecommendations(data.recommendations)
        }

        setError(null)
      } catch (err) {
        setError(
          err instanceof Error ? err : new Error('Failed to fetch paper')
        )
      } finally {
        setLoading(false)
      }
    }

    fetchPaper()
  }, [paperId])

  // Handle WebSocket messages
  useEffect(() => {
    if (!lastMessage) return

    const message = lastMessage as any

    switch (message.type) {
      case 'score_update': {
        const { category, score, confidence, explanation } = message.data
        setScores((prev) => ({
          ...prev,
          [category]: {
            category,
            score,
            confidence,
            explanation,
          } as CategoryScore,
        }))
        break
      }

      case 'overall_score': {
        const { score } = message.data
        setScores((prev) => ({
          ...prev,
          overall: score,
        }))
        break
      }

      case 'recommendation_added': {
        const recommendation: Recommendation = message.data
        setRecommendations((prev) => [...prev, recommendation])
        break
      }

      default:
        break
    }
  }, [lastMessage])

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="space-y-4">
          <div className="h-8 bg-secondary rounded animate-pulse" />
          <div className="h-4 bg-secondary rounded w-3/4 animate-pulse" />
          <div className="h-4 bg-secondary rounded w-1/2 animate-pulse" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
            <div className="h-64 bg-secondary rounded animate-pulse" />
            <div className="h-64 bg-secondary rounded animate-pulse" />
          </div>
        </div>
        <div className="text-center mt-4 text-muted-foreground">Loading...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center py-12">
          <h2 className="text-2xl font-bold text-red-600 mb-2">
            {error.message.includes('not found')
              ? 'Paper Not Found'
              : 'Error Loading Paper'}
          </h2>
          <p className="text-muted-foreground">{error.message}</p>
        </div>
      </div>
    )
  }

  if (!paperData) return null

  const { metadata } = paperData

  return (
    <div className="container mx-auto p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-3xl font-bold">{metadata.title}</h1>
          {wsError && (
            <div className="text-sm text-red-600">
              Connection Failed: {wsError.message}
            </div>
          )}
          {!wsError && (
            <div
              className={`flex items-center gap-2 text-sm ${
                isConnected ? 'text-green-600' : 'text-muted-foreground'
              }`}
            >
              <div
                className={`h-2 w-2 rounded-full ${
                  isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                }`}
              />
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          )}
        </div>

        {/* Authors */}
        {metadata.authors.length > 0 && (
          <div className="text-muted-foreground mb-2">
            {metadata.authors.join(', ')}
          </div>
        )}

        {/* Abstract */}
        {metadata.abstract && (
          <p className="text-sm text-muted-foreground mt-4">
            {metadata.abstract}
          </p>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="border-b mb-6">
        <div className="flex gap-4">
          {(['overview', 'scores', 'recommendations'] as TabType[]).map(
            (tab) => (
              <button
                key={tab}
                role="tab"
                aria-selected={activeTab === tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 border-b-2 transition-colors ${
                  activeTab === tab
                    ? 'border-primary text-primary font-semibold'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            )
          )}
        </div>
      </div>

      {/* Tab Content */}
      <div className="space-y-6">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <AnalysisPanel scores={scores} isLive={isConnected} />
            <RecommendationsPanel
              recommendations={recommendations}
              isLive={isConnected}
            />
          </div>
        )}

        {activeTab === 'scores' && (
          <AnalysisPanel scores={scores} isLive={isConnected} />
        )}

        {activeTab === 'recommendations' && (
          <RecommendationsPanel
            recommendations={recommendations}
            isLive={isConnected}
          />
        )}
      </div>
    </div>
  )
}
