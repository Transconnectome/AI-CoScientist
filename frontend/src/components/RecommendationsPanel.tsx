import { useState, memo, useMemo, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from './ui/card'
import type { Recommendation } from '@/types/paper'

interface RecommendationsPanelProps {
  recommendations: Recommendation[]
  isLive: boolean
}

type CategoryFilter = 'all' | 'novelty' | 'technical_quality' | 'clarity' | 'impact' | 'general'

const getPriorityColor = (priority: string): string => {
  switch (priority) {
    case 'high':
      return 'text-red-600 bg-red-50'
    case 'medium':
      return 'text-yellow-600 bg-yellow-50'
    case 'low':
      return 'text-blue-600 bg-blue-50'
    default:
      return 'text-gray-600 bg-gray-50'
  }
}

const getCategoryLabel = (category: string): string => {
  const labels: Record<string, string> = {
    novelty: 'Novelty',
    technical_quality: 'Technical',
    clarity: 'Clarity',
    impact: 'Impact',
    general: 'General',
  }
  return labels[category] || category
}

const sortByPriority = (a: Recommendation, b: Recommendation): number => {
  const priorityOrder = { high: 0, medium: 1, low: 2 }
  return priorityOrder[a.priority] - priorityOrder[b.priority]
}

export const RecommendationsPanel = memo(function RecommendationsPanel({
  recommendations,
  isLive,
}: RecommendationsPanelProps) {
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>('all')
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())

  const categories: CategoryFilter[] = useMemo(() => [
    'all',
    'clarity',
    'technical_quality',
    'novelty',
    'impact',
    'general',
  ], [])

  const filteredRecommendations = useMemo(
    () => recommendations
      .filter(
        (rec) => categoryFilter === 'all' || rec.category === categoryFilter
      )
      .sort(sortByPriority),
    [recommendations, categoryFilter]
  )

  const toggleExpanded = useCallback((id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [])

  const handleExport = useCallback(() => {
    const text = recommendations
      .map(
        (rec) =>
          `${rec.priority.toUpperCase()}: ${rec.title}\n${rec.description}\nImpact: ${rec.impact}\n`
      )
      .join('\n---\n\n')

    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'recommendations.txt'
    a.click()
    URL.revokeObjectURL(url)
  }, [recommendations])

  if (recommendations.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            {isLive
              ? 'Generating recommendations...'
              : 'No recommendations available yet'}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <CardTitle>Recommendations</CardTitle>
            <span className="text-sm text-muted-foreground">
              {filteredRecommendations.length}{' '}
              {filteredRecommendations.length === 1
                ? 'recommendation'
                : 'recommendations'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {isLive && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                <span>Live Streaming</span>
              </div>
            )}
            <button
              onClick={handleExport}
              className="text-sm px-3 py-1 rounded-md border hover:bg-accent transition-colors"
            >
              Export
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Category Filters */}
        <div className="flex flex-wrap gap-2 mb-4" role="group" aria-label="Filter recommendations by category">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setCategoryFilter(cat)}
              aria-pressed={categoryFilter === cat}
              aria-label={`Filter by ${cat === 'all' ? 'all categories' : getCategoryLabel(cat)}`}
              className={`px-3 py-1 rounded-md text-sm transition-colors ${
                categoryFilter === cat
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary hover:bg-accent'
              }`}
            >
              {cat === 'all' ? 'All' : getCategoryLabel(cat)}
            </button>
          ))}
        </div>

        {/* Recommendations List */}
        <div className="space-y-3" role="list" aria-label="Recommendations">
          {filteredRecommendations.map((rec, index) => (
            <div
              key={rec.id}
              role="listitem"
              className={`border rounded-lg p-4 ${
                isLive && index === filteredRecommendations.length - 1
                  ? 'animate-pulse transition-all duration-300'
                  : 'transition-all duration-300'
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className={`px-2 py-1 rounded text-xs font-semibold ${getPriorityColor(
                        rec.priority
                      )}`}
                    >
                      {rec.priority.toUpperCase()}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {getCategoryLabel(rec.category)}
                    </span>
                  </div>
                  <h3 className="font-semibold mb-2">{rec.title}</h3>

                  {/* Always show impact */}
                  <div className="text-sm text-muted-foreground mb-2">
                    {rec.impact}
                  </div>

                  {/* Expandable description */}
                  {expandedIds.has(rec.id) && (
                    <div className="space-y-2 mt-3">
                      <p className="text-sm text-muted-foreground">
                        {rec.description}
                      </p>
                    </div>
                  )}
                </div>

                <button
                  onClick={() => toggleExpanded(rec.id)}
                  className="text-sm text-primary hover:underline whitespace-nowrap"
                  aria-label={
                    expandedIds.has(rec.id)
                      ? 'Collapse details'
                      : 'Show More Details'
                  }
                >
                  {expandedIds.has(rec.id) ? 'Collapse' : 'Show More Details'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
})
