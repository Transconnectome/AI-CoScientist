import { describe, it, expect } from 'vitest'
import { render, screen } from '@/test/utils'
import { RecommendationsPanel } from '../RecommendationsPanel'
import type { Recommendation } from '@/types/paper'

describe('RecommendationsPanel Component - TDD RED Phase', () => {
  const mockRecommendations: Recommendation[] = [
    {
      id: 'rec-1',
      category: 'clarity',
      priority: 'high',
      title: 'Improve abstract structure',
      description: 'Reorganize abstract to follow IMRAD format',
      impact: 'Significantly improves readability',
    },
    {
      id: 'rec-2',
      category: 'technical_quality',
      priority: 'medium',
      title: 'Add statistical validation',
      description: 'Include p-values and confidence intervals',
      impact: 'Strengthens methodology rigor',
    },
    {
      id: 'rec-3',
      category: 'novelty',
      priority: 'low',
      title: 'Compare with recent work',
      description: 'Reference 2024 papers on similar topics',
      impact: 'Better contextualizes contribution',
    },
  ]

  describe('Rendering', () => {
    it('should render recommendations panel', () => {
      render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      expect(screen.getByText(/recommendations/i)).toBeInTheDocument()
    })

    it('should display all recommendations', () => {
      render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      expect(screen.getByText(/improve abstract structure/i)).toBeInTheDocument()
      expect(screen.getByText(/add statistical validation/i)).toBeInTheDocument()
      expect(screen.getByText(/compare with recent work/i)).toBeInTheDocument()
    })

    it('should show recommendation count', () => {
      render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      expect(screen.getByText(/3.*recommendation/i)).toBeInTheDocument()
    })
  })

  describe('Priority Indicators', () => {
    it('should show high priority badge', () => {
      render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      expect(screen.getByText(/high/i)).toBeInTheDocument()
    })

    it('should color-code priorities', () => {
      const { container } = render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      // High priority should have red/urgent styling
      const highPriority = container.querySelector('[class*="text-red"], [class*="bg-red"]')
      expect(highPriority).toBeInTheDocument()
    })

    it('should sort by priority by default', () => {
      render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      const titles = screen.getAllByRole('heading', { level: 3 })

      // First should be high priority
      expect(titles[0]).toHaveTextContent(/improve abstract/i)
    })
  })

  describe('Category Filtering', () => {
    it('should have category filter buttons', () => {
      render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      expect(screen.getByRole('button', { name: /all/i })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /clarity/i })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /technical/i })).toBeInTheDocument()
    })

    it('should filter recommendations by category', async () => {
      const { user } = render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      const clarityFilter = screen.getByRole('button', { name: /clarity/i })
      await user.click(clarityFilter)

      // Should only show clarity recommendations
      expect(screen.getByText(/improve abstract structure/i)).toBeInTheDocument()
      expect(screen.queryByText(/add statistical validation/i)).not.toBeInTheDocument()
    })
  })

  describe('Expandable Details', () => {
    it('should have expandable recommendation cards', () => {
      render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      const expandButtons = screen.getAllByRole('button', { name: /expand|show.*details/i })
      expect(expandButtons.length).toBeGreaterThan(0)
    })

    it('should show full description when expanded', async () => {
      const { user } = render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      const expandButton = screen.getAllByRole('button', { name: /expand|show.*details/i })[0]
      await user.click(expandButton)

      expect(screen.getByText(/reorganize abstract to follow IMRAD/i)).toBeInTheDocument()
    })

    it('should show impact assessment', () => {
      render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      expect(screen.getByText(/significantly improves readability/i)).toBeInTheDocument()
    })
  })

  describe('Live Streaming', () => {
    it('should show live indicator when streaming', () => {
      render(<RecommendationsPanel recommendations={mockRecommendations} isLive={true} />)

      expect(screen.getByText(/live|streaming/i)).toBeInTheDocument()
    })

    it('should animate new recommendations when added', () => {
      const { rerender } = render(<RecommendationsPanel recommendations={[mockRecommendations[0]]} isLive={true} />)

      rerender(<RecommendationsPanel recommendations={mockRecommendations} isLive={true} />)

      // Check for animation classes or transitions
      const newRec = screen.getByText(/add statistical validation/i)
      expect(newRec.parentElement?.parentElement).toHaveClass(/animate|transition/)
    })
  })

  describe('Empty State', () => {
    it('should show empty state when no recommendations', () => {
      render(<RecommendationsPanel recommendations={[]} isLive={false} />)

      expect(screen.getByText(/no recommendations|analyzing/i)).toBeInTheDocument()
    })

    it('should show analyzing state when live but empty', () => {
      render(<RecommendationsPanel recommendations={[]} isLive={true} />)

      expect(screen.getByText(/generating recommendations/i)).toBeInTheDocument()
    })
  })

  describe('Action Buttons', () => {
    it('should have export recommendations button', () => {
      render(<RecommendationsPanel recommendations={mockRecommendations} isLive={false} />)

      expect(screen.getByRole('button', { name: /export|download/i })).toBeInTheDocument()
    })
  })
})
