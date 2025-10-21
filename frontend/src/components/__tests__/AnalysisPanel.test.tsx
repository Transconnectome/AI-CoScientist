import { describe, it, expect } from 'vitest'
import { render, screen } from '@/test/utils'
import { AnalysisPanel } from '../AnalysisPanel'
import type { PaperScores } from '@/types/paper'

describe('AnalysisPanel Component - TDD RED Phase', () => {
  const mockScores: PaperScores = {
    overall: 82,
    novelty: {
      category: 'novelty',
      score: 85,
      confidence: 0.9,
      explanation: 'The paper presents novel approaches to machine learning',
    },
    technical_quality: {
      category: 'technical_quality',
      score: 80,
      confidence: 0.85,
      explanation: 'Strong methodology with minor improvements needed',
    },
    clarity: {
      category: 'clarity',
      score: 78,
      confidence: 0.88,
      explanation: 'Well-written but could improve structure',
    },
    impact: {
      category: 'impact',
      score: 85,
      confidence: 0.92,
      explanation: 'High potential for research community impact',
    },
  }

  describe('Rendering', () => {
    it('should render analysis panel with overall score', () => {
      render(<AnalysisPanel scores={mockScores} isLive={false} />)

      expect(screen.getByText(/overall.*82|82.*overall/i)).toBeInTheDocument()
    })

    it('should display all score categories', () => {
      render(<AnalysisPanel scores={mockScores} isLive={false} />)

      expect(screen.getByText(/novelty/i)).toBeInTheDocument()
      expect(screen.getByText(/technical quality/i)).toBeInTheDocument()
      expect(screen.getByText(/clarity/i)).toBeInTheDocument()
      expect(screen.getByText(/impact/i)).toBeInTheDocument()
    })

    it('should show individual scores', () => {
      render(<AnalysisPanel scores={mockScores} isLive={false} />)

      expect(screen.getByText('85')).toBeInTheDocument() // novelty
      expect(screen.getByText('80')).toBeInTheDocument() // technical
      expect(screen.getByText('78')).toBeInTheDocument() // clarity
    })
  })

  describe('Explanations', () => {
    it('should display score explanations', () => {
      render(<AnalysisPanel scores={mockScores} isLive={false} />)

      expect(screen.getByText(/novel approaches to machine learning/i)).toBeInTheDocument()
      expect(screen.getByText(/strong methodology/i)).toBeInTheDocument()
    })

    it('should have expandable explanation sections', async () => {
      const { user } = render(<AnalysisPanel scores={mockScores} isLive={false} />)

      // Look for expand/collapse buttons or icons
      const expandButtons = screen.getAllByRole('button', { name: /expand|show|more/i })
      expect(expandButtons.length).toBeGreaterThan(0)
    })
  })

  describe('Confidence Display', () => {
    it('should show confidence percentage for each score', () => {
      render(<AnalysisPanel scores={mockScores} isLive={false} />)

      expect(screen.getByText(/90%.*confidence/i)).toBeInTheDocument()
      expect(screen.getByText(/85%.*confidence/i)).toBeInTheDocument()
    })

    it('should visualize confidence with progress bars', () => {
      const { container } = render(<AnalysisPanel scores={mockScores} isLive={false} />)

      // Check for progress bars or confidence indicators
      const confidenceIndicators = container.querySelectorAll('[class*="progress"], [role="progressbar"]')
      expect(confidenceIndicators.length).toBeGreaterThan(0)
    })
  })

  describe('Live Updates Indicator', () => {
    it('should show live indicator when isLive is true', () => {
      render(<AnalysisPanel scores={mockScores} isLive={true} />)

      expect(screen.getByText(/live|updating/i)).toBeInTheDocument()
    })

    it('should not show live indicator when isLive is false', () => {
      render(<AnalysisPanel scores={mockScores} isLive={false} />)

      expect(screen.queryByText(/live/i)).not.toBeInTheDocument()
    })
  })

  describe('Score Color Coding', () => {
    it('should use green color for high scores (>=90)', () => {
      const highScores: PaperScores = {
        ...mockScores,
        novelty: { ...mockScores.novelty, score: 95 },
      }

      const { container } = render(<AnalysisPanel scores={highScores} isLive={false} />)

      const highScoreElement = container.querySelector('[class*="text-green"]')
      expect(highScoreElement).toBeInTheDocument()
    })

    it('should use yellow color for medium scores (60-75)', () => {
      const mediumScores: PaperScores = {
        ...mockScores,
        clarity: { ...mockScores.clarity, score: 65 },
      }

      const { container } = render(<AnalysisPanel scores={mediumScores} isLive={false} />)

      const mediumScoreElement = container.querySelector('[class*="text-yellow"]')
      expect(mediumScoreElement).toBeInTheDocument()
    })
  })

  describe('Empty State', () => {
    it('should show placeholder when no scores available', () => {
      const emptyScores: PaperScores = {
        overall: 0,
        novelty: { category: 'novelty', score: 0, confidence: 0 },
        technical_quality: { category: 'technical_quality', score: 0, confidence: 0 },
        clarity: { category: 'clarity', score: 0, confidence: 0 },
        impact: { category: 'impact', score: 0, confidence: 0 },
      }

      render(<AnalysisPanel scores={emptyScores} isLive={false} />)

      expect(screen.getByText(/analyzing|pending|calculating/i)).toBeInTheDocument()
    })
  })

  describe('Responsive Layout', () => {
    it('should render in grid layout', () => {
      const { container } = render(<AnalysisPanel scores={mockScores} isLive={false} />)

      const gridElement = container.querySelector('[class*="grid"]')
      expect(gridElement).toBeInTheDocument()
    })
  })
})
