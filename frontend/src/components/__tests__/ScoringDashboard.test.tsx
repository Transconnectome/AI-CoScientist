import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@/test/utils'
import { ScoringDashboard } from '../ScoringDashboard'

// Mock useWebSocket hook
vi.mock('@/hooks/useWebSocket', () => ({
  useWebSocket: vi.fn(() => ({
    isConnected: true,
    lastMessage: null,
    error: null,
    sendMessage: vi.fn(),
    readyState: 1,
  })),
}))

describe('ScoringDashboard Component - TDD RED Phase', () => {
  describe('Initial Rendering', () => {
    it('should render dashboard title', () => {
      render(<ScoringDashboard paperId="test-123" />)
      expect(screen.getByText(/paper analysis/i)).toBeInTheDocument()
    })

    it('should show connection status', () => {
      render(<ScoringDashboard paperId="test-123" />)
      expect(screen.getByText(/connected/i)).toBeInTheDocument()
    })

    it('should display paper ID', () => {
      render(<ScoringDashboard paperId="test-paper-456" />)
      expect(screen.getByText(/test-paper-456/i)).toBeInTheDocument()
    })
  })

  describe('Score Display', () => {
    it('should render all score categories', () => {
      render(<ScoringDashboard paperId="test-123" />)

      // Check for score category labels
      expect(screen.getByText(/novelty/i)).toBeInTheDocument()
      expect(screen.getByText(/technical quality/i)).toBeInTheDocument()
      expect(screen.getByText(/clarity/i)).toBeInTheDocument()
      expect(screen.getByText(/impact/i)).toBeInTheDocument()
    })

    it('should show initial scores as pending', () => {
      render(<ScoringDashboard paperId="test-123" />)

      // Should show placeholder state
      const pendingElements = screen.getAllByText(/analyzing/i)
      expect(pendingElements.length).toBeGreaterThan(0)
    })

    it('should display overall score prominently', () => {
      render(<ScoringDashboard paperId="test-123" />)

      expect(screen.getByText(/overall score/i)).toBeInTheDocument()
    })
  })

  describe('Real-time Updates', () => {
    it('should update scores when receiving WebSocket messages', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: true,
        lastMessage: {
          type: 'score_update',
          data: {
            category: 'novelty',
            score: 85,
            confidence: 0.9,
          },
        },
        error: null,
        sendMessage: vi.fn(),
        readyState: 1,
      })

      render(<ScoringDashboard paperId="test-123" />)

      // Score should be displayed
      expect(screen.getByText(/85/)).toBeInTheDocument()
    })

    it('should show progress indicator during analysis', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: true,
        lastMessage: {
          type: 'progress',
          data: {
            stage: 'analyzing_structure',
            percentage: 45,
          },
        },
        error: null,
        sendMessage: vi.fn(),
        readyState: 1,
      })

      render(<ScoringDashboard paperId="test-123" />)

      expect(screen.getByText(/45%/)).toBeInTheDocument()
      expect(screen.getByText(/analyzing structure/i)).toBeInTheDocument()
    })

    it('should update overall score based on category scores', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: true,
        lastMessage: {
          type: 'overall_score',
          data: {
            score: 82,
            breakdown: {
              novelty: 85,
              technical_quality: 80,
              clarity: 78,
              impact: 85,
            },
          },
        },
        error: null,
        sendMessage: vi.fn(),
        readyState: 1,
      })

      render(<ScoringDashboard paperId="test-123" />)

      expect(screen.getByText(/82/)).toBeInTheDocument()
    })
  })

  describe('Connection Status', () => {
    it('should show disconnected state when WebSocket is not connected', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: false,
        lastMessage: null,
        error: null,
        sendMessage: vi.fn(),
        readyState: 3, // CLOSED
      })

      render(<ScoringDashboard paperId="test-123" />)

      expect(screen.getByText(/disconnected/i)).toBeInTheDocument()
    })

    it('should show error state when WebSocket has error', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: false,
        lastMessage: null,
        error: new Error('Connection failed'),
        sendMessage: vi.fn(),
        readyState: 3,
      })

      render(<ScoringDashboard paperId="test-123" />)

      expect(screen.getByText(/connection.*failed/i)).toBeInTheDocument()
    })

    it('should show reconnecting state', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: false,
        lastMessage: null,
        error: null,
        sendMessage: vi.fn(),
        readyState: 0, // CONNECTING
      })

      render(<ScoringDashboard paperId="test-123" />)

      expect(screen.getByText(/connecting/i)).toBeInTheDocument()
    })
  })

  describe('Score Visualization', () => {
    it('should render score cards for each category', () => {
      render(<ScoringDashboard paperId="test-123" />)

      // Should have 4 category cards + 1 overall score card
      const cards = screen.getAllByRole('article')
      expect(cards.length).toBeGreaterThanOrEqual(4)
    })

    it('should show confidence indicators', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: true,
        lastMessage: {
          type: 'score_update',
          data: {
            category: 'novelty',
            score: 85,
            confidence: 0.95,
          },
        },
        error: null,
        sendMessage: vi.fn(),
        readyState: 1,
      })

      render(<ScoringDashboard paperId="test-123" />)

      expect(screen.getByText(/95%.*confidence/i)).toBeInTheDocument()
    })

    it('should color-code scores based on value', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: true,
        lastMessage: {
          type: 'score_update',
          data: {
            category: 'novelty',
            score: 95, // High score
            confidence: 0.9,
          },
        },
        error: null,
        sendMessage: vi.fn(),
        readyState: 1,
      })

      const { container } = render(<ScoringDashboard paperId="test-123" />)

      // Check for high score styling (green)
      const scoreElement = container.querySelector('[class*="text-green"]')
      expect(scoreElement).toBeInTheDocument()
    })
  })

  describe('Analysis Stages', () => {
    it('should display current analysis stage', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: true,
        lastMessage: {
          type: 'stage_update',
          data: {
            stage: 'extracting_methodology',
            description: 'Analyzing research methodology',
          },
        },
        error: null,
        sendMessage: vi.fn(),
        readyState: 1,
      })

      render(<ScoringDashboard paperId="test-123" />)

      expect(screen.getByText(/analyzing research methodology/i)).toBeInTheDocument()
    })

    it('should show completion status', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: true,
        lastMessage: {
          type: 'analysis_complete',
          data: {
            status: 'completed',
            timestamp: new Date().toISOString(),
          },
        },
        error: null,
        sendMessage: vi.fn(),
        readyState: 1,
      })

      render(<ScoringDashboard paperId="test-123" />)

      expect(screen.getByText(/analysis.*complete/i)).toBeInTheDocument()
    })
  })
})
