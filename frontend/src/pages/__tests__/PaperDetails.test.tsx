import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@/test/utils'
import { PaperDetails } from '../PaperDetails'
import { useParams } from 'react-router-dom'

// Mock react-router
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useParams: vi.fn(),
    useNavigate: () => vi.fn(),
  }
})

// Mock API
vi.mock('@/api/papers', () => ({
  papersApi: {
    getPaperById: vi.fn(),
  },
}))

// Mock WebSocket hook
vi.mock('@/hooks/useWebSocket', () => ({
  useWebSocket: vi.fn(() => ({
    isConnected: true,
    lastMessage: null,
    error: null,
    sendMessage: vi.fn(),
    readyState: 1,
  })),
}))

describe('PaperDetails Page - TDD RED Phase', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useParams).mockReturnValue({ paperId: 'test-paper-123' })
  })

  describe('Page Rendering', () => {
    it('should render paper details page with paper ID', async () => {
      const { papersApi } = await import('@/api/papers')
      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Test Research Paper',
          authors: ['John Doe', 'Jane Smith'],
          abstract: 'This is a test abstract',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByText(/test research paper/i)).toBeInTheDocument()
      })
    })

    it('should display paper title prominently', async () => {
      const { papersApi } = await import('@/api/papers')
      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Novel Machine Learning Approach',
          authors: ['Dr. Smith'],
          abstract: 'Abstract here',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByRole('heading', { name: /novel machine learning approach/i })).toBeInTheDocument()
      })
    })

    it('should display authors list', async () => {
      const { papersApi } = await import('@/api/papers')
      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Test Paper',
          authors: ['Alice Johnson', 'Bob Williams', 'Carol Davis'],
          abstract: 'Abstract',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByText(/alice johnson/i)).toBeInTheDocument()
        expect(screen.getByText(/bob williams/i)).toBeInTheDocument()
        expect(screen.getByText(/carol davis/i)).toBeInTheDocument()
      })
    })

    it('should display paper abstract', async () => {
      const { papersApi } = await import('@/api/papers')
      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Test Paper',
          authors: ['Author'],
          abstract: 'This is a detailed abstract about the research methodology and findings',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByText(/detailed abstract about the research/i)).toBeInTheDocument()
      })
    })
  })

  describe('WebSocket Connection', () => {
    it('should connect to paper-specific WebSocket', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')
      const { papersApi } = await import('@/api/papers')

      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'paper-456',
          title: 'Test',
          authors: [],
          abstract: '',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      vi.mocked(useParams).mockReturnValue({ paperId: 'paper-456' })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(useWebSocket).toHaveBeenCalledWith(
          expect.stringContaining('paper-456'),
          expect.any(Object)
        )
      })
    })

    it('should show connection status indicator', async () => {
      const { papersApi } = await import('@/api/papers')
      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Test',
          authors: [],
          abstract: '',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByText(/connected|live/i)).toBeInTheDocument()
      })
    })
  })

  describe('Real-time Score Updates', () => {
    it('should update scores when receiving WebSocket messages', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')
      const { papersApi } = await import('@/api/papers')

      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Test',
          authors: [],
          abstract: '',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: true,
        lastMessage: {
          type: 'score_update',
          data: {
            category: 'novelty',
            score: 88,
            confidence: 0.92,
          },
        },
        error: null,
        sendMessage: vi.fn(),
        readyState: 1,
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByText(/88/)).toBeInTheDocument()
      })
    })

    it('should handle overall score updates', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')
      const { papersApi } = await import('@/api/papers')

      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Test',
          authors: [],
          abstract: '',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: true,
        lastMessage: {
          type: 'overall_score',
          data: {
            score: 85,
          },
        },
        error: null,
        sendMessage: vi.fn(),
        readyState: 1,
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByText(/overall.*85|85.*overall/i)).toBeInTheDocument()
      })
    })
  })

  describe('Recommendations Display', () => {
    it('should show recommendations section', async () => {
      const { papersApi } = await import('@/api/papers')
      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Test',
          authors: [],
          abstract: '',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByText(/recommendations/i)).toBeInTheDocument()
      })
    })

    it('should display streaming recommendations', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')
      const { papersApi } = await import('@/api/papers')

      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Test',
          authors: [],
          abstract: '',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: true,
        lastMessage: {
          type: 'recommendation_added',
          data: {
            id: 'rec-1',
            category: 'clarity',
            priority: 'high',
            title: 'Improve abstract clarity',
            description: 'Consider restructuring the abstract',
            impact: 'High readability improvement',
          },
        },
        error: null,
        sendMessage: vi.fn(),
        readyState: 1,
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByText(/improve abstract clarity/i)).toBeInTheDocument()
      })
    })
  })

  describe('Loading States', () => {
    it('should show loading state while fetching paper', async () => {
      const { papersApi } = await import('@/api/papers')
      vi.mocked(papersApi.getPaperById).mockReturnValue(
        new Promise(() => {}) // Never resolves
      )

      render(<PaperDetails />)

      expect(screen.getByText(/loading/i)).toBeInTheDocument()
    })

    it('should show skeleton loaders for content', async () => {
      const { papersApi } = await import('@/api/papers')
      vi.mocked(papersApi.getPaperById).mockReturnValue(
        new Promise(() => {})
      )

      const { container } = render(<PaperDetails />)

      // Check for skeleton elements
      const skeletons = container.querySelectorAll('[class*="animate-pulse"]')
      expect(skeletons.length).toBeGreaterThan(0)
    })
  })

  describe('Error Handling', () => {
    it('should display error when paper not found', async () => {
      const { papersApi } = await import('@/api/papers')
      vi.mocked(papersApi.getPaperById).mockRejectedValue(
        new Error('Paper not found')
      )

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByText(/paper not found|not found/i)).toBeInTheDocument()
      })
    })

    it('should show error when WebSocket connection fails', async () => {
      const { useWebSocket } = await import('@/hooks/useWebSocket')
      const { papersApi } = await import('@/api/papers')

      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Test',
          authors: [],
          abstract: '',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      vi.mocked(useWebSocket).mockReturnValue({
        isConnected: false,
        lastMessage: null,
        error: new Error('Connection failed'),
        sendMessage: vi.fn(),
        readyState: 3,
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByText(/connection.*failed|disconnected/i)).toBeInTheDocument()
      })
    })
  })

  describe('Navigation and Tabs', () => {
    it('should have tab navigation', async () => {
      const { papersApi } = await import('@/api/papers')
      vi.mocked(papersApi.getPaperById).mockResolvedValue({
        metadata: {
          id: 'test-paper-123',
          title: 'Test',
          authors: [],
          abstract: '',
          uploadedAt: '2024-01-01T00:00:00Z',
          status: 'processing',
        },
      })

      render(<PaperDetails />)

      await waitFor(() => {
        expect(screen.getByRole('tab', { name: /overview/i })).toBeInTheDocument()
        expect(screen.getByRole('tab', { name: /scores/i })).toBeInTheDocument()
        expect(screen.getByRole('tab', { name: /recommendations/i })).toBeInTheDocument()
      })
    })
  })
})
