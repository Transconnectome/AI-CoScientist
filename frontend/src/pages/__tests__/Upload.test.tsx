import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@/test/utils'
import userEvent from '@testing-library/user-event'
import { Upload } from '../Upload'
import { papersApi } from '@/api/papers'

// Mock the papers API
vi.mock('@/api/papers', () => ({
  papersApi: {
    upload: vi.fn(),
  },
}))

// Mock useNavigate
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('Upload Page - API Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Page Rendering', () => {
    it('should render upload page with title and description', () => {
      render(<Upload />)

      expect(screen.getByText(/upload paper/i)).toBeInTheDocument()
      expect(screen.getByText(/AI-powered analysis/i)).toBeInTheDocument()
    })

    it('should render FileUpload component', () => {
      render(<Upload />)

      expect(screen.getByLabelText(/upload.*paper/i)).toBeInTheDocument()
    })
  })

  describe('Successful Upload Flow', () => {
    it('should call API with uploaded file', async () => {
      const mockResponse = {
        paper_id: 'test-paper-id-123',
        status: 'uploaded',
      }

      vi.mocked(papersApi.upload).mockResolvedValue(mockResponse)

      const user = userEvent.setup()
      render(<Upload />)

      const file = new File(['pdf content'], 'test-paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      await waitFor(() => {
        expect(papersApi.upload).toHaveBeenCalledWith(file)
      })
    })

    it('should show success message after upload', async () => {
      const mockResponse = {
        paper_id: 'test-paper-id-123',
        status: 'uploaded',
      }

      vi.mocked(papersApi.upload).mockResolvedValue(mockResponse)

      const user = userEvent.setup()
      render(<Upload />)

      const file = new File(['pdf content'], 'test-paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      await waitFor(() => {
        expect(screen.getByText(/upload successful/i)).toBeInTheDocument()
      })
    })

    it('should show View Paper Analysis button after upload', async () => {
      const mockResponse = {
        paper_id: 'test-paper-id-123',
        status: 'uploaded',
      }

      vi.mocked(papersApi.upload).mockResolvedValue(mockResponse)

      const user = userEvent.setup()
      render(<Upload />)

      const file = new File(['pdf content'], 'test-paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /view paper analysis/i })).toBeInTheDocument()
      })
    })

    it('should navigate to paper details on View button click', async () => {
      const mockResponse = {
        paper_id: 'test-paper-id-123',
        status: 'uploaded',
      }

      vi.mocked(papersApi.upload).mockResolvedValue(mockResponse)

      const user = userEvent.setup()
      render(<Upload />)

      const file = new File(['pdf content'], 'test-paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /view paper analysis/i })).toBeInTheDocument()
      })

      const viewButton = screen.getByRole('button', { name: /view paper analysis/i })
      await user.click(viewButton)

      expect(mockNavigate).toHaveBeenCalledWith('/papers/test-paper-id-123')
    })

    it('should show Upload Another button after success', async () => {
      const mockResponse = {
        paper_id: 'test-paper-id-123',
        status: 'uploaded',
      }

      vi.mocked(papersApi.upload).mockResolvedValue(mockResponse)

      const user = userEvent.setup()
      render(<Upload />)

      const file = new File(['pdf content'], 'test-paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /upload another/i })).toBeInTheDocument()
      })
    })
  })

  describe('Upload Error Handling', () => {
    it('should show error message when API fails', async () => {
      const mockError = new Error('Network error: Failed to connect')

      vi.mocked(papersApi.upload).mockRejectedValue(mockError)

      const user = userEvent.setup()
      render(<Upload />)

      const file = new File(['pdf content'], 'test-paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      await waitFor(() => {
        expect(screen.getByText(/upload failed/i)).toBeInTheDocument()
        const errorMessages = screen.getAllByText(/network error/i)
        expect(errorMessages.length).toBeGreaterThan(0)
      })
    })

    it('should not show success message when upload fails', async () => {
      const mockError = new Error('Upload failed')

      vi.mocked(papersApi.upload).mockRejectedValue(mockError)

      const user = userEvent.setup()
      render(<Upload />)

      const file = new File(['pdf content'], 'test-paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      await waitFor(() => {
        const errorMessages = screen.getAllByText(/upload failed/i)
        expect(errorMessages.length).toBeGreaterThan(0)
      })

      expect(screen.queryByText(/upload successful/i)).not.toBeInTheDocument()
    })

    it('should handle API errors with custom messages', async () => {
      const mockError = new Error('File size too large')

      vi.mocked(papersApi.upload).mockRejectedValue(mockError)

      const user = userEvent.setup()
      render(<Upload />)

      const file = new File(['pdf content'], 'test-paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      await waitFor(() => {
        const errorMessages = screen.getAllByText(/file size too large/i)
        expect(errorMessages.length).toBeGreaterThan(0)
      })
    })
  })

  describe('Upload State Management', () => {
    it('should start in idle state', () => {
      render(<Upload />)

      expect(screen.queryByText(/upload successful/i)).not.toBeInTheDocument()
      expect(screen.queryByText(/upload failed/i)).not.toBeInTheDocument()
    })

    it('should reset error when attempting new upload', async () => {
      const mockError = new Error('First upload failed')
      const mockSuccess = {
        paper_id: 'test-paper-id-123',
        status: 'uploaded',
      }

      vi.mocked(papersApi.upload)
        .mockRejectedValueOnce(mockError)
        .mockResolvedValueOnce(mockSuccess)

      const user = userEvent.setup()
      render(<Upload />)

      // First upload fails
      const file1 = new File(['pdf content'], 'test-paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file1)

      await waitFor(() => {
        const errorMessages = screen.getAllByText(/upload failed/i)
        expect(errorMessages.length).toBeGreaterThan(0)
      })

      // Clear input for retry
      if (input) {
        input.value = ''
      }

      // Second upload succeeds
      const file2 = new File(['pdf content 2'], 'test-paper-2.pdf', { type: 'application/pdf' })
      await user.upload(input, file2)

      await waitFor(() => {
        expect(screen.getByText(/upload successful/i)).toBeInTheDocument()
      })

      // Error message should be gone (check both possible locations)
      const errorMessages = screen.queryAllByText(/first upload failed/i)
      expect(errorMessages.length).toBe(0)
    })
  })
})
