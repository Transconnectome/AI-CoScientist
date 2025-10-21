import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@/test/utils'
import userEvent from '@testing-library/user-event'
import { FileUpload } from '../FileUpload'

describe('FileUpload Component - TDD RED Phase', () => {
  describe('Basic File Input', () => {
    it('should render file upload input', () => {
      render(<FileUpload onUpload={vi.fn()} />)
      expect(screen.getByLabelText(/upload.*paper/i)).toBeInTheDocument()
    })

    it('should accept .pdf files', async () => {
      const onUpload = vi.fn()
      const user = userEvent.setup()

      render(<FileUpload onUpload={onUpload} />)

      const file = new File(['pdf content'], 'paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      expect(onUpload).toHaveBeenCalledWith(file)
    })

    it('should accept .docx files', async () => {
      const onUpload = vi.fn()
      const user = userEvent.setup()

      render(<FileUpload onUpload={onUpload} />)

      const file = new File(
        ['docx content'],
        'paper.docx',
        { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' }
      )
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      expect(onUpload).toHaveBeenCalledWith(file)
    })

    it('should reject invalid file types', async () => {
      const onUpload = vi.fn()
      const user = userEvent.setup()

      render(<FileUpload onUpload={onUpload} />)

      const file = new File(['image content'], 'image.jpg', { type: 'image/jpeg' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      // Note: Browser file input with 'accept' attribute prevents invalid files
      // So we simulate this via drop instead
      const { container } = render(<FileUpload onUpload={onUpload} />)
      const dropzone = container.querySelector('[data-testid="drop-zone"]')

      const dropEvent = new Event('drop', { bubbles: true })
      Object.defineProperty(dropEvent, 'dataTransfer', {
        value: { files: [file], types: ['Files'] }
      })

      dropzone?.dispatchEvent(dropEvent)

      await waitFor(() => {
        expect(screen.getByText(/only.*pdf.*docx/i)).toBeInTheDocument()
      })

      expect(onUpload).not.toHaveBeenCalled()
    })
  })

  describe('Drag and Drop', () => {
    it('should show drop zone with instructions', () => {
      render(<FileUpload onUpload={vi.fn()} />)
      expect(screen.getByText(/drag.*drop/i)).toBeInTheDocument()
      expect(screen.getByText(/or.*click/i)).toBeInTheDocument()
    })

    it('should highlight drop zone on drag over', async () => {
      const { container } = render(<FileUpload onUpload={vi.fn()} />)

      const dropzone = container.querySelector('[data-testid="drop-zone"]')
      expect(dropzone).not.toHaveClass('border-primary')

      // Simulate drag over using fireEvent
      const dragOverEvent = new Event('dragover', { bubbles: true })
      Object.defineProperty(dragOverEvent, 'dataTransfer', {
        value: { types: ['Files'] }
      })

      dropzone?.dispatchEvent(dragOverEvent)

      await waitFor(() => {
        expect(dropzone).toHaveClass('border-primary')
      })
    })

    it('should accept dropped PDF file', async () => {
      const onUpload = vi.fn()
      const { container } = render(<FileUpload onUpload={onUpload} />)

      const file = new File(['pdf content'], 'paper.pdf', { type: 'application/pdf' })
      const dropzone = container.querySelector('[data-testid="drop-zone"]')

      const dropEvent = new Event('drop', { bubbles: true })
      Object.defineProperty(dropEvent, 'dataTransfer', {
        value: {
          files: [file],
          types: ['Files'],
        }
      })

      dropzone?.dispatchEvent(dropEvent)

      await waitFor(() => {
        expect(onUpload).toHaveBeenCalledWith(file)
      })
    })
  })

  describe('File Validation', () => {
    it('should reject files larger than 10MB', async () => {
      const onUpload = vi.fn()
      const { container } = render(<FileUpload onUpload={onUpload} maxSize={10 * 1024 * 1024} />)

      // Create a large file via drop (bypass browser file input)
      const file = new File(['content'], 'large.pdf', { type: 'application/pdf' })
      Object.defineProperty(file, 'size', { value: 11 * 1024 * 1024 })

      const dropzone = container.querySelector('[data-testid="drop-zone"]')
      const dropEvent = new Event('drop', { bubbles: true })
      Object.defineProperty(dropEvent, 'dataTransfer', {
        value: { files: [file], types: ['Files'] }
      })

      dropzone?.dispatchEvent(dropEvent)

      await waitFor(() => {
        expect(screen.getByText(/file too large/i)).toBeInTheDocument()
      })

      expect(onUpload).not.toHaveBeenCalled()
    })

    it('should show file name after successful upload', async () => {
      const onUpload = vi.fn()
      const user = userEvent.setup()

      render(<FileUpload onUpload={onUpload} />)

      const file = new File(['content'], 'research-paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      // Check file appears in the success message
      expect(screen.getByText(/file uploaded successfully/i)).toBeInTheDocument()

      // Check file appears in the file info section at the bottom (with getAllByText since it appears twice)
      const fileNames = screen.getAllByText('research-paper.pdf')
      expect(fileNames.length).toBeGreaterThan(0)
      expect(fileNames[0].closest('[class*="muted"]')).toBeInTheDocument()
    })
  })

  describe('Loading State', () => {
    it('should show loading state during upload', async () => {
      const onUpload = vi.fn(() => new Promise(resolve => setTimeout(resolve, 100)))
      const user = userEvent.setup()

      render(<FileUpload onUpload={onUpload} />)

      const file = new File(['content'], 'paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      expect(screen.getByText(/uploading/i)).toBeInTheDocument()
    })

    it('should disable input during upload', async () => {
      const onUpload = vi.fn(() => new Promise(resolve => setTimeout(resolve, 100)))
      const user = userEvent.setup()

      render(<FileUpload onUpload={onUpload} />)

      const file = new File(['content'], 'paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      expect(input).toBeDisabled()
    })
  })

  describe('Error Handling', () => {
    it('should display error message on upload failure', async () => {
      const onUpload = vi.fn(() => Promise.reject(new Error('Upload failed')))
      const user = userEvent.setup()

      render(<FileUpload onUpload={onUpload} />)

      const file = new File(['content'], 'paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      await waitFor(() => {
        expect(screen.getByText(/upload failed/i)).toBeInTheDocument()
      })
    })

    it('should allow retry after failure', async () => {
      const onUpload = vi.fn(() => Promise.reject(new Error('Upload failed')))
      const user = userEvent.setup()

      render(<FileUpload onUpload={onUpload} />)

      const file = new File(['content'], 'paper.pdf', { type: 'application/pdf' })
      const input = screen.getByLabelText(/upload.*paper/i) as HTMLInputElement

      await user.upload(input, file)

      await waitFor(() => {
        expect(screen.getByText(/try again/i)).toBeInTheDocument()
      })

      const retryButton = screen.getByRole('button', { name: /try again/i })
      expect(retryButton).toBeEnabled()
    })
  })
})
