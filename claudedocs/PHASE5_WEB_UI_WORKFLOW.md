# Phase 5: Web UI Implementation Workflow (TDD)

**Status**: Planning
**Estimated Duration**: 3-4 weeks
**Methodology**: Test-Driven Development
**Stack**: React + TypeScript + Vite + Vitest + Testing Library

---

## 🎯 Overview

Build a modern web interface for AI-CoScientist with emphasis on **test-first development**. Every feature will follow the TDD cycle: **RED → GREEN → REFACTOR**.

### Core Features
1. **Drag-and-drop paper upload** with file validation
2. **Real-time paper scoring** with live progress updates
3. **Interactive improvement suggestions** with preview/apply
4. **Visual version comparison** with diff highlighting
5. **Responsive dashboard** with analytics

### Technology Stack Decision

**Frontend Framework**: **React 18 + TypeScript**
- Rationale: Better testing ecosystem (Vitest, Testing Library), existing backend is Python-based so need clear separation
- Alternative considered: Vue 3 (easier learning curve but less mature testing tools)

**Build Tool**: **Vite 5**
- Rationale: Fast HMR, native ESM, excellent TypeScript support

**Testing Stack**:
- **Vitest**: Unit/integration tests (Jest-compatible, faster)
- **Testing Library**: Component testing (React Testing Library)
- **Playwright**: E2E tests (already in project)
- **MSW**: API mocking (Mock Service Worker)

**UI Components**: **shadcn/ui** (Radix + Tailwind)
- Rationale: Accessible, customizable, copy-paste approach (not npm dependency bloat)

**State Management**: **Zustand**
- Rationale: Lightweight, simple API, easy to test

**API Client**: **TanStack Query (React Query)**
- Rationale: Handles caching, loading states, error handling automatically

---

## 📋 Implementation Phases

### Phase 5.1: Project Setup & Infrastructure (Week 1)

**TDD Setup**:
```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install -D vitest @vitest/ui @testing-library/react @testing-library/jest-dom
npm install -D msw playwright @playwright/test
```

#### Task 5.1.1: Test Environment Setup (2 hours)
**Goal**: Configure complete testing infrastructure

**RED Phase** (Write failing tests):
```typescript
// __tests__/setup.test.ts
import { describe, it, expect } from 'vitest'

describe('Testing Infrastructure', () => {
  it('should run basic unit tests', () => {
    expect(true).toBe(true)
  })

  it('should have DOM testing utilities available', () => {
    expect(document.createElement).toBeDefined()
  })
})
```

**GREEN Phase** (Make tests pass):
- Configure `vitest.config.ts`
- Set up `@testing-library/react` with JSDOM environment
- Add test scripts to `package.json`

**REFACTOR Phase**:
- Extract test utilities to `src/test-utils.tsx`
- Create custom render function with providers

**Acceptance Criteria**:
- [ ] `npm test` runs successfully
- [ ] `npm run test:ui` opens Vitest UI
- [ ] DOM queries work in tests

---

#### Task 5.1.2: API Client Infrastructure (4 hours)
**Goal**: Set up TanStack Query with MSW mocking

**RED Phase** (Write failing tests):
```typescript
// src/api/__tests__/client.test.ts
import { describe, it, expect } from 'vitest'
import { apiClient } from '../client'

describe('API Client', () => {
  it('should fetch papers list', async () => {
    const papers = await apiClient.getPapers()
    expect(papers).toHaveLength(0)
  })

  it('should handle 404 errors gracefully', async () => {
    await expect(
      apiClient.getPaper('nonexistent-id')
    ).rejects.toThrow('Paper not found')
  })
})
```

**GREEN Phase** (Implement):
```typescript
// src/api/client.ts
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const apiClient = {
  getPapers: async () => {
    const res = await axios.get(`${API_BASE}/api/v1/papers`)
    return res.data
  },

  getPaper: async (id: string) => {
    try {
      const res = await axios.get(`${API_BASE}/api/v1/papers/${id}`)
      return res.data
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 404) {
        throw new Error('Paper not found')
      }
      throw error
    }
  }
}
```

**MSW Setup**:
```typescript
// src/mocks/handlers.ts
import { rest } from 'msw'

export const handlers = [
  rest.get('http://localhost:8000/api/v1/papers', (req, res, ctx) => {
    return res(ctx.json([]))
  }),

  rest.get('http://localhost:8000/api/v1/papers/:id', (req, res, ctx) => {
    return res(ctx.status(404))
  })
]
```

**REFACTOR Phase**:
- Extract error handling to utility
- Add TypeScript types for API responses
- Create custom hooks: `usePapers()`, `usePaper(id)`

**Acceptance Criteria**:
- [ ] All API client tests pass
- [ ] MSW intercepts requests in tests
- [ ] Error handling works correctly

---

#### Task 5.1.3: Routing Setup (2 hours)
**Goal**: Configure React Router with protected routes

**RED Phase**:
```typescript
// src/routes/__tests__/Router.test.tsx
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { AppRouter } from '../Router'

describe('Routing', () => {
  it('should render home page on /', () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <AppRouter />
      </MemoryRouter>
    )
    expect(screen.getByText(/dashboard/i)).toBeInTheDocument()
  })

  it('should render 404 on unknown route', () => {
    render(
      <MemoryRouter initialEntries={['/nonexistent']}>
        <AppRouter />
      </MemoryRouter>
    )
    expect(screen.getByText(/404/i)).toBeInTheDocument()
  })
})
```

**GREEN Phase**:
```typescript
// src/routes/Router.tsx
import { Routes, Route } from 'react-router-dom'
import { Dashboard } from '@/pages/Dashboard'
import { NotFound } from '@/pages/NotFound'

export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
```

**REFACTOR Phase**:
- Add lazy loading for code splitting
- Create route constants
- Add route guards for auth (future)

**Acceptance Criteria**:
- [ ] All routes render correctly
- [ ] 404 page works
- [ ] Navigation between routes works

---

### Phase 5.2: File Upload Component (Week 1-2)

#### Task 5.2.1: Drag-and-Drop Zone (6 hours)
**Goal**: Implement file upload with drag-and-drop, validation, and progress

**RED Phase** (Write failing tests FIRST):
```typescript
// src/components/FileUpload/__tests__/FileUpload.test.tsx
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { FileUpload } from '../FileUpload'

describe('FileUpload Component', () => {
  it('should accept .pdf and .docx files', async () => {
    const onUpload = vi.fn()
    render(<FileUpload onUpload={onUpload} />)

    const file = new File(['content'], 'paper.pdf', { type: 'application/pdf' })
    const input = screen.getByLabelText(/upload/i)

    await userEvent.upload(input, file)

    expect(onUpload).toHaveBeenCalledWith(file)
  })

  it('should reject invalid file types', async () => {
    const onUpload = vi.fn()
    render(<FileUpload onUpload={onUpload} />)

    const file = new File(['content'], 'image.jpg', { type: 'image/jpeg' })
    const input = screen.getByLabelText(/upload/i)

    await userEvent.upload(input, file)

    expect(screen.getByText(/invalid file type/i)).toBeInTheDocument()
    expect(onUpload).not.toHaveBeenCalled()
  })

  it('should show upload progress', async () => {
    render(<FileUpload onUpload={vi.fn()} />)

    const file = new File(['content'], 'paper.pdf', { type: 'application/pdf' })
    const input = screen.getByLabelText(/upload/i)

    await userEvent.upload(input, file)

    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })

  it('should handle drag-and-drop', async () => {
    const onUpload = vi.fn()
    render(<FileUpload onUpload={onUpload} />)

    const dropzone = screen.getByTestId('dropzone')
    const file = new File(['content'], 'paper.pdf', { type: 'application/pdf' })

    const dragEvent = new DragEvent('drop', {
      dataTransfer: { files: [file] }
    })

    fireEvent.drop(dropzone, dragEvent)

    await waitFor(() => {
      expect(onUpload).toHaveBeenCalledWith(file)
    })
  })
})
```

**GREEN Phase** (Implement minimal working code):
```typescript
// src/components/FileUpload/FileUpload.tsx
import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

interface FileUploadProps {
  onUpload: (file: File) => void
}

export function FileUpload({ onUpload }: FileUploadProps) {
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setError(null)

    if (acceptedFiles.length === 0) {
      setError('Invalid file type. Please upload .pdf or .docx')
      return
    }

    const file = acceptedFiles[0]
    setProgress(50) // Simulate upload progress
    onUpload(file)
    setProgress(100)
  }, [onUpload])

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    maxFiles: 1
  })

  return (
    <div {...getRootProps()} data-testid="dropzone">
      <input {...getInputProps()} aria-label="upload" />
      <p>Drag & drop a paper here, or click to select</p>

      {error && <div role="alert">{error}</div>}
      {progress > 0 && progress < 100 && (
        <progress value={progress} max={100} />
      )}
    </div>
  )
}
```

**REFACTOR Phase**:
- Extract file validation logic
- Add accessibility attributes
- Style with Tailwind/shadcn
- Add file size limit (10MB)
- Add preview of uploaded file name

**Acceptance Criteria**:
- [ ] All tests pass (4/4)
- [ ] Drag-and-drop works
- [ ] File validation works
- [ ] Progress indicator shows
- [ ] Error messages display correctly

---

#### Task 5.2.2: Upload API Integration (4 hours)
**Goal**: Connect upload to backend API with real progress tracking

**RED Phase**:
```typescript
// src/hooks/__tests__/useUploadPaper.test.ts
import { renderHook, waitFor } from '@testing-library/react'
import { useUploadPaper } from '../useUploadPaper'
import { server } from '@/mocks/server'
import { rest } from 'msw'

describe('useUploadPaper', () => {
  it('should upload paper and return paper_id', async () => {
    server.use(
      rest.post('http://localhost:8000/api/v1/papers/upload', (req, res, ctx) => {
        return res(ctx.json({ paper_id: '123', status: 'uploaded' }))
      })
    )

    const { result } = renderHook(() => useUploadPaper())
    const file = new File(['content'], 'paper.pdf', { type: 'application/pdf' })

    result.current.mutate(file)

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
      expect(result.current.data.paper_id).toBe('123')
    })
  })

  it('should track upload progress', async () => {
    const { result } = renderHook(() => useUploadPaper())
    const file = new File(['content'], 'paper.pdf', { type: 'application/pdf' })

    result.current.mutate(file)

    await waitFor(() => {
      expect(result.current.progress).toBeGreaterThan(0)
    })
  })

  it('should handle upload errors', async () => {
    server.use(
      rest.post('http://localhost:8000/api/v1/papers/upload', (req, res, ctx) => {
        return res(ctx.status(500), ctx.json({ error: 'Upload failed' }))
      })
    )

    const { result } = renderHook(() => useUploadPaper())
    const file = new File(['content'], 'paper.pdf', { type: 'application/pdf' })

    result.current.mutate(file)

    await waitFor(() => {
      expect(result.current.isError).toBe(true)
      expect(result.current.error.message).toContain('Upload failed')
    })
  })
})
```

**GREEN Phase**:
```typescript
// src/hooks/useUploadPaper.ts
import { useMutation } from '@tanstack/react-query'
import axios from 'axios'

export function useUploadPaper() {
  return useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(
        'http://localhost:8000/api/v1/papers/upload',
        formData,
        {
          onUploadProgress: (progressEvent) => {
            const progress = progressEvent.total
              ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
              : 0
            // Store progress in mutation context
          }
        }
      )

      return response.data
    }
  })
}
```

**REFACTOR Phase**:
- Add progress state management
- Extract API endpoint to constants
- Add retry logic (3 attempts)
- Add optimistic updates

**Acceptance Criteria**:
- [ ] Upload mutation tests pass
- [ ] Progress tracking works
- [ ] Error handling works
- [ ] Integration with FileUpload component works

---

### Phase 5.3: Real-Time Scoring Dashboard (Week 2)

#### Task 5.3.1: Scoring Progress Component (6 hours)
**Goal**: Display real-time scoring progress with WebSocket updates

**RED Phase**:
```typescript
// src/components/ScoringProgress/__tests__/ScoringProgress.test.tsx
import { render, screen, waitFor } from '@testing-library/react'
import { ScoringProgress } from '../ScoringProgress'

describe('ScoringProgress', () => {
  it('should show "Analyzing..." status initially', () => {
    render(<ScoringProgress paperId="123" />)
    expect(screen.getByText(/analyzing/i)).toBeInTheDocument()
  })

  it('should update progress in real-time', async () => {
    // Mock WebSocket connection
    const mockWS = new MockWebSocket()
    render(<ScoringProgress paperId="123" />)

    mockWS.emit('progress', {
      step: 'GPT-4 Analysis',
      progress: 33
    })

    await waitFor(() => {
      expect(screen.getByText(/gpt-4 analysis/i)).toBeInTheDocument()
      expect(screen.getByRole('progressbar')).toHaveAttribute('value', '33')
    })
  })

  it('should display final scores when complete', async () => {
    render(<ScoringProgress paperId="123" />)

    // Simulate scoring completion
    mockWS.emit('complete', {
      overall: 7.96,
      dimensions: {
        novelty: 7.46,
        methodology: 7.89,
        clarity: 7.45,
        significance: 7.40
      }
    })

    await waitFor(() => {
      expect(screen.getByText(/7.96/)).toBeInTheDocument()
      expect(screen.getByText(/novelty.*7.46/i)).toBeInTheDocument()
    })
  })

  it('should handle scoring errors', async () => {
    render(<ScoringProgress paperId="123" />)

    mockWS.emit('error', { message: 'LLM service unavailable' })

    await waitFor(() => {
      expect(screen.getByText(/llm service unavailable/i)).toBeInTheDocument()
    })
  })
})
```

**GREEN Phase**:
```typescript
// src/components/ScoringProgress/ScoringProgress.tsx
import { useEffect, useState } from 'react'
import { useWebSocket } from '@/hooks/useWebSocket'

interface ScoringProgressProps {
  paperId: string
}

interface ScoringState {
  status: 'analyzing' | 'complete' | 'error'
  step: string
  progress: number
  scores?: {
    overall: number
    dimensions: {
      novelty: number
      methodology: number
      clarity: number
      significance: number
    }
  }
  error?: string
}

export function ScoringProgress({ paperId }: ScoringProgressProps) {
  const [state, setState] = useState<ScoringState>({
    status: 'analyzing',
    step: 'Initializing...',
    progress: 0
  })

  const { sendMessage, lastMessage } = useWebSocket(
    `ws://localhost:8000/api/v1/papers/${paperId}/score/stream`
  )

  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data)

      if (data.type === 'progress') {
        setState(prev => ({
          ...prev,
          step: data.step,
          progress: data.progress
        }))
      } else if (data.type === 'complete') {
        setState({
          status: 'complete',
          step: 'Complete',
          progress: 100,
          scores: data.scores
        })
      } else if (data.type === 'error') {
        setState(prev => ({
          ...prev,
          status: 'error',
          error: data.message
        }))
      }
    }
  }, [lastMessage])

  if (state.status === 'error') {
    return <div role="alert">{state.error}</div>
  }

  if (state.status === 'complete' && state.scores) {
    return (
      <div>
        <h2>Overall Score: {state.scores.overall}/10</h2>
        <ul>
          <li>Novelty: {state.scores.dimensions.novelty}</li>
          <li>Methodology: {state.scores.dimensions.methodology}</li>
          <li>Clarity: {state.scores.dimensions.clarity}</li>
          <li>Significance: {state.scores.dimensions.significance}</li>
        </ul>
      </div>
    )
  }

  return (
    <div>
      <p>{state.step}</p>
      <progress value={state.progress} max={100} />
    </div>
  )
}
```

**REFACTOR Phase**:
- Extract WebSocket hook to reusable utility
- Add animations for progress updates
- Style with shadcn Card, Progress, Badge components
- Add color-coded score indicators (red <7.0, yellow 7.0-8.0, green >8.0)

**Acceptance Criteria**:
- [ ] All tests pass (4/4)
- [ ] WebSocket connection works
- [ ] Real-time updates display
- [ ] Final scores render correctly
- [ ] Error handling works

---

### Phase 5.4: Interactive Improvements UI (Week 2-3)

#### Task 5.4.1: Suggestion Cards Component (8 hours)
**Goal**: Display improvement suggestions with preview and apply functionality

**RED Phase**:
```typescript
// src/components/ImprovementSuggestions/__tests__/SuggestionCard.test.tsx
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SuggestionCard } from '../SuggestionCard'

describe('SuggestionCard', () => {
  const mockSuggestion = {
    id: '1',
    section: 'Abstract',
    title: 'Enhance clarity with crisis framing',
    impact: 0.8,
    effort: 'Medium',
    preview: 'This paper addresses the critical gap...',
    current: 'This paper presents a framework...'
  }

  it('should display suggestion details', () => {
    render(<SuggestionCard suggestion={mockSuggestion} />)

    expect(screen.getByText(/enhance clarity/i)).toBeInTheDocument()
    expect(screen.getByText(/\+0.8/)).toBeInTheDocument()
    expect(screen.getByText(/medium/i)).toBeInTheDocument()
  })

  it('should toggle preview on click', async () => {
    render(<SuggestionCard suggestion={mockSuggestion} />)

    const previewButton = screen.getByRole('button', { name: /preview/i })
    await userEvent.click(previewButton)

    expect(screen.getByText(/this paper addresses the critical gap/i)).toBeInTheDocument()
  })

  it('should show diff when previewing', async () => {
    render(<SuggestionCard suggestion={mockSuggestion} />)

    const previewButton = screen.getByRole('button', { name: /preview/i })
    await userEvent.click(previewButton)

    expect(screen.getByText(/this paper presents a framework/i)).toHaveClass('text-red-500 line-through')
    expect(screen.getByText(/this paper addresses the critical gap/i)).toHaveClass('text-green-500')
  })

  it('should call onApply when apply button clicked', async () => {
    const onApply = vi.fn()
    render(<SuggestionCard suggestion={mockSuggestion} onApply={onApply} />)

    const applyButton = screen.getByRole('button', { name: /apply/i })
    await userEvent.click(applyButton)

    expect(onApply).toHaveBeenCalledWith(mockSuggestion.id)
  })

  it('should disable apply button while applying', async () => {
    const onApply = vi.fn(() => new Promise(resolve => setTimeout(resolve, 1000)))
    render(<SuggestionCard suggestion={mockSuggestion} onApply={onApply} />)

    const applyButton = screen.getByRole('button', { name: /apply/i })
    await userEvent.click(applyButton)

    expect(applyButton).toBeDisabled()
    expect(screen.getByText(/applying/i)).toBeInTheDocument()
  })
})
```

**GREEN Phase**:
```typescript
// src/components/ImprovementSuggestions/SuggestionCard.tsx
import { useState } from 'react'
import { Card, CardHeader, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { DiffView } from './DiffView'

interface Suggestion {
  id: string
  section: string
  title: string
  impact: number
  effort: string
  preview: string
  current: string
}

interface SuggestionCardProps {
  suggestion: Suggestion
  onApply?: (id: string) => Promise<void>
}

export function SuggestionCard({ suggestion, onApply }: SuggestionCardProps) {
  const [showPreview, setShowPreview] = useState(false)
  const [isApplying, setIsApplying] = useState(false)

  const handleApply = async () => {
    if (!onApply) return

    setIsApplying(true)
    try {
      await onApply(suggestion.id)
    } finally {
      setIsApplying(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <Badge>{suggestion.section}</Badge>
            <h3>{suggestion.title}</h3>
          </div>
          <Badge variant="success">+{suggestion.impact}</Badge>
        </div>
        <p className="text-sm text-muted-foreground">{suggestion.effort} effort</p>
      </CardHeader>

      <CardContent>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => setShowPreview(!showPreview)}
          >
            {showPreview ? 'Hide' : 'Preview'}
          </Button>

          <Button
            onClick={handleApply}
            disabled={isApplying}
          >
            {isApplying ? 'Applying...' : 'Apply'}
          </Button>
        </div>

        {showPreview && (
          <DiffView
            oldText={suggestion.current}
            newText={suggestion.preview}
          />
        )}
      </CardContent>
    </Card>
  )
}
```

**REFACTOR Phase**:
- Extract DiffView to separate component
- Add undo functionality
- Add keyboard shortcuts (Enter to apply, Esc to close preview)
- Add optimistic updates for better UX

**Acceptance Criteria**:
- [ ] All tests pass (5/5)
- [ ] Preview toggle works
- [ ] Diff highlighting works
- [ ] Apply functionality works
- [ ] Loading states work

---

#### Task 5.4.2: Batch Apply with Undo (4 hours)
**Goal**: Allow applying multiple suggestions at once with rollback capability

**RED Phase**:
```typescript
// src/components/ImprovementSuggestions/__tests__/BatchActions.test.tsx
describe('BatchActions', () => {
  it('should select multiple suggestions', async () => {
    render(<ImprovementSuggestions suggestions={mockSuggestions} />)

    const checkboxes = screen.getAllByRole('checkbox')
    await userEvent.click(checkboxes[0])
    await userEvent.click(checkboxes[1])

    expect(screen.getByText(/2 selected/i)).toBeInTheDocument()
  })

  it('should apply selected suggestions in order', async () => {
    const onApplyBatch = vi.fn()
    render(<ImprovementSuggestions suggestions={mockSuggestions} onApplyBatch={onApplyBatch} />)

    await userEvent.click(screen.getAllByRole('checkbox')[0])
    await userEvent.click(screen.getAllByRole('checkbox')[1])
    await userEvent.click(screen.getByRole('button', { name: /apply selected/i }))

    expect(onApplyBatch).toHaveBeenCalledWith(['1', '2'])
  })

  it('should show undo toast after applying', async () => {
    render(<ImprovementSuggestions suggestions={mockSuggestions} />)

    await userEvent.click(screen.getAllByRole('checkbox')[0])
    await userEvent.click(screen.getByRole('button', { name: /apply selected/i }))

    await waitFor(() => {
      expect(screen.getByText(/applied successfully/i)).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /undo/i })).toBeInTheDocument()
    })
  })
})
```

**GREEN Phase**: Implement batch selection state and undo logic

**REFACTOR Phase**:
- Add confirmation dialog for batch operations
- Implement undo stack with version tracking
- Add keyboard shortcuts (Ctrl+A for select all, Ctrl+Z for undo)

**Acceptance Criteria**:
- [ ] Batch selection works
- [ ] Apply batch works
- [ ] Undo functionality works

---

### Phase 5.5: Version Comparison Viewer (Week 3)

#### Task 5.5.1: Diff Viewer Component (8 hours)
**Goal**: Visual side-by-side comparison with syntax highlighting

**RED Phase**:
```typescript
// src/components/VersionComparison/__tests__/DiffViewer.test.tsx
describe('DiffViewer', () => {
  it('should display side-by-side comparison', () => {
    const versions = {
      old: { version: '1.0.0', content: 'Original text' },
      new: { version: '1.2.0', content: 'Improved text' }
    }

    render(<DiffViewer versions={versions} />)

    expect(screen.getByText(/1.0.0/)).toBeInTheDocument()
    expect(screen.getByText(/1.2.0/)).toBeInTheDocument()
    expect(screen.getByText(/original text/i)).toBeInTheDocument()
    expect(screen.getByText(/improved text/i)).toBeInTheDocument()
  })

  it('should highlight changes', () => {
    render(<DiffViewer versions={versions} />)

    const deletions = screen.getAllByTestId('deletion')
    const additions = screen.getAllByTestId('addition')

    expect(deletions).toHaveLength(1)
    expect(additions).toHaveLength(1)
  })

  it('should toggle between unified and split view', async () => {
    render(<DiffViewer versions={versions} />)

    const toggleButton = screen.getByRole('button', { name: /unified/i })
    await userEvent.click(toggleButton)

    expect(screen.queryByText(/1.0.0/)).not.toBeInTheDocument()
    expect(screen.getByText(/\-original text/)).toBeInTheDocument()
    expect(screen.getByText(/\+improved text/)).toBeInTheDocument()
  })
})
```

**GREEN Phase**:
```typescript
// src/components/VersionComparison/DiffViewer.tsx
import { useState } from 'react'
import { diffLines } from 'diff'

export function DiffViewer({ versions }) {
  const [viewMode, setViewMode] = useState<'split' | 'unified'>('split')

  const changes = diffLines(versions.old.content, versions.new.content)

  if (viewMode === 'unified') {
    return (
      <div>
        {changes.map((change, idx) => (
          <div
            key={idx}
            className={change.added ? 'bg-green-50' : change.removed ? 'bg-red-50' : ''}
          >
            {change.added && <span>+</span>}
            {change.removed && <span>-</span>}
            {change.value}
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <h3>Version {versions.old.version}</h3>
        <pre>{versions.old.content}</pre>
      </div>
      <div>
        <h3>Version {versions.new.version}</h3>
        <pre>{versions.new.content}</pre>
      </div>
    </div>
  )
}
```

**REFACTOR Phase**:
- Add syntax highlighting with Prism.js
- Add line numbers
- Add expand/collapse for unchanged sections
- Add export diff as .patch file

**Acceptance Criteria**:
- [ ] Split view works
- [ ] Unified view works
- [ ] Change highlighting works
- [ ] View toggle works

---

### Phase 5.6: E2E Testing & Integration (Week 3-4)

#### Task 5.6.1: Playwright E2E Tests (6 hours)
**Goal**: Complete user journey tests

**Test Scenarios**:
```typescript
// e2e/paper-workflow.spec.ts
import { test, expect } from '@playwright/test'

test.describe('Complete Paper Improvement Workflow', () => {
  test('should upload, score, improve, and compare paper', async ({ page }) => {
    // 1. Navigate to app
    await page.goto('http://localhost:5173')

    // 2. Upload paper
    await page.setInputFiles('input[type="file"]', 'test-fixtures/sample-paper.pdf')
    await expect(page.locator('text=Uploaded successfully')).toBeVisible()

    // 3. Wait for scoring to complete
    await expect(page.locator('text=Overall Score')).toBeVisible({ timeout: 30000 })
    const score = await page.locator('[data-testid="overall-score"]').textContent()
    expect(parseFloat(score)).toBeGreaterThan(0)

    // 4. View suggestions
    await page.click('text=View Suggestions')
    await expect(page.locator('[data-testid="suggestion-card"]')).toHaveCount(5)

    // 5. Preview a suggestion
    await page.click('[data-testid="suggestion-card"] >> text=Preview')
    await expect(page.locator('[data-testid="diff-view"]')).toBeVisible()

    // 6. Apply suggestion
    await page.click('text=Apply')
    await expect(page.locator('text=Applied successfully')).toBeVisible()

    // 7. Compare versions
    await page.click('text=Version History')
    await page.selectOption('select[name="version-a"]', '1.0.0')
    await page.selectOption('select[name="version-b"]', '1.1.0')
    await page.click('text=Compare')
    await expect(page.locator('[data-testid="diff-viewer"]')).toBeVisible()

    // 8. Verify version comparison
    const additions = await page.locator('[data-testid="addition"]').count()
    expect(additions).toBeGreaterThan(0)
  })

  test('should handle upload errors gracefully', async ({ page }) => {
    await page.goto('http://localhost:5173')

    // Upload invalid file type
    await page.setInputFiles('input[type="file"]', 'test-fixtures/image.jpg')

    await expect(page.locator('text=Invalid file type')).toBeVisible()
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible()
  })

  test('should support undo functionality', async ({ page }) => {
    await page.goto('http://localhost:5173')

    // ... upload and apply suggestion ...

    await page.click('text=Undo')
    await expect(page.locator('text=Reverted successfully')).toBeVisible()

    // Verify version rolled back
    const currentVersion = await page.locator('[data-testid="current-version"]').textContent()
    expect(currentVersion).toBe('1.0.0')
  })
})
```

**Acceptance Criteria**:
- [ ] Complete workflow test passes
- [ ] Error handling tests pass
- [ ] Undo functionality test passes
- [ ] All E2E tests run in CI/CD

---

### Phase 5.7: Performance & Accessibility (Week 4)

#### Task 5.7.1: Lighthouse Audit & Optimization (4 hours)
**Goal**: Achieve Lighthouse scores >90 in all categories

**Test-Driven Approach**:
```typescript
// e2e/performance.spec.ts
import { test, expect } from '@playwright/test'

test('should meet Lighthouse performance targets', async ({ page }) => {
  const lighthouse = await runLighthouse(page, {
    onlyCategories: ['performance', 'accessibility', 'best-practices']
  })

  expect(lighthouse.lhr.categories.performance.score).toBeGreaterThan(0.9)
  expect(lighthouse.lhr.categories.accessibility.score).toBeGreaterThan(0.9)
  expect(lighthouse.lhr.categories['best-practices'].score).toBeGreaterThan(0.9)
})

test('should have no accessibility violations', async ({ page }) => {
  await page.goto('http://localhost:5173')

  const accessibilityScanResults = await injectAxe(page)
  expect(accessibilityScanResults.violations).toHaveLength(0)
})
```

**Optimization Tasks**:
- [ ] Code splitting for routes
- [ ] Image optimization (WebP, lazy loading)
- [ ] Bundle size analysis (<200KB gzip)
- [ ] Implement React.lazy for heavy components
- [ ] Add service worker for offline support

**Acceptance Criteria**:
- [ ] Performance: >90
- [ ] Accessibility: >90
- [ ] Best Practices: >90
- [ ] Zero accessibility violations

---

#### Task 5.7.2: Accessibility Testing (4 hours)
**Goal**: WCAG 2.1 AA compliance

**Automated Tests**:
```typescript
// src/components/__tests__/a11y.test.tsx
import { axe } from 'vitest-axe'

describe('Accessibility', () => {
  it('FileUpload should have no violations', async () => {
    const { container } = render(<FileUpload onUpload={vi.fn()} />)
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })

  it('ScoringProgress should have no violations', async () => {
    const { container } = render(<ScoringProgress paperId="123" />)
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })

  it('should support keyboard navigation', async () => {
    render(<ImprovementSuggestions suggestions={mockSuggestions} />)

    const firstCard = screen.getAllByRole('article')[0]
    firstCard.focus()

    await userEvent.keyboard('{Enter}')
    expect(screen.getByText(/preview/i)).toBeVisible()

    await userEvent.keyboard('{Escape}')
    expect(screen.queryByText(/preview/i)).not.toBeVisible()
  })
})
```

**Manual Testing Checklist**:
- [ ] All interactive elements keyboard accessible
- [ ] Focus indicators visible
- [ ] Screen reader announces changes
- [ ] Color contrast ratio ≥4.5:1
- [ ] Forms have proper labels
- [ ] Error messages associated with inputs

---

### Phase 5.8: Deployment & CI/CD (Week 4)

#### Task 5.8.1: Docker & Nginx Setup (4 hours)
```dockerfile
# frontend/Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**Test Deployment**:
```typescript
// e2e/deployment.spec.ts
test('should serve production build correctly', async ({ page }) => {
  await page.goto('http://localhost:80')

  // Check CSP headers
  const response = await page.goto('http://localhost:80')
  const headers = response.headers()
  expect(headers['content-security-policy']).toBeDefined()

  // Check compression
  expect(headers['content-encoding']).toBe('gzip')

  // Check caching
  expect(headers['cache-control']).toContain('max-age')
})
```

**Acceptance Criteria**:
- [ ] Docker build succeeds
- [ ] Nginx serves static files
- [ ] Gzip compression enabled
- [ ] Security headers configured

---

#### Task 5.8.2: GitHub Actions CI/CD (3 hours)
```yaml
# .github/workflows/frontend-ci.yml
name: Frontend CI/CD

on:
  push:
    branches: [main]
    paths:
      - 'frontend/**'
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 20
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        working-directory: frontend
        run: npm ci

      - name: Run linter
        working-directory: frontend
        run: npm run lint

      - name: Run type check
        working-directory: frontend
        run: npm run type-check

      - name: Run unit tests
        working-directory: frontend
        run: npm run test:ci

      - name: Run E2E tests
        working-directory: frontend
        run: |
          npm run build
          npm run test:e2e

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: frontend/coverage/lcov.info

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        working-directory: frontend
        run: docker build -t ai-coscientist-frontend .

      - name: Test Docker image
        run: |
          docker run -d -p 8080:80 ai-coscientist-frontend
          sleep 5
          curl -f http://localhost:8080 || exit 1
```

**Acceptance Criteria**:
- [ ] All tests run in CI
- [ ] Coverage reports uploaded
- [ ] Docker image built successfully
- [ ] Deploy preview generated for PRs

---

## 📊 Testing Metrics & Quality Gates

### Coverage Requirements
- **Unit Tests**: ≥80% coverage
- **Integration Tests**: All API interactions covered
- **E2E Tests**: All critical user paths covered

### Quality Gates (Must Pass)
```typescript
// vitest.config.ts
export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80
      },
      exclude: [
        'node_modules/',
        'src/test-utils/**',
        '**/*.test.{ts,tsx}',
        '**/*.spec.{ts,tsx}'
      ]
    }
  }
})
```

### Pre-Commit Hooks
```bash
# .husky/pre-commit
#!/bin/sh
npm run lint
npm run type-check
npm run test:changed
```

---

## 🚀 Deployment Strategy

### Development
```bash
npm run dev  # http://localhost:5173
```

### Staging
```bash
docker-compose -f docker-compose.staging.yml up
# Frontend: http://staging.localhost:80
# Backend: http://staging.localhost:8000
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
# Frontend: https://app.ai-coscientist.com
# Backend: https://api.ai-coscientist.com
```

---

## 📋 Final Checklist

### Week 1
- [ ] Project setup complete
- [ ] Testing infrastructure configured
- [ ] File upload component with tests
- [ ] API client with MSW mocking

### Week 2
- [ ] Real-time scoring dashboard
- [ ] WebSocket integration
- [ ] Improvement suggestions UI
- [ ] Preview/apply functionality

### Week 3
- [ ] Version comparison viewer
- [ ] Batch operations
- [ ] Undo functionality
- [ ] E2E tests written

### Week 4
- [ ] Performance optimization
- [ ] Accessibility audit
- [ ] Docker deployment
- [ ] CI/CD pipeline
- [ ] Documentation complete

---

## 🎯 Success Criteria

**Functional Requirements**:
- [ ] All 5 core features implemented and tested
- [ ] 100% of tests passing (unit + integration + E2E)
- [ ] Zero critical accessibility violations
- [ ] All API integrations working

**Performance Requirements**:
- [ ] Lighthouse Performance >90
- [ ] First Contentful Paint <1.5s
- [ ] Time to Interactive <3s
- [ ] Bundle size <200KB gzipped

**Quality Requirements**:
- [ ] Test coverage >80%
- [ ] TypeScript strict mode enabled
- [ ] Zero ESLint errors
- [ ] Zero console errors in production

---

**Total Estimated Time**: 3-4 weeks (120-160 hours)
**Risk Level**: Medium (new frontend stack, WebSocket complexity)
**Dependencies**: Backend API endpoints must be stable

**Next Steps After Completion**:
1. User acceptance testing
2. Beta deployment to staging
3. Performance monitoring setup
4. User feedback collection
5. Iteration planning for Phase 6
