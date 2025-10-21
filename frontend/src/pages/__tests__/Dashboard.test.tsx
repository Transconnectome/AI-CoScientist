import { describe, it, expect } from 'vitest'
import { render, screen } from '@/test/utils'
import { Dashboard } from '../Dashboard'

describe('Dashboard', () => {
  it('should render dashboard title', () => {
    render(<Dashboard />)
    expect(screen.getByText(/AI-CoScientist Dashboard/i)).toBeInTheDocument()
  })

  it('should render subtitle', () => {
    render(<Dashboard />)
    expect(screen.getByText(/Phase 5 Web UI/i)).toBeInTheDocument()
  })
})
