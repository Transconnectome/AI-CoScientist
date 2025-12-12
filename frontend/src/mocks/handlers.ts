import { rest } from 'msw'

const API_BASE = 'http://localhost:8000'

export const handlers = [
  rest.get(`${API_BASE}/api/v1/health`, (req, res, ctx) => {
    return res(ctx.json({ status: 'ok' }))
  }),

  rest.post(`${API_BASE}/api/v1/papers/upload`, (req, res, ctx) => {
    return res(
      ctx.status(201),
      ctx.json({
        paper_id: 'mock-paper-id',
        status: 'uploaded',
      })
    )
  }),
]
