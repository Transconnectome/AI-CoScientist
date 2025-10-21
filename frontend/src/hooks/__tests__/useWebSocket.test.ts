import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { useWebSocket } from '../useWebSocket'

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static instances: MockWebSocket[] = []

  readyState = MockWebSocket.CONNECTING
  onopen: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null

  constructor(public url: string) {
    MockWebSocket.instances.push(this)
    // Simulate connection opening after a tick
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN
      this.onopen?.(new Event('open'))
    }, 0)
  }

  send(data: string) {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open')
    }
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.(new CloseEvent('close'))
  }

  static getLastInstance(): MockWebSocket {
    return this.instances[this.instances.length - 1]
  }

  static reset() {
    this.instances = []
  }
}

describe('useWebSocket Hook - TDD RED Phase', () => {
  beforeEach(() => {
    MockWebSocket.reset()
    vi.stubGlobal('WebSocket', MockWebSocket)
  })

  afterEach(() => {
    MockWebSocket.reset()
    vi.unstubAllGlobals()
  })

  describe('Connection Management', () => {
    it('should connect to WebSocket on mount', async () => {
      const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

      expect(result.current.isConnected).toBe(false)

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })
    })

    it('should disconnect on unmount', async () => {
      const { result, unmount } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })

      const ws = MockWebSocket.getLastInstance()
      unmount()

      // Should close the connection
      expect(ws.readyState).toBe(MockWebSocket.CLOSED)
    })

    it('should handle connection URL changes', async () => {
      const { result, rerender } = renderHook(
        ({ url }) => useWebSocket(url),
        { initialProps: { url: 'ws://localhost:8000/ws/1' } }
      )

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })

      // Change URL
      rerender({ url: 'ws://localhost:8000/ws/2' })

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })
    })
  })

  describe('Message Handling', () => {
    it('should receive and store messages', async () => {
      const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })

      // Simulate receiving a message
      const ws = MockWebSocket.getLastInstance()
      const testMessage = { type: 'score_update', data: { score: 85 } }
      ws.onmessage?.(new MessageEvent('message', { data: JSON.stringify(testMessage) }))

      await waitFor(() => {
        expect(result.current.lastMessage).toEqual(testMessage)
      })
    })

    it('should handle multiple messages', async () => {
      const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })

      const ws = MockWebSocket.getLastInstance()

      // Send first message
      const message1 = { type: 'progress', data: { percentage: 50 } }
      ws.onmessage?.(new MessageEvent('message', { data: JSON.stringify(message1) }))

      await waitFor(() => {
        expect(result.current.lastMessage).toEqual(message1)
      })

      // Send second message
      const message2 = { type: 'progress', data: { percentage: 75 } }
      ws.onmessage?.(new MessageEvent('message', { data: JSON.stringify(message2) }))

      await waitFor(() => {
        expect(result.current.lastMessage).toEqual(message2)
      })
    })

    it('should handle malformed JSON messages gracefully', async () => {
      const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })

      const ws = MockWebSocket.getLastInstance()

      // Send invalid JSON
      ws.onmessage?.(new MessageEvent('message', { data: 'invalid json {' }))

      // Should not crash and error should be captured
      await waitFor(() => {
        expect(result.current.error).toBeTruthy()
      })
    })
  })

  describe('Sending Messages', () => {
    it('should send messages when connected', async () => {
      const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })

      const testMessage = { action: 'subscribe', paper_id: '123' }

      // Should not throw
      expect(() => {
        result.current.sendMessage(testMessage)
      }).not.toThrow()
    })

    it('should queue messages when not connected', async () => {
      const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

      // Try to send before connection is open
      const testMessage = { action: 'subscribe', paper_id: '123' }
      result.current.sendMessage(testMessage)

      // Should not throw, message should be queued
      expect(result.current.error).toBeNull()
    })
  })

  describe('Error Handling', () => {
    it('should handle connection errors', async () => {
      const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })

      const ws = MockWebSocket.getLastInstance()
      ws.onerror?.(new Event('error'))

      await waitFor(() => {
        expect(result.current.error).toBeTruthy()
      })
    })

    it('should handle connection close', async () => {
      const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })

      const ws = MockWebSocket.getLastInstance()
      ws.close()

      await waitFor(() => {
        expect(result.current.isConnected).toBe(false)
      })
    })

    it('should attempt reconnection after disconnect', async () => {
      const { result } = renderHook(() =>
        useWebSocket('ws://localhost:8000/ws', { reconnect: true, reconnectInterval: 100 })
      )

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })

      // Force disconnect
      const ws = MockWebSocket.getLastInstance()
      ws.readyState = MockWebSocket.CLOSED
      ws.onclose?.(new CloseEvent('close'))

      await waitFor(() => {
        expect(result.current.isConnected).toBe(false)
      })

      // Should attempt reconnection
      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      }, { timeout: 1000 })
    })
  })

  describe('Connection State', () => {
    it('should track connection state correctly', async () => {
      const { result } = renderHook(() => useWebSocket('ws://localhost:8000/ws'))

      // Initially connecting
      expect(result.current.isConnected).toBe(false)

      // Connected after open event
      await waitFor(() => {
        expect(result.current.isConnected).toBe(true)
      })

      // Get readyState
      expect(result.current.readyState).toBe(MockWebSocket.OPEN)
    })
  })
})
