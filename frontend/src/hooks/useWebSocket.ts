import { useState, useEffect, useRef, useCallback } from 'react'

interface WebSocketOptions {
  reconnect?: boolean
  reconnectInterval?: number
  reconnectAttempts?: number
}

interface WebSocketHookReturn<T = any> {
  isConnected: boolean
  lastMessage: T | null
  error: Error | null
  sendMessage: (message: any) => void
  readyState: number
}

export function useWebSocket<T = any>(
  url: string,
  options: WebSocketOptions = {}
): WebSocketHookReturn<T> {
  const {
    reconnect = true,
    reconnectInterval = 3000,
    reconnectAttempts = 5,
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [readyState, setReadyState] = useState<number>(WebSocket.CONNECTING)

  const ws = useRef<WebSocket | null>(null)
  const reconnectCount = useRef(0)
  const reconnectTimeout = useRef<NodeJS.Timeout>()
  const messageQueue = useRef<any[]>([])

  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(url)

      ws.current.onopen = () => {
        setIsConnected(true)
        setReadyState(WebSocket.OPEN)
        setError(null)
        reconnectCount.current = 0

        // Send queued messages
        while (messageQueue.current.length > 0) {
          const message = messageQueue.current.shift()
          ws.current?.send(JSON.stringify(message))
        }
      }

      ws.current.onclose = () => {
        setIsConnected(false)
        setReadyState(WebSocket.CLOSED)

        // Attempt reconnection
        if (reconnect && reconnectCount.current < reconnectAttempts) {
          reconnectCount.current++
          reconnectTimeout.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        }
      }

      ws.current.onerror = (event) => {
        setError(new Error('WebSocket connection error'))
        setIsConnected(false)
        setReadyState(WebSocket.CLOSED)
      }

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setLastMessage(data)
          setError(null)
        } catch (err) {
          setError(new Error('Failed to parse message'))
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Connection failed'))
    }
  }, [url, reconnect, reconnectAttempts, reconnectInterval])

  const sendMessage = useCallback((message: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
    } else {
      // Queue message for later
      messageQueue.current.push(message)
    }
  }, [])

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current)
      }
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [connect])

  return {
    isConnected,
    lastMessage,
    error,
    sendMessage,
    readyState,
  }
}
