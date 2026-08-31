/**
 * The API client.
 *
 * The token arrives on `window.__CIWS__` when the page is served by the hub
 * itself, and falls back to a query parameter (stashed in localStorage) for the
 * Vite dev server, where the page comes from :5173 and the API from :8787.
 */

declare global {
  interface Window {
    __CIWS__?: { token: string; version: string; requiresToken: boolean }
  }
}

const TOKEN_KEY = 'ciws.token'

function readToken(): string {
  const injected = window.__CIWS__?.token
  if (injected) {
    localStorage.setItem(TOKEN_KEY, injected)
    return injected
  }
  const fromUrl = new URLSearchParams(window.location.search).get('token')
  if (fromUrl) {
    localStorage.setItem(TOKEN_KEY, fromUrl)
    // Keep the token out of the address bar, and out of any link the user shares.
    window.history.replaceState({}, '', window.location.pathname)
    return fromUrl
  }
  return localStorage.getItem(TOKEN_KEY) ?? ''
}

export const token = readToken()
export const version = window.__CIWS__?.version ?? 'dev'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code = 'error',
    readonly detail: Record<string, unknown> = {},
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

function headers(json = true): HeadersInit {
  const h: Record<string, string> = {}
  if (json) h['content-type'] = 'application/json'
  if (token) h['authorization'] = `Bearer ${token}`
  return h
}

async function handle<T>(response: Response): Promise<T> {
  if (response.ok) {
    if (response.status === 204) return undefined as T
    return (await response.json()) as T
  }
  let message = `${response.status} ${response.statusText}`
  let code = 'error'
  let detail: Record<string, unknown> = {}
  try {
    const body = await response.json()
    message = body.message || body.detail || message
    code = body.error || code
    detail = body.detail || {}
  } catch {
    /* a non-JSON error body is still an error */
  }
  throw new ApiError(message, response.status, code, detail)
}

export const api = {
  async get<T>(path: string, params?: Record<string, unknown>): Promise<T> {
    const query = params
      ? '?' +
        new URLSearchParams(
          Object.entries(params)
            .filter(([, v]) => v !== undefined && v !== null && v !== '')
            .map(([k, v]) => [k, String(v)]),
        )
      : ''
    return handle<T>(await fetch(`/api${path}${query}`, { headers: headers(false) }))
  },

  async post<T>(path: string, body?: unknown): Promise<T> {
    return handle<T>(
      await fetch(`/api${path}`, {
        method: 'POST',
        headers: headers(),
        body: body === undefined ? undefined : JSON.stringify(body),
      }),
    )
  },

  async patch<T>(path: string, body: unknown): Promise<T> {
    return handle<T>(
      await fetch(`/api${path}`, { method: 'PATCH', headers: headers(), body: JSON.stringify(body) }),
    )
  },

  async put<T>(path: string, body: unknown): Promise<T> {
    return handle<T>(
      await fetch(`/api${path}`, { method: 'PUT', headers: headers(), body: JSON.stringify(body) }),
    )
  },

  async del<T>(path: string): Promise<T> {
    return handle<T>(await fetch(`/api${path}`, { method: 'DELETE', headers: headers(false) }))
  },

  async upload<T>(path: string, file: File, params?: Record<string, string>): Promise<T> {
    const form = new FormData()
    form.append('file', file)
    const query = params ? '?' + new URLSearchParams(params) : ''
    return handle<T>(
      await fetch(`/api${path}${query}`, { method: 'POST', headers: headers(false), body: form }),
    )
  },

  /** A media file URL that works in an <img> tag, where headers cannot be set. */
  mediaUrl(assetId: string, thumb = false): string {
    const query = new URLSearchParams({ token, ...(thumb ? { thumb: 'true' } : {}) })
    return `/api/media/${assetId}/file?${query}`
  },
}

// ---------------------------------------------------------------------------
// Chat streaming
// ---------------------------------------------------------------------------

export type ChatFrame =
  | { type: 'conversation'; conversation_id: string }
  | { type: 'start'; run_id: string; model: string }
  | { type: 'delta'; text: string }
  | { type: 'thinking'; text: string }
  | { type: 'error'; error: string }
  | {
      type: 'done'
      run_id: string
      model: string
      finish_reason: string
      usage: Record<string, number>
      duration_ms: number
    }

export interface ChatOptions {
  message: string
  conversation_id?: string | null
  project_id?: string | null
  agent?: string
  model?: string
  attachments?: { type: string; data: string; mime_type: string; name: string }[]
}

/**
 * Stream a chat turn.
 *
 * `fetch` with a reader rather than EventSource: EventSource cannot POST, and
 * the request body carries the message, history flags and attachments.
 */
export async function streamChat(
  options: ChatOptions,
  onFrame: (frame: ChatFrame) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify(options),
    signal,
  })

  if (!response.ok || !response.body) {
    await handle(response)
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE frames are separated by a blank line; a chunk may split one in half.
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const raw = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf('\n\n')

      const payload = raw
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())
        .join('')
      if (!payload || payload === '[DONE]') continue
      try {
        onFrame(JSON.parse(payload) as ChatFrame)
      } catch {
        /* a malformed frame is skipped, not fatal */
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Live event socket
// ---------------------------------------------------------------------------

export interface BusEvent {
  id: string
  topic: string
  ts: number
  data: Record<string, any>
}

type Listener = (event: BusEvent) => void

/**
 * One socket for the whole app, with reconnect.
 *
 * Panels subscribe to topic prefixes rather than each opening their own socket,
 * so a workspace with nine panels open still holds one connection.
 */
class EventStream {
  private socket: WebSocket | null = null
  private listeners = new Map<string, Set<Listener>>()
  private retry = 0
  private timer: number | null = null
  private closed = false

  connected = false
  onStatus: ((connected: boolean) => void) | null = null

  connect(): void {
    if (this.socket || this.closed) return
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${protocol}://${window.location.host}/api/ws?token=${encodeURIComponent(token)}`
    const socket = new WebSocket(url)
    this.socket = socket

    socket.onopen = () => {
      this.retry = 0
      this.connected = true
      this.onStatus?.(true)
    }
    socket.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data) as BusEvent
        if (event.topic === 'ping') return
        this.dispatch(event)
      } catch {
        /* ignore malformed frames */
      }
    }
    socket.onclose = () => {
      this.socket = null
      this.connected = false
      this.onStatus?.(false)
      this.scheduleReconnect()
    }
    socket.onerror = () => socket.close()
  }

  private scheduleReconnect(): void {
    if (this.closed || this.timer !== null) return
    // Exponential backoff to 15s: a server that is restarting should not be
    // hammered, but a transient blip should recover in under a second.
    const delay = Math.min(15000, 400 * 2 ** this.retry++)
    this.timer = window.setTimeout(() => {
      this.timer = null
      this.connect()
    }, delay)
  }

  private dispatch(event: BusEvent): void {
    for (const [prefix, set] of this.listeners) {
      if (prefix === '*' || event.topic === prefix || event.topic.startsWith(prefix)) {
        for (const listener of set) {
          try {
            listener(event)
          } catch (error) {
            console.error('event listener failed', error)
          }
        }
      }
    }
  }

  on(prefix: string, listener: Listener): () => void {
    let set = this.listeners.get(prefix)
    if (!set) {
      set = new Set()
      this.listeners.set(prefix, set)
    }
    set.add(listener)
    this.connect()
    return () => {
      set!.delete(listener)
      if (set!.size === 0) this.listeners.delete(prefix)
    }
  }

  close(): void {
    this.closed = true
    this.socket?.close()
  }
}

export const events = new EventStream()

// ---------------------------------------------------------------------------
// Shapes returned by the API
// ---------------------------------------------------------------------------

export interface ModelRow {
  id: string
  provider: string
  name: string
  display_name: string
  context_window: number
  max_output: number
  modalities: string[]
  capabilities: string[]
  input_cost_per_mtok: number
  output_cost_per_mtok: number
  local: boolean
  description: string
  configured: boolean
  favorite: boolean
  usage: {
    calls: number
    errors: number
    input_tokens: number
    output_tokens: number
    cost_usd: number
    avg_latency_ms: number
  }
}

export interface AgentRow {
  id: string
  slug: string
  name: string
  description: string
  system_prompt: string
  model: string
  tools: string[]
  max_steps: number
  color: string
  icon: string
  builtin: boolean
  enabled: boolean
}

export interface MessageRow {
  id: string
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  thinking: string
  parts: any[]
  model: string
  run_id: string | null
  error: string
  input_tokens: number
  output_tokens: number
  cost_usd: number
  created_at: string
}

export interface ConversationRow {
  id: string
  title: string
  model: string
  project_id: string | null
  pinned: boolean
  message_count?: number
  updated_at: string
  created_at: string
}

export interface MemoryRow {
  id: string
  kind: string
  content: string
  summary: string
  importance: number
  confidence: number
  access_count: number
  pinned: boolean
  archived: boolean
  tags: string[]
  source: string
  created_at: string
  score?: number
  reasons?: string[]
}

export interface GraphNode {
  id: string
  type: string
  type_label: string
  name: string
  description: string
  aliases: string[]
  properties: Record<string, unknown>
  salience: number
  mention_count: number
  degree: number
  color: string
  icon: string
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  type: string
  label: string
  weight: number
  directed: boolean
  color: string
}

export interface DocumentRow {
  id: string
  title: string
  source_type: string
  source_uri: string
  mime_type: string
  size_bytes: number
  chunk_count: number
  page_count: number
  status: string
  error: string
  summary: string
  tags: string[]
  created_at: string
}

export interface AssetRow {
  id: string
  kind: string
  status: string
  prompt: string
  negative_prompt: string
  provider: string
  model: string
  width: number
  height: number
  duration_s: number
  size_bytes: number
  cost_usd: number
  favorite: boolean
  error: string
  created_at: string
}

export interface HubRow {
  id: string
  slug: string
  name: string
  kind: string
  description: string
  config: Record<string, any>
  enabled: boolean
  autostart: boolean
  status: string
  status_detail: string
  tool_count: number
  tools: string[]
  builtin: boolean
  running: boolean
}

export interface ToolRow {
  name: string
  description: string
  parameters: Record<string, any>
  category: string
  risk: string
  source: string
  enabled: boolean
}

export interface RunRow {
  id: string
  agent_slug: string
  goal: string
  status: string
  model: string
  steps: number
  input_tokens: number
  output_tokens: number
  cost_usd: number
  duration_ms: number
  error: string
  started_at: string
}

export interface WorkflowRow {
  id: string
  name: string
  description: string
  graph: { nodes: any[]; edges: any[] }
  inputs: Record<string, unknown>
  enabled: boolean
}

export interface TaskRow {
  id: string
  title: string
  detail: string
  status: string
  priority: number
  assignee: string
  due_at: string | null
  tags: string[]
  created_at: string
}

export interface CredentialRow {
  name: string
  configured: boolean
  masked: string
  source: string
  env_names: string[]
}

export interface SystemInfo {
  version: string
  python: string
  platform: string
  paths: Record<string, string>
  settings: any
  providers: {
    id: string
    label: string
    configured: boolean
    local: boolean
    requires_key: boolean
    env_hint: string
    website: string
  }[]
  embeddings: { backend: string; dim: number; local: boolean; quality: string; note: string }
  tools: number
  hubs_running: string[]
}
