/**
 * The command console: conversation, live tool trace, and run telemetry.
 *
 * The trace column is the reason this looks different from a chat app. When an
 * agent takes fourteen steps to answer, the answer alone is not enough to trust
 * it -- you need to see which tools ran, what they returned, and what it cost.
 * The trace is fed by the event bus, so it fills in as the run happens rather
 * than being reconstructed afterwards.
 */

import {
  Bot,
  Brain,
  ChevronRight,
  CircleStop,
  Cpu,
  ImagePlus,
  MessageSquarePlus,
  Paperclip,
  Pin,
  Send,
  Terminal,
  Trash2,
  Wrench,
  X,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Markdown } from '../components/Markdown'
import {
  Empty,
  ErrorLine,
  Panel,
  Spinner,
  ago,
  classNames,
  compact,
  duration,
  useAsync,
  useLocalState,
  useToast,
  usd,
} from '../components/ui'
import {
  api,
  events,
  streamChat,
  type AgentRow,
  type ConversationRow,
  type MessageRow,
  type ModelRow,
} from '../lib/api'

interface TraceEntry {
  id: string
  kind: 'step' | 'tool' | 'memory' | 'graph' | 'media' | 'notice'
  label: string
  detail: string
  ok: boolean
  ms?: number
  at: number
}

interface Draft {
  content: string
  thinking: string
  runId: string
  model: string
}

export function Command({ projectId }: { projectId: string | null }) {
  const toast = useToast()
  const [conversationId, setConversationId] = useLocalState<string | null>('ciws.conversation', null)
  const [agent, setAgent] = useLocalState('ciws.agent', 'analyst')
  const [model, setModel] = useLocalState('ciws.model', '')
  const [input, setInput] = useState('')
  const [draft, setDraft] = useState<Draft | null>(null)
  const [trace, setTrace] = useState<TraceEntry[]>([])
  const [attachments, setAttachments] = useState<
    { type: string; data: string; mime_type: string; name: string }[]
  >([])
  const [error, setError] = useState('')
  const [showThinking, setShowThinking] = useLocalState('ciws.showThinking', false)

  const abort = useRef<AbortController | null>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const pinnedToBottom = useRef(true)
  const fileInput = useRef<HTMLInputElement>(null)

  const agents = useAsync(() => api.get<{ agents: AgentRow[] }>('/agents'), [])
  const models = useAsync(() => api.get<{ models: ModelRow[] }>('/models'), [])
  const conversations = useAsync(
    () => api.get<{ conversations: ConversationRow[] }>('/conversations', { project_id: projectId }),
    [projectId],
  )
  const conversation = useAsync(
    async () =>
      conversationId
        ? api.get<ConversationRow & { messages: MessageRow[] }>(`/conversations/${conversationId}`)
        : null,
    [conversationId],
  )

  const messages = conversation.data?.messages ?? []
  const streaming = draft !== null

  // Only auto-scroll when the user is already at the bottom; yanking them back
  // mid-read is the most annoying thing a streaming UI can do.
  const onScroll = useCallback(() => {
    const element = scroller.current
    if (!element) return
    pinnedToBottom.current =
      element.scrollHeight - element.scrollTop - element.clientHeight < 90
  }, [])

  useEffect(() => {
    if (pinnedToBottom.current) {
      scroller.current?.scrollTo({ top: scroller.current.scrollHeight })
    }
  }, [messages.length, draft?.content])

  // Live trace from the event bus.
  useEffect(() => {
    const push = (entry: Omit<TraceEntry, 'id' | 'at'>) =>
      setTrace((current) =>
        [...current, { ...entry, id: `${Date.now()}-${Math.random()}`, at: Date.now() }].slice(-160),
      )

    const unsubscribers = [
      events.on('run.step', (e) =>
        push({
          kind: 'step',
          label: `step ${e.data.step}/${e.data.max_steps}`,
          detail: '',
          ok: true,
        }),
      ),
      events.on('tool.start', (e) =>
        push({
          kind: 'tool',
          label: e.data.tool,
          detail: summarizeArgs(e.data.arguments),
          ok: true,
        }),
      ),
      events.on('tool.end', (e) =>
        setTrace((current) => {
          const index = [...current].reverse().findIndex(
            (t) => t.kind === 'tool' && t.label === e.data.tool && t.ms === undefined,
          )
          if (index === -1) return current
          const at = current.length - 1 - index
          const next = [...current]
          next[at] = {
            ...next[at],
            ok: e.data.ok,
            ms: e.data.duration_ms,
            detail: next[at].detail || String(e.data.preview ?? '').slice(0, 120),
          }
          return next
        }),
      ),
      events.on('tool.error', (e) =>
        push({ kind: 'tool', label: e.data.tool, detail: String(e.data.preview ?? ''), ok: false }),
      ),
      events.on('memory.write', (e) =>
        push({ kind: 'memory', label: 'remembered', detail: String(e.data.content ?? ''), ok: true }),
      ),
      events.on('graph.entity', (e) =>
        push({
          kind: 'graph',
          label: e.data.created ? 'entity created' : 'entity seen',
          detail: `${e.data.name} (${e.data.type})`,
          ok: true,
        }),
      ),
      events.on('media.done', (e) =>
        push({ kind: 'media', label: 'render complete', detail: String(e.data.model ?? ''), ok: true }),
      ),
    ]
    return () => unsubscribers.forEach((off) => off())
  }, [])

  const send = useCallback(async () => {
    const text = input.trim()
    if ((!text && attachments.length === 0) || streaming) return

    setInput('')
    setError('')
    setTrace([])
    setDraft({ content: '', thinking: '', runId: '', model: '' })
    pinnedToBottom.current = true

    const controller = new AbortController()
    abort.current = controller
    const outgoing = [...attachments]
    setAttachments([])

    try {
      await streamChat(
        {
          message: text,
          conversation_id: conversationId,
          project_id: projectId,
          agent,
          model,
          attachments: outgoing,
        },
        (frame) => {
          switch (frame.type) {
            case 'conversation':
              if (frame.conversation_id !== conversationId) {
                setConversationId(frame.conversation_id)
                conversations.reload()
              }
              break
            case 'start':
              setDraft((d) => (d ? { ...d, runId: frame.run_id, model: frame.model } : d))
              break
            case 'delta':
              setDraft((d) => (d ? { ...d, content: d.content + frame.text } : d))
              break
            case 'thinking':
              setDraft((d) => (d ? { ...d, thinking: d.thinking + frame.text } : d))
              break
            case 'error':
              setError(frame.error)
              break
            case 'done':
              break
          }
        },
        controller.signal,
      )
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError((err as Error).message)
        toast((err as Error).message, 'error')
      }
    } finally {
      abort.current = null
      setDraft(null)
      conversation.reload()
      conversations.reload()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input, attachments, streaming, conversationId, projectId, agent, model])

  const stop = useCallback(() => {
    const runId = draft?.runId
    abort.current?.abort()
    if (runId) api.post(`/chat/cancel/${runId}`).catch(() => undefined)
  }, [draft?.runId])

  const attach = useCallback(
    (files: FileList | null) => {
      if (!files) return
      for (const file of Array.from(files).slice(0, 4)) {
        if (file.size > 8 * 1024 * 1024) {
          toast(`${file.name} is over 8MB`, 'error')
          continue
        }
        const reader = new FileReader()
        reader.onload = () => {
          const result = String(reader.result)
          setAttachments((current) => [
            ...current,
            {
              type: file.type.startsWith('image/') ? 'image' : 'file',
              data: result.split(',')[1] ?? '',
              mime_type: file.type || 'application/octet-stream',
              name: file.name,
            },
          ])
        }
        reader.readAsDataURL(file)
      }
    },
    [toast],
  )

  const modelOptions = useMemo(() => {
    const rows = (models.data?.models ?? []).filter((m) => m.configured && !m.capabilities.includes('embedding'))
    rows.sort((a, b) => a.id.localeCompare(b.id))
    return rows
  }, [models.data])

  const activeAgent = agents.data?.agents.find((a) => a.slug === agent)

  return (
    <div className="h-full grid grid-cols-[220px_1fr_300px] gap-2 min-h-0 max-[1500px]:grid-cols-[200px_1fr] max-[1100px]:grid-cols-1">
      {/* Conversations */}
      <Panel
        title="Sessions"
        className="max-[1100px]:hidden"
        actions={
          <button
            className="btn btn-ghost !p-1"
            title="New session"
            onClick={() => {
              setConversationId(null)
              setTrace([])
              setError('')
            }}
          >
            <MessageSquarePlus size={13} />
          </button>
        }
      >
        <div className="h-full overflow-y-auto scroll">
          {conversations.loading && !conversations.data ? (
            <div className="p-3 text-faint text-xs flex items-center gap-2">
              <Spinner /> loading
            </div>
          ) : (conversations.data?.conversations ?? []).length === 0 ? (
            <div className="p-3 text-faint text-xs leading-relaxed">
              No sessions yet. Ask something below to start one.
            </div>
          ) : (
            (conversations.data?.conversations ?? []).map((c) => (
              <button
                key={c.id}
                onClick={() => {
                  setConversationId(c.id)
                  setTrace([])
                }}
                className={classNames(
                  'group w-full text-left px-3 py-2 border-b border-line/50 transition-colors',
                  c.id === conversationId ? 'bg-cyan/10 border-l-2 border-l-cyan' : 'hover:bg-raised',
                )}
              >
                <div className="flex items-start gap-1.5">
                  {c.pinned && <Pin size={9} className="text-amber mt-1 shrink-0" />}
                  <span
                    className={classNames(
                      'text-xs leading-snug line-clamp-2 flex-1',
                      c.id === conversationId ? 'text-cyan' : 'text-dim',
                    )}
                  >
                    {c.title}
                  </span>
                  <span
                    className="opacity-0 group-hover:opacity-100 text-faint hover:text-rose transition-opacity shrink-0"
                    onClick={async (e) => {
                      e.stopPropagation()
                      await api.del(`/conversations/${c.id}`)
                      if (c.id === conversationId) setConversationId(null)
                      conversations.reload()
                    }}
                  >
                    <Trash2 size={10} />
                  </span>
                </div>
                <div className="text-2xs text-faint mt-0.5 flex items-center gap-2">
                  <span>{ago(c.updated_at)}</span>
                  {!!c.message_count && <span>{c.message_count} msg</span>}
                </div>
              </button>
            ))
          )}
        </div>
      </Panel>

      {/* Conversation */}
      <Panel
        title={conversation.data?.title ?? 'Command'}
        subtitle={activeAgent?.description}
        bodyClass="flex flex-col"
        actions={
          <>
            <button
              className={classNames('btn btn-ghost !px-1.5 !py-1', showThinking && 'text-violet')}
              title="Show reasoning"
              onClick={() => setShowThinking(!showThinking)}
            >
              <Brain size={12} />
            </button>
            <select
              value={agent}
              onChange={(e) => setAgent(e.target.value)}
              className="bg-void border border-line rounded px-1.5 py-1 text-2xs text-dim outline-none focus:border-cyan/50 cursor-pointer"
            >
              {(agents.data?.agents ?? []).map((a) => (
                <option key={a.slug} value={a.slug}>
                  {a.name}
                </option>
              ))}
            </select>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="bg-void border border-line rounded px-1.5 py-1 text-2xs text-dim outline-none focus:border-cyan/50 cursor-pointer max-w-[190px]"
            >
              <option value="">agent default</option>
              {modelOptions.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.display_name || m.name}
                  {m.local ? ' (local)' : ''}
                </option>
              ))}
            </select>
          </>
        }
      >
        <div ref={scroller} onScroll={onScroll} className="flex-1 overflow-y-auto scroll min-h-0 px-4 py-3">
          {messages.length === 0 && !draft ? (
            <Welcome
              agent={activeAgent}
              configured={modelOptions.length > 0}
              onPick={(text) => setInput(text)}
            />
          ) : (
            <div className="max-w-3xl mx-auto space-y-5">
              {messages.map((m) => (
                <Message key={m.id} message={m} showThinking={showThinking} />
              ))}
              {draft && (
                <div className="animate-fade-in">
                  {showThinking && draft.thinking && (
                    <ThinkingBlock text={draft.thinking} live />
                  )}
                  <div className="flex items-center gap-2 mb-1.5">
                    <Cpu size={11} className="text-cyan" />
                    <span className="label text-cyan">{draft.model || 'thinking'}</span>
                  </div>
                  {draft.content ? (
                    <div className="caret">
                      <Markdown text={draft.content} />
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-faint text-xs py-1">
                      <Spinner /> working
                    </div>
                  )}
                </div>
              )}
              {error && (
                <div className="max-w-3xl">
                  <ErrorLine error={error} />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="shrink-0 border-t border-line p-2.5">
          {attachments.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-2">
              {attachments.map((a, i) => (
                <span key={i} className="chip">
                  {a.type === 'image' ? <ImagePlus size={9} /> : <Paperclip size={9} />}
                  <span className="max-w-[130px] truncate">{a.name}</span>
                  <button
                    onClick={() => setAttachments((c) => c.filter((_, index) => index !== i))}
                    className="hover:text-rose"
                  >
                    <X size={9} />
                  </button>
                </span>
              ))}
            </div>
          )}
          <div className="flex items-end gap-2">
            <input
              ref={fileInput}
              type="file"
              multiple
              className="hidden"
              onChange={(e) => {
                attach(e.target.files)
                e.target.value = ''
              }}
            />
            <button
              className="btn btn-ghost !p-2 shrink-0"
              onClick={() => fileInput.current?.click()}
              title="Attach"
            >
              <Paperclip size={14} />
            </button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  void send()
                }
              }}
              onPaste={(e) => {
                const files = Array.from(e.clipboardData.files)
                if (files.length) {
                  e.preventDefault()
                  attach(e.clipboardData.files)
                }
              }}
              rows={1}
              placeholder={streaming ? 'Working...' : 'Ask, or give an instruction.  Enter to send, Shift+Enter for a newline.'}
              disabled={streaming}
              className="field resize-none max-h-40 min-h-[36px] py-2 leading-relaxed disabled:opacity-60"
              style={{ height: 'auto' }}
              onInput={(e) => {
                const el = e.currentTarget
                el.style.height = 'auto'
                el.style.height = `${Math.min(el.scrollHeight, 160)}px`
              }}
            />
            {streaming ? (
              <button className="btn btn-danger !p-2 shrink-0 border-rose/40 text-rose" onClick={stop} title="Stop">
                <CircleStop size={14} />
              </button>
            ) : (
              <button
                className="btn btn-primary !p-2 shrink-0"
                onClick={() => void send()}
                disabled={!input.trim() && attachments.length === 0}
                title="Send"
              >
                <Send size={14} />
              </button>
            )}
          </div>
        </div>
      </Panel>

      {/* Trace */}
      <Panel
        title="Trace"
        subtitle={streaming ? 'live' : trace.length ? `${trace.length} events` : ''}
        className="max-[1500px]:hidden"
        actions={
          trace.length > 0 ? (
            <button className="btn btn-ghost !p-1" onClick={() => setTrace([])} title="Clear">
              <Trash2 size={11} />
            </button>
          ) : null
        }
      >
        <div className="h-full overflow-y-auto scroll">
          {trace.length === 0 ? (
            <Empty
              icon={Terminal}
              title="No activity"
              hint="Tool calls, memory writes and graph updates appear here as the agent works."
            />
          ) : (
            <div className="divide-y divide-line/50">
              {trace.map((entry) => (
                <TraceRow key={entry.id} entry={entry} />
              ))}
            </div>
          )}
        </div>
      </Panel>
    </div>
  )
}

function TraceRow({ entry }: { entry: TraceEntry }) {
  const icons = {
    step: ChevronRight,
    tool: Wrench,
    memory: Brain,
    graph: Bot,
    media: ImagePlus,
    notice: Terminal,
  }
  const Icon = icons[entry.kind]
  return (
    <div className="px-2.5 py-1.5 hover:bg-raised/50 transition-colors">
      <div className="flex items-center gap-1.5">
        <Icon size={10} className={entry.ok ? 'text-faint shrink-0' : 'text-rose shrink-0'} />
        <span className={classNames('text-2xs font-medium truncate flex-1', entry.ok ? 'text-dim' : 'text-rose')}>
          {entry.label}
        </span>
        {entry.ms !== undefined && (
          <span className="text-2xs text-faint tabular-nums">{duration(entry.ms)}</span>
        )}
      </div>
      {entry.detail && (
        <div className="text-2xs text-faint mt-0.5 pl-4 leading-snug line-clamp-2 break-words">
          {entry.detail}
        </div>
      )}
    </div>
  )
}

function ThinkingBlock({ text, live = false }: { text: string; live?: boolean }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mb-2 border-l-2 border-violet/40 pl-3">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-2xs text-violet/80 hover:text-violet"
      >
        <Brain size={10} />
        <span className="uppercase tracking-wider">
          reasoning {live && <span className="animate-pulse-slow">·</span>}
        </span>
        <ChevronRight size={9} className={classNames('transition-transform', open && 'rotate-90')} />
      </button>
      {open && (
        <div className="text-xs text-faint mt-1 whitespace-pre-wrap leading-relaxed max-h-72 overflow-y-auto scroll">
          {text}
        </div>
      )}
    </div>
  )
}

function Message({ message, showThinking }: { message: MessageRow; showThinking: boolean }) {
  if (message.role === 'user') {
    const images = (message.parts ?? []).filter((p: any) => p?.type === 'image')
    return (
      <div className="flex justify-end animate-fade-in">
        <div className="max-w-[85%]">
          {images.length > 0 && (
            <div className="flex flex-wrap gap-1.5 justify-end mb-1.5">
              {images.map((p: any, i: number) => (
                <img
                  key={i}
                  src={p.url || `data:${p.mime_type};base64,${p.data}`}
                  alt=""
                  className="max-h-40 rounded border border-line"
                />
              ))}
            </div>
          )}
          {message.content && (
            <div className="bg-cyan/10 border border-cyan/25 rounded px-3 py-2 text-base text-ink whitespace-pre-wrap break-words">
              {message.content}
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="animate-fade-in">
      {showThinking && message.thinking && <ThinkingBlock text={message.thinking} />}
      <div className="flex items-center gap-2 mb-1.5">
        <Cpu size={11} className="text-faint" />
        <span className="label">{message.model || 'assistant'}</span>
        {!!message.output_tokens && (
          <span className="text-2xs text-faint tabular-nums">
            {compact(message.input_tokens)}→{compact(message.output_tokens)} tok
          </span>
        )}
        {message.cost_usd > 0 && (
          <span className="text-2xs text-faint tabular-nums">{usd(message.cost_usd)}</span>
        )}
      </div>
      {message.error ? (
        <ErrorLine error={message.error} />
      ) : (
        <Markdown text={message.content} />
      )}
    </div>
  )
}

function Welcome({
  agent,
  configured,
  onPick,
}: {
  agent?: AgentRow
  configured: boolean
  onPick: (text: string) => void
}) {
  const prompts = [
    'What do you remember about me and what I am working on?',
    'Ingest my notes folder and build the entity graph from it.',
    'Research how local-first software handles sync, and write me a sourced brief.',
    'Show me the most connected entities in my workspace and why they matter.',
  ]
  return (
    <div className="h-full grid place-items-center">
      <div className="max-w-lg text-center">
        <div className="inline-flex items-center justify-center w-11 h-11 rounded-full border border-cyan/30 bg-cyan/5 mb-4">
          <Bot size={18} className="text-cyan" />
        </div>
        <h2 className="text-md text-ink font-semibold tracking-tight">
          {agent?.name ?? 'Analyst'} standing by
        </h2>
        <p className="text-xs text-faint mt-1.5 leading-relaxed">
          {agent?.description ?? 'Your workspace, your models, your machine.'}
        </p>

        {!configured && (
          <div className="mt-4 text-xs text-amber border border-amber/30 bg-amber/5 rounded px-3 py-2 text-left leading-relaxed">
            No model provider is configured yet. Add an API key under{' '}
            <span className="text-ink">Systems → Credentials</span>, or run Ollama locally for a
            fully offline workspace.
          </div>
        )}

        <div className="mt-5 space-y-1.5 text-left">
          {prompts.map((prompt) => (
            <button
              key={prompt}
              onClick={() => onPick(prompt)}
              className="w-full text-left px-3 py-2 rounded border border-line bg-hull text-xs text-dim hover:border-cyan/40 hover:text-cyan transition-colors"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

function summarizeArgs(args: unknown): string {
  if (!args || typeof args !== 'object') return ''
  const entries = Object.entries(args as Record<string, unknown>)
  if (!entries.length) return ''
  return entries
    .map(([key, value]) => {
      const rendered =
        typeof value === 'string'
          ? value
          : Array.isArray(value)
            ? `[${value.length}]`
            : JSON.stringify(value)
      return `${key}=${String(rendered).slice(0, 70)}`
    })
    .join('  ')
    .slice(0, 160)
}
