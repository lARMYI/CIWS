/**
 * Memory and Corpus.
 *
 * Both are search-first. A memory store you can only browse chronologically is
 * a diary; the useful question is always "what do you know about X", so the
 * search box runs the same hybrid recall the agent uses, and shows why each
 * result matched.
 */

import {
  Archive,
  Brain,
  FileText,
  Link2,
  Loader2,
  Pin,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import { useCallback, useRef, useState } from 'react'

import {
  Badge,
  Empty,
  ErrorLine,
  Field,
  Modal,
  Panel,
  Spinner,
  Stat,
  StatusDot,
  ago,
  bytes,
  classNames,
  compact,
  useAsync,
  useSearch,
  useToast,
} from '../components/ui'
import { api, events, type DocumentRow, type MemoryRow } from '../lib/api'
import { useEffect } from 'react'

const KINDS = ['fact', 'preference', 'identity', 'decision', 'procedure', 'event', 'insight', 'task']

const KIND_TONE: Record<string, 'cyan' | 'amber' | 'jade' | 'violet' | 'rose' | 'default'> = {
  identity: 'violet',
  preference: 'cyan',
  decision: 'amber',
  procedure: 'jade',
  insight: 'violet',
  event: 'default',
  task: 'amber',
  fact: 'default',
}

export function Knowledge({ projectId }: { projectId: string | null }) {
  const [tab, setTab] = useState<'memory' | 'corpus'>('memory')
  return (
    <div className="h-full flex flex-col gap-2 min-h-0">
      <div className="flex gap-1 shrink-0">
        {(['memory', 'corpus'] as const).map((key) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={classNames(
              'btn !text-2xs uppercase tracking-widest',
              tab === key && 'btn-primary',
            )}
          >
            {key === 'memory' ? <Brain size={11} /> : <FileText size={11} />}
            {key}
          </button>
        ))}
      </div>
      <div className="flex-1 min-h-0">
        {tab === 'memory' ? <MemoryView projectId={projectId} /> : <CorpusView projectId={projectId} />}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

function MemoryView({ projectId }: { projectId: string | null }) {
  const toast = useToast()
  const [query, setQuery, debounced] = useSearch('')
  const [kind, setKind] = useState('')
  const [showArchived, setShowArchived] = useState(false)
  const [adding, setAdding] = useState(false)
  const [busy, setBusy] = useState(false)

  // A search runs semantic recall; an empty box lists chronologically. They are
  // different endpoints because they answer different questions.
  const semantic = useAsync(
    async () =>
      debounced.trim()
        ? api.post<{ hits: MemoryRow[] }>('/memory/search', {
            query: debounced,
            limit: 40,
            project_id: projectId,
          })
        : null,
    [debounced, projectId],
  )
  const listing = useAsync(
    () =>
      api.get<{ memories: MemoryRow[]; stats: any }>('/memory', {
        project_id: projectId,
        kind,
        archived: showArchived,
        limit: 200,
      }),
    [projectId, kind, showArchived],
  )

  useEffect(() => events.on('memory.write', () => listing.reload()), [listing.reload])

  const rows = debounced.trim() ? (semantic.data?.hits ?? []) : (listing.data?.memories ?? [])
  const stats = listing.data?.stats
  const loading = debounced.trim() ? semantic.loading : listing.loading

  const consolidate = useCallback(async () => {
    setBusy(true)
    try {
      const result = await api.post<any>('/memory/consolidate', { project_id: projectId })
      toast(
        `Consolidated: ${result.merged} merged, ${result.insights} insights, ${result.archived} archived`,
        'success',
      )
      listing.reload()
    } catch (err) {
      toast((err as Error).message, 'error')
    } finally {
      setBusy(false)
    }
  }, [projectId, toast, listing])

  return (
    <div className="h-full grid grid-rows-[auto_1fr] gap-2 min-h-0">
      <div className="grid grid-cols-4 gap-2 max-[900px]:grid-cols-2">
        <Stat label="Memories" value={stats?.total ?? '--'} tone="cyan" />
        <Stat label="Pinned" value={stats?.pinned ?? '--'} tone="amber" />
        <Stat label="Avg importance" value={stats?.avg_importance ?? '--'} />
        <Stat label="Recalls" value={compact(stats?.total_accesses ?? 0)} hint="times used" />
      </div>

      <Panel
        title="Memory"
        subtitle={debounced ? 'ranked by relevance' : 'most recent first'}
        bodyClass="flex flex-col"
        actions={
          <>
            <select
              value={kind}
              onChange={(e) => setKind(e.target.value)}
              className="bg-void border border-line rounded px-1.5 py-1 text-2xs text-dim outline-none focus:border-cyan/50 cursor-pointer"
            >
              <option value="">all kinds</option>
              {KINDS.map((k) => (
                <option key={k} value={k}>
                  {k} {stats?.by_kind?.[k] ? `(${stats.by_kind[k]})` : ''}
                </option>
              ))}
            </select>
            <button
              className={classNames('btn btn-ghost !p-1', showArchived && 'text-amber')}
              onClick={() => setShowArchived(!showArchived)}
              title="Show archived"
            >
              <Archive size={12} />
            </button>
            <button className="btn btn-ghost !p-1" onClick={consolidate} disabled={busy} title="Consolidate">
              {busy ? <Spinner size={12} /> : <Sparkles size={12} />}
            </button>
            <button className="btn btn-ghost !p-1" onClick={() => setAdding(true)} title="Add memory">
              <Plus size={12} />
            </button>
          </>
        }
      >
        <div className="p-2 border-b border-line shrink-0">
          <div className="relative">
            <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-faint" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What do you know about..."
              className="field !pl-7 !py-1.5"
            />
            {query && (
              <button
                onClick={() => setQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-faint hover:text-ink"
              >
                <X size={12} />
              </button>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto scroll min-h-0">
          {loading && rows.length === 0 ? (
            <div className="p-4 text-faint text-xs flex items-center gap-2">
              <Spinner /> loading
            </div>
          ) : rows.length === 0 ? (
            <Empty
              icon={Brain}
              title={debounced ? 'Nothing matched' : 'No memories yet'}
              hint={
                debounced
                  ? 'Try different words. Recall is semantic, but the built-in embeddings without an API key are closer to keyword matching.'
                  : 'Memories accumulate as you work. Agents record durable facts, preferences and decisions automatically.'
              }
              action={
                !debounced && (
                  <button className="btn btn-primary" onClick={() => setAdding(true)}>
                    <Plus size={11} /> Add one manually
                  </button>
                )
              }
            />
          ) : (
            <div className="divide-y divide-line/50">
              {rows.map((memory) => (
                <MemoryRowView key={memory.id} memory={memory} onChange={() => listing.reload()} />
              ))}
            </div>
          )}
        </div>
      </Panel>

      <AddMemory
        open={adding}
        onClose={() => setAdding(false)}
        projectId={projectId}
        onAdded={() => listing.reload()}
      />
    </div>
  )
}

function MemoryRowView({ memory, onChange }: { memory: MemoryRow; onChange: () => void }) {
  const [busy, setBusy] = useState(false)

  const patch = async (body: Record<string, unknown>) => {
    setBusy(true)
    try {
      await api.patch(`/memory/${memory.id}`, body)
      onChange()
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="group px-3 py-2.5 hover:bg-raised/40 transition-colors">
      <div className="flex items-start gap-2">
        <div className="flex-1 min-w-0">
          <p
            className={classNames(
              'text-sm leading-relaxed break-words',
              memory.archived ? 'text-faint line-through' : 'text-ink',
            )}
          >
            {memory.content}
          </p>
          <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
            <Badge tone={KIND_TONE[memory.kind] ?? 'default'}>{memory.kind}</Badge>
            <span className="text-2xs text-faint tabular-nums" title="importance">
              w{memory.importance.toFixed(2)}
            </span>
            {memory.access_count > 0 && (
              <span className="text-2xs text-faint">used {memory.access_count}x</span>
            )}
            <span className="text-2xs text-faint">{ago(memory.created_at)}</span>
            {memory.tags?.map((tag) => (
              <Badge key={tag}>#{tag}</Badge>
            ))}
            {memory.score !== undefined && (
              <span className="text-2xs text-cyan tabular-nums" title={memory.reasons?.join(', ')}>
                {memory.score.toFixed(3)}
              </span>
            )}
            {memory.reasons?.map((reason) => (
              <span key={reason} className="text-2xs text-faint">
                · {reason}
              </span>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
          {busy && <Loader2 size={11} className="animate-spin text-faint" />}
          <button
            className={classNames('btn btn-ghost !p-1', memory.pinned && 'text-amber opacity-100')}
            onClick={() => patch({ pinned: !memory.pinned })}
            title={memory.pinned ? 'Unpin' : 'Pin'}
          >
            <Pin size={11} />
          </button>
          <button
            className="btn btn-ghost !p-1"
            onClick={() => patch({ archived: !memory.archived })}
            title={memory.archived ? 'Restore' : 'Archive'}
          >
            <Archive size={11} />
          </button>
          <button
            className="btn btn-ghost !p-1 hover:text-rose"
            onClick={async () => {
              await api.del(`/memory/${memory.id}`)
              onChange()
            }}
            title="Delete"
          >
            <Trash2 size={11} />
          </button>
        </div>
      </div>
    </div>
  )
}

function AddMemory({
  open,
  onClose,
  projectId,
  onAdded,
}: {
  open: boolean
  onClose: () => void
  projectId: string | null
  onAdded: () => void
}) {
  const toast = useToast()
  const [content, setContent] = useState('')
  const [kind, setKind] = useState('fact')
  const [importance, setImportance] = useState(0.6)
  const [error, setError] = useState('')

  const submit = async () => {
    if (!content.trim()) return
    try {
      await api.post('/memory', { content, kind, importance, project_id: projectId })
      toast('Remembered', 'success')
      setContent('')
      onAdded()
      onClose()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="New memory">
      <div className="space-y-3">
        <Field
          label="Content"
          hint="Write it so it still makes sense months from now with no other context."
        >
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            rows={3}
            className="field resize-none"
            placeholder="I prefer concise answers with the conclusion first."
            autoFocus
          />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Kind">
            <select value={kind} onChange={(e) => setKind(e.target.value)} className="field">
              {KINDS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </Field>
          <Field label={`Importance  ${importance.toFixed(2)}`}>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={importance}
              onChange={(e) => setImportance(Number(e.target.value))}
              className="w-full accent-cyan mt-2"
            />
          </Field>
        </div>
        <ErrorLine error={error} />
        <div className="flex justify-end gap-2">
          <button className="btn" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={submit} disabled={!content.trim()}>
            Remember
          </button>
        </div>
      </div>
    </Modal>
  )
}

// ---------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------

function CorpusView({ projectId }: { projectId: string | null }) {
  const toast = useToast()
  const [query, setQuery, debounced] = useSearch('')
  const [selected, setSelected] = useState<DocumentRow | null>(null)
  const [ingesting, setIngesting] = useState(false)
  const [progress, setProgress] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)

  const listing = useAsync(
    () => api.get<{ documents: DocumentRow[]; stats: any; supported: string[] }>('/corpus', { project_id: projectId }),
    [projectId],
  )
  const passages = useAsync(
    async () =>
      debounced.trim()
        ? api.post<{ results: any[] }>('/corpus/search', {
            query: debounced,
            limit: 20,
            project_id: projectId,
          })
        : null,
    [debounced, projectId],
  )

  useEffect(() => {
    const off = [
      events.on('ingest.progress', (e) =>
        setProgress(
          e.data.percent !== undefined
            ? `${e.data.done}/${e.data.total} files`
            : String(e.data.status ?? ''),
        ),
      ),
      events.on('ingest.done', () => {
        setProgress('')
        listing.reload()
      }),
    ]
    return () => off.forEach((fn) => fn())
  }, [listing.reload])

  const upload = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return
      setIngesting(true)
      let ok = 0
      for (const file of Array.from(files)) {
        try {
          await api.upload('/corpus/upload', file, projectId ? { project_id: projectId } : undefined)
          ok++
        } catch (err) {
          toast(`${file.name}: ${(err as Error).message}`, 'error')
        }
      }
      setIngesting(false)
      setProgress('')
      if (ok) toast(`Ingested ${ok} document${ok === 1 ? '' : 's'}`, 'success')
      listing.reload()
    },
    [projectId, toast, listing],
  )

  const stats = listing.data?.stats
  const documents = listing.data?.documents ?? []

  return (
    <div className="h-full grid grid-rows-[auto_1fr] gap-2 min-h-0">
      <div className="grid grid-cols-4 gap-2 max-[900px]:grid-cols-2">
        <Stat label="Documents" value={stats?.documents ?? '--'} tone="cyan" />
        <Stat label="Chunks" value={compact(stats?.chunks ?? 0)} hint="searchable passages" />
        <Stat label="On disk" value={bytes(stats?.bytes ?? 0)} />
        <Stat
          label="Failed"
          value={stats?.by_status?.failed ?? 0}
          tone={stats?.by_status?.failed ? 'rose' : 'default'}
        />
      </div>

      <div className="grid grid-cols-[1fr_320px] gap-2 min-h-0 max-[1100px]:grid-cols-1">
        <Panel
          title="Corpus"
          subtitle={debounced ? `passages matching "${debounced}"` : `${documents.length} documents`}
          bodyClass="flex flex-col"
          actions={
            <>
              {(ingesting || progress) && (
                <span className="text-2xs text-cyan flex items-center gap-1">
                  <Spinner size={10} /> {progress || 'ingesting'}
                </span>
              )}
              <button className="btn btn-ghost !p-1" onClick={() => fileInput.current?.click()} title="Upload">
                <Upload size={12} />
              </button>
              <IngestUrl projectId={projectId} onDone={() => listing.reload()} />
              <button className="btn btn-ghost !p-1" onClick={listing.reload} title="Refresh">
                <RefreshCw size={12} />
              </button>
            </>
          }
        >
          <input
            ref={fileInput}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              void upload(e.target.files)
              e.target.value = ''
            }}
          />
          <div className="p-2 border-b border-line shrink-0">
            <div className="relative">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-faint" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search inside your documents"
                className="field !pl-7 !py-1.5"
              />
            </div>
          </div>

          <div
            className="flex-1 overflow-y-auto scroll min-h-0"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              void upload(e.dataTransfer.files)
            }}
          >
            {debounced.trim() ? (
              passages.loading ? (
                <div className="p-4 text-faint text-xs flex items-center gap-2">
                  <Spinner /> searching
                </div>
              ) : (passages.data?.results.length ?? 0) === 0 ? (
                <Empty icon={Search} title="No passages matched" hint="Try different words." />
              ) : (
                <div className="divide-y divide-line/50">
                  {passages.data!.results.map((hit) => (
                    <div key={hit.chunk_id} className="px-3 py-2.5 hover:bg-raised/40">
                      <div className="flex items-center gap-2 mb-1">
                        <FileText size={10} className="text-faint shrink-0" />
                        <span className="text-2xs text-cyan truncate">{hit.document_title}</span>
                        {hit.heading && <span className="text-2xs text-faint truncate">› {hit.heading}</span>}
                        {!!hit.page && <span className="text-2xs text-faint">p.{hit.page}</span>}
                        <span className="text-2xs text-faint tabular-nums ml-auto">
                          {hit.score.toFixed(3)}
                        </span>
                      </div>
                      <p className="text-xs text-dim leading-relaxed line-clamp-4 break-words">
                        {hit.text}
                      </p>
                    </div>
                  ))}
                </div>
              )
            ) : documents.length === 0 ? (
              <Empty
                icon={FileText}
                title="No documents yet"
                hint="Drop files here, or use the upload button. PDF, Word, Excel, PowerPoint, Markdown, CSV, HTML and source code are all understood."
              />
            ) : (
              <div className="divide-y divide-line/50">
                {documents.map((doc) => (
                  <button
                    key={doc.id}
                    onClick={() => setSelected(doc)}
                    className={classNames(
                      'w-full text-left px-3 py-2 hover:bg-raised/50 transition-colors',
                      selected?.id === doc.id && 'bg-cyan/5',
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <StatusDot status={doc.status} />
                      <span className="text-sm text-ink truncate flex-1">{doc.title}</span>
                      <span className="text-2xs text-faint tabular-nums shrink-0">
                        {doc.chunk_count} chunks
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-0.5 pl-3.5">
                      <span className="text-2xs text-faint">{doc.source_type}</span>
                      <span className="text-2xs text-faint">{bytes(doc.size_bytes)}</span>
                      <span className="text-2xs text-faint">{ago(doc.created_at)}</span>
                      {doc.error && <span className="text-2xs text-rose truncate">{doc.error}</span>}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </Panel>

        <Panel
          title={selected ? 'Document' : 'Details'}
          className="max-[1100px]:hidden"
          bodyClass="overflow-y-auto scroll"
          actions={
            selected && (
              <button className="btn btn-ghost !p-1" onClick={() => setSelected(null)}>
                <X size={12} />
              </button>
            )
          }
        >
          {selected ? (
            <DocumentDetail
              document={selected}
              onChange={() => {
                listing.reload()
                setSelected(null)
              }}
            />
          ) : (
            <div className="p-3 text-xs text-faint leading-relaxed">
              Select a document to see its summary, chunks and provenance.
              <div className="mt-3">
                <div className="label mb-1">Supported formats</div>
                <div className="flex flex-wrap gap-1">
                  {(listing.data?.supported ?? []).slice(0, 26).map((ext) => (
                    <Badge key={ext}>{ext}</Badge>
                  ))}
                </div>
              </div>
            </div>
          )}
        </Panel>
      </div>
    </div>
  )
}

function DocumentDetail({ document: doc, onChange }: { document: DocumentRow; onChange: () => void }) {
  const toast = useToast()
  const [summary, setSummary] = useState(doc.summary)
  const [busy, setBusy] = useState(false)
  const chunks = useAsync(
    () => api.get<{ chunks: any[] }>(`/corpus/${doc.id}`, { chunks: true }),
    [doc.id],
  )

  return (
    <div className="p-3 space-y-3">
      <div>
        <div className="text-sm text-ink font-semibold break-words">{doc.title}</div>
        <div className="flex flex-wrap gap-1 mt-1.5">
          <Badge tone={doc.status === 'ready' ? 'jade' : doc.status === 'failed' ? 'rose' : 'amber'}>
            {doc.status}
          </Badge>
          <Badge>{doc.source_type}</Badge>
          <Badge>{bytes(doc.size_bytes)}</Badge>
          {!!doc.page_count && <Badge>{doc.page_count}p</Badge>}
          <Badge>{doc.chunk_count} chunks</Badge>
        </div>
      </div>

      {doc.source_uri && (
        <div className="text-2xs text-faint break-all flex items-start gap-1.5">
          <Link2 size={10} className="shrink-0 mt-0.5" />
          {doc.source_uri}
        </div>
      )}

      {doc.error && <ErrorLine error={doc.error} />}

      <div>
        <div className="flex items-center justify-between mb-1">
          <span className="label">Summary</span>
          <button
            className="btn btn-ghost !py-0.5 !px-1.5 !text-2xs"
            disabled={busy}
            onClick={async () => {
              setBusy(true)
              try {
                const result = await api.post<{ summary: string }>(
                  `/corpus/${doc.id}/summarize?force=true`,
                )
                setSummary(result.summary)
              } catch (err) {
                toast((err as Error).message, 'error')
              } finally {
                setBusy(false)
              }
            }}
          >
            {busy ? <Spinner size={9} /> : <Sparkles size={9} />} generate
          </button>
        </div>
        <p className="text-xs text-dim leading-relaxed">
          {summary || <span className="text-faint">No summary yet.</span>}
        </p>
      </div>

      <div>
        <div className="label mb-1">Chunks</div>
        <div className="surface divide-y divide-line/60 max-h-72 overflow-y-auto scroll">
          {(chunks.data?.chunks ?? []).map((chunk) => (
            <div key={chunk.id} className="px-2 py-1.5">
              <div className="flex items-center gap-1.5">
                <span className="text-2xs text-faint tabular-nums">#{chunk.idx}</span>
                {chunk.heading && <span className="text-2xs text-cyan truncate">{chunk.heading}</span>}
                <span className="text-2xs text-faint ml-auto tabular-nums">{chunk.token_count}t</span>
              </div>
              <p className="text-2xs text-faint line-clamp-3 mt-0.5 break-words">{chunk.text}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="flex gap-1.5">
        <button
          className="btn flex-1"
          onClick={async () => {
            await api.post(`/corpus/${doc.id}/reindex`)
            toast('Re-indexed', 'success')
            onChange()
          }}
        >
          <RefreshCw size={11} /> reindex
        </button>
        <button
          className="btn btn-danger !px-2"
          onClick={async () => {
            await api.del(`/corpus/${doc.id}`)
            onChange()
          }}
        >
          <Trash2 size={11} />
        </button>
      </div>
    </div>
  )
}

function IngestUrl({ projectId, onDone }: { projectId: string | null; onDone: () => void }) {
  const toast = useToast()
  const [open, setOpen] = useState(false)
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)

  return (
    <>
      <button className="btn btn-ghost !p-1" onClick={() => setOpen(true)} title="Ingest a URL">
        <Link2 size={12} />
      </button>
      <Modal open={open} onClose={() => setOpen(false)} title="Ingest a web page">
        <div className="space-y-3">
          <Field label="URL" hint="The page is fetched, reduced to readable text, chunked and indexed.">
            <input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/article"
              className="field"
              autoFocus
            />
          </Field>
          <div className="flex justify-end gap-2">
            <button className="btn" onClick={() => setOpen(false)}>
              Cancel
            </button>
            <button
              className="btn btn-primary"
              disabled={!url.trim() || busy}
              onClick={async () => {
                setBusy(true)
                try {
                  await api.post('/corpus/ingest', { url, project_id: projectId })
                  toast('Ingested', 'success')
                  setUrl('')
                  setOpen(false)
                  onDone()
                } catch (err) {
                  toast((err as Error).message, 'error')
                } finally {
                  setBusy(false)
                }
              }}
            >
              {busy ? <Spinner size={11} /> : <Upload size={11} />} Ingest
            </button>
          </div>
        </div>
      </Modal>
    </>
  )
}
