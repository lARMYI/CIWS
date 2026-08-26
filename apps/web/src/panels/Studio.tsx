/**
 * The media studio.
 *
 * Assets appear as placeholder tiles the moment a render starts and fill in as
 * results land, driven by the event bus. For video that gap is minutes, and a
 * grid that shows you what is in flight is the difference between "working" and
 * "did I click it?".
 */

import {
  Download,
  Film,
  Heart,
  ImageIcon,
  Loader2,
  Play,
  RefreshCw,
  Sparkles,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'

import {
  Badge,
  Empty,
  ErrorLine,
  Field,
  Modal,
  Panel,
  Spinner,
  Stat,
  ago,
  bytes,
  classNames,
  useAsync,
  useLocalState,
  useToast,
  usd,
} from '../components/ui'
import { api, events, type AssetRow } from '../lib/api'

interface MediaModelRow {
  id: string
  display_name: string
  kind: string
  local: boolean
  configured: boolean
  env_hint: string
  description: string
  sizes: string[]
  supports_negative_prompt: boolean
  max_duration_s: number
}

export function Studio({ projectId }: { projectId: string | null }) {
  const toast = useToast()
  const [kind, setKind] = useLocalState<'image' | 'video'>('ciws.studio.kind', 'image')
  const [prompt, setPrompt] = useState('')
  const [negative, setNegative] = useState('')
  const [model, setModel] = useLocalState('ciws.studio.model', '')
  const [size, setSize] = useLocalState('ciws.studio.size', '1024x1024')
  const [count, setCount] = useState(1)
  const [duration, setDuration] = useState(5)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [preview, setPreview] = useState<AssetRow | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const models = useAsync(
    () => api.get<{ models: MediaModelRow[] }>('/media/models', { kind }),
    [kind],
  )
  const assets = useAsync(
    () => api.get<{ assets: AssetRow[]; stats: any }>('/media', { project_id: projectId, limit: 120 }),
    [projectId],
  )

  useEffect(() => {
    const off = [
      events.on('media.done', () => assets.reload()),
      events.on('media.error', (e) => {
        toast(String(e.data.error ?? 'Render failed'), 'error')
        assets.reload()
      }),
      events.on('media.start', () => assets.reload()),
    ]
    return () => off.forEach((fn) => fn())
  }, [assets.reload, toast])

  const usable = (models.data?.models ?? []).filter((m) => m.configured)
  const active = usable.find((m) => m.id === model) ?? usable[0]

  const generate = useCallback(async () => {
    if (!prompt.trim()) return
    setBusy(true)
    setError('')
    try {
      if (kind === 'image') {
        await api.post('/media/image' + (projectId ? `?project_id=${projectId}` : ''), {
          prompt,
          negative_prompt: negative,
          model: active?.id ?? '',
          size,
          n: count,
        })
        toast('Image ready', 'success')
      } else {
        await api.post('/media/video' + (projectId ? `?project_id=${projectId}` : ''), {
          prompt,
          negative_prompt: negative,
          model: active?.id ?? '',
          duration_s: duration,
        })
        toast('Rendering — it will appear when finished', 'success')
      }
      assets.reload()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }, [prompt, negative, kind, active, size, count, duration, projectId, toast, assets])

  const stats = assets.data?.stats
  const rows = (assets.data?.assets ?? []).filter((a) => a.kind === kind || kind === 'image')

  return (
    <div className="h-full grid grid-cols-[300px_1fr] gap-2 min-h-0 max-[980px]:grid-cols-1">
      <Panel title="Compose" bodyClass="overflow-y-auto scroll">
        <div className="p-3 space-y-3">
          <div className="flex gap-1">
            {(['image', 'video'] as const).map((k) => (
              <button
                key={k}
                onClick={() => {
                  setKind(k)
                  setModel('')
                }}
                className={classNames('btn flex-1 !text-2xs uppercase tracking-wider', kind === k && 'btn-primary')}
              >
                {k === 'image' ? <ImageIcon size={11} /> : <Film size={11} />}
                {k}
              </button>
            ))}
          </div>

          <Field
            label="Prompt"
            hint="Subject, composition, lighting, palette, medium. Concrete direction beats adjectives."
          >
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              className="field resize-none"
              placeholder="Overhead shot of a rain-slick observation deck, hard side light, muted teal and rust, 35mm"
            />
          </Field>

          {active?.supports_negative_prompt && (
            <Field label="Negative prompt">
              <input
                value={negative}
                onChange={(e) => setNegative(e.target.value)}
                className="field"
                placeholder="blurry, text, watermark"
              />
            </Field>
          )}

          <Field label="Model">
            {usable.length === 0 ? (
              <div className="text-2xs text-amber border border-amber/30 bg-amber/5 rounded px-2 py-1.5 leading-relaxed">
                No {kind} provider configured. Add a key under Systems → Credentials, or run ComfyUI
                / Automatic1111 locally for free generation.
              </div>
            ) : (
              <select
                value={active?.id ?? ''}
                onChange={(e) => setModel(e.target.value)}
                className="field"
              >
                {usable.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.display_name}
                    {m.local ? ' (local)' : ''}
                  </option>
                ))}
              </select>
            )}
          </Field>

          {active && <div className="text-2xs text-faint leading-relaxed">{active.description}</div>}

          {kind === 'image' ? (
            <div className="grid grid-cols-2 gap-3">
              <Field label="Size">
                <select value={size} onChange={(e) => setSize(e.target.value)} className="field">
                  {(active?.sizes?.length ? active.sizes : ['1024x1024']).map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label={`Count  ${count}`}>
                <input
                  type="range"
                  min={1}
                  max={4}
                  value={count}
                  onChange={(e) => setCount(Number(e.target.value))}
                  className="w-full accent-cyan mt-2"
                />
              </Field>
            </div>
          ) : (
            <Field label={`Duration  ${duration}s`}>
              <input
                type="range"
                min={2}
                max={Math.max(4, active?.max_duration_s ?? 8)}
                step={1}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                className="w-full accent-cyan mt-2"
              />
            </Field>
          )}

          <ErrorLine error={error} />

          <button
            className="btn btn-primary w-full justify-center !py-2"
            disabled={!prompt.trim() || busy || usable.length === 0}
            onClick={generate}
          >
            {busy ? <Spinner size={13} /> : <Sparkles size={13} />}
            {busy ? 'generating' : `generate ${kind}`}
          </button>

          <div className="pt-2 border-t border-line">
            <input
              ref={fileInput}
              type="file"
              accept="image/*,video/*"
              className="hidden"
              onChange={async (e) => {
                const file = e.target.files?.[0]
                e.target.value = ''
                if (!file) return
                try {
                  await api.upload('/media/upload', file, projectId ? { project_id: projectId } : undefined)
                  toast('Imported', 'success')
                  assets.reload()
                } catch (err) {
                  toast((err as Error).message, 'error')
                }
              }}
            />
            <button className="btn w-full justify-center" onClick={() => fileInput.current?.click()}>
              <Upload size={11} /> import a file
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 pt-1">
            <Stat label="Assets" value={Object.values(stats?.by_kind ?? {}).reduce((a: number, b) => a + Number(b), 0) || 0} />
            <Stat label="Spend" value={usd(stats?.cost_usd ?? 0)} tone="amber" />
          </div>
        </div>
      </Panel>

      <Panel
        title="Library"
        subtitle={`${rows.length} assets · ${bytes(stats?.bytes ?? 0)}`}
        bodyClass="overflow-y-auto scroll"
        actions={
          <button className="btn btn-ghost !p-1" onClick={assets.reload} title="Refresh">
            <RefreshCw size={12} />
          </button>
        }
      >
        {assets.loading && !assets.data ? (
          <div className="p-4 text-faint text-xs flex items-center gap-2">
            <Spinner /> loading
          </div>
        ) : rows.length === 0 ? (
          <Empty
            icon={ImageIcon}
            title="Nothing generated yet"
            hint="Everything you make is stored on your own disk, under the assets folder in your CIWS home."
          />
        ) : (
          <div className="p-2 grid gap-2 grid-cols-[repeat(auto-fill,minmax(168px,1fr))]">
            {rows.map((asset) => (
              <AssetTile
                key={asset.id}
                asset={asset}
                onOpen={() => setPreview(asset)}
                onChange={assets.reload}
              />
            ))}
          </div>
        )}
      </Panel>

      <Modal
        open={!!preview}
        onClose={() => setPreview(null)}
        title={preview?.model ?? 'Asset'}
        width="max-w-4xl"
      >
        {preview && (
          <div className="space-y-3">
            {preview.kind === 'video' ? (
              <video src={api.mediaUrl(preview.id)} controls className="w-full rounded border border-line" />
            ) : (
              <img
                src={api.mediaUrl(preview.id)}
                alt={preview.prompt}
                className="w-full rounded border border-line"
              />
            )}
            <p className="text-sm text-dim leading-relaxed break-words">{preview.prompt}</p>
            {preview.negative_prompt && (
              <p className="text-2xs text-faint">negative: {preview.negative_prompt}</p>
            )}
            <div className="flex flex-wrap gap-1">
              <Badge tone="cyan">{preview.provider}</Badge>
              <Badge>{preview.model}</Badge>
              {!!preview.width && (
                <Badge>
                  {preview.width}×{preview.height}
                </Badge>
              )}
              {!!preview.duration_s && <Badge>{preview.duration_s}s</Badge>}
              <Badge>{bytes(preview.size_bytes)}</Badge>
              {preview.cost_usd > 0 && <Badge tone="amber">{usd(preview.cost_usd)}</Badge>}
              <Badge>{ago(preview.created_at)}</Badge>
            </div>
            <div className="flex gap-2">
              <a className="btn" href={api.mediaUrl(preview.id)} download target="_blank" rel="noreferrer">
                <Download size={11} /> download
              </a>
              <button
                className="btn btn-danger"
                onClick={async () => {
                  await api.del(`/media/${preview.id}`)
                  setPreview(null)
                  assets.reload()
                }}
              >
                <Trash2 size={11} /> delete
              </button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

function AssetTile({
  asset,
  onOpen,
  onChange,
}: {
  asset: AssetRow
  onOpen: () => void
  onChange: () => void
}) {
  const pending = asset.status === 'queued' || asset.status === 'running'
  return (
    <div className="group relative surface overflow-hidden aspect-square">
      {pending ? (
        <div className="absolute inset-0 grid place-items-center">
          <div className="text-center">
            <Loader2 size={18} className="animate-spin text-cyan mx-auto" />
            <div className="text-2xs text-faint mt-1.5">{asset.status}</div>
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-0.5 overflow-hidden bg-line">
            <div className="h-full w-1/3 bg-cyan animate-sweep" />
          </div>
        </div>
      ) : asset.status === 'failed' ? (
        <button onClick={onOpen} className="absolute inset-0 p-2.5 text-left">
          <div className="text-2xs text-rose leading-snug line-clamp-4">{asset.error || 'failed'}</div>
        </button>
      ) : (
        <button onClick={onOpen} className="absolute inset-0 w-full h-full">
          {asset.kind === 'video' ? (
            <div className="w-full h-full grid place-items-center bg-void">
              <Play size={22} className="text-cyan" />
            </div>
          ) : (
            <img
              src={api.mediaUrl(asset.id, true)}
              onError={(e) => {
                // No thumbnail (Pillow absent) -- fall back to the original.
                const img = e.currentTarget
                if (!img.dataset.fallback) {
                  img.dataset.fallback = '1'
                  img.src = api.mediaUrl(asset.id)
                }
              }}
              alt={asset.prompt}
              loading="lazy"
              className="w-full h-full object-cover"
            />
          )}
        </button>
      )}

      <div className="absolute inset-x-0 bottom-0 p-1.5 bg-gradient-to-t from-void via-void/85 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
        <div className="text-2xs text-dim line-clamp-2 leading-snug">{asset.prompt}</div>
      </div>

      <div className="absolute top-1 right-1 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          className={classNames(
            'p-1 rounded bg-void/85 border border-line',
            asset.favorite ? 'text-rose' : 'text-faint hover:text-rose',
          )}
          onClick={async (e) => {
            e.stopPropagation()
            await api.patch(`/media/${asset.id}`, { favorite: !asset.favorite })
            onChange()
          }}
        >
          <Heart size={10} fill={asset.favorite ? 'currentColor' : 'none'} />
        </button>
        <button
          className="p-1 rounded bg-void/85 border border-line text-faint hover:text-rose"
          onClick={async (e) => {
            e.stopPropagation()
            await api.del(`/media/${asset.id}`)
            onChange()
          }}
        >
          <X size={10} />
        </button>
      </div>
    </div>
  )
}
