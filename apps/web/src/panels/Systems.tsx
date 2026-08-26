/**
 * Models, credentials, hubs, tools and settings.
 *
 * The credential form is the one place a secret is typed, so it is explicit
 * about where the value goes: environment variables win over the vault, and the
 * source column says which one is actually in play. A settings screen that
 * silently ignores an env var is how people lose an afternoon.
 */

import {
  Activity,
  Boxes,
  Check,
  Cpu,
  Download,
  Eye,
  EyeOff,
  Key,
  Play,
  Plug,
  RefreshCw,
  Settings2,
  Square,
  Trash2,
  Wrench,
  Zap,
} from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import {
  Badge,
  ErrorLine,
  Field,
  Modal,
  Panel,
  Select,
  Spinner,
  Stat,
  StatusDot,
  Toggle,
  bytes,
  classNames,
  compact,
  riskTone,
  useAsync,
  useToast,
  usd,
} from '../components/ui'
import {
  api,
  events,
  type CredentialRow,
  type HubRow,
  type ModelRow,
  type SystemInfo,
  type ToolRow,
} from '../lib/api'

type Tab = 'models' | 'credentials' | 'hubs' | 'tools' | 'settings' | 'ops'

const TABS: { key: Tab; label: string; icon: typeof Cpu }[] = [
  { key: 'models', label: 'Models', icon: Cpu },
  { key: 'credentials', label: 'Credentials', icon: Key },
  { key: 'hubs', label: 'Hubs', icon: Plug },
  { key: 'tools', label: 'Tools', icon: Wrench },
  { key: 'settings', label: 'Settings', icon: Settings2 },
  { key: 'ops', label: 'Ops', icon: Activity },
]

export function Systems() {
  const [tab, setTab] = useState<Tab>('models')
  return (
    <div className="h-full flex flex-col gap-2 min-h-0">
      <div className="flex gap-1 shrink-0 flex-wrap">
        {TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={classNames('btn !text-2xs uppercase tracking-widest', tab === key && 'btn-primary')}
          >
            <Icon size={11} /> {label}
          </button>
        ))}
      </div>
      <div className="flex-1 min-h-0">
        {tab === 'models' && <Models />}
        {tab === 'credentials' && <Credentials />}
        {tab === 'hubs' && <Hubs />}
        {tab === 'tools' && <Tools />}
        {tab === 'settings' && <SettingsView />}
        {tab === 'ops' && <Ops />}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------

function Models() {
  const [pulling, setPulling] = useState('')
  const models = useAsync(() => api.get<{ models: ModelRow[] }>('/models'), [])
  const health = useAsync(() => api.get<{ providers: Record<string, any> }>('/models/health'), [])

  const rows = models.data?.models ?? []
  const configured = rows.filter((m) => m.configured)
  const totalCost = rows.reduce((sum, m) => sum + m.usage.cost_usd, 0)
  const totalCalls = rows.reduce((sum, m) => sum + m.usage.calls, 0)

  const byProvider = rows.reduce<Record<string, ModelRow[]>>((acc, model) => {
    ;(acc[model.provider] ??= []).push(model)
    return acc
  }, {})

  return (
    <div className="h-full grid grid-rows-[auto_1fr] gap-2 min-h-0">
      <div className="grid grid-cols-4 gap-2 max-[900px]:grid-cols-2">
        <Stat label="Available" value={configured.length} hint={`of ${rows.length} known`} tone="cyan" />
        <Stat label="Providers up" value={Object.values(health.data?.providers ?? {}).filter((p: any) => p.ok).length} />
        <Stat label="Calls" value={compact(totalCalls)} />
        <Stat label="Spend" value={usd(totalCost)} tone="amber" hint="tracked locally" />
      </div>

      <Panel
        title="Model registry"
        subtitle="pricing is editable — click a rate to correct it"
        bodyClass="overflow-y-auto scroll"
        actions={
          <>
            <PullModel onDone={models.reload} pulling={pulling} setPulling={setPulling} />
            <button
              className="btn btn-ghost !p-1"
              onClick={() => {
                models.reload()
                health.reload()
              }}
              title="Refresh"
            >
              <RefreshCw size={12} />
            </button>
          </>
        }
      >
        {models.error && (
          <div className="p-3">
            <ErrorLine error={models.error} />
          </div>
        )}
        {Object.entries(byProvider)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([provider, list]) => {
            const status = health.data?.providers?.[provider]
            return (
              <div key={provider}>
                <div className="sticky top-0 z-10 flex items-center gap-2 px-3 py-1.5 bg-panel border-y border-line">
                  <StatusDot status={status?.ok ? 'ready' : status?.has_key === false ? 'stopped' : 'error'} />
                  <span className="text-xs text-ink font-semibold">{provider}</span>
                  <span className="text-2xs text-faint truncate flex-1">{status?.detail ?? ''}</span>
                  <span className="text-2xs text-faint">{list.length}</span>
                </div>
                {list.map((model) => (
                  <ModelRowView key={model.id} model={model} onChange={models.reload} />
                ))}
              </div>
            )
          })}
      </Panel>
    </div>
  )
}

function ModelRowView({ model, onChange }: { model: ModelRow; onChange: () => void }) {
  const [editing, setEditing] = useState(false)
  const [input, setInput] = useState(String(model.input_cost_per_mtok))
  const [output, setOutput] = useState(String(model.output_cost_per_mtok))

  return (
    <div
      className={classNames(
        'grid grid-cols-[1fr_auto_auto_auto] gap-3 items-center px-3 py-2 border-b border-line/50 hover:bg-raised/40',
        !model.configured && 'opacity-45',
      )}
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm text-ink truncate">{model.display_name || model.name}</span>
          {model.local && <Badge tone="jade">local</Badge>}
          {model.capabilities.includes('vision') && <Badge tone="violet">vision</Badge>}
          {model.capabilities.includes('thinking') && <Badge tone="cyan">thinking</Badge>}
          {model.capabilities.includes('embedding') && <Badge>embed</Badge>}
        </div>
        <div className="text-2xs text-faint truncate">{model.id}</div>
      </div>

      <div className="text-2xs text-faint tabular-nums text-right">
        {model.context_window ? `${compact(model.context_window)} ctx` : '--'}
      </div>

      <button
        className="text-2xs text-faint tabular-nums text-right hover:text-cyan"
        onClick={() => setEditing(true)}
        title="Edit pricing"
      >
        {model.local
          ? 'free'
          : model.input_cost_per_mtok || model.output_cost_per_mtok
            ? `$${model.input_cost_per_mtok}/$${model.output_cost_per_mtok}`
            : 'set price'}
      </button>

      <div className="text-2xs text-faint tabular-nums text-right w-24">
        {model.usage.calls > 0 ? (
          <>
            <div className="text-dim">{model.usage.calls} calls</div>
            <div>
              {usd(model.usage.cost_usd)} · {Math.round(model.usage.avg_latency_ms)}ms
            </div>
          </>
        ) : (
          <span>unused</span>
        )}
      </div>

      <Modal open={editing} onClose={() => setEditing(false)} title={`Pricing — ${model.name}`} width="max-w-sm">
        <div className="space-y-3">
          <div className="text-2xs text-faint leading-relaxed">
            Dollars per million tokens. CIWS seeds what it knows and leaves the rest at zero rather
            than inventing a number — set it here and your figure survives refreshes.
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Input $/Mtok">
              <input value={input} onChange={(e) => setInput(e.target.value)} className="field" />
            </Field>
            <Field label="Output $/Mtok">
              <input value={output} onChange={(e) => setOutput(e.target.value)} className="field" />
            </Field>
          </div>
          <div className="flex justify-end gap-2">
            <button className="btn" onClick={() => setEditing(false)}>
              Cancel
            </button>
            <button
              className="btn btn-primary"
              onClick={async () => {
                await api.patch(`/models/${model.id}`, {
                  input_cost_per_mtok: Number(input) || 0,
                  output_cost_per_mtok: Number(output) || 0,
                })
                setEditing(false)
                onChange()
              }}
            >
              Save
            </button>
          </div>
        </div>
      </Modal>
    </div>
  )
}

function PullModel({
  onDone,
  pulling,
  setPulling,
}: {
  onDone: () => void
  pulling: string
  setPulling: (v: string) => void
}) {
  const toast = useToast()
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('llama3.2')

  return (
    <>
      <button className="btn btn-ghost !p-1" onClick={() => setOpen(true)} title="Pull a local model">
        <Download size={12} />
      </button>
      <Modal open={open} onClose={() => setOpen(false)} title="Pull a local model" width="max-w-md">
        <div className="space-y-3">
          <Field
            label="Ollama model"
            hint="Downloads through your local Ollama install. Large models take a while and the request stays open until it finishes."
          >
            <input value={name} onChange={(e) => setName(e.target.value)} className="field" />
          </Field>
          <div className="flex flex-wrap gap-1">
            {['llama3.2', 'qwen2.5:7b', 'mistral', 'nomic-embed-text', 'llava'].map((suggestion) => (
              <button key={suggestion} className="chip hover:border-cyan/50" onClick={() => setName(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
          <div className="flex justify-end gap-2">
            <button className="btn" onClick={() => setOpen(false)}>
              Cancel
            </button>
            <button
              className="btn btn-primary"
              disabled={!!pulling}
              onClick={async () => {
                setPulling(name)
                try {
                  await api.post('/models/pull', { model: name })
                  toast(`Pulled ${name}`, 'success')
                  setOpen(false)
                  onDone()
                } catch (err) {
                  toast((err as Error).message, 'error')
                } finally {
                  setPulling('')
                }
              }}
            >
              {pulling ? <Spinner size={11} /> : <Download size={11} />} {pulling || 'Pull'}
            </button>
          </div>
        </div>
      </Modal>
    </>
  )
}

// ---------------------------------------------------------------------------

function Credentials() {
  const toast = useToast()
  const credentials = useAsync(() => api.get<{ credentials: CredentialRow[] }>('/credentials'), [])
  const system = useAsync(() => api.get<SystemInfo>('/system'), [])
  const [editing, setEditing] = useState<string | null>(null)
  const [value, setValue] = useState('')
  const [reveal, setReveal] = useState(false)

  const providers = system.data?.providers ?? []
  const hintFor = (name: string) => providers.find((p) => p.id === name)?.env_hint ?? ''

  return (
    <Panel
      title="Credentials"
      subtitle="encrypted at rest — environment variables take precedence"
      bodyClass="overflow-y-auto scroll"
      actions={
        <button className="btn btn-ghost !p-1" onClick={credentials.reload} title="Refresh">
          <RefreshCw size={12} />
        </button>
      }
    >
      <div className="p-3 text-2xs text-faint leading-relaxed border-b border-line">
        Keys live in an encrypted vault inside your CIWS home directory, never in a config file and
        never sent anywhere but the provider they belong to. Anything set as an environment variable
        wins over the vault — the source column tells you which one is live.
      </div>
      <div className="divide-y divide-line/50">
        {(credentials.data?.credentials ?? []).map((row) => (
          <div key={row.name} className="grid grid-cols-[1fr_auto_auto] gap-3 items-center px-3 py-2 hover:bg-raised/40">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <StatusDot status={row.configured ? 'ready' : 'stopped'} />
                <span className="text-sm text-ink">{row.name}</span>
                {row.configured && <span className="text-2xs text-faint font-mono">{row.masked}</span>}
              </div>
              <div className="text-2xs text-faint truncate">
                {row.source === 'missing'
                  ? row.env_names.length
                    ? `set ${row.env_names[0]} or add it here`
                    : 'not configured'
                  : `from ${row.source}`}
              </div>
            </div>
            <div>
              {row.source.startsWith('env:') && <Badge tone="jade">env</Badge>}
              {row.source === 'vault' && <Badge tone="cyan">vault</Badge>}
            </div>
            <div className="flex gap-1">
              <button
                className="btn !py-1 !px-2 !text-2xs"
                onClick={() => {
                  setEditing(row.name)
                  setValue('')
                }}
              >
                {row.configured ? 'replace' : 'add'}
              </button>
              {row.source === 'vault' && (
                <button
                  className="btn btn-danger !py-1 !px-1.5"
                  onClick={async () => {
                    await api.del(`/credentials/${row.name}`)
                    credentials.reload()
                  }}
                >
                  <Trash2 size={10} />
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      <Modal open={!!editing} onClose={() => setEditing(null)} title={`Key — ${editing}`} width="max-w-md">
        <div className="space-y-3">
          <Field
            label="API key"
            hint={
              hintFor(editing ?? '')
                ? `Equivalent environment variable: ${hintFor(editing ?? '')}`
                : 'Pasted values are encrypted before they touch the disk.'
            }
          >
            <div className="relative">
              <input
                type={reveal ? 'text' : 'password'}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                className="field pr-8 font-mono"
                placeholder="sk-..."
                autoFocus
              />
              <button
                className="absolute right-2 top-1/2 -translate-y-1/2 text-faint hover:text-ink"
                onClick={() => setReveal(!reveal)}
                type="button"
              >
                {reveal ? <EyeOff size={12} /> : <Eye size={12} />}
              </button>
            </div>
          </Field>
          <div className="flex justify-end gap-2">
            <button className="btn" onClick={() => setEditing(null)}>
              Cancel
            </button>
            <button
              className="btn btn-primary"
              disabled={!value.trim()}
              onClick={async () => {
                try {
                  await api.put('/credentials', { name: editing, value })
                  toast(`${editing} saved`, 'success')
                  setEditing(null)
                  setValue('')
                  credentials.reload()
                } catch (err) {
                  toast((err as Error).message, 'error')
                }
              }}
            >
              <Check size={11} /> Save
            </button>
          </div>
        </div>
      </Modal>
    </Panel>
  )
}

// ---------------------------------------------------------------------------

function Hubs() {
  const toast = useToast()
  const hubs = useAsync(() => api.get<{ hubs: HubRow[]; kinds: string[] }>('/hubs'), [])
  const [busy, setBusy] = useState('')

  useEffect(() => events.on('hub.status', () => hubs.reload()), [hubs.reload])

  const act = useCallback(
    async (slug: string, action: 'start' | 'stop' | 'test') => {
      setBusy(slug)
      try {
        const result = await api.post<any>(`/hubs/${slug}/${action}`)
        if (action === 'test') {
          toast(result.ok ? `OK — ${result.detail}` : `Failed — ${result.detail}`, result.ok ? 'success' : 'error')
        }
        hubs.reload()
      } catch (err) {
        toast((err as Error).message, 'error')
      } finally {
        setBusy('')
      }
    },
    [toast, hubs],
  )

  return (
    <Panel
      title="Hubs"
      subtitle="MCP servers, folders and search backends"
      bodyClass="overflow-y-auto scroll"
      actions={
        <button className="btn btn-ghost !p-1" onClick={hubs.reload} title="Refresh">
          <RefreshCw size={12} />
        </button>
      }
    >
      <div className="p-3 text-2xs text-faint leading-relaxed border-b border-line">
        MCP hubs run as local subprocesses and are disabled until you enable them — CIWS will not
        spawn a process you did not ask for. Their tools appear to agents namespaced as{' '}
        <span className="text-dim">hub__tool</span>.
      </div>
      <div className="divide-y divide-line/50">
        {(hubs.data?.hubs ?? []).map((hub) => (
          <div key={hub.slug} className="px-3 py-2.5 hover:bg-raised/40">
            <div className="flex items-start gap-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <StatusDot status={hub.status} />
                  <span className="text-sm text-ink">{hub.name}</span>
                  <Badge>{hub.kind}</Badge>
                  {hub.tool_count > 0 && <Badge tone="cyan">{hub.tool_count} tools</Badge>}
                </div>
                <div className="text-2xs text-faint mt-0.5 leading-relaxed">{hub.description}</div>
                {hub.status_detail && (
                  <div
                    className={classNames(
                      'text-2xs mt-0.5 break-words',
                      hub.status === 'error' ? 'text-rose' : 'text-faint',
                    )}
                  >
                    {hub.status_detail}
                  </div>
                )}
                {hub.config?.command && (
                  <div className="text-2xs text-faint font-mono mt-0.5 truncate">
                    {hub.config.command} {(hub.config.args ?? []).join(' ')}
                  </div>
                )}
              </div>

              <div className="flex items-center gap-1 shrink-0">
                {busy === hub.slug && <Spinner size={11} />}
                <button
                  className="btn !py-1 !px-1.5 !text-2xs"
                  onClick={() => act(hub.slug, 'test')}
                  title="Test"
                >
                  <Zap size={10} />
                </button>
                {['mcp_stdio', 'mcp_http'].includes(hub.kind) &&
                  (hub.running ? (
                    <button
                      className="btn !py-1 !px-1.5 !text-2xs text-amber"
                      onClick={() => act(hub.slug, 'stop')}
                      title="Stop"
                    >
                      <Square size={10} />
                    </button>
                  ) : (
                    <button
                      className="btn !py-1 !px-1.5 !text-2xs"
                      onClick={() => act(hub.slug, 'start')}
                      title="Start"
                      disabled={!hub.enabled}
                    >
                      <Play size={10} />
                    </button>
                  ))}
                <button
                  className={classNames('btn !py-1 !px-1.5 !text-2xs', hub.enabled && 'text-jade')}
                  onClick={async () => {
                    await api.patch(`/hubs/${hub.slug}`, { enabled: !hub.enabled })
                    hubs.reload()
                  }}
                  title={hub.enabled ? 'Disable' : 'Enable'}
                >
                  {hub.enabled ? 'on' : 'off'}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

// ---------------------------------------------------------------------------

function Tools() {
  const tools = useAsync(() => api.get<{ tools: ToolRow[]; categories: Record<string, number> }>('/tools'), [])
  const [category, setCategory] = useState('')

  const rows = (tools.data?.tools ?? []).filter((t) => !category || t.category === category)

  return (
    <Panel
      title="Tools"
      subtitle={`${tools.data?.tools.length ?? 0} available to agents`}
      bodyClass="overflow-y-auto scroll"
      actions={
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="bg-void border border-line rounded px-1.5 py-1 text-2xs text-dim outline-none focus:border-cyan/50 cursor-pointer"
        >
          <option value="">all categories</option>
          {Object.entries(tools.data?.categories ?? {}).map(([name, count]) => (
            <option key={name} value={name}>
              {name} ({count})
            </option>
          ))}
        </select>
      }
    >
      <div className="divide-y divide-line/50">
        {rows.map((tool) => (
          <div key={tool.name} className="px-3 py-2 hover:bg-raised/40">
            <div className="flex items-center gap-2">
              <span className="text-sm text-cyan font-mono">{tool.name}</span>
              <Badge tone={riskTone(tool.risk)}>{tool.risk}</Badge>
              <Badge>{tool.category}</Badge>
              {tool.source !== 'builtin' && <Badge tone="violet">{tool.source}</Badge>}
            </div>
            <p className="text-2xs text-faint mt-0.5 leading-relaxed line-clamp-2">
              {tool.description.split('\n')[0]}
            </p>
          </div>
        ))}
      </div>
    </Panel>
  )
}

// ---------------------------------------------------------------------------

function SettingsView() {
  const toast = useToast()
  const system = useAsync(() => api.get<SystemInfo>('/system'), [])
  const models = useAsync(() => api.get<{ models: ModelRow[] }>('/models'), [])
  const [saving, setSaving] = useState(false)

  const settings = system.data?.settings
  if (!settings) {
    return (
      <Panel title="Settings">
        <div className="p-4 text-faint text-xs flex items-center gap-2">
          <Spinner /> loading
        </div>
      </Panel>
    )
  }

  const patch = async (body: Record<string, unknown>) => {
    setSaving(true)
    try {
      await api.patch('/settings', body)
      system.reload()
    } catch (err) {
      toast((err as Error).message, 'error')
    } finally {
      setSaving(false)
    }
  }

  const chatModels = (models.data?.models ?? [])
    .filter((m) => m.configured && !m.capabilities.includes('embedding'))
    .map((m) => ({ value: m.id, label: `${m.display_name || m.name}${m.local ? ' (local)' : ''}` }))
  const embedModels = [
    { value: 'local/hash-embed-768', label: 'Built-in hashed features (no key, lexical)' },
    ...(models.data?.models ?? [])
      .filter((m) => m.configured && m.capabilities.includes('embedding'))
      .map((m) => ({ value: m.id, label: m.display_name || m.name })),
  ]

  return (
    <div className="h-full grid grid-cols-2 gap-2 min-h-0 max-[1000px]:grid-cols-1">
      <Panel title="Routing" subtitle={saving ? 'saving' : ''} bodyClass="overflow-y-auto scroll">
        <div className="p-3 space-y-3">
          <div className="text-2xs text-faint leading-relaxed">
            Agents ask for a capability, not a model name. These map those requests onto real models,
            so switching provider is one change here rather than an edit to every agent.
          </div>
          {(['deep', 'balanced', 'fast', 'vision', 'local'] as const).map((role) => (
            <Field key={role} label={role}>
              <Select
                value={settings.routing[role]}
                onChange={(value) => patch({ routing: { [role]: value } })}
                options={
                  chatModels.length
                    ? chatModels
                    : [{ value: settings.routing[role], label: settings.routing[role] }]
                }
              />
            </Field>
          ))}
          <Field
            label="embeddings"
            hint={system.data?.embeddings.note || `Currently: ${system.data?.embeddings.quality}`}
          >
            <Select
              value={settings.routing.embed}
              onChange={(value) => patch({ routing: { embed: value } })}
              options={embedModels}
            />
          </Field>
        </div>
      </Panel>

      <div className="grid grid-rows-2 gap-2 min-h-0">
        <Panel title="Security" bodyClass="overflow-y-auto scroll">
          <div className="p-3 space-y-1">
            <Toggle
              checked={settings.security.allow_shell_tool}
              onChange={(v) => patch({ security: { allow_shell_tool: v } })}
              label="Allow the shell tool"
              hint="Agents can run shell commands. They run with your permissions — this is not sandboxed."
            />
            <Toggle
              checked={settings.security.allow_python_tool}
              onChange={(v) => patch({ security: { allow_python_tool: v } })}
              label="Allow the Python tool"
              hint="Runs in a subprocess with a timeout. Isolated from the server process, not from your files."
            />
            <Toggle
              checked={settings.security.approve_writes_outside_workspace}
              onChange={(v) => patch({ security: { approve_writes_outside_workspace: v } })}
              label="Restrict writes to the workspace"
              hint="When on, file writes outside the workspace directory are refused."
            />
            <Toggle
              checked={settings.security.audit_all_tool_calls}
              onChange={(v) => patch({ security: { audit_all_tool_calls: v } })}
              label="Audit every tool call"
              hint="Arguments and results are recorded so any claim can be traced back to its source."
            />
            <Toggle
              checked={settings.security.require_token}
              onChange={(v) => patch({ security: { require_token: v } })}
              label="Require the access token"
              hint="Without it, any page you visit can drive this hub through localhost. Leave it on."
            />
          </div>
        </Panel>

        <Panel title="Memory & agents" bodyClass="overflow-y-auto scroll">
          <div className="p-3 space-y-1">
            <Toggle
              checked={settings.memory.enabled}
              onChange={(v) => patch({ memory: { enabled: v } })}
              label="Enable memory recall"
              hint="Injects relevant memories into every agent's system prompt."
            />
            <Toggle
              checked={settings.memory.auto_capture}
              onChange={(v) => patch({ memory: { auto_capture: v } })}
              label="Capture memories automatically"
              hint="After each answer, durable facts are extracted in the background."
            />
            <Toggle
              checked={settings.agents.parallel_tools}
              onChange={(v) => patch({ agents: { parallel_tools: v } })}
              label="Run tool calls in parallel"
              hint="Four searches at once instead of four in a row."
            />
            <div className="pt-2 grid grid-cols-2 gap-3">
              <Field label={`Recall limit  ${settings.memory.recall_limit}`}>
                <input
                  type="range"
                  min={4}
                  max={30}
                  value={settings.memory.recall_limit}
                  onChange={(e) => patch({ memory: { recall_limit: Number(e.target.value) } })}
                  className="w-full accent-cyan mt-2"
                />
              </Field>
              <Field label={`Max steps  ${settings.agents.max_steps}`}>
                <input
                  type="range"
                  min={4}
                  max={60}
                  value={settings.agents.max_steps}
                  onChange={(e) => patch({ agents: { max_steps: Number(e.target.value) } })}
                  className="w-full accent-cyan mt-2"
                />
              </Field>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------

function Ops() {
  const toast = useToast()
  const system = useAsync(() => api.get<SystemInfo>('/system'), [])
  const stats = useAsync(() => api.get<any>('/system/stats'), [])
  const logs = useAsync(() => api.get<{ lines: string[] }>('/system/logs', { lines: 300 }), [])
  const [busy, setBusy] = useState(false)

  const db = stats.data?.database ?? {}
  const vectors = stats.data?.vectors ?? {}
  const vectorTotal = Object.values(vectors).reduce((sum: number, v: any) => sum + (v?.count ?? 0), 0)

  return (
    <div className="h-full grid grid-rows-[auto_auto_1fr] gap-2 min-h-0">
      <div className="grid grid-cols-4 gap-2 max-[900px]:grid-cols-2">
        <Stat label="Database" value={bytes(db._size_bytes ?? 0)} hint={db._path?.split('/').pop()} />
        <Stat label="Vectors" value={compact(vectorTotal)} tone="cyan" hint={system.data?.embeddings.backend} />
        <Stat label="Tools" value={system.data?.tools ?? '--'} />
        <Stat label="Hubs running" value={system.data?.hubs_running.length ?? 0} tone="jade" />
      </div>

      <Panel title="Workspace">
        <div className="p-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-2xs max-[900px]:grid-cols-1">
          {Object.entries(system.data?.paths ?? {}).map(([key, value]) => (
            <div key={key} className="grid grid-cols-[70px_1fr] gap-2">
              <span className="text-faint">{key}</span>
              <span className="text-dim font-mono truncate" title={value}>
                {value}
              </span>
            </div>
          ))}
          <div className="grid grid-cols-[70px_1fr] gap-2">
            <span className="text-faint">platform</span>
            <span className="text-dim">
              {system.data?.platform} · python {system.data?.python}
            </span>
          </div>
        </div>
      </Panel>

      <Panel
        title="Log"
        bodyClass="overflow-y-auto scroll"
        actions={
          <>
            <button
              className="btn btn-ghost !p-1"
              disabled={busy}
              title="Compact the database and drop orphaned vectors"
              onClick={async () => {
                setBusy(true)
                try {
                  const result = await api.post<any>('/system/vacuum')
                  toast(`Compacted. ${result.orphaned_vectors_removed} orphaned vectors removed.`, 'success')
                  stats.reload()
                } catch (err) {
                  toast((err as Error).message, 'error')
                } finally {
                  setBusy(false)
                }
              }}
            >
              {busy ? <Spinner size={12} /> : <Boxes size={12} />}
            </button>
            <button className="btn btn-ghost !p-1" onClick={logs.reload} title="Refresh">
              <RefreshCw size={12} />
            </button>
          </>
        }
      >
        <pre className="p-3 text-2xs text-faint leading-relaxed whitespace-pre-wrap break-words">
          {(logs.data?.lines ?? []).join('\n') || 'No log output yet.'}
        </pre>
      </Panel>
    </div>
  )
}
