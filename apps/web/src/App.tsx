/**
 * The shell: a rail of panels, a status bar, and the active workspace.
 *
 * The status bar is not decoration. In a hub where an agent might be running a
 * fourteen-step task in the background, "is the socket up, is a run in flight,
 * what has it cost" needs to be visible without leaving whatever panel you are
 * in.
 */

import {
  Activity,
  Boxes,
  Brain,
  CircleDot,
  Layers,
  Radio,
  ScanEye,
  Server,
  Sparkles,
  Terminal,
  WifiOff,
  Workflow as WorkflowIcon,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import { Badge, ToastHost, classNames, useAsync, useLocalState, usd } from './components/ui'
import { api, events, version, type SystemInfo } from './lib/api'
import { Command } from './panels/Command'
import { Flows } from './panels/Flows'
import { Knowledge } from './panels/Knowledge'
import { Ontology } from './panels/Ontology'
import { Studio } from './panels/Studio'
import { Systems } from './panels/Systems'

type PanelKey = 'command' | 'ontology' | 'knowledge' | 'studio' | 'flows' | 'systems'

const PANELS: {
  key: PanelKey
  label: string
  icon: typeof Terminal
  hint: string
}[] = [
  { key: 'command', label: 'Command', icon: Terminal, hint: 'Talk to your agents' },
  { key: 'ontology', label: 'Ontology', icon: ScanEye, hint: 'The entity graph' },
  { key: 'knowledge', label: 'Knowledge', icon: Brain, hint: 'Memory and documents' },
  { key: 'studio', label: 'Studio', icon: Sparkles, hint: 'Image and video' },
  { key: 'flows', label: 'Flows', icon: WorkflowIcon, hint: 'Automated pipelines' },
  { key: 'systems', label: 'Systems', icon: Server, hint: 'Models, keys, hubs, settings' },
]

export default function App() {
  const [panel, setPanel] = useLocalState<PanelKey>('ciws.panel', 'command')
  const [projectId, setProjectId] = useLocalState<string | null>('ciws.project', null)
  const [connected, setConnected] = useState(false)
  const [activity, setActivity] = useState<string>('')
  const [runs, setRuns] = useState(0)
  const [spend, setSpend] = useState(0)

  const system = useAsync(() => api.get<SystemInfo>('/system'), [])
  const projects = useAsync(() => api.get<{ projects: any[] }>('/projects'), [])

  useEffect(() => {
    events.onStatus = setConnected
    events.connect()
    return () => {
      events.onStatus = null
    }
  }, [])

  useEffect(() => {
    let clearTimer: number | undefined
    const flash = (text: string) => {
      setActivity(text)
      window.clearTimeout(clearTimer)
      clearTimer = window.setTimeout(() => setActivity(''), 2600)
    }

    const off = [
      events.on('run.start', () => setRuns((n) => n + 1)),
      events.on('run.end', (e) => {
        setRuns((n) => Math.max(0, n - 1))
        setSpend((total) => total + Number(e.data.cost_usd ?? 0))
      }),
      events.on('run.error', () => setRuns((n) => Math.max(0, n - 1))),
      events.on('tool.start', (e) => flash(`${e.data.tool}`)),
      events.on('memory.write', () => flash('memory written')),
      events.on('graph.entity', (e) => flash(`entity: ${e.data.name}`)),
      events.on('media.start', () => flash('rendering')),
      events.on('ingest.start', (e) => flash(`ingesting ${e.data.title ?? ''}`)),
      events.on('hub.status', (e) => flash(`hub ${e.data.hub}: ${e.data.status}`)),
    ]
    return () => {
      off.forEach((fn) => fn())
      window.clearTimeout(clearTimer)
    }
  }, [])

  const providersUp = useMemo(
    () => (system.data?.providers ?? []).filter((p) => p.configured).length,
    [system.data],
  )

  const Active = {
    command: Command,
    ontology: Ontology,
    knowledge: Knowledge,
    studio: Studio,
    flows: Flows,
    systems: Systems,
  }[panel]

  const noProviders = system.data && providersUp === 0

  return (
    <ToastHost>
      <div className="h-full flex flex-col bg-void">
        {/* Top bar */}
        <header className="h-10 shrink-0 flex items-center gap-3 px-3 border-b border-line bg-hull">
          <div className="flex items-center gap-2 shrink-0">
            <CircleDot size={15} className="text-cyan" />
            <span className="text-sm font-semibold tracking-[0.2em] text-ink">CIWS</span>
            <span className="text-2xs text-faint hidden sm:inline">v{version}</span>
          </div>

          <div className="w-px h-4 bg-line shrink-0" />

          <select
            value={projectId ?? ''}
            onChange={(e) => setProjectId(e.target.value || null)}
            className="bg-void border border-line rounded px-1.5 py-1 text-2xs text-dim outline-none focus:border-cyan/50 cursor-pointer max-w-[170px]"
            title="Scope everything to a project"
          >
            <option value="">all projects</option>
            {(projects.data?.projects ?? []).map((project) => (
              <option key={project.id} value={project.id}>
                {project.name}
              </option>
            ))}
          </select>

          <div className="flex-1 min-w-0 text-center">
            {activity && (
              <span className="text-2xs text-cyan/80 truncate animate-fade-in">{activity}</span>
            )}
          </div>

          <div className="flex items-center gap-2 shrink-0 text-2xs">
            {runs > 0 && (
              <span className="flex items-center gap-1 text-cyan">
                <Activity size={11} className="animate-pulse-slow" />
                {runs} running
              </span>
            )}
            {spend > 0 && <span className="text-faint tabular-nums">{usd(spend)} this session</span>}
            <span
              className={classNames('flex items-center gap-1', connected ? 'text-jade' : 'text-rose')}
              title={connected ? 'Live event stream connected' : 'Reconnecting to the event stream'}
            >
              {connected ? <Radio size={11} /> : <WifiOff size={11} />}
              <span className="hidden sm:inline">{connected ? 'live' : 'offline'}</span>
            </span>
          </div>
        </header>

        {noProviders && panel !== 'systems' && (
          <button
            onClick={() => setPanel('systems')}
            className="shrink-0 px-3 py-1.5 text-2xs text-amber bg-amber/10 border-b border-amber/25 text-left hover:bg-amber/15 transition-colors"
          >
            No model provider is configured — CIWS is running but cannot think yet. Add an API key in
            Systems → Credentials, or start Ollama for a fully local workspace.
          </button>
        )}

        <div className="flex-1 flex min-h-0">
          {/* Rail */}
          <nav className="w-14 shrink-0 flex flex-col items-center gap-1 py-2 border-r border-line bg-hull">
            {PANELS.map(({ key, label, icon: Icon, hint }) => (
              <button
                key={key}
                onClick={() => setPanel(key)}
                title={`${label} — ${hint}`}
                className={classNames(
                  'group relative w-11 h-11 rounded flex flex-col items-center justify-center gap-0.5 transition-colors',
                  panel === key ? 'bg-cyan/12 text-cyan' : 'text-faint hover:text-dim hover:bg-raised',
                )}
              >
                {panel === key && (
                  <span className="absolute left-0 top-2 bottom-2 w-0.5 rounded-r bg-cyan" />
                )}
                <Icon size={16} />
                <span className="text-[8.5px] uppercase tracking-wider leading-none">{label}</span>
              </button>
            ))}

            <div className="mt-auto flex flex-col items-center gap-1 text-faint">
              <div className="text-[8.5px] text-center leading-tight" title="Configured providers">
                <Boxes size={13} className="mx-auto" />
                <span className="tabular-nums">{providersUp}</span>
              </div>
              <div className="text-[8.5px] text-center leading-tight" title="Tools available">
                <Layers size={13} className="mx-auto" />
                <span className="tabular-nums">{system.data?.tools ?? 0}</span>
              </div>
            </div>
          </nav>

          <main className="flex-1 min-w-0 p-2">
            <Active projectId={projectId} />
          </main>
        </div>

        {/* Status bar */}
        <footer className="h-6 shrink-0 flex items-center gap-3 px-3 border-t border-line bg-hull text-2xs text-faint">
          <span className="truncate">{system.data?.paths?.home ?? 'local workspace'}</span>
          <span className="w-px h-3 bg-line" />
          <span title="Embedding backend in use">
            embeddings: {system.data?.embeddings.backend ?? '--'}
          </span>
          {system.data?.embeddings.local && (
            <Badge tone="jade">
              <span className="text-[9px]">offline capable</span>
            </Badge>
          )}
          <span className="ml-auto hidden md:inline">{system.data?.platform}</span>
        </footer>
      </div>
    </ToastHost>
  )
}
