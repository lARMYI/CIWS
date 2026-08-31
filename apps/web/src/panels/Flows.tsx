/**
 * Workflows: a DAG view with live node states, plus run history.
 *
 * The canvas is a viewer, not a drag-and-drop editor — the graph is edited as
 * JSON, validated by the server before it can run. That is a deliberate trade:
 * a half-working node editor is worse than a clear diagram next to the source
 * of truth, and the validator names cycles and dangling edges precisely enough
 * to fix by hand.
 *
 * Node states stream in over the event bus while a run executes, so the diagram
 * animates rather than reporting after the fact.
 */

import {
  Pencil,
  Play,
  RefreshCw,
  Workflow as WorkflowIcon,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import {
  Empty,
  ErrorLine,
  Field,
  Modal,
  Panel,
  Spinner,
  StatusDot,
  ago,
  classNames,
  useAsync,
  useToast,
  usd,
} from '../components/ui'
import { FlowEditor } from '../components/FlowEditor'
import { api, events, type WorkflowRow } from '../lib/api'

interface NodeType {
  type: string
  label: string
  category: string
  description: string
  color: string
  inputs: string[]
  outputs: string[]
}

interface RunRecord {
  id: string
  status: string
  outputs: Record<string, unknown>
  node_states: Record<string, { status: string; error?: string; preview?: string; duration_ms?: number }>
  error: string
  cost_usd: number
  started_at: string
  ended_at: string | null
}

const NODE_W = 138
const NODE_H = 44

export function Flows({ projectId }: { projectId: string | null }) {
  const toast = useToast()
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [live, setLive] = useState<Record<string, { status: string; error?: string }>>({})
  const [running, setRunning] = useState(false)
  const [editing, setEditing] = useState(false)
  const [inputs, setInputs] = useState<Record<string, string>>({})

  const workflows = useAsync(
    () => api.get<{ workflows: WorkflowRow[]; node_types: NodeType[] }>('/workflows', { project_id: projectId }),
    [projectId],
  )
  const detail = useAsync(
    async () =>
      selectedId
        ? api.get<WorkflowRow & { problems: string[]; runs: RunRecord[] }>(`/workflows/${selectedId}`)
        : null,
    [selectedId],
  )

  const list = workflows.data?.workflows ?? []

  useEffect(() => {
    if (!selectedId && list.length) setSelectedId(list[0].id)
  }, [list, selectedId])

  useEffect(() => {
    if (detail.data) {
      setInputs(
        Object.fromEntries(
          Object.entries(detail.data.inputs ?? {}).map(([k, v]) => [k, String(v ?? '')]),
        ),
      )
    }
  }, [detail.data?.id])

  useEffect(
    () =>
      events.on('workflow.node', (e) =>
        setLive((current) => ({
          ...current,
          [String(e.data.node_id)]: { status: String(e.data.status), error: e.data.error },
        })),
      ),
    [],
  )

  const run = useCallback(async () => {
    if (!selectedId) return
    setRunning(true)
    setLive({})
    try {
      const result = await api.post<RunRecord>(`/workflows/${selectedId}/run`, inputs)
      toast(
        result.status === 'completed' ? 'Workflow completed' : `Workflow ${result.status}`,
        result.status === 'completed' ? 'success' : 'error',
      )
      detail.reload()
    } catch (err) {
      toast((err as Error).message, 'error')
    } finally {
      setRunning(false)
    }
  }, [selectedId, inputs, toast, detail])

  const workflow = detail.data
  const lastRun = workflow?.runs?.[0]
  const nodeStates = useMemo(() => {
    const base: Record<string, { status: string; error?: string }> = {}
    if (lastRun?.node_states && Object.keys(live).length === 0) {
      for (const [id, state] of Object.entries(lastRun.node_states)) base[id] = state
    }
    return { ...base, ...live }
  }, [lastRun, live])

  return (
    <div className="h-full grid grid-cols-[220px_1fr_280px] gap-2 min-h-0 max-[1300px]:grid-cols-[200px_1fr] max-[900px]:grid-cols-1">
      <Panel
        title="Workflows"
        actions={
          <button className="btn btn-ghost !p-1" onClick={workflows.reload} title="Refresh">
            <RefreshCw size={12} />
          </button>
        }
      >
        <div className="h-full overflow-y-auto scroll">
          {list.length === 0 ? (
            <div className="p-3 text-2xs text-faint leading-relaxed">
              No workflows yet. Examples are seeded on first boot — restart the server if this stays
              empty.
            </div>
          ) : (
            list.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  setSelectedId(item.id)
                  setLive({})
                }}
                className={classNames(
                  'w-full text-left px-3 py-2 border-b border-line/50 transition-colors',
                  item.id === selectedId ? 'bg-cyan/10 border-l-2 border-l-cyan' : 'hover:bg-raised',
                )}
              >
                <div className={classNames('text-xs', item.id === selectedId ? 'text-cyan' : 'text-dim')}>
                  {item.name}
                </div>
                <div className="text-2xs text-faint line-clamp-2 mt-0.5 leading-snug">
                  {item.description}
                </div>
                <div className="text-2xs text-faint mt-0.5">{item.graph?.nodes?.length ?? 0} nodes</div>
              </button>
            ))
          )}
        </div>
      </Panel>

      <Panel
        title={workflow?.name ?? 'Graph'}
        subtitle={workflow?.description}
        bodyClass="relative overflow-auto scroll"
        actions={
          <>
            {workflow && (
              <button className="btn btn-ghost !p-1" onClick={() => setEditing(true)} title="Edit graph">
                <Pencil size={12} />
              </button>
            )}
            <button
              className="btn btn-primary !py-1 !px-2 !text-2xs"
              onClick={run}
              disabled={!workflow || running || (workflow?.problems?.length ?? 0) > 0}
            >
              {running ? <Spinner size={11} /> : <Play size={11} />} run
            </button>
          </>
        }
      >
        {!workflow ? (
          <Empty icon={WorkflowIcon} title="Select a workflow" />
        ) : (
          <>
            {workflow.problems.length > 0 && (
              <div className="m-3">
                <ErrorLine error={`This workflow will not run:\n- ${workflow.problems.join('\n- ')}`} />
              </div>
            )}
            <GraphView graph={workflow.graph} states={nodeStates} nodeTypes={workflows.data?.node_types ?? []} />
          </>
        )}
      </Panel>

      <Panel title="Run" bodyClass="overflow-y-auto scroll" className="max-[1300px]:hidden">
        {workflow ? (
          <div className="p-3 space-y-3">
            {Object.keys(inputs).length > 0 && (
              <div>
                <div className="label mb-1.5">Inputs</div>
                <div className="space-y-2">
                  {Object.entries(inputs).map(([key, value]) => (
                    <Field key={key} label={key}>
                      <textarea
                        value={value}
                        onChange={(e) => setInputs({ ...inputs, [key]: e.target.value })}
                        rows={2}
                        className="field resize-none text-xs"
                      />
                    </Field>
                  ))}
                </div>
              </div>
            )}

            {lastRun && (
              <div>
                <div className="label mb-1.5">Last run</div>
                <div className="surface p-2 space-y-1.5">
                  <div className="flex items-center gap-2">
                    <StatusDot status={lastRun.status} />
                    <span className="text-xs text-dim">{lastRun.status}</span>
                    <span className="text-2xs text-faint ml-auto">{ago(lastRun.started_at)}</span>
                  </div>
                  {lastRun.cost_usd > 0 && (
                    <div className="text-2xs text-faint">cost {usd(lastRun.cost_usd)}</div>
                  )}
                  {lastRun.error && <div className="text-2xs text-rose break-words">{lastRun.error}</div>}
                  {Object.entries(lastRun.outputs ?? {}).map(([key, value]) => (
                    <div key={key}>
                      <div className="text-2xs text-cyan">{key}</div>
                      <div className="text-2xs text-dim whitespace-pre-wrap break-words max-h-52 overflow-y-auto scroll">
                        {typeof value === 'string' ? value : JSON.stringify(value, null, 2)}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {(workflow.runs?.length ?? 0) > 1 && (
              <div>
                <div className="label mb-1.5">History</div>
                <div className="surface divide-y divide-line/60">
                  {workflow.runs.slice(1, 9).map((record) => (
                    <div key={record.id} className="flex items-center gap-2 px-2 py-1.5">
                      <StatusDot status={record.status} />
                      <span className="text-2xs text-dim">{record.status}</span>
                      <span className="text-2xs text-faint ml-auto">{ago(record.started_at)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <Empty icon={Play} title="Nothing selected" />
        )}
      </Panel>

      {workflow && (
        <GraphEditor
          open={editing}
          onClose={() => setEditing(false)}
          workflow={workflow}
          nodeTypes={workflows.data?.node_types ?? []}
          onSaved={() => {
            detail.reload()
            workflows.reload()
          }}
        />
      )}
    </div>
  )
}

function GraphView({
  graph,
  states,
  nodeTypes,
}: {
  graph: { nodes: any[]; edges: any[] }
  states: Record<string, { status: string; error?: string }>
  nodeTypes: NodeType[]
}) {
  const nodes = graph?.nodes ?? []
  const edges = graph?.edges ?? []
  const colorOf = (type: string) => nodeTypes.find((n) => n.type === type)?.color ?? '#64748b'

  // Fall back to a layered layout when a node carries no stored position, so a
  // hand-written graph still renders sensibly instead of stacking at the origin.
  const positions = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>()
    const depth = new Map<string, number>()
    const incoming = new Map<string, string[]>()
    for (const node of nodes) incoming.set(String(node.id), [])
    for (const edge of edges) incoming.get(String(edge.target))?.push(String(edge.source))

    const compute = (id: string, seen = new Set<string>()): number => {
      if (depth.has(id)) return depth.get(id)!
      if (seen.has(id)) return 0
      seen.add(id)
      const parents = incoming.get(id) ?? []
      const value = parents.length ? Math.max(...parents.map((p) => compute(p, seen))) + 1 : 0
      depth.set(id, value)
      return value
    }

    const perLevel = new Map<number, number>()
    for (const node of nodes) {
      const id = String(node.id)
      if (node.position?.x !== undefined) {
        map.set(id, { x: node.position.x, y: node.position.y })
        continue
      }
      const level = compute(id)
      const index = perLevel.get(level) ?? 0
      perLevel.set(level, index + 1)
      map.set(id, { x: 40 + level * 210, y: 40 + index * 88 })
    }
    return map
  }, [nodes, edges])

  const bounds = useMemo(() => {
    let width = 600
    let height = 320
    for (const point of positions.values()) {
      width = Math.max(width, point.x + NODE_W + 60)
      height = Math.max(height, point.y + NODE_H + 60)
    }
    return { width, height }
  }, [positions])

  if (!nodes.length) return <Empty icon={WorkflowIcon} title="This workflow has no nodes" />

  return (
    <div className="p-3">
      <svg width={bounds.width} height={bounds.height} className="min-w-full">
        <defs>
          <marker id="wf-arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#2c3d52" />
          </marker>
        </defs>

        {edges.map((edge, index) => {
          const from = positions.get(String(edge.source))
          const to = positions.get(String(edge.target))
          if (!from || !to) return null
          const x1 = from.x + NODE_W
          const y1 = from.y + NODE_H / 2
          const x2 = to.x
          const y2 = to.y + NODE_H / 2
          const mid = (x1 + x2) / 2
          const active = states[String(edge.source)]?.status === 'done'
          return (
            <path
              key={index}
              d={`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`}
              fill="none"
              stroke={active ? '#22d3ee' : '#2c3d52'}
              strokeWidth={active ? 1.8 : 1.2}
              markerEnd="url(#wf-arrow)"
              opacity={active ? 0.9 : 0.6}
            />
          )
        })}

        {nodes.map((node) => {
          const id = String(node.id)
          const point = positions.get(id)!
          const state = states[id]?.status
          const stroke =
            state === 'done'
              ? '#34d399'
              : state === 'running'
                ? '#22d3ee'
                : state === 'failed'
                  ? '#fb7185'
                  : state === 'skipped'
                    ? '#475569'
                    : '#1a2432'
          return (
            <g key={id} transform={`translate(${point.x},${point.y})`}>
              <rect
                width={NODE_W}
                height={NODE_H}
                rx={4}
                fill="#0d1219"
                stroke={stroke}
                strokeWidth={state && state !== 'pending' ? 1.8 : 1}
              />
              <rect width={3} height={NODE_H} rx={1.5} fill={colorOf(String(node.type))} />
              <text x={11} y={17} fill="#dbe6f3" fontSize={11} fontFamily="ui-monospace, monospace">
                {id.length > 15 ? `${id.slice(0, 14)}…` : id}
              </text>
              <text x={11} y={31} fill="#55677d" fontSize={9.5} fontFamily="ui-monospace, monospace">
                {node.type}
              </text>
              {state === 'running' && (
                <circle cx={NODE_W - 11} cy={13} r={3} fill="#22d3ee">
                  <animate attributeName="opacity" values="1;0.25;1" dur="1.1s" repeatCount="indefinite" />
                </circle>
              )}
              {state === 'done' && <circle cx={NODE_W - 11} cy={13} r={3} fill="#34d399" />}
              {state === 'failed' && <circle cx={NODE_W - 11} cy={13} r={3} fill="#fb7185" />}
              {state === 'skipped' && <circle cx={NODE_W - 11} cy={13} r={3} fill="#475569" />}
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function GraphEditor({
  open,
  onClose,
  workflow,
  nodeTypes,
  onSaved,
}: {
  open: boolean
  onClose: () => void
  workflow: WorkflowRow
  nodeTypes: NodeType[]
  onSaved: () => void
}) {
  const toast = useToast()
  const [error, setError] = useState('')

  return (
    <Modal open={open} onClose={onClose} title={`Graph \u2014 ${workflow.name}`} width="max-w-6xl">
      <div className="space-y-3">
        <ErrorLine error={error} />
        {open && (
          <FlowEditor
            initial={(workflow.graph as any) ?? { nodes: [], edges: [] }}
            nodeTypes={nodeTypes as any}
            onCancel={onClose}
            onSave={async (graph) => {
              try {
                await api.patch(`/workflows/${workflow.id}`, { graph })
                toast('Saved', 'success')
                onSaved()
                onClose()
              } catch (err) {
                setError((err as Error).message)
              }
            }}
          />
        )}
      </div>
    </Modal>
  )
}
