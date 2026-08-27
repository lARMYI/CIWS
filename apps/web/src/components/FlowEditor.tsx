/**
 * A direct-manipulation editor for a workflow graph.
 *
 * The canvas used to be a viewer, and graphs were edited as JSON beside it.
 * That was the right call at v0.1 -- a half-working node editor is worse than a
 * clear diagram next to the source of truth -- and it stops being the right call
 * once loops make workflows the primary surface.
 *
 * Two decisions worth knowing about:
 *
 * - **The server keeps deciding what is valid.** Every change re-validates
 *   against `/workflows/validate`. The editor makes malformed graphs harder to
 *   draw; it does not become a second, drifting copy of the rules.
 * - **JSON stays reachable.** Some things are still faster to type than to
 *   drag, and an editor that traps you is worse than one you can step out of.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, Check, Link2, Plus, Trash2, X } from 'lucide-react'

import { api } from '../lib/api'
import { useToast } from './ui'

export type NodeType = {
  type: string
  label: string
  category: string
  description: string
  inputs: string[]
  outputs: string[]
  config_schema: Record<string, unknown>
  color: string
  icon: string
}

type GraphNode = {
  id: string
  type: string
  label?: string
  config?: Record<string, unknown>
  position?: { x: number; y: number }
}

type GraphEdge = {
  source: string
  target: string
  source_output?: string
  target_input?: string
}

export type Graph = { nodes: GraphNode[]; edges: GraphEdge[] }

const NODE_W = 168
const NODE_H = 52
const GRID = 8

/** Snap to a grid so a hand-dragged graph still reads as deliberate. */
const snap = (value: number) => Math.round(value / GRID) * GRID

function freshId(type: string, existing: Set<string>): string {
  const base = type.replace(/[^a-z0-9]/gi, '').slice(0, 8) || 'node'
  let n = 1
  while (existing.has(`${base}${n}`)) n += 1
  return `${base}${n}`
}

/** Lay out any node that has never been positioned, by dependency depth. */
function withPositions(graph: Graph): Map<string, { x: number; y: number }> {
  const map = new Map<string, { x: number; y: number }>()
  const depth = new Map<string, number>()
  const incoming = new Map<string, string[]>()
  for (const node of graph.nodes) incoming.set(String(node.id), [])
  for (const edge of graph.edges) incoming.get(String(edge.target))?.push(String(edge.source))

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
  for (const node of graph.nodes) {
    const id = String(node.id)
    if (node.position && typeof node.position.x === 'number') {
      map.set(id, { x: node.position.x, y: node.position.y })
      continue
    }
    const level = compute(id)
    const index = perLevel.get(level) ?? 0
    perLevel.set(level, index + 1)
    map.set(id, { x: 40 + level * 210, y: 40 + index * 88 })
  }
  return map
}

export function FlowEditor({
  initial,
  nodeTypes,
  onSave,
  onCancel,
}: {
  initial: Graph
  nodeTypes: NodeType[]
  onSave: (graph: Graph) => Promise<void>
  onCancel: () => void
}) {
  const toast = useToast()
  const svgRef = useRef<SVGSVGElement | null>(null)

  const [graph, setGraph] = useState<Graph>(() => ({
    nodes: (initial?.nodes ?? []).map((n) => ({ ...n })),
    edges: (initial?.edges ?? []).map((e) => ({ ...e })),
  }))
  const [selected, setSelected] = useState<string | null>(null)
  const [problems, setProblems] = useState<string[]>([])
  const [linkFrom, setLinkFrom] = useState<{ id: string; output: string } | null>(null)
  const [drag, setDrag] = useState<{ id: string; dx: number; dy: number } | null>(null)
  const [saving, setSaving] = useState(false)
  const [showJson, setShowJson] = useState(false)
  const [jsonText, setJsonText] = useState('')

  const positions = useMemo(() => withPositions(graph), [graph])
  const specOf = useCallback(
    (type: string) => nodeTypes.find((n) => n.type === type),
    [nodeTypes],
  )

  // The server is the authority on validity; re-ask it on every change rather
  // than growing a second copy of the rules here.
  useEffect(() => {
    let cancelled = false
    const timer = window.setTimeout(async () => {
      try {
        const result = await api.post<{ ok: boolean; problems: string[] }>(
          '/workflows/validate',
          graph,
        )
        if (!cancelled) setProblems(result.problems ?? [])
      } catch {
        /* a validation hiccup must not block editing */
      }
    }, 250)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [graph])

  const bounds = useMemo(() => {
    let width = 640
    let height = 300
    positions.forEach((p) => {
      width = Math.max(width, p.x + NODE_W + 60)
      height = Math.max(height, p.y + NODE_H + 60)
    })
    return { width, height }
  }, [positions])

  const pointAt = useCallback((event: React.PointerEvent) => {
    const svg = svgRef.current
    if (!svg) return { x: 0, y: 0 }
    const rect = svg.getBoundingClientRect()
    return { x: event.clientX - rect.left, y: event.clientY - rect.top }
  }, [])

  const addNode = (type: string) => {
    setGraph((current) => {
      const ids = new Set(current.nodes.map((n) => String(n.id)))
      const id = freshId(type, ids)
      const spec = specOf(type)
      // Drop it below everything already placed. A fixed origin lands the new
      // node on top of an existing one, which reads as if nothing happened.
      const placed = withPositions(current)
      let lowest = 0
      placed.forEach((point) => {
        lowest = Math.max(lowest, point.y + NODE_H)
      })
      return {
        ...current,
        nodes: [
          ...current.nodes,
          {
            id,
            type,
            config: { ...((spec?.config_schema as Record<string, unknown>) ?? {}) },
            position: { x: 40, y: snap(lowest + 28) },
          },
        ],
      }
    })
    setSelected(null)
  }

  const removeNode = (id: string) => {
    setGraph((current) => ({
      nodes: current.nodes.filter((n) => String(n.id) !== id),
      // Edges to a node that no longer exists would be dangling; drop them
      // with it rather than leaving the user to find them in the problem list.
      edges: current.edges.filter(
        (e) => String(e.source) !== id && String(e.target) !== id,
      ),
    }))
    setSelected((s) => (s === id ? null : s))
  }

  const connect = (targetId: string) => {
    if (!linkFrom) return
    if (linkFrom.id === targetId) {
      toast('A node cannot feed itself', 'error')
      setLinkFrom(null)
      return
    }
    setGraph((current) => {
      const exists = current.edges.some(
        (e) =>
          String(e.source) === linkFrom.id &&
          String(e.target) === targetId &&
          (e.source_output ?? 'out') === linkFrom.output,
      )
      if (exists) return current
      return {
        ...current,
        edges: [
          ...current.edges,
          linkFrom.output === 'out'
            ? { source: linkFrom.id, target: targetId }
            : { source: linkFrom.id, target: targetId, source_output: linkFrom.output },
        ],
      }
    })
    setLinkFrom(null)
  }

  const patchNode = (id: string, patch: Partial<GraphNode>) => {
    setGraph((current) => ({
      ...current,
      nodes: current.nodes.map((n) => (String(n.id) === id ? { ...n, ...patch } : n)),
    }))
  }

  const selectedNode = graph.nodes.find((n) => String(n.id) === selected) ?? null
  const selectedSpec = selectedNode ? specOf(selectedNode.type) : undefined

  const byCategory = useMemo(() => {
    const groups = new Map<string, NodeType[]>()
    for (const spec of nodeTypes) {
      const list = groups.get(spec.category) ?? []
      list.push(spec)
      groups.set(spec.category, list)
    }
    return [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]))
  }, [nodeTypes])

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-2xs uppercase tracking-wider text-faint">add</span>
        {byCategory.map(([category, specs]) => (
          <div key={category} className="flex items-center gap-1">
            <span className="text-2xs text-faint">{category}</span>
            {specs.map((spec) => (
              <button
                key={spec.type}
                className="btn text-2xs"
                title={spec.description}
                onClick={() => addNode(spec.type)}
              >
                <Plus size={9} /> {spec.label}
              </button>
            ))}
          </div>
        ))}
      </div>

      {linkFrom && (
        <div className="flex items-center gap-2 rounded border border-cyan/40 bg-cyan/5 px-2 py-1 text-2xs text-cyan">
          <Link2 size={11} />
          Connecting from <span className="font-mono">{linkFrom.id}</span>
          {linkFrom.output !== 'out' && <span className="font-mono">·{linkFrom.output}</span>}
          — click a node to finish.
          <button className="ml-auto text-faint hover:text-ink" onClick={() => setLinkFrom(null)}>
            <X size={11} />
          </button>
        </div>
      )}

      <div className="flex gap-3">
        <div className="flex-1 overflow-auto rounded border border-line bg-panel">
          <svg
            ref={svgRef}
            width={bounds.width}
            height={bounds.height}
            className="min-w-full select-none"
            onPointerMove={(event) => {
              if (!drag) return
              const point = pointAt(event)
              patchNode(drag.id, {
                position: { x: snap(point.x - drag.dx), y: snap(point.y - drag.dy) },
              })
            }}
            onPointerUp={() => setDrag(null)}
            onPointerLeave={() => setDrag(null)}
          >
            <defs>
              <marker id="fe-arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto">
                <path d="M0,0 L0,6 L8,3 z" fill="#2c3d52" />
              </marker>
            </defs>

            {graph.edges.map((edge, index) => {
              const from = positions.get(String(edge.source))
              const to = positions.get(String(edge.target))
              if (!from || !to) return null
              const x1 = from.x + NODE_W
              const y1 = from.y + NODE_H / 2
              const x2 = to.x
              const y2 = to.y + NODE_H / 2
              const mid = (x1 + x2) / 2
              return (
                <g key={index}>
                  <path
                    d={`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`}
                    fill="none"
                    stroke="#2c3d52"
                    strokeWidth={1.2}
                    markerEnd="url(#fe-arrow)"
                  />
                  {/* A fat invisible hit area: a 1px curve is not a click target. */}
                  <path
                    d={`M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`}
                    fill="none"
                    stroke="transparent"
                    strokeWidth={12}
                    className="cursor-pointer"
                    onClick={() =>
                      setGraph((current) => ({
                        ...current,
                        edges: current.edges.filter((_, i) => i !== index),
                      }))
                    }
                  >
                    <title>Click to remove this connection</title>
                  </path>
                  {edge.source_output && edge.source_output !== 'out' && (
                    <text
                      x={mid}
                      y={(y1 + y2) / 2 - 4}
                      textAnchor="middle"
                      className="fill-faint"
                      style={{ fontSize: 9, fontFamily: 'ui-monospace, monospace' }}
                    >
                      {edge.source_output}
                    </text>
                  )}
                </g>
              )
            })}

            {graph.nodes.map((node) => {
              const id = String(node.id)
              const point = positions.get(id)!
              const spec = specOf(node.type)
              const isSelected = selected === id
              return (
                <g key={id} transform={`translate(${point.x},${point.y})`}>
                  <rect
                    width={NODE_W}
                    height={NODE_H}
                    rx={3}
                    fill="#111a22"
                    stroke={isSelected ? '#22d3ee' : (spec?.color ?? '#2c3d52')}
                    strokeWidth={isSelected ? 1.8 : 1.1}
                    className={linkFrom ? 'cursor-crosshair' : 'cursor-move'}
                    onPointerDown={(event) => {
                      if (linkFrom) return
                      event.currentTarget.setPointerCapture(event.pointerId)
                      const p = pointAt(event)
                      setSelected(id)
                      setDrag({ id, dx: p.x - point.x, dy: p.y - point.y })
                    }}
                    onClick={() => (linkFrom ? connect(id) : setSelected(id))}
                  />
                  <text
                    x={9}
                    y={20}
                    className="fill-ink pointer-events-none"
                    style={{ fontSize: 11, fontFamily: 'ui-monospace, monospace' }}
                  >
                    {id}
                  </text>
                  <text
                    x={9}
                    y={35}
                    className="fill-faint pointer-events-none"
                    style={{ fontSize: 9.5 }}
                  >
                    {spec?.label ?? node.type}
                  </text>

                  {/* Output ports: one per declared output, so a branch or a
                      loop can be wired from the right side of the node. */}
                  {(spec?.outputs ?? ['out']).map((output, index, all) => (
                    <circle
                      key={output}
                      cx={NODE_W}
                      cy={(NODE_H / (all.length + 1)) * (index + 1)}
                      r={4.5}
                      fill={linkFrom?.id === id && linkFrom.output === output ? '#22d3ee' : '#0b1218'}
                      stroke={spec?.color ?? '#2c3d52'}
                      strokeWidth={1.2}
                      className="cursor-crosshair"
                      onClick={(event) => {
                        event.stopPropagation()
                        setLinkFrom({ id, output })
                      }}
                    >
                      <title>{`Drag a connection from "${output}"`}</title>
                    </circle>
                  ))}
                  {(spec?.inputs ?? ['in']).length > 0 && (
                    <circle cx={0} cy={NODE_H / 2} r={4} fill="#0b1218" stroke="#2c3d52" />
                  )}
                </g>
              )
            })}
          </svg>
        </div>

        <aside className="w-64 shrink-0 space-y-3">
          {selectedNode ? (
            <div className="space-y-2 rounded border border-line bg-panel p-3">
              <div className="flex items-center justify-between">
                <span className="text-2xs uppercase tracking-wider text-faint">node</span>
                <button
                  className="text-rose hover:opacity-80"
                  title="Delete this node"
                  onClick={() => removeNode(String(selectedNode.id))}
                >
                  <Trash2 size={12} />
                </button>
              </div>

              <label className="block">
                <span className="text-2xs text-faint">id</span>
                <input
                  className="field font-mono text-xs"
                  value={String(selectedNode.id)}
                  onChange={(event) => {
                    const next = event.target.value.trim()
                    if (!next) return
                    const previous = String(selectedNode.id)
                    setGraph((current) => ({
                      nodes: current.nodes.map((n) =>
                        String(n.id) === previous ? { ...n, id: next } : n,
                      ),
                      // Renaming a node must carry its edges, or the graph
                      // silently loses every connection it had.
                      edges: current.edges.map((e) => ({
                        ...e,
                        source: String(e.source) === previous ? next : e.source,
                        target: String(e.target) === previous ? next : e.target,
                      })),
                    }))
                    setSelected(next)
                  }}
                />
              </label>

              <div className="text-2xs leading-relaxed text-faint">
                {selectedSpec?.description}
              </div>

              <div className="space-y-2">
                <span className="text-2xs uppercase tracking-wider text-faint">config</span>
                {Object.keys(selectedSpec?.config_schema ?? {}).length === 0 && (
                  <div className="text-2xs text-faint">This node takes no configuration.</div>
                )}
                {Object.entries(selectedSpec?.config_schema ?? {}).map(([key, sample]) => {
                  const value = (selectedNode.config ?? {})[key]
                  const isObject = sample !== null && typeof sample === 'object'
                  return (
                    <label key={key} className="block">
                      <span className="text-2xs text-dim">{key}</span>
                      {isObject ? (
                        <textarea
                          rows={3}
                          spellCheck={false}
                          className="field font-mono text-2xs resize-y"
                          value={
                            typeof value === 'string' ? value : JSON.stringify(value ?? sample, null, 1)
                          }
                          onChange={(event) => {
                            let parsed: unknown = event.target.value
                            try {
                              parsed = JSON.parse(event.target.value)
                            } catch {
                              /* keep the raw text while it is mid-edit */
                            }
                            patchNode(String(selectedNode.id), {
                              config: { ...(selectedNode.config ?? {}), [key]: parsed },
                            })
                          }}
                        />
                      ) : (
                        <input
                          className="field font-mono text-2xs"
                          value={String(value ?? '')}
                          placeholder={String(sample)}
                          onChange={(event) => {
                            const raw = event.target.value
                            const next =
                              typeof sample === 'number' && raw !== '' && !Number.isNaN(Number(raw))
                                ? Number(raw)
                                : typeof sample === 'boolean'
                                  ? raw === 'true'
                                  : raw
                            patchNode(String(selectedNode.id), {
                              config: { ...(selectedNode.config ?? {}), [key]: next },
                            })
                          }}
                        />
                      )}
                    </label>
                  )
                })}
              </div>
            </div>
          ) : (
            <div className="rounded border border-line bg-panel p-3 text-2xs leading-relaxed text-faint">
              Click a node to configure it. Drag to move. Click an output port, then a node, to
              connect them. Click a connection to remove it.
            </div>
          )}

          <div className="rounded border border-line bg-panel p-3">
            <span className="text-2xs uppercase tracking-wider text-faint">checks</span>
            {problems.length === 0 ? (
              <div className="mt-1 flex items-center gap-1.5 text-2xs text-green">
                <Check size={11} /> valid
              </div>
            ) : (
              <div className="mt-1 space-y-1">
                {problems.map((problem, index) => (
                  <div key={index} className="flex items-start gap-1.5 text-2xs text-amber">
                    <AlertTriangle size={10} className="mt-0.5 shrink-0" />
                    {problem}
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>

      {showJson && (
        <textarea
          rows={12}
          spellCheck={false}
          className="field font-mono text-xs resize-y"
          value={jsonText}
          onChange={(event) => setJsonText(event.target.value)}
          onBlur={() => {
            try {
              setGraph(JSON.parse(jsonText))
            } catch {
              toast('That is not valid JSON', 'error')
            }
          }}
        />
      )}

      <div className="flex items-center justify-end gap-2">
        <button
          className="btn"
          onClick={() => {
            setJsonText(JSON.stringify(graph, null, 2))
            setShowJson((v) => !v)
          }}
        >
          {showJson ? 'hide JSON' : 'edit as JSON'}
        </button>
        <button className="btn" onClick={onCancel}>
          Cancel
        </button>
        <button
          className="btn btn-primary"
          disabled={saving || problems.length > 0}
          title={problems.length > 0 ? 'Fix the problems listed before saving' : undefined}
          onClick={async () => {
            setSaving(true)
            try {
              await onSave(graph)
            } finally {
              setSaving(false)
            }
          }}
        >
          {saving ? 'Saving…' : 'Save graph'}
        </button>
      </div>
    </div>
  )
}
