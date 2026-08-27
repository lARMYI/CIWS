/**
 * The ontology panel: a force-directed entity graph on a canvas.
 *
 * Canvas rather than SVG because a few hundred nodes with per-frame position
 * updates means a few hundred DOM mutations per frame in SVG, and the browser
 * gives up somewhere around 400. Canvas redraws the whole scene in one pass and
 * stays smooth into the low thousands.
 *
 * The simulation is deliberately hand-rolled and small: repulsion between all
 * pairs, springs along edges, a weak pull to centre, and velocity damping. It
 * cools to a stop rather than jittering forever, and any drag reheats it.
 */

import {
  Crosshair,
  ArrowLeftRight,
  GitMerge,
  Layers,
  Maximize2,
  RotateCw,
  Route,
  Search,
  Trash2,
  X,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  Badge,
  Empty,
  ErrorLine,
  Panel,
  Spinner,
  classNames,
  useAsync,
  useSearch,
  useToast,
} from '../components/ui'
import { api, type GraphEdge, type GraphNode } from '../lib/api'

interface Simulated extends GraphNode {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  fixed?: boolean
}

interface GraphPayload {
  nodes: GraphNode[]
  edges: GraphEdge[]
  truncated: boolean
  stats: {
    entities: number
    edges: number
    by_type: Record<string, number>
    by_edge_type: Record<string, number>
    density: number
  }
}

const REPULSION = 5200
const SPRING = 0.014
const SPRING_LENGTH = 92
const CENTER_PULL = 0.0022
const DAMPING = 0.86
const COOL_RATE = 0.994
const MIN_ALPHA = 0.004

type DuplicatePair = {
  a: GraphNode
  b: GraphNode
  similarity: number
  reason: string
}

export function Ontology({ projectId }: { projectId: string | null }) {
  const toast = useToast()
  const [typeFilter, setTypeFilter] = useState<string>('')
  const [selected, setSelected] = useState<string | null>(null)
  const [pathFrom, setPathFrom] = useState<string | null>(null)
  const [pathResult, setPathResult] = useState<{ nodes: GraphNode[]; edges: GraphEdge[]; hops: number } | null>(null)
  const [query, setQuery, debounced] = useSearch('')

  const graph = useAsync(
    () => api.get<GraphPayload>('/graph', { project_id: projectId, types: typeFilter, limit: 600 }),
    [projectId, typeFilter],
  )
  const searchResults = useAsync(
    async () =>
      debounced.trim()
        ? api.get<{ entities: GraphNode[] }>('/graph/search', { q: debounced, limit: 20 })
        : { entities: [] },
    [debounced],
  )

  const canvas = useRef<HTMLCanvasElement>(null)
  const wrapper = useRef<HTMLDivElement>(null)
  const nodes = useRef<Map<string, Simulated>>(new Map())
  const edgesRef = useRef<GraphEdge[]>([])
  const view = useRef({ x: 0, y: 0, scale: 1 })
  const alpha = useRef(1)
  const hovered = useRef<string | null>(null)
  const dragging = useRef<{ id: string | null; panning: boolean; lastX: number; lastY: number }>({
    id: null,
    panning: false,
    lastX: 0,
    lastY: 0,
  })
  const selectedRef = useRef<string | null>(null)
  const highlightRef = useRef<Set<string>>(new Set())
  const frame = useRef(0)

  useEffect(() => {
    selectedRef.current = selected
  }, [selected])

  // Seed the simulation whenever the data changes, preserving positions for
  // nodes that were already on screen so the layout does not jump.
  useEffect(() => {
    const payload = graph.data
    if (!payload) return
    const next = new Map<string, Simulated>()
    const count = payload.nodes.length || 1
    payload.nodes.forEach((node, index) => {
      const existing = nodes.current.get(node.id)
      // Golden-angle spiral: an even initial spread, so the simulation starts
      // from something reasonable instead of a random clump.
      const angle = index * 2.399963
      const radius = 26 * Math.sqrt(index + 1) * (1 + 220 / count)
      next.set(node.id, {
        ...node,
        x: existing?.x ?? Math.cos(angle) * radius,
        y: existing?.y ?? Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
        radius: 5 + Math.min(11, Math.sqrt(node.degree + 1) * 2.6),
      })
    })
    nodes.current = next
    edgesRef.current = payload.edges
    alpha.current = 1
  }, [graph.data])

  useEffect(() => {
    const ids = new Set<string>()
    if (selected) {
      ids.add(selected)
      for (const edge of edgesRef.current) {
        if (edge.source === selected) ids.add(edge.target)
        if (edge.target === selected) ids.add(edge.source)
      }
    }
    highlightRef.current = ids
  }, [selected])

  const tick = useCallback(() => {
    const list = [...nodes.current.values()]
    if (alpha.current > MIN_ALPHA && list.length) {
      const a = alpha.current

      // All-pairs repulsion. O(n^2) is fine to ~800 nodes, and the API caps the
      // result set below that; a quadtree here would be complexity without payoff.
      for (let i = 0; i < list.length; i++) {
        const p = list[i]
        for (let j = i + 1; j < list.length; j++) {
          const q = list[j]
          let dx = q.x - p.x
          let dy = q.y - p.y
          let distanceSq = dx * dx + dy * dy
          if (distanceSq < 0.01) {
            // Perfectly coincident nodes produce NaN; nudge them apart.
            dx = (Math.random() - 0.5) * 2
            dy = (Math.random() - 0.5) * 2
            distanceSq = dx * dx + dy * dy
          }
          if (distanceSq > 360_000) continue // ignore distant pairs
          const distance = Math.sqrt(distanceSq)
          const force = (REPULSION * a) / distanceSq
          const fx = (dx / distance) * force
          const fy = (dy / distance) * force
          p.vx -= fx
          p.vy -= fy
          q.vx += fx
          q.vy += fy
        }
      }

      for (const edge of edgesRef.current) {
        const source = nodes.current.get(edge.source)
        const target = nodes.current.get(edge.target)
        if (!source || !target) continue
        const dx = target.x - source.x
        const dy = target.y - source.y
        const distance = Math.hypot(dx, dy) || 1
        const force = (distance - SPRING_LENGTH) * SPRING * a * Math.min(2.2, edge.weight)
        const fx = (dx / distance) * force
        const fy = (dy / distance) * force
        source.vx += fx
        source.vy += fy
        target.vx -= fx
        target.vy -= fy
      }

      for (const node of list) {
        node.vx -= node.x * CENTER_PULL * a
        node.vy -= node.y * CENTER_PULL * a
        if (node.fixed) {
          node.vx = 0
          node.vy = 0
          continue
        }
        node.vx *= DAMPING
        node.vy *= DAMPING
        node.x += Math.max(-24, Math.min(24, node.vx))
        node.y += Math.max(-24, Math.min(24, node.vy))
      }
      alpha.current *= COOL_RATE
    }
    draw()
    frame.current = requestAnimationFrame(tick)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const draw = useCallback(() => {
    const element = canvas.current
    const context = element?.getContext('2d')
    if (!element || !context) return

    const ratio = window.devicePixelRatio || 1
    const width = element.clientWidth
    const height = element.clientHeight
    if (element.width !== width * ratio || element.height !== height * ratio) {
      element.width = width * ratio
      element.height = height * ratio
    }

    context.setTransform(ratio, 0, 0, ratio, 0, 0)
    context.clearRect(0, 0, width, height)
    context.save()
    context.translate(width / 2 + view.current.x, height / 2 + view.current.y)
    context.scale(view.current.scale, view.current.scale)

    const highlight = highlightRef.current
    const dimAll = highlight.size > 0
    const pathIds = new Set(pathResult?.nodes.map((n) => n.id) ?? [])

    // Edges first, so nodes sit on top of them.
    context.lineCap = 'round'
    for (const edge of edgesRef.current) {
      const source = nodes.current.get(edge.source)
      const target = nodes.current.get(edge.target)
      if (!source || !target) continue
      const inPath = pathIds.has(edge.source) && pathIds.has(edge.target)
      const active = !dimAll || (highlight.has(edge.source) && highlight.has(edge.target))
      context.globalAlpha = inPath ? 0.95 : active ? 0.42 : 0.07
      context.strokeStyle = inPath ? '#fbbf24' : edge.color
      context.lineWidth = (inPath ? 2.2 : 1) / view.current.scale + Math.min(1.4, edge.weight * 0.18)
      context.beginPath()
      context.moveTo(source.x, source.y)
      context.lineTo(target.x, target.y)
      context.stroke()

      if (edge.directed && (active || inPath)) {
        const angle = Math.atan2(target.y - source.y, target.x - source.x)
        const tipX = target.x - Math.cos(angle) * (target.radius + 2.5)
        const tipY = target.y - Math.sin(angle) * (target.radius + 2.5)
        const size = 5 / view.current.scale + 1.5
        context.beginPath()
        context.moveTo(tipX, tipY)
        context.lineTo(
          tipX - Math.cos(angle - 0.42) * size,
          tipY - Math.sin(angle - 0.42) * size,
        )
        context.lineTo(
          tipX - Math.cos(angle + 0.42) * size,
          tipY - Math.sin(angle + 0.42) * size,
        )
        context.closePath()
        context.fillStyle = inPath ? '#fbbf24' : edge.color
        context.fill()
      }
    }

    const labelThreshold = view.current.scale > 0.62
    for (const node of nodes.current.values()) {
      const isSelected = node.id === selectedRef.current
      const isHovered = node.id === hovered.current
      const inPath = pathIds.has(node.id)
      const active = !dimAll || highlight.has(node.id)
      context.globalAlpha = inPath ? 1 : active ? 1 : 0.16

      if (isSelected || isHovered || inPath) {
        context.beginPath()
        context.arc(node.x, node.y, node.radius + 6, 0, Math.PI * 2)
        context.fillStyle = inPath ? 'rgba(251,191,36,0.16)' : 'rgba(34,211,238,0.14)'
        context.fill()
      }

      context.beginPath()
      context.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
      context.fillStyle = node.color
      context.fill()
      context.lineWidth = (isSelected ? 2.4 : 1.2) / view.current.scale
      context.strokeStyle = isSelected ? '#e2f4ff' : inPath ? '#fbbf24' : '#05070a'
      context.stroke()

      if (labelThreshold && (active || inPath)) {
        const label = node.name.length > 26 ? `${node.name.slice(0, 25)}…` : node.name
        context.font = `${isSelected ? 600 : 400} ${11 / view.current.scale}px ui-monospace, monospace`
        context.textAlign = 'center'
        context.textBaseline = 'top'
        context.fillStyle = isSelected ? '#e2f4ff' : '#8296ad'
        context.fillText(label, node.x, node.y + node.radius + 4 / view.current.scale)
      }
    }

    context.restore()
    context.globalAlpha = 1
  }, [pathResult])

  useEffect(() => {
    frame.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame.current)
  }, [tick])

  const toWorld = useCallback((clientX: number, clientY: number) => {
    const element = canvas.current!
    const rect = element.getBoundingClientRect()
    return {
      x: (clientX - rect.left - rect.width / 2 - view.current.x) / view.current.scale,
      y: (clientY - rect.top - rect.height / 2 - view.current.y) / view.current.scale,
    }
  }, [])

  const nodeAt = useCallback(
    (clientX: number, clientY: number): Simulated | null => {
      const point = toWorld(clientX, clientY)
      let best: Simulated | null = null
      let bestDistance = Infinity
      for (const node of nodes.current.values()) {
        const distance = Math.hypot(node.x - point.x, node.y - point.y)
        if (distance < node.radius + 7 && distance < bestDistance) {
          best = node
          bestDistance = distance
        }
      }
      return best
    },
    [toWorld],
  )

  const focus = useCallback((id: string) => {
    const node = nodes.current.get(id)
    if (!node) return
    view.current.x = -node.x * view.current.scale
    view.current.y = -node.y * view.current.scale
    setSelected(id)
  }, [])

  const fit = useCallback(() => {
    const list = [...nodes.current.values()]
    if (!list.length || !canvas.current) return
    const xs = list.map((n) => n.x)
    const ys = list.map((n) => n.y)
    const width = Math.max(140, Math.max(...xs) - Math.min(...xs))
    const height = Math.max(140, Math.max(...ys) - Math.min(...ys))
    const scale = Math.min(
      2.2,
      Math.max(0.14, Math.min(canvas.current.clientWidth / (width + 130), canvas.current.clientHeight / (height + 130))),
    )
    view.current.scale = scale
    view.current.x = -((Math.min(...xs) + Math.max(...xs)) / 2) * scale
    view.current.y = -((Math.min(...ys) + Math.max(...ys)) / 2) * scale
  }, [])

  const selectedNode = selected ? nodes.current.get(selected) : null
  const stats = graph.data?.stats

  const findPath = useCallback(async () => {
    if (!pathFrom || !selected || pathFrom === selected) return
    try {
      const result = await api.get<{ found: boolean; nodes: GraphNode[]; edges: GraphEdge[]; hops: number }>(
        '/graph/path',
        { source: pathFrom, target: selected },
      )
      if (!result.found) {
        toast('No path between those entities', 'error')
        setPathResult(null)
      } else {
        setPathResult(result)
        toast(`Path found: ${result.hops} hops`, 'success')
      }
    } catch (err) {
      toast((err as Error).message, 'error')
    }
    setPathFrom(null)
  }, [pathFrom, selected, toast])

  const typeCounts = useMemo(() => Object.entries(stats?.by_type ?? {}).sort((a, b) => b[1] - a[1]), [stats])

  return (
    <div className="h-full grid grid-cols-[1fr_290px] gap-2 min-h-0 max-[1000px]:grid-cols-1">
      <Panel
        title="Ontology"
        subtitle={
          stats ? `${stats.entities} entities · ${stats.edges} links · density ${stats.density}` : ''
        }
        bodyClass="relative"
        actions={
          <>
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="bg-void border border-line rounded px-1.5 py-1 text-2xs text-dim outline-none focus:border-cyan/50 cursor-pointer"
            >
              <option value="">all types</option>
              {typeCounts.map(([type, count]) => (
                <option key={type} value={type}>
                  {type} ({count})
                </option>
              ))}
            </select>
            <button className="btn btn-ghost !p-1" onClick={fit} title="Fit to view">
              <Maximize2 size={12} />
            </button>
            <button
              className="btn btn-ghost !p-1"
              onClick={() => {
                alpha.current = 1
              }}
              title="Re-run layout"
            >
              <RotateCw size={12} />
            </button>
          </>
        }
      >
        <div ref={wrapper} className="absolute inset-0">
          <canvas
            ref={canvas}
            className="w-full h-full cursor-grab active:cursor-grabbing"
            onMouseDown={(e) => {
              const hit = nodeAt(e.clientX, e.clientY)
              if (hit) {
                dragging.current = { id: hit.id, panning: false, lastX: e.clientX, lastY: e.clientY }
                hit.fixed = true
                setSelected(hit.id)
              } else {
                dragging.current = { id: null, panning: true, lastX: e.clientX, lastY: e.clientY }
              }
            }}
            onMouseMove={(e) => {
              const state = dragging.current
              if (state.id) {
                const node = nodes.current.get(state.id)
                if (node) {
                  const point = toWorld(e.clientX, e.clientY)
                  node.x = point.x
                  node.y = point.y
                  alpha.current = Math.max(alpha.current, 0.32)
                }
              } else if (state.panning) {
                view.current.x += e.clientX - state.lastX
                view.current.y += e.clientY - state.lastY
                state.lastX = e.clientX
                state.lastY = e.clientY
              } else {
                const hit = nodeAt(e.clientX, e.clientY)
                hovered.current = hit?.id ?? null
                e.currentTarget.style.cursor = hit ? 'pointer' : 'grab'
              }
            }}
            onMouseUp={() => {
              if (dragging.current.id) {
                const node = nodes.current.get(dragging.current.id)
                if (node) node.fixed = false
              }
              dragging.current = { id: null, panning: false, lastX: 0, lastY: 0 }
            }}
            onMouseLeave={() => {
              dragging.current = { id: null, panning: false, lastX: 0, lastY: 0 }
              hovered.current = null
            }}
            onWheel={(e) => {
              // Zoom toward the cursor, not the centre -- otherwise the thing
              // you are aiming at slides away as you zoom in.
              const rect = e.currentTarget.getBoundingClientRect()
              const px = e.clientX - rect.left - rect.width / 2
              const py = e.clientY - rect.top - rect.height / 2
              const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12
              const next = Math.max(0.1, Math.min(4, view.current.scale * factor))
              const applied = next / view.current.scale
              view.current.x = px - (px - view.current.x) * applied
              view.current.y = py - (py - view.current.y) * applied
              view.current.scale = next
            }}
          />

          {graph.loading && !graph.data && (
            <div className="absolute inset-0 grid place-items-center text-faint text-xs">
              <span className="flex items-center gap-2">
                <Spinner /> building graph
              </span>
            </div>
          )}

          {!graph.loading && (graph.data?.nodes.length ?? 0) === 0 && (
            <div className="absolute inset-0">
              <Empty
                icon={Layers}
                title="The graph is empty"
                hint="Entities appear as agents work, or when you ingest documents. Ask the analyst to extract the entities from something you have written."
              />
            </div>
          )}

          {/* Search overlay */}
          <div className="absolute top-2.5 left-2.5 w-64">
            <div className="relative">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-faint" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Find an entity"
                className="field !pl-7 !py-1.5 text-xs bg-panel/95 backdrop-blur"
              />
              {query && (
                <button
                  onClick={() => setQuery('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-faint hover:text-ink"
                >
                  <X size={11} />
                </button>
              )}
            </div>
            {debounced && (searchResults.data?.entities.length ?? 0) > 0 && (
              <div className="mt-1 panel max-h-60 overflow-y-auto scroll bg-panel/97 backdrop-blur">
                {searchResults.data!.entities.map((entity) => (
                  <button
                    key={entity.id}
                    onClick={() => {
                      focus(entity.id)
                      setQuery('')
                    }}
                    className="w-full text-left px-2.5 py-1.5 hover:bg-raised border-b border-line/50 last:border-0"
                  >
                    <div className="flex items-center gap-1.5">
                      <span
                        className="w-1.5 h-1.5 rounded-full shrink-0"
                        style={{ background: entity.color }}
                      />
                      <span className="text-xs text-ink truncate">{entity.name}</span>
                    </div>
                    <div className="text-2xs text-faint pl-3">{entity.type_label}</div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Legend */}
          {typeCounts.length > 0 && (
            <div className="absolute bottom-2.5 left-2.5 flex flex-wrap gap-1 max-w-[60%]">
              {typeCounts.slice(0, 9).map(([type, count]) => (
                <button
                  key={type}
                  onClick={() => setTypeFilter(typeFilter === type ? '' : type)}
                  className={classNames(
                    'chip transition-colors',
                    typeFilter === type ? 'border-cyan/50 text-cyan' : 'hover:border-line2',
                  )}
                >
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ background: colorFor(type, graph.data?.nodes ?? []) }}
                  />
                  {type} <span className="text-faint">{count}</span>
                </button>
              ))}
            </div>
          )}

          {graph.data?.truncated && (
            <div className="absolute bottom-2.5 right-2.5 chip border-amber/40 text-amber">
              showing first 600 — filter by type to narrow
            </div>
          )}
        </div>
      </Panel>

      <Panel
        title={selectedNode ? 'Entity' : 'Inspector'}
        bodyClass="overflow-y-auto scroll"
        actions={
          selectedNode && (
            <button className="btn btn-ghost !p-1" onClick={() => setSelected(null)}>
              <X size={12} />
            </button>
          )
        }
      >
        {graph.error && (
          <div className="p-3">
            <ErrorLine error={graph.error} />
          </div>
        )}
        {selectedNode ? (
          <EntityDetail
            node={selectedNode}
            edges={edgesRef.current}
            allNodes={nodes.current}
            pathFrom={pathFrom}
            onFocus={focus}
            onStartPath={() => setPathFrom(selectedNode.id)}
            onFindPath={findPath}
            onCancelPath={() => {
              setPathFrom(null)
              setPathResult(null)
            }}
            onDeleted={() => {
              setSelected(null)
              graph.reload()
            }}
          />
        ) : (
          <div className="p-3 space-y-3">
            <div className="text-xs text-faint leading-relaxed">
              Click any node to inspect it. Drag to reposition, scroll to zoom, drag the background
              to pan.
            </div>
            <Duplicates onMerged={graph.reload} />
            <Central projectId={projectId} onFocus={focus} />
          </div>
        )}
      </Panel>
    </div>
  )
}

function colorFor(type: string, nodes: GraphNode[]): string {
  return nodes.find((n) => n.type === type)?.color ?? '#64748b'
}

function EntityDetail({
  node,
  edges,
  allNodes,
  pathFrom,
  onFocus,
  onStartPath,
  onFindPath,
  onCancelPath,
  onDeleted,
}: {
  node: Simulated
  edges: GraphEdge[]
  allNodes: Map<string, Simulated>
  pathFrom: string | null
  onFocus: (id: string) => void
  onStartPath: () => void
  onFindPath: () => void
  onCancelPath: () => void
  onDeleted: () => void
}) {
  const connections = useMemo(
    () =>
      edges
        .filter((e) => e.source === node.id || e.target === node.id)
        .map((e) => {
          const otherId = e.source === node.id ? e.target : e.source
          return { edge: e, other: allNodes.get(otherId), outgoing: e.source === node.id }
        })
        .filter((c) => c.other),
    [edges, node.id, allNodes],
  )

  return (
    <div className="p-3 space-y-3">
      <div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: node.color }} />
          <span className="text-md text-ink font-semibold leading-tight break-words">{node.name}</span>
        </div>
        <div className="flex flex-wrap gap-1 mt-1.5">
          <Badge tone="cyan">{node.type_label}</Badge>
          <Badge>{node.degree} links</Badge>
          <Badge>seen {node.mention_count}x</Badge>
        </div>
      </div>

      {node.description && (
        <p className="text-xs text-dim leading-relaxed break-words">{node.description}</p>
      )}

      {node.aliases.length > 0 && (
        <div>
          <div className="label mb-1">Also known as</div>
          <div className="flex flex-wrap gap-1">
            {node.aliases.map((alias) => (
              <Badge key={alias}>{alias}</Badge>
            ))}
          </div>
        </div>
      )}

      {Object.keys(node.properties ?? {}).length > 0 && (
        <div>
          <div className="label mb-1">Properties</div>
          <div className="surface divide-y divide-line/60">
            {Object.entries(node.properties).map(([key, value]) => (
              <div key={key} className="grid grid-cols-[80px_1fr] gap-2 px-2 py-1.5">
                <span className="text-2xs text-faint truncate">{key}</span>
                <span className="text-2xs text-dim break-words">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-1.5">
        {pathFrom === node.id ? (
          <button className="btn flex-1 text-amber border-amber/40" onClick={onCancelPath}>
            <Route size={11} /> pick a target
          </button>
        ) : pathFrom ? (
          <button className="btn btn-primary flex-1" onClick={onFindPath}>
            <Route size={11} /> path to here
          </button>
        ) : (
          <button className="btn flex-1" onClick={onStartPath}>
            <Route size={11} /> trace path
          </button>
        )}
        <button
          className="btn btn-danger !px-2"
          title="Delete entity"
          onClick={async () => {
            await api.del(`/graph/entity/${node.id}`)
            onDeleted()
          }}
        >
          <Trash2 size={11} />
        </button>
      </div>

      <div>
        <div className="label mb-1">Connections ({connections.length})</div>
        {connections.length === 0 ? (
          <div className="text-2xs text-faint">Nothing links to this yet.</div>
        ) : (
          <div className="surface divide-y divide-line/60 max-h-72 overflow-y-auto scroll">
            {connections.map(({ edge, other, outgoing }) => (
              <button
                key={edge.id}
                onClick={() => onFocus(other!.id)}
                className="w-full text-left px-2 py-1.5 hover:bg-raised transition-colors"
              >
                <div className="text-2xs" style={{ color: edge.color }}>
                  {outgoing ? '→' : '←'} {edge.label}
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span
                    className="w-1.5 h-1.5 rounded-full shrink-0"
                    style={{ background: other!.color }}
                  />
                  <span className="text-xs text-dim truncate">{other!.name}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Duplicates({ onMerged }: { onMerged: () => void }) {
  const toast = useToast()
  const [threshold, setThreshold] = useState(0.9)
  const [swapped, setSwapped] = useState<Record<string, boolean>>({})
  const [busy, setBusy] = useState<string | null>(null)

  const duplicates = useAsync(
    () => api.get<{ candidates: DuplicatePair[] }>('/graph/duplicates', { threshold }),
    [threshold],
  )
  const candidates = duplicates.data?.candidates ?? []

  const keyOf = (pair: DuplicatePair) => `${pair.a.id}:${pair.b.id}`

  return (
    <div>
      <div className="label mb-1 flex items-center gap-1.5">
        <GitMerge size={10} /> Curate duplicates
        {candidates.length > 0 && <span className="text-faint">({candidates.length})</span>}
      </div>

      <label className="mb-2 flex items-center gap-2 text-2xs text-faint">
        similarity
        <input
          type="range"
          min={0.7}
          max={0.99}
          step={0.01}
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
          className="flex-1 accent-cyan"
        />
        <span className="font-mono text-dim tabular-nums">{threshold.toFixed(2)}</span>
      </label>

      {duplicates.loading && <Spinner />}

      {!duplicates.loading && candidates.length === 0 && (
        <div className="text-2xs text-faint">
          Nothing above this similarity. Lower the threshold to look harder.
        </div>
      )}

      <div className="space-y-1.5">
        {candidates.slice(0, 12).map((pair) => {
          const key = keyOf(pair)
          const flipped = swapped[key] ?? false
          // The survivor keeps its own name and description; everything on the
          // other side folds into it as an alias. Which one survives is a
          // judgement the person curating has to be able to make.
          const keep = flipped ? pair.b : pair.a
          const drop = flipped ? pair.a : pair.b
          return (
            <div key={key} className="surface p-2">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-2xs text-dim">
                    <span className="text-green">keep</span> {keep.name}
                    {keep.mention_count > 0 && (
                      <span className="text-faint"> · {keep.mention_count} mentions</span>
                    )}
                  </div>
                  <div className="truncate text-2xs text-faint">
                    <span className="text-rose">fold</span> {drop.name}
                    {drop.mention_count > 0 && (
                      <span> · {drop.mention_count} mentions</span>
                    )}
                  </div>
                </div>
                <button
                  className="btn !py-0.5 !px-1.5 !text-2xs"
                  title="Swap which one survives"
                  onClick={() => setSwapped((s) => ({ ...s, [key]: !flipped }))}
                >
                  <ArrowLeftRight size={9} />
                </button>
              </div>

              <div className="mt-1.5 flex items-center justify-between gap-2">
                <span className="truncate text-2xs text-faint">
                  {pair.reason} · {(pair.similarity * 100).toFixed(0)}%
                </span>
                <div className="flex gap-1">
                  <button
                    className="btn !py-0.5 !px-1.5 !text-2xs"
                    disabled={busy === key}
                    title="These are genuinely different things"
                    onClick={async () => {
                      setBusy(key)
                      try {
                        await api.post('/graph/duplicates/dismiss', {
                          a_id: pair.a.id,
                          b_id: pair.b.id,
                        })
                        toast('Marked as different', 'success')
                        duplicates.reload()
                      } catch (err) {
                        toast((err as Error).message, 'error')
                      } finally {
                        setBusy(null)
                      }
                    }}
                  >
                    not a match
                  </button>
                  <button
                    className="btn btn-primary !py-0.5 !px-1.5 !text-2xs"
                    disabled={busy === key}
                    onClick={async () => {
                      setBusy(key)
                      try {
                        await api.post('/graph/merge', {
                          keep_id: keep.id,
                          merge_ids: [drop.id],
                        })
                        toast(`Merged into ${keep.name}`, 'success')
                        duplicates.reload()
                        onMerged()
                      } catch (err) {
                        toast((err as Error).message, 'error')
                      } finally {
                        setBusy(null)
                      }
                    }}
                  >
                    merge
                  </button>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Central({
  projectId,
  onFocus,
}: {
  projectId: string | null
  onFocus: (id: string) => void
}) {
  const central = useAsync(
    () => api.get<{ rows: any[] }>('/graph/centrality', { project_id: projectId, limit: 10 }),
    [projectId],
  )
  const rows = central.data?.rows ?? []
  if (!rows.length) return null

  return (
    <div>
      <div className="label mb-1 flex items-center gap-1.5">
        <Crosshair size={10} /> Most central
      </div>
      <div className="surface divide-y divide-line/60">
        {rows.map((row) => (
          <button
            key={row.entity_id}
            onClick={() => onFocus(row.entity_id)}
            className="w-full grid grid-cols-[1fr_auto] gap-2 items-center px-2 py-1.5 hover:bg-raised text-left"
          >
            <span className="flex items-center gap-1.5 min-w-0">
              <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: row.color }} />
              <span className="text-xs text-dim truncate">{row.name}</span>
            </span>
            <span className="text-2xs text-faint tabular-nums">{row.degree}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
