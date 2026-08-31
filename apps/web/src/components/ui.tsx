/** Shared primitives. Deliberately small: the panels do the interesting work. */

import { AlertTriangle, Check, ChevronDown, Loader2, X, type LucideIcon } from 'lucide-react'
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = '',
  bodyClass = '',
}: {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
  bodyClass?: string
}) {
  return (
    <section className={`flex flex-col min-h-0 panel ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 px-3 h-9 shrink-0 border-b border-line">
          <div className="flex items-baseline gap-2 min-w-0">
            {title && <span className="label truncate">{title}</span>}
            {subtitle && <span className="text-2xs text-faint truncate">{subtitle}</span>}
          </div>
          {actions && <div className="flex items-center gap-1.5 shrink-0">{actions}</div>}
        </header>
      )}
      <div className={`flex-1 min-h-0 ${bodyClass}`}>{children}</div>
    </section>
  )
}

export function Stat({
  label,
  value,
  hint,
  tone = 'default',
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: 'default' | 'cyan' | 'amber' | 'jade' | 'rose'
}) {
  const tones = {
    default: 'text-ink',
    cyan: 'text-cyan',
    amber: 'text-amber',
    jade: 'text-jade',
    rose: 'text-rose',
  }
  return (
    <div className="surface px-3 py-2.5 min-w-0">
      <div className="label truncate">{label}</div>
      <div className={`text-md font-semibold tabular-nums mt-0.5 ${tones[tone]}`}>{value}</div>
      {hint && <div className="text-2xs text-faint truncate mt-0.5">{hint}</div>}
    </div>
  )
}

export function Empty({
  icon: Icon,
  title,
  hint,
  action,
}: {
  // lucide icons are forwardRef components whose `size` accepts string | number,
  // which does not unify with a narrower local signature. LucideIcon is the type
  // the library exports for exactly this.
  icon?: LucideIcon
  title: string
  hint?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="h-full grid place-items-center p-8 text-center">
      <div className="max-w-sm">
        {Icon && <Icon size={26} className="mx-auto mb-3 text-line2" />}
        <div className="text-dim text-sm">{title}</div>
        {hint && <div className="text-xs text-faint mt-1.5 leading-relaxed">{hint}</div>}
        {action && <div className="mt-4 flex justify-center">{action}</div>}
      </div>
    </div>
  )
}

export function Spinner({ size = 13, className = '' }: { size?: number; className?: string }) {
  return <Loader2 size={size} className={`animate-spin ${className}`} />
}

export function Badge({
  children,
  tone = 'default',
  title,
}: {
  children: ReactNode
  tone?: 'default' | 'cyan' | 'amber' | 'jade' | 'rose' | 'violet'
  title?: string
}) {
  const tones = {
    default: 'border-line text-faint',
    cyan: 'border-cyan/40 text-cyan bg-cyan/5',
    amber: 'border-amber/40 text-amber bg-amber/5',
    jade: 'border-jade/40 text-jade bg-jade/5',
    rose: 'border-rose/40 text-rose bg-rose/5',
    violet: 'border-violet/40 text-violet bg-violet/5',
  }
  return (
    <span title={title} className={`chip ${tones[tone]}`}>
      {children}
    </span>
  )
}

export function StatusDot({ status }: { status: string }) {
  const map: Record<string, string> = {
    ready: 'bg-jade', ok: 'bg-jade', completed: 'bg-jade', done: 'bg-jade',
    running: 'bg-cyan animate-pulse-slow', starting: 'bg-cyan animate-pulse-slow',
    queued: 'bg-amber', pending: 'bg-amber', embedding: 'bg-amber', chunking: 'bg-amber',
    extracting: 'bg-amber', awaiting_approval: 'bg-amber animate-pulse-slow',
    error: 'bg-rose', failed: 'bg-rose',
    stopped: 'bg-line2', disabled: 'bg-line2', cancelled: 'bg-line2', skipped: 'bg-line2',
  }
  return (
    <span
      title={status}
      className={`inline-block w-1.5 h-1.5 rounded-full shrink-0 ${map[status] ?? 'bg-line2'}`}
    />
  )
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

export function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: ReactNode
  children: ReactNode
}) {
  return (
    <label className="block">
      <div className="label mb-1">{label}</div>
      {children}
      {hint && <div className="text-2xs text-faint mt-1 leading-relaxed">{hint}</div>}
    </label>
  )
}

export function Select<T extends string>({
  value,
  onChange,
  options,
  className = '',
}: {
  value: T
  onChange: (value: T) => void
  options: { value: T; label: string; disabled?: boolean }[]
  className?: string
}) {
  return (
    <div className={`relative ${className}`}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="field appearance-none pr-7 cursor-pointer"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </option>
        ))}
      </select>
      <ChevronDown
        size={13}
        className="absolute right-2 top-1/2 -translate-y-1/2 text-faint pointer-events-none"
      />
    </div>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  label: string
  hint?: string
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex items-start gap-2.5 w-full text-left group py-1"
    >
      <span
        className={`mt-0.5 w-7 h-4 rounded-full border shrink-0 relative transition-colors ${
          checked ? 'bg-cyan/25 border-cyan/50' : 'bg-void border-line'
        }`}
      >
        <span
          className={`absolute top-0.5 w-2.5 h-2.5 rounded-full transition-all ${
            checked ? 'left-3.5 bg-cyan' : 'left-0.5 bg-faint'
          }`}
        />
      </span>
      <span className="min-w-0">
        <span className="text-sm text-ink group-hover:text-cyan transition-colors">{label}</span>
        {hint && <span className="block text-2xs text-faint leading-relaxed">{hint}</span>}
      </span>
    </button>
  )
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

export function Modal({
  open,
  onClose,
  title,
  children,
  width = 'max-w-lg',
}: {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  width?: string
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null
  return (
    <div
      className="fixed inset-0 z-50 grid place-items-center bg-void/80 backdrop-blur-sm p-6"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className={`w-full ${width} panel shadow-2xl animate-fade-in max-h-[85vh] flex flex-col`}>
        <header className="flex items-center justify-between px-3 h-9 border-b border-line shrink-0">
          <span className="label">{title}</span>
          <button className="btn btn-ghost !p-1" onClick={onClose} aria-label="Close">
            <X size={13} />
          </button>
        </header>
        <div className="p-4 overflow-y-auto scroll min-h-0">{children}</div>
      </div>
    </div>
  )
}

export function Confirm({
  open,
  onClose,
  onConfirm,
  title,
  body,
  confirmLabel = 'Confirm',
}: {
  open: boolean
  onClose: () => void
  onConfirm: () => void
  title: string
  body: ReactNode
  confirmLabel?: string
}) {
  return (
    <Modal open={open} onClose={onClose} title={title} width="max-w-md">
      <div className="text-sm text-dim leading-relaxed">{body}</div>
      <div className="flex justify-end gap-2 mt-5">
        <button className="btn" onClick={onClose}>
          Cancel
        </button>
        <button
          className="btn btn-danger border-rose/40 text-rose"
          onClick={() => {
            onConfirm()
            onClose()
          }}
        >
          {confirmLabel}
        </button>
      </div>
    </Modal>
  )
}

// ---------------------------------------------------------------------------
// Toasts
// ---------------------------------------------------------------------------

type Toast = { id: number; message: string; tone: 'info' | 'error' | 'success' }
const ToastContext = createContext<(message: string, tone?: Toast['tone']) => void>(() => {})

export function useToast() {
  return useContext(ToastContext)
}

export function ToastHost({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const counter = useRef(0)

  const push = useCallback((message: string, tone: Toast['tone'] = 'info') => {
    const id = ++counter.current
    setToasts((current) => [...current, { id, message, tone }])
    window.setTimeout(
      () => setToasts((current) => current.filter((t) => t.id !== id)),
      tone === 'error' ? 8000 : 4000,
    )
  }, [])

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2 max-w-md pointer-events-none">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`panel px-3 py-2 text-xs animate-fade-in flex items-start gap-2 pointer-events-auto ${
              toast.tone === 'error'
                ? 'border-rose/50 text-rose'
                : toast.tone === 'success'
                  ? 'border-jade/50 text-jade'
                  : 'text-dim'
            }`}
          >
            {toast.tone === 'error' ? (
              <AlertTriangle size={13} className="shrink-0 mt-0.5" />
            ) : toast.tone === 'success' ? (
              <Check size={13} className="shrink-0 mt-0.5" />
            ) : null}
            <span className="whitespace-pre-wrap break-words">{toast.message}</span>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/**
 * A small async-resource hook.
 *
 * Every panel needs load / loading / error / reload with the same semantics, and
 * a stale response from a superseded request must never overwrite a newer one --
 * which is what the sequence guard below prevents.
 */
export function useAsync<T>(
  loader: () => Promise<T>,
  deps: unknown[] = [],
): { data: T | null; loading: boolean; error: string; reload: () => void } {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [nonce, setNonce] = useState(0)
  const sequence = useRef(0)

  useEffect(() => {
    const current = ++sequence.current
    let alive = true
    setLoading(true)
    loader()
      .then((result) => {
        if (!alive || current !== sequence.current) return
        setData(result)
        setError('')
      })
      .catch((err: Error) => {
        if (!alive || current !== sequence.current) return
        setError(err.message || String(err))
      })
      .finally(() => {
        if (alive && current === sequence.current) setLoading(false)
      })
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  return { data, loading, error, reload: useCallback(() => setNonce((n) => n + 1), []) }
}

export function ErrorLine({ error }: { error: string }) {
  if (!error) return null
  return (
    <div className="flex items-start gap-2 px-3 py-2 text-xs text-rose border border-rose/30 bg-rose/5 rounded">
      <AlertTriangle size={13} className="shrink-0 mt-0.5" />
      <span className="whitespace-pre-wrap break-words">{error}</span>
    </div>
  )
}

export function useDebounced<T>(value: T, delay = 250): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay)
    return () => window.clearTimeout(timer)
  }, [value, delay])
  return debounced
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

export function bytes(n: number): string {
  if (!n) return '0B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const index = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)))
  const value = n / 1024 ** index
  return `${index === 0 ? value.toFixed(0) : value.toFixed(1)}${units[index]}`
}

export function compact(n: number): string {
  if (n < 1000) return String(n)
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`
  return `${(n / 1_000_000).toFixed(1)}M`
}

export function usd(n: number): string {
  if (!n) return '$0'
  if (n < 0.01) return `$${n.toFixed(4)}`
  return `$${n.toFixed(2)}`
}

export function ago(iso: string | null | undefined): string {
  if (!iso) return '--'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return '--'
  const seconds = Math.max(0, (Date.now() - then) / 1000)
  if (seconds < 60) return 'just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  if (seconds < 2_592_000) return `${Math.floor(seconds / 86400)}d ago`
  return new Date(iso).toLocaleDateString()
}

export function duration(ms: number): string {
  if (!ms) return '--'
  if (ms < 1000) return `${Math.round(ms)}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`
}

export function useLocalState<T>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key)
      return stored ? (JSON.parse(stored) as T) : initial
    } catch {
      return initial
    }
  })
  const set = useCallback(
    (next: T) => {
      setValue(next)
      try {
        localStorage.setItem(key, JSON.stringify(next))
      } catch {
        /* private mode, quota -- the UI still works, it just forgets */
      }
    },
    [key],
  )
  return [value, set]
}

export function useNow(intervalMs = 30_000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs)
    return () => window.clearInterval(timer)
  }, [intervalMs])
  return now
}

export function riskTone(risk: string): 'default' | 'cyan' | 'amber' | 'jade' | 'rose' {
  return (
    { safe: 'jade', network: 'cyan', write: 'amber', dangerous: 'rose' } as const
  )[risk] ?? 'default'
}

export function useSearch(initial = ''): [string, (v: string) => void, string] {
  const [raw, setRaw] = useState(initial)
  const debounced = useDebounced(raw, 220)
  return [raw, setRaw, debounced]
}

export function classNames(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}

export function useMounted(): boolean {
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  return mounted
}

export function useKey(key: string, handler: (e: KeyboardEvent) => void, deps: unknown[] = []) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === key) handler(e)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}

export function useMemoized<T>(factory: () => T, deps: unknown[]): T {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(factory, deps)
}
