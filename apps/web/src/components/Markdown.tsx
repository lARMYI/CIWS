/**
 * A small Markdown renderer that emits React elements.
 *
 * The point is what it does NOT do: there is no `dangerouslySetInnerHTML`
 * anywhere in it. Model output is untrusted -- it routinely contains text
 * fetched from the open web -- and the usual pattern of markdown-to-HTML plus a
 * sanitiser puts an XSS filter on the critical path of every message. Building
 * React nodes directly means a `<script>` in a model's answer renders as the
 * literal characters, because that is the only thing React can do with a string.
 *
 * It covers what models actually emit: fenced code, headings, lists, tables,
 * blockquotes, links, and inline emphasis. Anything it does not recognise falls
 * through as plain text rather than disappearing.
 */

import { Check, Copy } from 'lucide-react'
import { memo, useState, type ReactNode } from 'react'

// ---------------------------------------------------------------------------
// Inline
// ---------------------------------------------------------------------------

const INLINE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(\*[^*\n]+\*)|(~~[^~]+~~)|(\[[^\]]+\]\([^)]+\))|(https?:\/\/[^\s<>()]+)/g

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let cursor = 0
  let index = 0

  for (const match of text.matchAll(INLINE)) {
    const start = match.index ?? 0
    if (start > cursor) nodes.push(text.slice(cursor, start))
    const token = match[0]
    const key = `${keyPrefix}-${index++}`

    if (token.startsWith('`')) {
      nodes.push(<code key={key}>{token.slice(1, -1)}</code>)
    } else if (token.startsWith('**') || token.startsWith('__')) {
      nodes.push(
        <strong key={key} className="font-semibold text-ink">
          {token.slice(2, -2)}
        </strong>,
      )
    } else if (token.startsWith('~~')) {
      nodes.push(
        <span key={key} className="line-through text-faint">
          {token.slice(2, -2)}
        </span>,
      )
    } else if (token.startsWith('*')) {
      nodes.push(<em key={key}>{token.slice(1, -1)}</em>)
    } else if (token.startsWith('[')) {
      const parsed = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token)
      if (parsed) {
        nodes.push(
          <SafeLink key={key} href={parsed[2]}>
            {parsed[1]}
          </SafeLink>,
        )
      } else {
        nodes.push(token)
      }
    } else {
      nodes.push(
        <SafeLink key={key} href={token}>
          {token}
        </SafeLink>,
      )
    }
    cursor = start + token.length
  }

  if (cursor < text.length) nodes.push(text.slice(cursor))
  return nodes
}

function SafeLink({ href, children }: { href: string; children: ReactNode }) {
  // Only http(s) and mailto get to be links. `javascript:` and `data:` URLs in
  // model output render as text.
  const safe = /^(https?:|mailto:)/i.test(href)
  if (!safe) return <span className="text-faint">{children}</span>
  return (
    <a href={href} target="_blank" rel="noopener noreferrer nofollow">
      {children}
    </a>
  )
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

function CodeBlock({ code, language }: { code: string; language: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="relative group my-2.5">
      <div className="flex items-center justify-between px-2.5 h-6 border border-b-0 border-line bg-raised rounded-t">
        <span className="text-2xs text-faint uppercase tracking-wider">{language || 'text'}</span>
        <button
          className="opacity-0 group-hover:opacity-100 transition-opacity text-faint hover:text-cyan"
          onClick={() => {
            navigator.clipboard?.writeText(code).then(
              () => {
                setCopied(true)
                window.setTimeout(() => setCopied(false), 1400)
              },
              () => undefined,
            )
          }}
          aria-label="Copy code"
        >
          {copied ? <Check size={11} className="text-jade" /> : <Copy size={11} />}
        </button>
      </div>
      <pre className="!my-0 !rounded-t-none">
        <code>{code}</code>
      </pre>
    </div>
  )
}

function Table({ rows }: { rows: string[][] }) {
  const [header, ...body] = rows
  return (
    <div className="overflow-x-auto scroll my-2">
      <table>
        <thead>
          <tr>
            {header.map((cell, i) => (
              <th key={i}>{renderInline(cell, `th-${i}`)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, r) => (
            <tr key={r}>
              {row.map((cell, c) => (
                <td key={c}>{renderInline(cell, `td-${r}-${c}`)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function splitRow(line: string): string[] {
  return line
    .replace(/^\s*\|/, '')
    .replace(/\|\s*$/, '')
    .split('|')
    .map((cell) => cell.trim())
}

function isDivider(line: string): boolean {
  return /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(line) && line.includes('-')
}

export const Markdown = memo(function Markdown({ text }: { text: string }) {
  if (!text) return null
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  const blocks: ReactNode[] = []
  let i = 0
  let key = 0

  const flushList = (items: string[], ordered: boolean) => {
    if (!items.length) return
    const List = ordered ? 'ol' : 'ul'
    blocks.push(
      <List key={`l-${key++}`}>
        {items.map((item, index) => (
          <li key={index} className={ordered ? '' : 'pl-3 before:content-["\\2022"] before:absolute before:left-0 before:text-cyan/60'}>
            {renderInline(item, `li-${key}-${index}`)}
          </li>
        ))}
      </List>,
    )
  }

  while (i < lines.length) {
    const line = lines[i]

    // Fenced code
    const fence = /^```([\w+-]*)\s*$/.exec(line.trim())
    if (fence) {
      const language = fence[1]
      const code: string[] = []
      i++
      while (i < lines.length && !/^```\s*$/.test(lines[i].trim())) code.push(lines[i++])
      i++ // closing fence (or end of input on a stream mid-flight)
      blocks.push(<CodeBlock key={`c-${key++}`} code={code.join('\n')} language={language} />)
      continue
    }

    // Table: a header row followed by a divider
    if (line.includes('|') && i + 1 < lines.length && isDivider(lines[i + 1])) {
      const rows: string[][] = [splitRow(line)]
      i += 2
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        rows.push(splitRow(lines[i++]))
      }
      blocks.push(<Table key={`t-${key++}`} rows={rows} />)
      continue
    }

    // Headings
    const heading = /^(#{1,4})\s+(.*)$/.exec(line)
    if (heading) {
      const level = heading[1].length
      const Tag = (['h1', 'h2', 'h3', 'h3'] as const)[level - 1]
      blocks.push(<Tag key={`h-${key++}`}>{renderInline(heading[2], `h-${key}`)}</Tag>)
      i++
      continue
    }

    // Horizontal rule
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      blocks.push(<hr key={`hr-${key++}`} />)
      i++
      continue
    }

    // Blockquote
    if (/^\s*>\s?/.test(line)) {
      const quoted: string[] = []
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quoted.push(lines[i++].replace(/^\s*>\s?/, ''))
      }
      blocks.push(
        <blockquote key={`q-${key++}`}>{renderInline(quoted.join(' '), `q-${key}`)}</blockquote>,
      )
      continue
    }

    // Lists
    if (/^\s*[-*+]\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(lines[i++].replace(/^\s*[-*+]\s+/, ''))
      }
      flushList(items, false)
      continue
    }
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(lines[i++].replace(/^\s*\d+[.)]\s+/, ''))
      }
      flushList(items, true)
      continue
    }

    // Blank
    if (!line.trim()) {
      i++
      continue
    }

    // Paragraph: consume until a blank line or the start of another block
    const paragraph: string[] = []
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^```/.test(lines[i].trim()) &&
      !/^(#{1,4})\s/.test(lines[i]) &&
      !/^\s*[-*+]\s+/.test(lines[i]) &&
      !/^\s*\d+[.)]\s+/.test(lines[i]) &&
      !/^\s*>\s?/.test(lines[i])
    ) {
      paragraph.push(lines[i++])
    }
    if (paragraph.length) {
      blocks.push(<p key={`p-${key++}`}>{renderInline(paragraph.join(' '), `p-${key}`)}</p>)
    } else {
      i++
    }
  }

  return <div className="prose-ciws">{blocks}</div>
})
