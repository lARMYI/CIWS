# Architecture

Notes on why CIWS is built the way it is. The README covers what it does; this covers the
decisions that would otherwise look arbitrary.

---

## The shape of it

Four layers, each depending only on the ones below it:

```
  api/                    HTTP + WebSocket. Knows about every subsystem.
  agents/  workflows/     Orchestration. Knows about tools and the gateway.
  memory/  ontology/      Knowledge. Knows about the gateway and the database.
  ingest/  media/  hubs/
  gateway/                Model access. Knows about the database and core.
  db/                     Persistence.
  core/                   Config, paths, secrets, events, errors. Depends on nothing.
```

The rule that keeps it honest: **nothing above `gateway/` imports a provider SDK or knows a
provider's wire format.** Adding a fourteenth provider touches one file.

---

## Why SQLite

A single-user hub does not need a database server. SQLite means no daemon to babysit, no
connection string to configure, and a workspace that is one file you can copy to a USB stick.

With WAL enabled it handles a UI, several concurrent agents and a background ingest without
readers blocking on writers. Without WAL, a long ingest transaction blocks every read and the UI
appears to hang — which is why `db/base.py` sets the pragmas before anything else touches a
connection.

Full-text search is FTS5 with trigger-maintained indexes, so nothing in the application layer has
to remember to keep the search index in sync with the base table.

## Why a hand-rolled vector index

Vectors live as float32 blobs in one `embeddings` table keyed by `(kind, ref_id)`, mirrored into a
NumPy matrix per kind. Search is an exact dense dot product.

For a personal workspace — tens of thousands of vectors — this runs in single-digit milliseconds
and never returns the wrong neighbour, which an approximate index can. It also means no extra
dependency, no index to rebuild, and no separate process. Vectors are L2-normalised on write, so
cosine similarity is a dot product.

One index across memories, chunks, entities and messages, partitioned by kind, so a memory search
never scans document chunks.

## Why three embedding backends

In order of preference: a provider model, `sentence-transformers` if installed, and a hashed-feature
projection computed in-process.

The third one is the interesting decision. It has no dependencies, no network and no key, which
means a fresh install recalls memories and searches documents on first launch instead of showing an
empty state and a request for a credit card. It is genuinely lexical rather than semantic — word
unigrams, bigrams and character trigrams hashed into 768 dimensions with a signed hash so
collisions cancel rather than accumulate — and the UI says so rather than implying otherwise.

Vectors from different backends are not comparable, so the index records which backend produced
each one and rebuilds when you switch.

## Why recall is hybrid

Vector search alone misses exact tokens: an error code, a version number, a surname. Keyword search
alone misses paraphrase. CIWS computes both, then:

1. Rescales each signal against the best candidate **in this result set**, because a strong
   hash-embedding match scores ~0.2 where a transformer scores ~0.8 and the weights must not depend
   on which backend is live.
2. Fuses them with reciprocal-rank fusion — as an *agreement* signal, not the ranking itself. RRF
   alone is nearly flat on a short candidate list, which lets importance and pinning swamp
   relevance. (This was a real bug during development: a pinned identity memory outranked a direct
   match on every query.)
3. Weights by importance and by exponential recency decay.
4. Applies MMR, so twelve results are twelve *different* facts rather than twelve wordings of one.

Pinning is a small additive nudge, not a multiplier. It should surface a memory when relevance is
close, never outrank a direct match.

## Why the ontology is open

Enterprise ontology tools ship a closed schema that a data engineer curates up front. That is right
for an organisation with a governance team and wrong for one person: the moment a model wants to
record a `spacecraft` or a `lease_agreement`, you would rather it did than dropped the fact.

So the entity types in `ontology/schema.py` are the *known* vocabulary — they get colours, icons and
suggested properties, and models are steered toward them — but `normalize_type()` lets an unknown
type through as a slug. Curation happens afterwards, in the Ontology panel, where duplicate
detection and merging live.

Merges are soft: losing entities keep a `merged_into` pointer rather than being deleted, so an old
id in a memory or a document citation still resolves.

## Why the agent loop looks like this

Three things it takes seriously:

**Parallel tools return in one turn.** A model that asks for four searches gets four searches at
once, and all four results come back in a single tool message. Splitting them across messages
silently trains the model to stop batching — a subtle, expensive regression, and one the test suite
now guards.

**Context compaction drops the middle.** When a transcript will not fit, the first two messages
(the goal) and the last six (the thread) are preserved, and the middle is dropped with a note
telling the model what happened and to look things up again rather than guess. Dropping the oldest
loses the goal; dropping the newest loses the thread.

**Cancellation is immediate.** The cancel event is checked between steps and passed into every
tool, so stopping a runaway run does not mean waiting out its next sixty-second call.

A tool that raises becomes an *observation*, never a crashed run. An exception would abort the
whole turn; a failed tool should instead be something the agent can reason about and route around.

## Why the tool registry owns policy

Three things are centralised there rather than left to each tool:

- **Permission.** A tool declares a risk level; the registry decides whether policy lets it run.
  Individual tools must not be able to grant themselves rights.
- **Provenance.** Every call is written to `tool_calls` before the agent sees the result.
- **Blast radius.** Timeouts, output truncation and cancellation are enforced centrally, so one
  runaway tool cannot hang a run or evict the conversation from the context window.

MCP tools are surfaced through the same interface, namespaced `<hub>__<tool>`, so an agent cannot
tell a built-in from a connector — and the same policy applies to both.

## Why the graph is canvas, not SVG

A few hundred nodes with per-frame position updates means a few hundred DOM mutations per frame in
SVG, and browsers give up around 400. Canvas redraws the scene in one pass and stays smooth into
the low thousands.

The simulation is deliberately small: all-pairs repulsion, springs along edges, a weak pull to
centre, velocity damping, and cooling to a stop. O(n²) repulsion is fine to ~800 nodes, and the API
caps the result set below that — a quadtree here would be complexity without payoff.

## Why Markdown is rendered as React elements

`components/Markdown.tsx` contains no `dangerouslySetInnerHTML`. Model output is untrusted — it
routinely contains text fetched from the open web — and the usual markdown-to-HTML-plus-sanitiser
pattern puts an XSS filter on the critical path of every message. Building React nodes directly
means a `<script>` in a model's answer renders as literal characters, because that is the only
thing React can do with a string.

Links are filtered to `http(s)` and `mailto` for the same reason.

## Why chat is SSE and everything else is a WebSocket

The WebSocket carries workspace-wide events — what every panel needs to stay live. A chat turn is a
request with one consumer and a definite end. Keeping them apart means a dropped socket does not
lose a half-written answer, and a laggy browser tab drops its own oldest events rather than
applying backpressure to the agent producing them.

---

## Failure posture

The hub boots stage by stage — database, vectors, tools, seeds, hubs — and every stage after the
database is wrapped. A broken MCP server should not be the reason you cannot reach your own notes.

Errors are typed (`core/errors.py`) and carry an HTTP status plus a message written for a human.
Anything else shows the user a stack trace they cannot act on.

Where a number is not known — pricing for a provider that does not publish it — CIWS stores zero and
says "not seeded" rather than inventing a figure. The Models panel lets you type the real one, and
your value survives a catalogue refresh.
