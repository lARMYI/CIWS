# CIWS

**Cognitive Intelligence Workspace System** — a local-first agentic intelligence hub for your PC.

Every model you have access to, one persistent memory, an entity graph of everything you know,
your documents, image and video generation, MCP connectors, and an agent runtime that ties them
together. It runs on your machine, stores everything in one directory you own, and works with no
API keys at all.

```
   ___ ___ _   _ ___
  / __|_ _| | | / __|   your models · your memory · your machine
 | (__ | || |/\| \__ \
  \___|___|\_/\_/|___/
```

---

## What it actually is

Most "AI hubs" are a chat box with a model dropdown. The difference here is that **the workspace
accumulates**. Every conversation can write to a memory store, extract entities into a graph, and
pull from a corpus of your own documents — so the hub gets more useful the longer you use it,
rather than starting from zero every session.

And the agent at the centre of it learns. **Prime**, the main agent you talk to by default,
reflects on its own run record and your feedback, maintains a set of learned directives in its
own prompt, and can even write new tools for itself — code that runs only after its tests pass
and you approve it.

| | |
|---|---|
| **Command** | Agent console with a live tool trace. You see every tool call, its arguments, its result and its cost as it happens — and thumbs on every answer feed the main agent's reflection loop. |
| **Ontology** | A force-directed entity graph. People, systems, projects and concepts, with typed links, link analysis, and a curation queue for merging duplicates. |
| **Knowledge** | Persistent memory with hybrid recall, plus a searchable corpus of your PDFs, notes, spreadsheets and code. |
| **Studio** | Image and video generation across seven backends, including two that run entirely on your own GPU. |
| **Flows** | Visual DAG workflows you edit on the canvas — chain models, agents, tools, memory and media, and loop until a critic approves. |
| **Systems** | Model registry with live health and cost tracking, an encrypted credential vault, MCP hub management, and the tool catalogue. |

---

## Install

**Requirements:** Python 3.10+ and Node 18+.

```bash
git clone https://github.com/lARMYI/CIWS.git
cd CIWS

# macOS / Linux
./scripts/setup.sh && ./scripts/start.sh

# Windows
scripts\setup.bat
scripts\start.bat
```

That builds the interface, installs the server, and opens `http://127.0.0.1:8787` in your browser.

Want it in the dock instead of a tab? `cd apps/desktop && npm install && npm start` wraps the same
hub in a native window.

### It works with nothing configured

CIWS boots and is useful with **zero API keys**. Memory recall, the entity graph, document
ingestion and search all run on a built-in embedding backend that needs no network and no
credentials. Point it at [Ollama](https://ollama.com) and the whole thing — reasoning included —
runs offline on your own hardware.

Add keys when you want frontier models. Either export them, or paste them into
**Systems → Credentials**, where they are encrypted at rest.

```bash
export ANTHROPIC_API_KEY=...    # or OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, ...
```

See `.env.example` for the full list.

---

## What's connected

**Language models** — Anthropic, OpenAI, Google Gemini, xAI, Groq, Mistral, DeepSeek, OpenRouter,
Together, Perplexity, and locally via Ollama, LM Studio or vLLM. Thirteen providers behind one
streaming interface, with capability-based routing (`fast`, `balanced`, `deep`, `vision`, `local`)
so you switch provider in one place rather than editing every agent.

**Image and video** — OpenAI Images, Google Imagen and Veo, Replicate, fal.ai, Stability, plus
ComfyUI and Automatic1111 running on your own GPU.

**Connectors** — any MCP server over stdio or HTTP (filesystem, git, SQLite, GitHub, fetch, and
whatever else you run), watched folders, and web search via Tavily, Brave, Serper, Exa or a keyless
DuckDuckGo fallback.

**Documents** — PDF, Word, Excel, PowerPoint, HTML, Markdown, CSV, JSON, and 30+ source code
formats. Images are captioned by a vision model rather than OCR'd, which handles diagrams and
screenshots far better.

---

## How it works

```
                        ┌──────────────────────────────┐
   browser / desktop ──▶│  React UI      WebSocket bus  │
                        └───────────────┬──────────────┘
                                        │  REST + SSE
                        ┌───────────────▼──────────────┐
                        │        FastAPI surface        │
                        └───────────────┬──────────────┘
                                        │
              ┌───────────┬─────────────┼─────────────┬───────────┐
              ▼           ▼             ▼             ▼           ▼
         ┌────────┐  ┌────────┐   ┌──────────┐  ┌────────┐  ┌────────┐
         │ Agent  │  │ Memory │   │ Ontology │  │ Corpus │  │ Media  │
         │runtime │  │ recall │   │  graph   │  │ search │  │ studio │
         └───┬────┘  └────┬───┘   └────┬─────┘  └───┬────┘  └───┬────┘
             │            └────────────┼────────────┘           │
        ┌────▼─────┐            ┌──────▼──────┐           ┌─────▼─────┐
        │  Tools   │            │   SQLite    │           │ Providers │
        │ + MCP    │            │ + vectors   │           │  (13)     │
        └──────────┘            └─────────────┘           └───────────┘
```

**One SQLite file** holds everything — conversations, memories, entities, documents, runs, assets.
It lives in `~/.ciws` alongside your original documents and generated media. Copy that directory
and you have moved your entire workspace.

**Hybrid recall.** Pure vector search misses exact tokens — an error code, a version number, a
surname. Pure keyword search misses paraphrase. CIWS runs both, fuses them by reciprocal rank,
weights by importance and recency, then applies MMR so the context window gets twelve *different*
facts rather than twelve wordings of one.

**An open ontology.** Entity types are suggested, not enforced. When a model wants to record a
`spacecraft` or a `lease_agreement`, it can — you curate afterwards in the Ontology panel, where
duplicate detection and merging live.

**Provenance by default.** Every tool call — arguments, result, duration, outcome — is written to
the database before the agent sees the result. An agent that cannot explain where a claim came from
is not much use, and this is where that trail lives.

**A main agent that improves.** Prime's run record, tool error rates and your thumbs up/down are
gathered into an evidence pack, and a scheduled reflection pass distils them into *lessons*
(memories of kind `lesson`, recalled like any other) and *directives* — standing rules appended to
its own system prompt, each with an id, a reason, and one-click retirement. When the gap is a
capability rather than a rule, the agent can forge a **skill**: a `run()` function plus a test
that proves it. Skills whose tests fail are not stored; skills whose tests pass still do not
execute until you activate them, unless you explicitly opt into auto-activation. Prompt changes
apply themselves because they are bounded, visible and reversible; code waits for a human yes
because it is neither.

---

## Security

This is software that runs models with tool access on your own machine. The honest position:

- **The server binds to loopback** and requires a token. Without one, any web page you visit could
  drive your agents and read your memory through `127.0.0.1`. The UI receives the token
  automatically when served by the hub, so you never type it.
- **Credentials are encrypted at rest** (Fernet, key at `0600` beside the vault). That protects
  against a synced folder, a backup tarball, or another account on the machine. It does not protect
  against malware already running as you — nothing on the same disk can.
- **The shell tool is off by default.** The Python tool runs in a subprocess with a timeout, which
  is isolation, not a sandbox: it has your filesystem and network access. It is documented that way
  in the tool description the model reads.
- **File writes are confined to the workspace** unless you relax it, and a deny-list refuses
  credential paths (`.ssh`, `.aws/credentials`, `.netrc`) regardless of setting.
- **Web and document content is treated as untrusted data.** Every agent's system prompt says so
  explicitly, because a model that treats a search snippet as an instruction is the most common way
  a browsing agent gets hijacked.
- **Self-written code is consent-gated and honest about what it is.** A skill the agent forges for
  itself must pass its own tests, then sits inert as *proposed* until you activate it — revisions
  drop back to proposed, because new code voids the old approval. Skills execute in a subprocess
  with a timeout, which is isolation, not a sandbox, exactly like the python tool they are gated
  behind; every proposal, activation and directive change lands in the audit trail.

Nothing leaves your machine except calls to the providers whose keys you supplied.

---

## Development

```bash
./scripts/dev.sh          # API with auto-reload + Vite with hot reload on :5173
```

```bash
cd apps/server && ../../.venv/bin/python -m pytest tests/ -q   # 355 tests, no keys needed
cd apps/web && npm run typecheck
```

The suite drives the agent loop with a scripted provider and the HTTP surface through an
in-process transport, so tool dispatch, parallel calls, failure handling, cancellation, step
ceilings, every API contract, the file deny-list, token auth and the credential vault are all
covered without a network call or a key. CI runs it on Python 3.10 and 3.12 and holds three
ratchets: statement coverage, a gzipped bundle ceiling, and a clean-room keyless boot that writes
a memory and recalls it.

```
apps/
  server/ciws/
    core/       config, paths, encrypted secrets, event bus, logging, errors
    db/         schema, async engine, vector index, FTS
    gateway/    provider-neutral model access + 13 adapters
    memory/     embeddings, hybrid recall, extraction, consolidation
    ontology/   entity graph, link analysis, entity resolution
    ingest/     extraction, heading-aware chunking, corpus search
    agents/     the reason/act loop, personas, run traces, the reflection engine
    tools/      registry with risk policy and audit, 43 built-in tools, the skill forge
    mcp/        dependency-free MCP client (stdio + HTTP)
    hubs/       connectors and web search
    media/      image and video generation, asset library
    workflows/  DAG engine with level-wise parallelism and bounded loop nodes
    api/        REST routers + WebSocket
    scheduler   due tasks, watched folders and reflection, on a background tick
    backup.py   snapshot, verify, restore, optional passphrase encryption
  web/          React + TypeScript + Tailwind
  desktop/      Electron shell
```

Further reading: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Known limits

Stated plainly, because the alternative is you discovering them yourself:

- **The keyless embedding backend is lexical, not semantic.** It captures word and character-level
  overlap, so it will match "deploy schedule" to "deploys run on Friday" but not "colour scheme" to
  "dark theme". Configure a real embedding model and that gap closes.
- **Non-Anthropic model pricing is seeded where it is well-published and left at zero otherwise.**
  A zero means "not seeded", not "free" — set the real figure in the Models panel and it sticks.
  Anthropic figures come from the current API reference; Ollama and local models genuinely are free.
- **Video generation is fire-and-forget.** Renders take minutes; the asset appears in the Studio
  when it lands.
- **Media provider adapters are written to documented REST endpoints but have not been run against
  live paid APIs** in this build. Their request building and response parsing are covered by
  fixture tests; the endpoints themselves are unconfirmed. The local backends (ComfyUI,
  Automatic1111) and the whole keyless path are verified.
- **No live model round-trip has ever happened here.** The gateway is proven against a scripted
  provider and recorded-shape fixtures, and the per-model request quirks are pinned by tests, but
  this build has never held a provider credential. `scripts/record_fixture.py` turns a real call
  into a replayable fixture once you have one.
- **The database is not encrypted at rest.** The credential vault is; the SQLite file beside it is
  not, because that needs SQLCipher — a build of SQLite rather than a Python package. Backups can
  be encrypted with a passphrase, and full-disk encryption covers the live workspace.

---

## Licence

MIT.
