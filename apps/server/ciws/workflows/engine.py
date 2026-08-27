"""The workflow DAG engine.

Nodes execute level-wise: every node whose inputs are ready runs concurrently,
then the next level. A pipeline of independent branches finishes in the time of
its slowest branch rather than the sum of all of them, which is the whole point
of drawing a graph instead of writing a script.

A failed node marks its descendants ``skipped`` rather than hanging the run --
a workflow that stalls silently is much worse than one that tells you which
branch died and finishes the rest.

Every cross-subsystem import is lazy, inside the executor that needs it, so the
node catalogue can be listed without pulling in the whole application.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from sqlalchemy import delete as sa_delete
from sqlalchemy import select

from ..core.errors import NotFound, ValidationFailed
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..core.util import new_id, now, truncate
from ..db.base import session_scope
from ..db.models import Workflow, WorkflowRun

log = get_logger("workflows")

MAX_FANOUT = 32
_active: dict[str, asyncio.Event] = {}


@dataclass
class RunContext:
    run_id: str
    workflow_id: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    node_states: dict[str, Any] = field(default_factory=dict)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    project_id: str | None = None
    cost_usd: float = 0.0


Executor = Callable[[dict[str, Any], dict[str, Any], RunContext], Awaitable[dict[str, Any]]]


@dataclass
class NodeSpec:
    type: str
    label: str
    category: str
    description: str
    run: Executor
    inputs: list[str] = field(default_factory=lambda: ["in"])
    outputs: list[str] = field(default_factory=lambda: ["out"])
    config_schema: dict[str, Any] = field(default_factory=dict)
    color: str = "#22d3ee"
    icon: str = "box"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type, "label": self.label, "category": self.category,
            "description": self.description, "inputs": self.inputs, "outputs": self.outputs,
            "config_schema": self.config_schema, "color": self.color, "icon": self.icon,
        }


NODE_TYPES: dict[str, NodeSpec] = {}


def node(
    type_: str,
    label: str,
    category: str,
    description: str,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    config_schema: dict[str, Any] | None = None,
    color: str = "#22d3ee",
    icon: str = "box",
) -> Callable[[Executor], Executor]:
    def wrap(fn: Executor) -> Executor:
        NODE_TYPES[type_] = NodeSpec(
            type=type_, label=label, category=category, description=description,
            run=fn, inputs=inputs if inputs is not None else ["in"],
            outputs=outputs if outputs is not None else ["out"],
            config_schema=config_schema or {}, color=color, icon=icon,
        )
        return fn

    return wrap


def _first(inputs: dict[str, Any]) -> Any:
    for value in inputs.values():
        if value is not None:
            return value
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(v) for v in value)
    if isinstance(value, dict):
        from ..core.util import dumps

        return dumps(value)
    return str(value)


# ---------------------------------------------------------------------------
# Node executors
# ---------------------------------------------------------------------------


@node("input", "Input", "io", "Reads a value from the run's inputs.",
      inputs=[], config_schema={"key": "string", "default": "string"},
      color="#34d399", icon="log-in")
async def _input(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    key = str(config.get("key") or "input")
    return {"out": ctx.inputs.get(key, config.get("default", ""))}


@node("output", "Output", "io", "Captures a value into the run's outputs.",
      outputs=[], config_schema={"key": "string"}, color="#34d399", icon="log-out")
async def _output(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    key = str(config.get("key") or "output")
    ctx.outputs[key] = _first(inputs)
    return {}


@node("prompt", "Prompt Template", "text",
      "Fills {placeholders} from inputs and the run's inputs.",
      config_schema={"template": "string"}, color="#818cf8", icon="type")
async def _prompt(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    template = str(config.get("template") or "{in}")
    values = {**ctx.inputs, **{k: _text(v) for k, v in inputs.items()}}
    try:
        return {"out": template.format(**values)}
    except KeyError as exc:
        # A missing placeholder should say which one, not raise a bare KeyError.
        raise ValidationFailed(f"Template refers to {exc} which no input provides") from exc


@node("model", "Model Call", "ai", "Sends the input to a model and returns its answer.",
      config_schema={"model": "string", "system": "string", "temperature": 0.7, "max_tokens": 4000},
      color="#22d3ee", icon="cpu")
async def _model(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..gateway.registry import gateway
    from ..gateway.types import ChatMessage, ChatRequest

    response = await gateway.chat(
        ChatRequest(
            model=str(config.get("model") or "balanced"),
            system=str(config.get("system") or ""),
            messages=[ChatMessage.user(_text(_first(inputs)))],
            temperature=float(config.get("temperature", 0.7)),
            max_tokens=int(config.get("max_tokens", 4000)),
        )
    )
    ctx.cost_usd += response.usage.cost_usd
    return {"out": response.content}


@node("agent", "Agent", "ai", "Runs a full agent with tools on the input.",
      config_schema={"agent": "analyst", "model": "string"}, color="#a78bfa", icon="bot")
async def _agent(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    try:
        from ..agents import runtime
    except ImportError as exc:  # pragma: no cover
        return {"out": f"[agent node unavailable: {exc}]"}

    result = await runtime.run_agent(
        agent_slug=str(config.get("agent") or "analyst"),
        prompt=_text(_first(inputs)),
        model_override=str(config.get("model") or ""),
        project_id=ctx.project_id,
    )
    ctx.cost_usd += result.usage.cost_usd
    if result.error:
        raise ValidationFailed(f"Agent '{config.get('agent')}' failed: {result.error}")
    return {"out": result.content}


@node("tool", "Tool", "ai", "Invokes a single tool with arguments from the input.",
      config_schema={"tool": "string", "arguments": {}}, color="#4ade80", icon="wrench")
async def _tool(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..tools.registry import ToolContext, registry

    arguments = dict(config.get("arguments") or {})
    incoming = _first(inputs)
    if isinstance(incoming, dict):
        arguments.update(incoming)
    elif incoming is not None and "query" not in arguments:
        arguments["query"] = _text(incoming)

    output = await registry.call(
        str(config.get("tool") or ""),
        arguments,
        ToolContext(run_id=ctx.run_id, project_id=ctx.project_id, auto_approve=True),
    )
    if output.is_error:
        raise ValidationFailed(output.content)
    return {"out": output.content}


@node("memory_recall", "Recall Memory", "knowledge", "Retrieves relevant memories.",
      config_schema={"limit": 8}, color="#fbbf24", icon="brain")
async def _memory_recall(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..memory import store

    hits = await store.recall(
        _text(_first(inputs)), limit=int(config.get("limit", 8)), project_id=ctx.project_id
    )
    return {"out": "\n".join(f"- {h.memory.content}" for h in hits)}


@node("memory_write", "Write Memory", "knowledge", "Stores the input as a memory.",
      config_schema={"kind": "fact", "importance": 0.5}, color="#fbbf24", icon="save")
async def _memory_write(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..memory import store

    memory = await store.remember(
        _text(_first(inputs)),
        kind=str(config.get("kind") or "fact"),
        importance=float(config.get("importance", 0.5)),
        project_id=ctx.project_id,
        source="workflow",
    )
    return {"out": memory.id}


@node("corpus_search", "Search Corpus", "knowledge", "Searches ingested documents.",
      config_schema={"limit": 8}, color="#94a3b8", icon="library")
async def _corpus_search(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..ingest import pipeline

    results = await pipeline.search_corpus(
        _text(_first(inputs)), limit=int(config.get("limit", 8)), project_id=ctx.project_id
    )
    return {
        "out": "\n\n".join(f"[{r['document_title']}] {r['text']}" for r in results),
        "results": results,
    }


@node("graph_query", "Query Graph", "knowledge", "Searches entities or expands a neighbourhood.",
      config_schema={"mode": "search", "depth": 1, "limit": 20}, color="#a78bfa", icon="share-2")
async def _graph_query(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..ontology import graph

    value = _text(_first(inputs))
    if str(config.get("mode")) == "neighbors":
        result = await graph.neighbors(value, depth=int(config.get("depth", 1)))
        names = ", ".join(n["name"] for n in result["nodes"])
        return {"out": names, "graph": result}
    found = await graph.search_entities(
        value, limit=int(config.get("limit", 20)), project_id=ctx.project_id
    )
    return {
        "out": "\n".join(f"- {e.name} ({e.type}) id={e.id}" for e in found),
        "entities": [graph.node_dict(e) for e in found],
    }


@node("graph_extract", "Extract Graph", "knowledge", "Mines entities and links from text.",
      color="#a78bfa", icon="git-branch")
async def _graph_extract(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..ontology import graph

    result = await graph.extract_graph(_text(_first(inputs)), project_id=ctx.project_id)
    return {
        "out": f"{len(result['entities'])} entities, {len(result['edges'])} links",
        "graph": result,
    }


@node("image", "Generate Image", "media", "Generates an image from the input prompt.",
      config_schema={"model": "string", "size": "1024x1024", "n": 1},
      color="#f472b6", icon="image")
async def _image(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..media import studio
    from ..media.types import ImageRequest

    assets = await studio.generate_image(
        ImageRequest(
            prompt=_text(_first(inputs)),
            model=str(config.get("model") or ""),
            size=str(config.get("size") or "1024x1024"),
            n=int(config.get("n", 1)),
        ),
        project_id=ctx.project_id,
    )
    ctx.cost_usd += sum(a.cost_usd for a in assets)
    return {"out": [a.id for a in assets], "assets": [a.to_dict() for a in assets]}


@node("video", "Generate Video", "media", "Starts a video render from the input prompt.",
      config_schema={"model": "string", "duration_s": 5}, color="#f472b6", icon="film")
async def _video(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..media import studio
    from ..media.types import VideoRequest

    asset = await studio.generate_video_background(
        VideoRequest(
            prompt=_text(_first(inputs)),
            model=str(config.get("model") or ""),
            duration_s=float(config.get("duration_s", 5)),
        ),
        project_id=ctx.project_id,
    )
    return {"out": asset.id}


@node("http", "HTTP Request", "io", "Calls an HTTP endpoint.",
      config_schema={"method": "GET", "url": "string", "headers": {}, "body": "string"},
      color="#60a5fa", icon="globe")
async def _http(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..gateway.base import http_client

    url = str(config.get("url") or "")
    if not url.startswith(("http://", "https://")):
        raise ValidationFailed("The http node needs an absolute http(s) URL")
    body = config.get("body")
    if body is None:
        body = _first(inputs)

    response = await http_client("workflow", timeout=60).request(
        str(config.get("method") or "GET").upper(),
        url,
        headers={str(k): str(v) for k, v in (config.get("headers") or {}).items()},
        content=_text(body) if body else None,
    )
    return {"out": response.text[:100_000], "status": response.status_code}


@node("branch", "Branch", "logic", "Routes to 'true' or 'false' on a condition.",
      outputs=["true", "false"], config_schema={"equals": "string", "contains": "string"},
      color="#facc15", icon="git-fork")
async def _branch(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    value = _first(inputs)
    text = _text(value)
    if config.get("equals") is not None and str(config["equals"]) != "":
        passed = text.strip() == str(config["equals"])
    elif config.get("contains"):
        passed = str(config["contains"]).lower() in text.lower()
    else:
        passed = bool(value) and text.strip().lower() not in ("false", "no", "0", "")
    return {"true": value, "false": None} if passed else {"true": None, "false": value}


@node("map", "Map", "logic", "Runs the downstream branch once per list item.",
      config_schema={"concurrency": 4, "node_type": "model", "node_config": {}},
      color="#facc15", icon="repeat")
async def _map(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    value = _first(inputs)
    items = value if isinstance(value, list) else [line for line in _text(value).splitlines() if line.strip()]
    items = items[:MAX_FANOUT]
    inner_type = str(config.get("node_type") or "model")
    spec = NODE_TYPES.get(inner_type)
    if spec is None:
        raise ValidationFailed(f"map: unknown inner node type '{inner_type}'")

    semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency", 4))))

    async def one(item: Any) -> Any:
        async with semaphore:
            try:
                result = await spec.run(dict(config.get("node_config") or {}), {"in": item}, ctx)
                return result.get("out")
            except Exception as exc:  # noqa: BLE001 - one bad item must not sink the batch
                return f"[failed: {exc}]"

    results = await asyncio.gather(*(one(i) for i in items))
    return {"out": list(results)}


#: Hard ceiling on any loop node, whatever its config says. A workflow that
#: wants more than this wants a different design, and an unbounded loop in an
#: unattended run is how a workspace quietly spends a month's budget overnight.
MAX_ITERATIONS = 25


def _condition_met(config: dict[str, Any], value: Any) -> bool:
    """Shared stop test for the loop nodes.

    Deliberately the same vocabulary as ``branch`` -- ``equals`` / ``contains``
    / truthiness -- so a workflow author who has used one already knows this.
    """
    text = _text(value)
    if config.get("until_equals") is not None and str(config["until_equals"]) != "":
        return text.strip() == str(config["until_equals"])
    if config.get("until_contains"):
        return str(config["until_contains"]).lower() in text.lower()
    return bool(value) and text.strip().lower() not in ("false", "no", "0", "")


@node("until", "Loop Until", "logic",
      "Runs an inner node repeatedly, feeding each result back in, until a "
      "condition holds or the iteration ceiling is reached.",
      outputs=["out", "iterations", "exit_reason"],
      config_schema={
          "node_type": "model",
          "node_config": {},
          "max_iterations": 5,
          "until_contains": "",
          "until_equals": "",
          "stop_on_repeat": True,
      },
      color="#facc15", icon="rotate-cw")
async def _until(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    """The loop the DAG cannot express.

    ``validate_graph`` rejects cycles, and rightly so -- a cyclic graph has no
    topological order to execute and no obvious place to stop. The iteration
    lives inside a node instead, which keeps the graph acyclic while letting a
    workflow refine something until it is good enough.

    Three ways out, and the caller is always told which one was taken: the
    condition held, the output stopped changing, or the ceiling was reached.
    A loop that ends without saying why is a loop nobody can debug.
    """
    inner_type = str(config.get("node_type") or "model")
    spec = NODE_TYPES.get(inner_type)
    if spec is None:
        raise ValidationFailed(f"until: unknown inner node type '{inner_type}'")
    if inner_type in ("until", "supervisor"):
        raise ValidationFailed("until: a loop node cannot be its own inner node")

    ceiling = max(1, min(int(config.get("max_iterations", 5) or 5), MAX_ITERATIONS))
    inner_config = dict(config.get("node_config") or {})
    stop_on_repeat = bool(config.get("stop_on_repeat", True))

    value = _first(inputs)
    previous = _text(value)
    iterations = 0
    exit_reason = "ceiling"

    for iteration in range(ceiling):
        if ctx.cancelled.is_set():
            exit_reason = "cancelled"
            break

        result = await spec.run(inner_config, {"in": value}, ctx)
        value = result.get("out")
        iterations = iteration + 1

        current = _text(value)
        if _condition_met(config, value):
            exit_reason = "condition"
            break
        # A loop whose output has stopped moving will not start again, and
        # every further pass costs a model call for nothing.
        if stop_on_repeat and iteration > 0 and current == previous:
            exit_reason = "converged"
            break
        previous = current

    bus.publish(
        Topic.WORKFLOW_NODE,
        run_id=ctx.run_id,
        node="until",
        iterations=iterations,
        exit_reason=exit_reason,
    )
    return {"out": value, "iterations": iterations, "exit_reason": exit_reason}


@node("supervisor", "Supervisor", "logic",
      "Runs a worker node, has a critic judge the result, and repeats until the "
      "critic approves or the ceiling is reached.",
      outputs=["out", "iterations", "exit_reason", "verdict"],
      config_schema={
          "worker_type": "agent",
          "worker_config": {},
          "critic_type": "model",
          "critic_config": {},
          "approve_when_contains": "APPROVED",
          "max_iterations": 3,
      },
      color="#facc15", icon="shield-check")
async def _supervisor(
    config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext
) -> dict[str, Any]:
    """Route on a critic's judgement rather than on a boolean expression.

    ``until`` can only test the worker's own output, which means the workflow
    has to be able to recognise "good enough" with a string match. A critic can
    read the work and say so, and its rejection carries a reason the next
    attempt can act on -- which is the difference between retrying and
    improving.
    """
    worker = NODE_TYPES.get(str(config.get("worker_type") or "agent"))
    critic = NODE_TYPES.get(str(config.get("critic_type") or "model"))
    if worker is None:
        raise ValidationFailed(f"supervisor: unknown worker type '{config.get('worker_type')}'")
    if critic is None:
        raise ValidationFailed(f"supervisor: unknown critic type '{config.get('critic_type')}'")

    ceiling = max(1, min(int(config.get("max_iterations", 3) or 3), MAX_ITERATIONS))
    approve_marker = str(config.get("approve_when_contains") or "APPROVED")
    worker_config = dict(config.get("worker_config") or {})
    critic_config = dict(config.get("critic_config") or {})

    task = _first(inputs)
    work: Any = None
    verdict = ""
    iterations = 0
    exit_reason = "ceiling"

    for iteration in range(ceiling):
        if ctx.cancelled.is_set():
            exit_reason = "cancelled"
            break

        # After the first pass the worker sees the critic's objection, so the
        # next attempt is a revision rather than a re-roll.
        brief = task if iteration == 0 else (
            f"{_text(task)}\n\n"
            f"A previous attempt was rejected. Address this and try again.\n\n"
            f"PREVIOUS ATTEMPT:\n{_text(work)}\n\n"
            f"REVIEWER SAID:\n{verdict}"
        )
        work = (await worker.run(worker_config, {"in": brief}, ctx)).get("out")
        iterations = iteration + 1

        judged = await critic.run(
            critic_config,
            {"in": f"TASK:\n{_text(task)}\n\nWORK TO REVIEW:\n{_text(work)}"},
            ctx,
        )
        verdict = _text(judged.get("out"))
        if approve_marker.lower() in verdict.lower():
            exit_reason = "approved"
            break

    bus.publish(
        Topic.WORKFLOW_NODE,
        run_id=ctx.run_id,
        node="supervisor",
        iterations=iterations,
        exit_reason=exit_reason,
    )
    return {"out": work, "iterations": iterations, "exit_reason": exit_reason, "verdict": verdict}


@node("merge", "Merge", "logic", "Combines several inputs into one block of text.",
      inputs=["a", "b", "c"], config_schema={"separator": "\n\n"}, color="#facc15", icon="merge")
async def _merge(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    separator = str(config.get("separator", "\n\n"))
    parts = [_text(v) for v in inputs.values() if v is not None]
    return {"out": separator.join(parts)}


@node("code", "Expression", "logic",
      "Evaluates a small Python expression over the inputs. Not a sandbox.",
      config_schema={"expression": "value"}, color="#f87171", icon="code")
async def _code(config: dict[str, Any], inputs: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from ..core.config import get_settings

    if not get_settings().security.allow_python_tool:
        raise ValidationFailed("Expression nodes are disabled in Settings -> Security")

    expression = str(config.get("expression") or "value")
    # A restricted namespace, not a security boundary -- it stops accidents, not
    # attacks. The gate above is the actual control.
    safe = {
        "len": len, "sum": sum, "min": min, "max": max, "sorted": sorted, "abs": abs,
        "round": round, "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "set": set, "any": any, "all": all,
        "enumerate": enumerate, "zip": zip, "range": range, "reversed": reversed,
    }
    try:
        result = eval(  # noqa: S307 - gated, documented, and user-authored
            expression,
            {"__builtins__": safe},
            {"value": _first(inputs), "inputs": inputs, "run_inputs": ctx.inputs},
        )
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailed(f"Expression failed: {exc}") from exc
    return {"out": result}


# ---------------------------------------------------------------------------
# Validation and execution
# ---------------------------------------------------------------------------


def validate_graph(graph: dict[str, Any]) -> list[str]:
    """Every problem with a graph, in one pass, phrased for a human."""
    problems: list[str] = []
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    seen: set[str] = set()
    for node_data in nodes:
        node_id = str(node_data.get("id") or "")
        if not node_id:
            problems.append("A node has no id.")
            continue
        if node_id in seen:
            problems.append(f"Duplicate node id '{node_id}'.")
        seen.add(node_id)
        node_type = str(node_data.get("type") or "")
        if node_type not in NODE_TYPES:
            problems.append(
                f"Node '{node_id}' has unknown type '{node_type}'. "
                f"Known types: {', '.join(sorted(NODE_TYPES))}"
            )

    for edge in edges:
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        if source not in seen:
            problems.append(f"Edge points from '{source}', which is not a node.")
        if target not in seen:
            problems.append(f"Edge points to '{target}', which is not a node.")

    # Cycle detection, naming the cycle rather than just reporting one exists.
    adjacency: dict[str, list[str]] = {n: [] for n in seen}
    for edge in edges:
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        if source in adjacency and target in seen:
            adjacency[source].append(target)

    WHITE, GREY, BLACK = 0, 1, 2
    colour = dict.fromkeys(seen, WHITE)
    stack: list[str] = []

    def visit(node_id: str) -> None:
        colour[node_id] = GREY
        stack.append(node_id)
        for nxt in adjacency.get(node_id, []):
            if colour[nxt] == GREY:
                cycle = stack[stack.index(nxt):] + [nxt]
                problems.append("Cycle: " + " -> ".join(cycle))
            elif colour[nxt] == WHITE:
                visit(nxt)
        stack.pop()
        colour[node_id] = BLACK

    for node_id in list(seen):
        if colour[node_id] == WHITE:
            visit(node_id)

    for node_data in nodes:
        node_type = str(node_data.get("type") or "")
        spec = NODE_TYPES.get(node_type)
        if spec is None or not spec.inputs:
            continue
        node_id = str(node_data.get("id") or "")
        has_incoming = any(str(e.get("target")) == node_id for e in edges)
        if not has_incoming and node_type not in ("input",):
            problems.append(f"Node '{node_id}' ({node_type}) has no input connected.")

    return problems


async def run_workflow(
    workflow_id: str,
    inputs: dict[str, Any] | None = None,
    *,
    run_id: str = "",
    project_id: str | None = None,
) -> WorkflowRun:
    workflow = await get_workflow(workflow_id)
    if workflow is None:
        raise NotFound(f"No workflow {workflow_id}")

    graph = workflow.graph or {}
    problems = validate_graph(graph)
    if problems:
        raise ValidationFailed("This workflow will not run:\n- " + "\n- ".join(problems))

    run_id = run_id or new_id("wfr")
    ctx = RunContext(
        run_id=run_id,
        workflow_id=workflow.id,
        inputs={**(workflow.inputs or {}), **(inputs or {})},
        project_id=project_id or workflow.project_id,
    )
    _active[run_id] = ctx.cancelled

    nodes = {str(n["id"]): n for n in graph.get("nodes") or []}
    edges = [e for e in (graph.get("edges") or [])]
    incoming: dict[str, list[dict[str, Any]]] = {nid: [] for nid in nodes}
    outgoing: dict[str, list[str]] = {nid: [] for nid in nodes}
    for edge in edges:
        target, source = str(edge.get("target")), str(edge.get("source"))
        if target in incoming:
            incoming[target].append(edge)
        if source in outgoing:
            outgoing[source].append(target)

    async with session_scope() as s:
        s.add(
            WorkflowRun(
                id=run_id, workflow_id=workflow.id, status="running", inputs=ctx.inputs
            )
        )

    results: dict[str, dict[str, Any]] = {}
    state: dict[str, str] = {nid: "pending" for nid in nodes}
    started = time.perf_counter()

    def publish(node_id: str, status: str, **extra: Any) -> None:
        state[node_id] = status
        ctx.node_states[node_id] = {"status": status, **extra}
        bus.publish(
            Topic.WORKFLOW_NODE, run_id=run_id, workflow_id=workflow.id,
            node_id=node_id, status=status, **extra,
        )

    async def execute(node_id: str) -> None:
        node_data = nodes[node_id]
        spec = NODE_TYPES[str(node_data["type"])]
        publish(node_id, "running")
        node_started = time.perf_counter()

        collected: dict[str, Any] = {}
        for edge in incoming[node_id]:
            source = str(edge.get("source"))
            source_port = str(edge.get("sourcePort") or edge.get("source_port") or "out")
            target_port = str(edge.get("targetPort") or edge.get("target_port") or "in")
            collected[target_port] = (results.get(source) or {}).get(source_port)

        try:
            output = await spec.run(dict(node_data.get("config") or {}), collected, ctx)
            results[node_id] = output or {}
            publish(
                node_id, "done",
                duration_ms=int((time.perf_counter() - node_started) * 1000),
                preview=truncate(_text(_first(output or {})), 300),
            )
        except Exception as exc:  # noqa: BLE001 - one node failing is a node state
            log.warning("Workflow node %s failed: %s", node_id, exc)
            results[node_id] = {}
            publish(node_id, "failed", error=str(exc)[:500])
            # Everything downstream of a failure is skipped, not left pending.
            frontier = list(outgoing.get(node_id, []))
            while frontier:
                nxt = frontier.pop()
                if state.get(nxt) == "pending":
                    publish(nxt, "skipped", error=f"upstream '{node_id}' failed")
                    frontier.extend(outgoing.get(nxt, []))

    # Level-wise execution: everything whose inputs are settled runs together.
    guard = 0
    while guard < len(nodes) + 5:
        guard += 1
        if ctx.cancelled.is_set():
            break
        ready = [
            nid
            for nid, node_data in nodes.items()
            if state[nid] == "pending"
            and all(
                state.get(str(e.get("source")), "pending") in ("done", "failed", "skipped")
                for e in incoming[nid]
            )
        ]
        if not ready:
            break
        await asyncio.gather(*(execute(nid) for nid in ready))

    _active.pop(run_id, None)
    duration = int((time.perf_counter() - started) * 1000)
    failed = [nid for nid, st in state.items() if st == "failed"]
    status = (
        "cancelled" if ctx.cancelled.is_set()
        else "failed" if failed and not ctx.outputs
        else "completed"
    )

    async with session_scope() as s:
        row = (
            await s.execute(select(WorkflowRun).where(WorkflowRun.id == run_id))
        ).scalar_one_or_none()
        if row is not None:
            row.status = status
            row.outputs = ctx.outputs
            row.node_states = ctx.node_states
            row.cost_usd = ctx.cost_usd
            row.ended_at = now()
            row.error = f"{len(failed)} node(s) failed: {', '.join(failed)}" if failed else ""
        result_row = row

    log.info("Workflow %s %s in %dms", workflow.name, status, duration)
    assert result_row is not None
    return result_row


def cancel_run(run_id: str) -> bool:
    event = _active.get(run_id)
    if event is None:
        return False
    event.set()
    return True


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


async def list_workflows(project_id: str | None = None) -> list[Workflow]:
    async with session_scope() as s:
        stmt = select(Workflow)
        if project_id:
            stmt = stmt.where(Workflow.project_id == project_id)
        return list((await s.execute(stmt.order_by(Workflow.name))).scalars().all())


async def get_workflow(workflow_id: str) -> Workflow | None:
    async with session_scope() as s:
        return (
            await s.execute(select(Workflow).where(Workflow.id == workflow_id))
        ).scalar_one_or_none()


async def create_workflow(**fields: Any) -> Workflow:
    async with session_scope() as s:
        workflow = Workflow(**{k: v for k, v in fields.items() if hasattr(Workflow, k)})
        s.add(workflow)
        await s.flush()
        return workflow


async def update_workflow(workflow_id: str, **fields: Any) -> Workflow | None:
    async with session_scope() as s:
        workflow = (
            await s.execute(select(Workflow).where(Workflow.id == workflow_id))
        ).scalar_one_or_none()
        if workflow is None:
            return None
        for key, value in fields.items():
            if key not in {"id", "created_at"} and hasattr(workflow, key):
                setattr(workflow, key, value)
        await s.flush()
        return workflow


async def delete_workflow(workflow_id: str) -> bool:
    async with session_scope() as s:
        result = await s.execute(sa_delete(Workflow).where(Workflow.id == workflow_id))
        return bool(result.rowcount)


async def list_runs(workflow_id: str | None = None, limit: int = 50) -> list[WorkflowRun]:
    async with session_scope() as s:
        stmt = select(WorkflowRun)
        if workflow_id:
            stmt = stmt.where(WorkflowRun.workflow_id == workflow_id)
        return list(
            (await s.execute(stmt.order_by(WorkflowRun.started_at.desc()).limit(limit))).scalars().all()
        )


async def get_run(run_id: str) -> WorkflowRun | None:
    async with session_scope() as s:
        return (
            await s.execute(select(WorkflowRun).where(WorkflowRun.id == run_id))
        ).scalar_one_or_none()


def node_catalog() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in NODE_TYPES.values()]


def example_workflows() -> list[dict[str, Any]]:
    """Ready-to-run graphs, so the canvas is never a blank page."""
    return [
        {
            "name": "Research Brief",
            "description": "Search the web on a topic, read the results, write a sourced brief, and remember it.",
            "inputs": {"topic": "the future of local-first software"},
            "graph": {
                "nodes": [
                    {"id": "topic", "type": "input", "position": {"x": 40, "y": 160},
                     "config": {"key": "topic"}},
                    {"id": "research", "type": "agent", "position": {"x": 280, "y": 160},
                     "config": {"agent": "researcher"}},
                    {"id": "brief", "type": "model", "position": {"x": 540, "y": 160},
                     "config": {
                         "model": "balanced",
                         "system": "You write tight executive briefs. Lead with the answer.",
                         "max_tokens": 2000,
                     }},
                    {"id": "remember", "type": "memory_write", "position": {"x": 800, "y": 60},
                     "config": {"kind": "insight", "importance": 0.7}},
                    {"id": "out", "type": "output", "position": {"x": 800, "y": 250},
                     "config": {"key": "brief"}},
                ],
                "edges": [
                    {"source": "topic", "target": "research"},
                    {"source": "research", "target": "brief"},
                    {"source": "brief", "target": "remember"},
                    {"source": "brief", "target": "out"},
                ],
            },
        },
        {
            "name": "Corpus to Ontology",
            "description": "Search your documents on a theme and fold what they contain into the entity graph.",
            "inputs": {"theme": "architecture decisions"},
            "graph": {
                "nodes": [
                    {"id": "theme", "type": "input", "position": {"x": 40, "y": 160},
                     "config": {"key": "theme"}},
                    {"id": "search", "type": "corpus_search", "position": {"x": 280, "y": 160},
                     "config": {"limit": 12}},
                    {"id": "extract", "type": "graph_extract", "position": {"x": 540, "y": 160},
                     "config": {}},
                    {"id": "out", "type": "output", "position": {"x": 800, "y": 160},
                     "config": {"key": "added"}},
                ],
                "edges": [
                    {"source": "theme", "target": "search"},
                    {"source": "search", "target": "extract"},
                    {"source": "extract", "target": "out"},
                ],
            },
        },
        {
            "name": "Concept to Image Set",
            "description": "Turn one concept into four art directions and render each.",
            "inputs": {"concept": "a rain-slick observation deck above a data centre at night"},
            "graph": {
                "nodes": [
                    {"id": "concept", "type": "input", "position": {"x": 40, "y": 160},
                     "config": {"key": "concept"}},
                    {"id": "directions", "type": "model", "position": {"x": 280, "y": 160},
                     "config": {
                         "model": "balanced",
                         "system": "Return exactly four image prompts, one per line, no numbering. "
                                   "Each names subject, composition, lighting, palette and medium.",
                         "max_tokens": 600,
                     }},
                    {"id": "render", "type": "map", "position": {"x": 540, "y": 160},
                     "config": {"concurrency": 2, "node_type": "image",
                                "node_config": {"size": "1024x1024", "n": 1}}},
                    {"id": "out", "type": "output", "position": {"x": 800, "y": 160},
                     "config": {"key": "assets"}},
                ],
                "edges": [
                    {"source": "concept", "target": "directions"},
                    {"source": "directions", "target": "render"},
                    {"source": "render", "target": "out"},
                ],
            },
        },
    ]


async def seed_examples() -> int:
    """Install the example workflows if the user has none."""
    existing = await list_workflows()
    if existing:
        return 0
    for example in example_workflows():
        await create_workflow(**example)
    return len(example_workflows())
