"""The agent loop.

Reason, act, observe, repeat, until the model stops asking for tools or a limit
stops it. Every step is streamed onto the event bus and written to ``run_steps``
and ``tool_calls``, so a finished run can be replayed rather than merely
summarised -- which is what makes the trace panel useful rather than decorative.

Three things this loop takes seriously:

* **Tools run in parallel.** A model that asks for four searches at once should
  get four searches at once, and all four results must return in a single tool
  turn -- splitting them teaches the model to stop batching.
* **Context is finite.** Before each model call the transcript is measured, and
  the middle is compacted when it will not fit. Dropping the oldest messages
  loses the goal; dropping the newest loses the thread. The middle is where the
  redundancy is.
* **Cancellation is immediate.** A cancel event is checked between steps and
  passed into every tool, so stopping a runaway run does not mean waiting for
  its next 60-second call to finish.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from ..core.config import get_settings
from ..core.errors import NotFound
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..core.util import estimate_tokens, new_id, now, truncate
from ..db.base import session_scope
from ..db.models import AgentDef, Run, RunStep
from ..gateway import catalog
from ..gateway.registry import gateway
from ..gateway.types import (
    ChatMessage,
    ChatRequest,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
    Usage,
)
from ..tools.registry import ToolContext, ToolOutput, registry
from . import presets

log = get_logger("agents")

#: Fraction of the context window the transcript may occupy before compaction.
CONTEXT_BUDGET = 0.65
#: Runs currently in flight, so the API can cancel them.
_active: dict[str, asyncio.Event] = {}


@dataclass
class AgentResult:
    run_id: str
    content: str = ""
    thinking: str = ""
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    error: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[ChatMessage] = field(default_factory=list)
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return not self.error


def cancel(run_id: str) -> bool:
    """Signal a running agent to stop at its next checkpoint."""
    event = _active.get(run_id)
    if event is None:
        return False
    event.set()
    bus.publish(Topic.RUN_CANCEL, run_id=run_id)
    return True


def active_runs() -> list[str]:
    return sorted(_active)


def _fit_context(messages: list[ChatMessage], model_id: str, system_chars: int) -> list[ChatMessage]:
    """Compact the middle of a transcript that will not fit.

    The first user turn holds the goal and the last few turns hold the thread;
    everything between them is where a long tool loop accumulates redundancy.
    """
    info = catalog.lookup(model_id)
    window = (info.context_window if info and info.context_window else 128_000)
    budget = int(window * CONTEXT_BUDGET) - estimate_tokens(" " * system_chars)
    if budget <= 0:
        budget = window // 2

    def size(items: list[ChatMessage]) -> int:
        total = 0
        for m in items:
            total += estimate_tokens(m.text()) + estimate_tokens(m.thinking)
            for r in m.tool_results:
                total += estimate_tokens(r.content)
            for c in m.tool_calls:
                total += estimate_tokens(str(c.arguments))
        return total

    if size(messages) <= budget or len(messages) <= 6:
        return messages

    head = messages[:2]
    tail = messages[-6:]
    middle = messages[2:-6]
    if not middle:
        return messages

    dropped = 0
    kept: list[ChatMessage] = []
    # Walk the middle backwards, keeping the most recent of it that fits.
    for message in reversed(middle):
        if size(head + [message, *kept] + tail) > budget:
            dropped += 1
            continue
        kept.insert(0, message)

    if dropped:
        note = ChatMessage(
            role="user",
            content=(
                f"[{dropped} earlier steps were compacted out of this transcript to fit the "
                f"context window. Their results are already reflected in what follows. If you "
                f"need something from them, use your tools to look it up again rather than "
                f"guessing.]"
            ),
        )
        log.info("Compacted %d messages out of the transcript", dropped)
        return head + [note, *kept] + tail
    return head + kept + tail


def _tool_results_message(results: list[ToolResult]) -> ChatMessage:
    return ChatMessage(role="tool", tool_results=results)


async def _execute_tools(
    calls: list[ToolCall], ctx: ToolContext, parallel: bool
) -> list[ToolResult]:
    """Run a batch of tool calls and return their results in request order."""

    async def one(call: ToolCall) -> ToolResult:
        output: ToolOutput = await registry.call(call.name, call.arguments, ctx, call_id=call.id)
        return ToolResult(
            call_id=call.id,
            name=call.name,
            content=output.content or "(no output)",
            is_error=output.is_error,
            parts=output.parts,
        )

    if parallel and len(calls) > 1:
        return list(await asyncio.gather(*(one(c) for c in calls)))
    return [await one(c) for c in calls]


async def _load_agent(agent_slug: str) -> AgentDef:
    agent = await presets.get_agent(agent_slug)
    if agent is None:
        await presets.seed_presets()
        agent = await presets.get_agent(agent_slug) or await presets.get_agent("analyst")
    if agent is None:
        raise NotFound(f"No agent '{agent_slug}' and no default available")
    return agent


#: Told to the model whenever recalled context is present. Kept short on
#: purpose: a long citation policy competes with the agent's actual task, and
#: the marker format is already visible in the context block itself.
CITATION_RULE = """\
## Citing what you were given

The context above carries a reference in backticks after each item, like
`[mem_1a2b]` for a memory or `[chk_9f8e]` for a passage from the user's own
documents. When a statement of yours rests on one of those items, put its
reference at the end of the sentence.

Cite only references that appear above -- never invent one, and never cite a
claim that came from your own knowledge rather than from the workspace. An
unsupported sentence with no marker is fine and expected; a marker on a
sentence it does not support is not."""


async def _build_system(
    agent: AgentDef, prompt: str, project_id: str | None, extra: str
) -> tuple[str, list[dict[str, Any]]]:
    """Assemble the system prompt: persona, recalled memory, workspace facts.

    Returns the prompt and the descriptors for everything recalled into it, so
    the run can record what its answer was allowed to cite.
    """
    blocks = [agent.system_prompt.strip()]
    sources: list[dict[str, Any]] = []

    cfg = get_settings().memory
    if cfg.enabled and agent.memory_scope != "none":
        try:
            from ..memory import store

            context, sources = await store.build_context_with_sources(
                prompt, limit=cfg.recall_limit, project_id=project_id
            )
            if context:
                blocks.append(context)
                blocks.append(CITATION_RULE)
        except Exception as exc:  # noqa: BLE001 - memory is an enhancement, not a dependency
            log.debug("Memory context skipped: %s", exc)

    if extra:
        blocks.append(extra)

    blocks.append(
        f"Current date: {now().strftime('%Y-%m-%d')} (UTC). "
        f"Use the current_time tool if you need more precision."
    )
    return "\n\n---\n\n".join(b for b in blocks if b), sources


async def stream_agent(
    *,
    agent_slug: str = "analyst",
    prompt: str = "",
    history: list[ChatMessage] | None = None,
    conversation_id: str | None = None,
    project_id: str | None = None,
    model_override: str = "",
    parent_run_id: str | None = None,
    depth: int = 0,
    system_extra: str = "",
    run_id: str = "",
    auto_approve: bool = False,
) -> AsyncIterator[StreamEvent]:
    """Run an agent, yielding stream events as they happen.

    The final event is always DONE or ERROR, so a consumer can rely on the
    stream terminating rather than watching for silence.
    """
    settings = get_settings()
    agent = await _load_agent(agent_slug)
    run_id = run_id or new_id("run")
    model = model_override or agent.model or settings.routing.balanced
    resolved_model = gateway.resolve(model)

    cancelled = asyncio.Event()
    _active[run_id] = cancelled
    started = time.perf_counter()

    messages: list[ChatMessage] = list(history or [])
    if prompt:
        messages.append(ChatMessage.user(prompt))

    system, sources = await _build_system(
        agent, prompt or (messages[-1].text() if messages else ""), project_id, system_extra
    )
    tools = registry.specs(agent.tools or ["*"])

    async with session_scope() as s:
        s.add(
            Run(
                id=run_id,
                conversation_id=conversation_id,
                project_id=project_id,
                agent_id=agent.id,
                agent_slug=agent.slug,
                parent_run_id=parent_run_id,
                goal=truncate(prompt, 2000),
                model=resolved_model,
                status="running",
                # Recorded up front: what the answer was allowed to cite is
                # part of the provenance record, whether or not it cited any.
                meta={"depth": depth, "tool_count": len(tools), "sources": sources},
            )
        )

    bus.publish(
        Topic.RUN_START,
        run_id=run_id,
        agent=agent.slug,
        model=resolved_model,
        conversation_id=conversation_id,
        goal=truncate(prompt, 400),
        tools=len(tools),
        depth=depth,
    )
    yield StreamEvent(type=StreamEventType.START, model=resolved_model, text=run_id)

    ctx = ToolContext(
        run_id=run_id,
        conversation_id=conversation_id,
        project_id=project_id,
        agent_slug=agent.slug,
        cancelled=cancelled,
        auto_approve=auto_approve,
        meta={"depth": depth},
    )

    total = Usage()
    final_text: list[str] = []
    final_thinking: list[str] = []
    tool_log: list[dict[str, Any]] = []
    error = ""
    step = 0
    max_steps = min(agent.max_steps or settings.agents.max_steps, 60)

    try:
        while step < max_steps:
            if cancelled.is_set():
                error = "Cancelled by the user."
                break

            step += 1
            step_id = new_id("stp")
            ctx.step_id = step_id
            bus.publish(Topic.RUN_STEP, run_id=run_id, step=step, max_steps=max_steps)

            request = ChatRequest(
                model=resolved_model,
                system=system,
                messages=_fit_context(messages, resolved_model, len(system)),
                tools=tools,
                temperature=agent.temperature,
                max_tokens=8000,
                stream=True,
                thinking_budget=int(agent.meta.get("thinking_budget", 0) or 0),
            )

            text_parts: list[str] = []
            think_parts: list[str] = []
            calls: list[ToolCall] = []
            step_usage = Usage()
            step_started = time.perf_counter()
            step_error = ""

            async for event in gateway.stream(request):
                if cancelled.is_set():
                    break
                if event.type is StreamEventType.TEXT:
                    text_parts.append(event.text)
                    bus.publish(Topic.RUN_DELTA, run_id=run_id, text=event.text, step=step)
                    yield event
                elif event.type is StreamEventType.THINKING:
                    think_parts.append(event.text)
                    bus.publish(Topic.RUN_THINKING, run_id=run_id, text=event.text, step=step)
                    yield event
                elif event.type is StreamEventType.TOOL_CALL and event.tool_call:
                    calls.append(event.tool_call)
                elif event.type is StreamEventType.USAGE and event.usage:
                    step_usage = event.usage
                elif event.type is StreamEventType.ERROR:
                    step_error = event.error

            content = "".join(text_parts)
            thinking = "".join(think_parts)
            total = total + step_usage
            latency = int((time.perf_counter() - step_started) * 1000)

            async with session_scope() as s:
                s.add(
                    RunStep(
                        id=step_id,
                        run_id=run_id,
                        idx=step,
                        kind="model",
                        content=content,
                        thinking=thinking,
                        tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
                        input_tokens=step_usage.input_tokens,
                        output_tokens=step_usage.output_tokens,
                        latency_ms=latency,
                        error=step_error,
                    )
                )

            if step_error:
                error = step_error
                break

            if content:
                final_text.append(content)
            if thinking:
                final_thinking.append(thinking)

            if not calls:
                break  # the model answered instead of asking for tools

            messages.append(
                ChatMessage(role="assistant", content=content, thinking=thinking, tool_calls=calls)
            )

            batch = calls[: settings.agents.max_tool_calls_per_step]
            results = await _execute_tools(batch, ctx, settings.agents.parallel_tools)
            for call, result in zip(batch, results):
                tool_log.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "ok": not result.is_error,
                        "preview": truncate(result.content, 400),
                    }
                )
            # Any calls beyond the per-step cap still need a result, or the
            # provider rejects the next turn for an unanswered tool_use block.
            for call in calls[len(batch):]:
                results.append(
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        content=(
                            f"Skipped: more than {settings.agents.max_tool_calls_per_step} tool "
                            f"calls in one step. Ask for this again next step if you still need it."
                        ),
                        is_error=True,
                    )
                )
            messages.append(_tool_results_message(results))

            if cancelled.is_set():
                error = "Cancelled by the user."
                break
        else:
            error = error or f"Stopped after {max_steps} steps without a final answer."

    except Exception as exc:  # noqa: BLE001 - a run failure must be recorded, not raised
        log.exception("Run %s failed", run_id)
        error = f"{type(exc).__name__}: {exc}"
    finally:
        _active.pop(run_id, None)

    duration = int((time.perf_counter() - started) * 1000)
    answer = "\n\n".join(t for t in final_text if t.strip())
    status = "cancelled" if cancelled.is_set() else ("failed" if error else "completed")

    async with session_scope() as s:
        run = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
        if run is not None:
            run.status = status
            run.steps = step
            run.input_tokens = total.input_tokens
            run.output_tokens = total.output_tokens
            run.cost_usd = total.cost_usd
            run.result = truncate(answer, 20_000)
            run.error = error
            run.ended_at = now()
            run.duration_ms = duration
            # Which of the offered sources the answer actually leaned on. The
            # interface renders these; a marker the model invented resolves to
            # nothing and is dropped here rather than shown as a real citation.
            meta = dict(run.meta or {})
            meta["citations"] = _resolve_citations(answer, sources)
            run.meta = meta

    bus.publish(
        Topic.RUN_END if not error else Topic.RUN_ERROR,
        run_id=run_id,
        status=status,
        steps=step,
        error=error,
        cost_usd=round(total.cost_usd, 6),
        input_tokens=total.input_tokens,
        output_tokens=total.output_tokens,
        duration_ms=duration,
    )

    if error and not answer:
        yield StreamEvent(type=StreamEventType.ERROR, error=error, model=resolved_model)
    yield StreamEvent(
        type=StreamEventType.DONE,
        finish_reason=status,
        usage=total,
        model=resolved_model,
        text=run_id,
        index=duration,
    )

    # Capture after the answer is out, so accumulation never delays the user.
    # An agent whose memory_scope is "none" is deliberately amnesiac -- the
    # scout, for one -- and must not write to a shared store behind the user's
    # back.
    if (
        get_settings().memory.auto_capture
        and agent.memory_scope != "none"
        and answer
        and not error
        and depth == 0
    ):
        task = asyncio.create_task(
            _capture(prompt, answer, project_id, conversation_id, run_id)
        )
        # Hold a reference. asyncio only keeps a weak one, so an unreferenced
        # task can be garbage collected mid-flight and the capture silently
        # never happens -- which looks exactly like "extraction found nothing".
        _capture_tasks.add(task)
        task.add_done_callback(_capture_tasks.discard)


#: A reference marker as it appears in an answer: [mem_1a2b] or [chk_9f8e].
_CITATION = re.compile(r"\[((?:mem|chk|doc|ent)_[0-9a-f]{4,})\]")


def _resolve_citations(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match the markers in an answer against what was actually offered.

    A model can emit a plausible-looking reference it was never given. Resolving
    against the offered set means an invented marker resolves to nothing and is
    dropped, so the interface never renders a citation that does not lead
    anywhere -- which would be worse than no citation at all.
    """
    if not answer or not sources:
        return []
    by_ref = {str(s.get("ref")): s for s in sources}
    seen: set[str] = set()
    cited: list[dict[str, Any]] = []
    for ref in _CITATION.findall(answer):
        if ref in seen or ref not in by_ref:
            continue
        seen.add(ref)
        cited.append(by_ref[ref])
    return cited


#: Live capture tasks, held so the event loop cannot collect them mid-flight.
_capture_tasks: set[asyncio.Task[None]] = set()


async def _capture(
    prompt: str,
    answer: str,
    project_id: str | None,
    conversation_id: str | None,
    run_id: str | None = None,
) -> None:
    """Turn a finished exchange into memories and entities.

    This is the loop that makes the workspace compound: without it the user has
    to write every memory by hand, and the graph only ever contains what they
    typed into it. Both halves degrade to a no-op when no model is configured,
    so a keyless install is unaffected rather than broken.
    """
    from ..core.config import get_settings as _settings
    from ..memory import store

    cfg = _settings().memory
    transcript = f"USER: {prompt}\n\nASSISTANT: {answer}"
    source_ref = run_id or conversation_id

    created: list[Any] = []
    try:
        created = await store.extract_memories(
            transcript, project_id=project_id, source_ref=source_ref
        )
    except Exception as exc:  # noqa: BLE001 - capture is best-effort by design
        log.debug("Auto memory capture skipped: %s", exc)

    if cfg.extract_entities:
        try:
            from ..ontology import graph

            result = await graph.extract_graph(
                transcript, project_id=project_id, source_ref=source_ref
            )
            if result.get("entities") or result.get("edges"):
                log.info(
                    "Captured %d entities and %d links from run %s",
                    len(result.get("entities") or []),
                    len(result.get("edges") or []),
                    run_id,
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("Auto entity capture skipped: %s", exc)

    if created:
        await _maybe_consolidate(len(created), project_id)


#: Memories written since the last consolidation pass, by project.
_since_consolidation: dict[str, int] = {}


async def _maybe_consolidate(new_memories: int, project_id: str | None) -> None:
    """Run a consolidation pass once enough has accumulated to be worth it.

    Consolidation costs a model call and rewrites the store, so it is not worth
    doing per capture. ``memory.consolidate_after`` is the threshold; counting
    happens per project so a busy project does not drag a quiet one into a
    pass it does not need.
    """
    from ..core.config import get_settings as _settings

    threshold = _settings().memory.consolidate_after
    if threshold <= 0:
        return

    key = project_id or ""
    _since_consolidation[key] = _since_consolidation.get(key, 0) + new_memories
    if _since_consolidation[key] < threshold:
        return

    _since_consolidation[key] = 0
    try:
        from ..memory import store

        report = await store.consolidate(project_id=project_id)
        log.info("Consolidation pass: %s", report)
    except Exception as exc:  # noqa: BLE001
        log.debug("Consolidation skipped: %s", exc)


async def run_agent(
    *,
    agent_slug: str = "analyst",
    prompt: str = "",
    history: list[ChatMessage] | None = None,
    conversation_id: str | None = None,
    project_id: str | None = None,
    model_override: str = "",
    parent_run_id: str | None = None,
    depth: int = 0,
    system_extra: str = "",
    persist_messages: bool = True,
    auto_approve: bool = True,
) -> AgentResult:
    """Run an agent to completion and return the result.

    ``persist_messages`` is False for delegated sub-runs: their transcript
    belongs to the run trace, not to the user's conversation.
    """
    run_id = new_id("run")
    content: list[str] = []
    thinking: list[str] = []
    usage = Usage()
    error = ""
    model = ""
    started = time.perf_counter()

    async for event in stream_agent(
        agent_slug=agent_slug,
        prompt=prompt,
        history=history,
        conversation_id=conversation_id if persist_messages else None,
        project_id=project_id,
        model_override=model_override,
        parent_run_id=parent_run_id,
        depth=depth,
        system_extra=system_extra,
        run_id=run_id,
        auto_approve=auto_approve,
    ):
        if event.type is StreamEventType.TEXT:
            content.append(event.text)
        elif event.type is StreamEventType.THINKING:
            thinking.append(event.text)
        elif event.type is StreamEventType.USAGE and event.usage:
            usage = event.usage
        elif event.type is StreamEventType.ERROR:
            error = event.error
        elif event.type is StreamEventType.DONE:
            model = event.model
            usage = event.usage or usage

    async with session_scope() as s:
        run = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
        steps = run.steps if run else 0
        # The stream only emits an ERROR event when there is no answer at all --
        # a run that hit its step ceiling mid-thought still produced text. The
        # persisted run row is the authority on whether it actually finished.
        if run is not None and run.error and not error:
            error = run.error

    return AgentResult(
        run_id=run_id,
        content="".join(content),
        thinking="".join(thinking),
        steps=steps,
        usage=usage,
        model=model,
        error=error,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


async def get_run(run_id: str) -> dict[str, Any] | None:
    """A run with its full step and tool trace, for the trace panel."""
    from ..db.models import ToolCall as ToolCallRow

    async with session_scope() as s:
        run = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
        if run is None:
            return None
        steps = (
            await s.execute(select(RunStep).where(RunStep.run_id == run_id).order_by(RunStep.idx))
        ).scalars().all()
        calls = (
            await s.execute(
                select(ToolCallRow).where(ToolCallRow.run_id == run_id).order_by(ToolCallRow.created_at)
            )
        ).scalars().all()

    meta = run.meta or {}
    return {
        **run.to_dict(),
        "steps": [s.to_dict() for s in steps],
        "tool_calls": [c.to_dict() for c in calls],
        # Lifted out of meta so the trace panel does not have to know where
        # provenance happens to be stored.
        "citations": meta.get("citations") or [],
        "sources_offered": meta.get("sources") or [],
    }


async def list_runs(
    *, conversation_id: str | None = None, project_id: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    async with session_scope() as s:
        stmt = select(Run)
        if conversation_id:
            stmt = stmt.where(Run.conversation_id == conversation_id)
        if project_id:
            stmt = stmt.where(Run.project_id == project_id)
        rows = (
            await s.execute(stmt.order_by(Run.started_at.desc()).limit(limit))
        ).scalars().all()
    return [r.to_dict() for r in rows]
