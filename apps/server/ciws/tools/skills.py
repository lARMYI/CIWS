"""The skill forge: tools an agent writes for itself.

This is the code-enhancing half of the self-improvement loop. When reflection
(or the agent mid-task) concludes that the missing thing is a *capability*
rather than a rule, the agent can author a new tool here: a ``run()`` function,
a test that proves it works, and a parameter schema for the model to call it
with.

The lifecycle is deliberate:

* **Proposed.** The code parsed, its tests passed in a subprocess, and it is
  stored -- but it does not execute. Nothing self-authored runs from this state.
* **Active.** A human approved it (or the user opted into auto-activation in
  Settings), and it is registered in the tool registry like any built-in.
* **Disabled / rejected.** Kept for the record, never registered.

Execution honesty, same as the python tool: skills run in a subprocess with a
timeout, which is isolation, not a sandbox. They are the python-tool capability
wearing a name, so they are gated on ``security.allow_python_tool`` and carry
``Risk.DANGEROUS``. The code lives in the database, so a workspace backup
carries the agent's acquired skills with everything else.
"""

from __future__ import annotations

import ast
import asyncio
import sys
import time
from typing import Any

from sqlalchemy import select

from ..core import paths
from ..core.config import get_settings
from ..core.errors import NotFound, ValidationFailed
from ..core.events import Topic, bus
from ..core.logging import get_logger
from ..core.util import dumps, slugify, truncate
from ..db.base import session_scope
from ..db.models import AuditEvent, Skill
from .registry import Risk, Tool, ToolOutput, registry

log = get_logger("tools.skills")

MAX_CODE_CHARS = 24_000
MAX_TEST_CHARS = 12_000
TEST_TIMEOUT_S = 45
CALL_TIMEOUT_S = 60

#: Slugs that would collide with the forge's own management tools.
RESERVED_SLUGS = {"forge", "list", "test", "update", "disable", "approve"}

#: Runs inside the subprocess. It reads one JSON payload from stdin:
#: {"mode": "call"|"test", "code": str, "test_code": str, "args": {...}}
_HARNESS = r"""
import asyncio, inspect, json, sys, traceback

def _main():
    payload = json.loads(sys.stdin.read())
    namespace = {"__name__": "__skill__"}
    try:
        exec(compile(payload["code"], "<skill>", "exec"), namespace)
    except BaseException:
        traceback.print_exc()
        return 3
    if payload["mode"] == "test":
        try:
            exec(compile(payload.get("test_code") or "", "<skill_test>", "exec"), namespace)
        except BaseException:
            traceback.print_exc()
            return 1
        sys.stdout.write("ok")
        return 0
    fn = namespace.get("run")
    if not callable(fn):
        sys.stderr.write("The skill does not define run()")
        return 3
    try:
        result = fn(**(payload.get("args") or {}))
        if inspect.isawaitable(result):
            result = asyncio.run(result)
    except BaseException:
        traceback.print_exc()
        return 1
    sys.stdout.write(result if isinstance(result, str) else json.dumps(result, default=str))
    return 0

sys.exit(_main())
"""


# ---------------------------------------------------------------------------
# Validation and execution
# ---------------------------------------------------------------------------


def validate_skill(code: str, test_code: str, parameters: dict[str, Any]) -> str:
    """Static checks. Returns "" when clean, otherwise what is wrong."""
    if not code.strip():
        return "No code given."
    if len(code) > MAX_CODE_CHARS:
        return f"Code is over {MAX_CODE_CHARS} characters. A skill should be one focused tool."
    if len(test_code) > MAX_TEST_CHARS:
        return f"Test code is over {MAX_TEST_CHARS} characters."
    if not test_code.strip():
        return "A skill needs a test. Untested self-written code does not get stored."
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Code does not parse: {exc}"
    try:
        ast.parse(test_code)
    except SyntaxError as exc:
        return f"Test code does not parse: {exc}"
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
        for node in tree.body
    ):
        return "The code must define run() at module level -- that is the tool's entry point."
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        return 'parameters must be a JSON schema object: {"type": "object", "properties": {...}}.'
    return ""


async def _execute(payload: dict[str, Any], timeout: float) -> tuple[int, str, str]:
    """Run the harness in a subprocess. Returns (exit_code, stdout, stderr)."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",  # isolated: ignore PYTHON* env vars and the user site directory
        "-c",
        _HARNESS,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(paths.workspace_dir()),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(dumps(payload).encode()), timeout=timeout
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return 124, "", f"Timed out after {timeout:.0f}s. The process was killed."
    return (
        process.returncode or 0,
        stdout.decode("utf-8", "replace").strip(),
        stderr.decode("utf-8", "replace").strip(),
    )


async def run_tests(code: str, test_code: str) -> dict[str, Any]:
    """Execute the skill's own tests in a subprocess and report the outcome."""
    started = time.perf_counter()
    exit_code, out, err = await _execute(
        {"mode": "test", "code": code, "test_code": test_code}, TEST_TIMEOUT_S
    )
    return {
        "passed": exit_code == 0,
        "exit_code": exit_code,
        "output": truncate(out, 2000),
        "error": truncate(err, 4000),
        "duration_ms": int((time.perf_counter() - started) * 1000),
    }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


async def list_skills(*, status: str | None = None) -> list[Skill]:
    async with session_scope() as s:
        stmt = select(Skill)
        if status:
            stmt = stmt.where(Skill.status == status)
        rows = (await s.execute(stmt.order_by(Skill.created_at))).scalars().all()
        for row in rows:
            s.expunge(row)
        return list(rows)


async def get_skill(slug: str) -> Skill | None:
    async with session_scope() as s:
        row = (await s.execute(select(Skill).where(Skill.slug == slug))).scalar_one_or_none()
        if row is not None:
            s.expunge(row)
        return row


async def propose_skill(
    name: str,
    description: str,
    parameters: dict[str, Any],
    code: str,
    test_code: str,
    *,
    agent_slug: str = "",
    origin_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate, test, and store a skill. Never activates past the consent policy.

    An existing slug is a revision: the version bumps and -- unless the user has
    opted into auto-activation -- an active skill drops back to ``proposed``,
    because new code means the old approval no longer covers what would run.
    """
    cfg = get_settings().improve
    if not get_settings().security.allow_python_tool:
        return {
            "ok": False,
            "error": "Skills execute through the python capability, which is disabled "
            "in Settings -> Security.",
        }

    parameters = parameters or {"type": "object", "properties": {}}
    problem = validate_skill(code, test_code, parameters)
    if problem:
        return {"ok": False, "error": problem}

    slug = slugify(name, max_len=48)
    if slug in RESERVED_SLUGS:
        return {"ok": False, "error": f"'{slug}' is a reserved name. Pick another."}

    report = await run_tests(code, test_code)
    if not report["passed"]:
        return {
            "ok": False,
            "error": "The skill's tests failed, so it was not stored. Fix the code or "
            "the tests and propose it again.",
            "test_report": report,
        }

    auto = cfg.auto_activate_skills
    async with session_scope() as s:
        existing = (
            await s.execute(select(Skill).where(Skill.slug == slug))
        ).scalar_one_or_none()

        if existing is None:
            count = len(
                (await s.execute(select(Skill.id).where(Skill.status != "rejected"))).all()
            )
            if count >= cfg.max_skills:
                return {
                    "ok": False,
                    "error": f"The skill limit ({cfg.max_skills}) is reached. Retire one first.",
                }
            skill = Skill(
                slug=slug,
                name=name.strip() or slug,
                description=description.strip(),
                parameters=parameters,
                code=code,
                test_code=test_code,
                status="active" if auto else "proposed",
                agent_slug=agent_slug,
                origin_run_id=origin_run_id,
                last_test_report=report,
            )
            s.add(skill)
            await s.flush()
        else:
            skill = existing
            skill.name = name.strip() or skill.name
            skill.description = description.strip() or skill.description
            skill.parameters = parameters
            skill.code = code
            skill.test_code = test_code
            skill.version += 1
            skill.last_test_report = report
            # New code means any earlier decision no longer covers what would
            # run -- a revision always faces the consent policy afresh.
            skill.status = "active" if auto else "proposed"

        s.add(
            AuditEvent(
                actor=agent_slug or "agent",
                action="improve:skill",
                target=slug,
                outcome=skill.status,
                detail={"version": skill.version, "run_id": origin_run_id},
            )
        )
        result = {
            "ok": True,
            "skill": _summary(skill),
            "test_report": report,
            "note": (
                f"Registered as tool 'skill_{slug}' and callable now."
                if skill.status == "active"
                else "Tests passed. The skill is stored as PROPOSED and will not execute "
                "until a human activates it in the Improve panel."
            ),
        }

    if result["skill"]["status"] == "active":
        await register_skill(slug)
    else:
        registry.unregister(f"skill_{slug}")
    bus.publish(Topic.IMPROVE_SKILL, **result["skill"])
    return result


async def set_skill_status(slug: str, status: str, *, actor: str = "user") -> dict[str, Any]:
    """Activate, disable or reject a skill. Activation re-runs its tests first."""
    if status not in ("active", "disabled", "rejected", "proposed"):
        raise ValidationFailed(f"'{status}' is not a skill status")
    skill = await get_skill(slug)
    if skill is None:
        raise NotFound(f"No skill '{slug}'")

    report: dict[str, Any] | None = None
    if status == "active":
        # The stored report proved the code once; approval should prove it now.
        report = await run_tests(skill.code, skill.test_code)
        if not report["passed"]:
            return {"ok": False, "error": "Tests failed on activation.", "test_report": report}

    async with session_scope() as s:
        row = (await s.execute(select(Skill).where(Skill.slug == slug))).scalar_one_or_none()
        if row is None:
            raise NotFound(f"No skill '{slug}'")
        row.status = status
        if report is not None:
            row.last_test_report = report
        s.add(
            AuditEvent(
                actor=actor,
                action="improve:skill",
                target=slug,
                outcome=status,
                detail={"version": row.version},
            )
        )
        summary = _summary(row)

    if status == "active":
        await register_skill(slug)
    else:
        registry.unregister(f"skill_{slug}")
    bus.publish(Topic.IMPROVE_SKILL, **summary)
    return {"ok": True, "skill": summary, "test_report": report}


async def test_skill(slug: str) -> dict[str, Any]:
    """Re-run a stored skill's tests and record the fresh report."""
    skill = await get_skill(slug)
    if skill is None:
        raise NotFound(f"No skill '{slug}'")
    report = await run_tests(skill.code, skill.test_code)
    async with session_scope() as s:
        row = (await s.execute(select(Skill).where(Skill.slug == slug))).scalar_one_or_none()
        if row is not None:
            row.last_test_report = report
    return report


def _summary(skill: Skill) -> dict[str, Any]:
    return {
        "slug": skill.slug,
        "name": skill.name,
        "status": skill.status,
        "version": skill.version,
        "description": skill.description,
    }


# ---------------------------------------------------------------------------
# Registration and execution
# ---------------------------------------------------------------------------


def _executor_for(slug: str) -> Any:
    """The tool function for one skill.

    It re-reads the row on every call, so a revision or a status change applies
    immediately rather than after a restart -- and a skill deactivated behind
    the registry's back refuses to run.
    """

    async def executor(ctx: Any = None, **kwargs: Any) -> ToolOutput:
        if not get_settings().security.allow_python_tool:
            return ToolOutput.error(
                "Skills execute through the python capability, which is disabled "
                "in Settings -> Security."
            )
        skill = await get_skill(slug)
        if skill is None or skill.status != "active":
            registry.unregister(f"skill_{slug}")
            return ToolOutput.error(f"Skill '{slug}' is no longer active.")

        exit_code, out, err = await _execute(
            {"mode": "call", "code": skill.code, "args": kwargs}, CALL_TIMEOUT_S
        )

        async with session_scope() as s:
            row = (await s.execute(select(Skill).where(Skill.slug == slug))).scalar_one_or_none()
            if row is not None:
                row.use_count += 1
                if exit_code != 0:
                    row.error_count += 1

        if exit_code != 0:
            return ToolOutput.error(
                f"skill_{slug} exited with code {exit_code}.\n\n"
                f"{truncate(err, 4000) or truncate(out, 2000) or '(no output)'}"
            )
        body = out or "(ran successfully, but returned nothing)"
        if err:
            body += f"\n\n[stderr]\n{truncate(err, 2000)}"
        return ToolOutput.text(body, skill=slug)

    return executor


async def register_skill(slug: str) -> bool:
    """Put an active skill into the tool registry as ``skill_<slug>``."""
    skill = await get_skill(slug)
    if skill is None or skill.status != "active":
        return False
    registry.register(
        Tool(
            name=f"skill_{skill.slug}",
            description=(
                f"{skill.description or skill.name}\n\n"
                f"(Self-authored skill v{skill.version}. Runs in a subprocess with a "
                f"timeout -- isolation, not a sandbox, exactly like the python tool.)"
            ),
            parameters=skill.parameters or {"type": "object", "properties": {}},
            fn=_executor_for(skill.slug),
            category="skills",
            risk=Risk.DANGEROUS,
            source="skill",
            timeout_s=CALL_TIMEOUT_S + 15,
        )
    )
    return True


async def load_active_skills() -> int:
    """Register every active skill at boot. Returns how many."""
    count = 0
    for skill in await list_skills(status="active"):
        try:
            if await register_skill(skill.slug):
                count += 1
        except Exception as exc:  # noqa: BLE001 - one broken skill must not stop boot
            log.warning("Skill %s failed to register: %s", skill.slug, exc)
    if count:
        log.info("Registered %d self-authored skills", count)
    return count
