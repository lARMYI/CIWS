"""Running Python.

This runs in a subprocess with a timeout and a working directory pinned to the
workspace. That is isolation, not a sandbox: the code has the same filesystem
and network access as CIWS itself, because a genuinely sandboxed interpreter
needs a container or a WASM runtime, and pretending otherwise would be worse
than saying so.

The subprocess is what makes it survivable -- an infinite loop or a segfault
takes out the child, not the hub. The tool is on by default because data work
is most of what it gets used for; turn it off in Settings -> Security if that
tradeoff is wrong for you.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from ...core import paths
from ...core.util import truncate
from ..registry import Risk, ToolOutput, registry

DEFAULT_TIMEOUT = 60


@registry.tool(
    "python",
    category="compute",
    risk=Risk.DANGEROUS,
    timeout_s=180,
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Python source. print() what you want to see -- the last "
                    "expression is not echoed automatically."
                ),
            },
            "timeout": {"type": "integer", "description": "Seconds, default 60.", "default": 60},
        },
        "required": ["code"],
    },
)
async def python(code: str, timeout: int = DEFAULT_TIMEOUT, ctx: Any = None) -> ToolOutput:
    """Run Python in a subprocess and return its output.

    Use for calculation, data wrangling, parsing, plotting, and anything easier
    to compute than to reason about. The working directory is the CIWS
    workspace, so relative paths land somewhere you can read back afterwards.

    This is NOT sandboxed -- it can read and write files and reach the network.
    Do not run code you would not run yourself.
    """
    if not code.strip():
        return ToolOutput.error("No code given")

    timeout = max(1, min(int(timeout), 600))
    workspace = paths.workspace_dir()

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",  # isolated: ignore PYTHON* env vars and the user site directory
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
    except OSError as exc:
        return ToolOutput.error(f"Could not start Python: {exc}")

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return ToolOutput.error(f"Timed out after {timeout}s. The process was killed.")

    out = stdout.decode("utf-8", "replace").strip()
    err = stderr.decode("utf-8", "replace").strip()

    if process.returncode == 0:
        body = out or "(ran successfully, but printed nothing)"
        if err:
            body += f"\n\n[stderr]\n{truncate(err, 4000)}"
        return ToolOutput.text(body, exit_code=0)

    return ToolOutput.error(
        f"Exited with code {process.returncode}.\n\n"
        f"{truncate(err, 6000) or truncate(out, 2000) or '(no output)'}"
    )
