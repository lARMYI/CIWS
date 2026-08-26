"""Shell access -- off unless you turn it on.

The registry refuses this tool entirely while ``security.allow_shell_tool`` is
false, which is the default. When enabled, commands run through the platform
shell with a timeout, in the workspace directory.

There is a refusal list below for the handful of commands that are
catastrophic and never plausibly intended (``rm -rf /``, ``mkfs``, fork bombs).
It is a guard against a model doing something stupid, not a security boundary:
a shell is a shell, and anything that reaches it can do what you can do. The
real control is the setting.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from typing import Any

from ...core import paths, secrets
from ...core.util import truncate
from ..registry import Risk, ToolOutput, registry

#: Patterns refused outright. Catastrophic and never a real request.
REFUSE = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rR][a-zA-Z]*f?[a-zA-Z]*\s+/(\s|$)"), "recursive delete of /"),
    (re.compile(r"\bmkfs(\.|\s)"), "filesystem format"),
    (re.compile(r"\bdd\s+.*of=/dev/(sd|nvme|disk)"), "raw write to a block device"),
    (re.compile(r":\(\)\s*\{.*\|.*&.*\}\s*;?\s*:"), "fork bomb"),
    (re.compile(r">\s*/dev/(sd|nvme|disk)"), "raw write to a block device"),
    (re.compile(r"\bchmod\s+(-[a-zA-Z]+\s+)*777\s+/(\s|$)"), "chmod 777 on /"),
    (re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b"), "shutting down the machine"),
]


def _refusal(command: str) -> str:
    for pattern, reason in REFUSE:
        if pattern.search(command):
            return reason
    return ""


@registry.tool(
    "shell",
    category="compute",
    risk=Risk.DANGEROUS,
    timeout_s=300,
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command line to run."},
            "cwd": {"type": "string", "description": "Working directory. Defaults to the workspace."},
            "timeout": {"type": "integer", "description": "Seconds, default 90.", "default": 90},
        },
        "required": ["command"],
    },
)
async def shell(command: str, cwd: str = "", timeout: int = 90, ctx: Any = None) -> ToolOutput:
    """Run a shell command.

    Disabled by default -- if this returns a policy error, the user has to
    enable it in Settings -> Security. Output is truncated, so pipe through
    head or grep rather than dumping large files.
    """
    command = command.strip()
    if not command:
        return ToolOutput.error("No command given")

    reason = _refusal(command)
    if reason:
        return ToolOutput.error(
            f"Refused: that command looks like {reason}. "
            f"If you genuinely need it, run it yourself in a terminal."
        )

    workdir = cwd or str(paths.workspace_dir())
    timeout = max(1, min(int(timeout), 600))

    if os.name == "nt":
        shell_exe = os.environ.get("COMSPEC", "cmd.exe")
        args = [shell_exe, "/c", command]
    else:
        shell_exe = shutil.which("bash") or "/bin/sh"
        args = [shell_exe, "-lc", command]

    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
    except OSError as exc:
        return ToolOutput.error(f"Could not start a shell: {exc}")

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return ToolOutput.error(f"Timed out after {timeout}s and was killed.")

    out = secrets.redact(stdout.decode("utf-8", "replace").strip())
    err = secrets.redact(stderr.decode("utf-8", "replace").strip())
    body = out or "(no stdout)"
    if err:
        body += f"\n\n[stderr]\n{truncate(err, 4000)}"

    if process.returncode == 0:
        return ToolOutput.text(body, exit_code=0, cwd=workdir)
    return ToolOutput.error(f"Exit code {process.returncode}\n\n{truncate(body, 8000)}")
