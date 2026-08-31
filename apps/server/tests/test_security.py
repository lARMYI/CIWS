"""Security controls, tested as controls rather than as intentions.

Three things are asserted here, and each one is a claim the README makes to the
user:

* the file deny-list refuses credential paths regardless of configuration
* the token is actually required, through every accepted channel
* the vault encrypts at rest, prefers the environment, and keeps keys out of logs

A security control with no test is a comment. These are the tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ciws.core import paths, secrets
from ciws.core.config import get_settings
from ciws.core.errors import ToolError
from ciws.tools.builtin import file_tools


# ---------------------------------------------------------------------------
# The file deny-list
# ---------------------------------------------------------------------------


DENIED = [
    "~/.ssh/id_rsa",
    "~/.aws/credentials",
    "~/.netrc",
    "~/.gnupg/secring.gpg",
    "~/.config/gh/hosts.yml",
    "/etc/shadow",
    "id_ed25519",
]


@pytest.mark.parametrize("candidate", DENIED)
async def test_denied_paths_are_refused_for_read(candidate: str):
    with pytest.raises(ToolError) as caught:
        await file_tools.file_read(candidate)
    assert "credential" in str(caught.value).lower()


@pytest.mark.parametrize("candidate", DENIED)
async def test_denied_paths_are_refused_for_write(candidate: str):
    with pytest.raises(ToolError):
        await file_tools.file_write(candidate, "pwned")


async def test_the_vault_itself_is_refused():
    """vault.key and vault.enc are in the deny-list; the agent cannot read its own keys."""
    for name in ("vault.key", "vault.enc"):
        with pytest.raises(ToolError):
            await file_tools.file_read(str(paths.home() / name))


async def test_a_symlink_cannot_launder_a_denied_path(tmp_path: Path):
    """Resolution happens before the check, so a friendly-looking link still fails.

    This is the interesting case: if the deny-list were applied to the raw
    string, ``notes.txt -> ~/.ssh/id_rsa`` would sail straight through it.
    """
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY", "utf-8")
    link = tmp_path / "notes.txt"
    link.symlink_to(secret)

    with pytest.raises(ToolError) as caught:
        await file_tools.file_read(str(link))
    assert "credential" in str(caught.value).lower()


async def test_directory_traversal_still_resolves_before_the_check(tmp_path: Path):
    workspace = paths.workspace_dir()
    sneaky = f"{workspace}/../../../../etc/shadow"
    with pytest.raises(ToolError):
        await file_tools.file_read(sneaky)


async def test_writes_outside_the_workspace_are_refused_by_default(tmp_path: Path):
    assert get_settings().security.approve_writes_outside_workspace is True
    target = tmp_path / "outside.txt"
    with pytest.raises(ToolError) as caught:
        await file_tools.file_write(str(target), "nope")
    assert "workspace" in str(caught.value).lower()
    assert not target.exists(), "the file was written despite the refusal"


async def test_writes_inside_the_workspace_are_allowed():
    target = paths.workspace_dir() / "allowed.txt"
    result = await file_tools.file_write(str(target), "fine")
    assert not result.is_error
    assert target.read_text("utf-8") == "fine"
    target.unlink(missing_ok=True)


async def test_an_empty_path_is_rejected_not_defaulted():
    with pytest.raises(ToolError):
        await file_tools.file_read("   ")


# ---------------------------------------------------------------------------
# Token auth
# ---------------------------------------------------------------------------


async def test_protected_route_rejects_no_token(anon):
    response = await anon.get("/api/memory")
    assert response.status_code == 401


async def test_protected_route_rejects_a_wrong_token(anon):
    response = await anon.get("/api/memory", headers={"x-ciws-token": "not-the-token"})
    assert response.status_code == 401


async def test_a_prefix_of_the_real_token_is_still_rejected(anon):
    from ciws.api.deps import get_token

    response = await anon.get("/api/memory", headers={"x-ciws-token": get_token()[:10]})
    assert response.status_code == 401


@pytest.mark.parametrize("channel", ["bearer", "header", "query"])
async def test_every_documented_token_channel_works(anon, channel: str):
    """EventSource cannot set headers, which is why the query param exists."""
    from ciws.api.deps import get_token

    token = get_token()
    if channel == "bearer":
        response = await anon.get("/api/memory", headers={"authorization": f"Bearer {token}"})
    elif channel == "header":
        response = await anon.get("/api/memory", headers={"x-ciws-token": token})
    else:
        response = await anon.get("/api/memory", params={"token": token})
    assert response.status_code == 200


async def test_health_stays_public_while_everything_else_does_not(anon):
    assert (await anon.get("/api/health")).status_code == 200
    assert (await anon.get("/api/system")).status_code == 401


async def test_the_token_file_is_owner_only():
    from ciws.api.deps import get_token

    get_token()
    token_file = paths.home() / "token"
    assert token_file.exists()
    if os.name != "nt":
        assert token_file.stat().st_mode & 0o077 == 0, "token file is readable by other users"


async def test_rotating_the_token_invalidates_the_old_one(anon):
    from ciws.api import deps

    old = deps.get_token()
    new = deps.rotate_token()
    assert new != old

    assert (await anon.get("/api/memory", headers={"x-ciws-token": old})).status_code == 401
    assert (await anon.get("/api/memory", headers={"x-ciws-token": new})).status_code == 200


# ---------------------------------------------------------------------------
# The credential vault
# ---------------------------------------------------------------------------


def test_secrets_round_trip_through_the_vault():
    secrets.put("test_provider", "sk-secret-value")
    assert secrets.get("test_provider") == "sk-secret-value"
    secrets.delete("test_provider")
    assert secrets.get("test_provider") is None


def test_the_vault_file_is_not_plaintext():
    secrets.put("test_provider", "sk-unmistakable-marker")
    vault = paths.home() / "vault.enc"
    assert vault.exists()
    assert b"sk-unmistakable-marker" not in vault.read_bytes(), "the vault stored plaintext"
    secrets.delete("test_provider")


def test_the_environment_beats_the_vault(monkeypatch):
    """A key exported in the shell must win, so `export` is a reliable override."""
    secrets.put("anthropic", "from-vault")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert secrets.get("anthropic") == "from-env"
    assert secrets.source_of("anthropic").startswith("env")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert secrets.get("anthropic") == "from-vault"
    assert secrets.source_of("anthropic") == "vault"
    secrets.delete("anthropic")


def test_status_never_returns_the_secret_itself():
    secrets.put("openai", "sk-should-never-appear")
    rows = secrets.status()
    blob = repr(rows)
    assert "sk-should-never-appear" not in blob
    assert any(r.get("name") == "openai" for r in rows)
    secrets.delete("openai")


def test_redact_scrubs_known_secrets_from_text():
    secrets.put("groq", "gsk-abcdefghijklmnop")
    line = "calling groq with key gsk-abcdefghijklmnop now"
    assert "gsk-abcdefghijklmnop" not in secrets.redact(line)
    secrets.delete("groq")


def test_mask_keeps_enough_to_recognise_but_not_to_use():
    masked = secrets.mask("sk-ant-1234567890abcdef")
    assert "1234567890" not in masked
    assert masked != "sk-ant-1234567890abcdef"


# ---------------------------------------------------------------------------
# Tool policy
# ---------------------------------------------------------------------------


async def test_shell_is_off_by_default(tools_loaded):
    from ciws.tools.registry import registry

    assert get_settings().security.allow_shell_tool is False
    names = {spec.name for spec in registry.specs(["*"])}
    assert "shell" not in names, "the shell tool was offered to an agent by default"


async def test_a_disabled_tool_reports_rather_than_raises(tools_loaded):
    from ciws.tools.registry import ToolContext, registry

    result = await registry.call("shell", {"command": "echo hi"}, ToolContext())
    assert result.is_error
    assert "disabled" in result.content.lower()
