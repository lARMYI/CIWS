"""The CLI entry point and the run/conversation surface.

The CLI is the first thing a new user touches -- `scripts/start.sh` is a wrapper
around it -- so its flags are part of the public interface. `--token` in
particular is the documented way to recover a token for a direct API call, and
it must print the token and exit without starting a server.

The run routes are what the Command panel reads to draw a trace after the fact,
which is the difference between an agent that can explain itself and one that
cannot.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def restore_settings():
    """The CLI writes settings to disk; without this they leak into later tests.

    Flags like --no-token and --port are persisted deliberately (so a restart
    keeps them), which makes an unrestored CLI test a source of failures a long
    way from its own file.
    """
    from ciws.core import config, paths

    config_file = paths.config_file()
    before = config_file.read_text("utf-8") if config_file.exists() else None
    home_before = os.environ.get("CIWS_HOME")

    yield

    if before is None:
        config_file.unlink(missing_ok=True)
    else:
        config_file.write_text(before, "utf-8")
    if home_before is not None:
        os.environ["CIWS_HOME"] = home_before
    # get_settings() memoises into a module global; drop it so the restored
    # file is what the next test reads.
    config._settings = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch, argv: list[str], serve=None):
    """Invoke main() with uvicorn stubbed out, capturing what it would serve."""
    import ciws.__main__ as cli

    captured: dict = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(sys, "argv", ["ciws", *argv])
    monkeypatch.setattr("uvicorn.run", serve or fake_run, raising=False)
    code = cli.main()
    return code, captured


def test_token_flag_prints_the_token_and_exits(monkeypatch, capsys):
    from ciws.api.deps import get_token

    expected = get_token()
    code, captured = _run_cli(monkeypatch, ["--token"])

    assert code == 0
    assert "app" not in captured, "--token started a server instead of exiting"
    assert expected in capsys.readouterr().out


def test_host_and_port_flags_reach_the_server(monkeypatch):
    code, captured = _run_cli(monkeypatch, ["--host", "0.0.0.0", "--port", "9123"])
    assert code == 0
    assert captured.get("host") == "0.0.0.0"
    assert captured.get("port") == 9123


def test_no_token_flag_turns_the_requirement_off(monkeypatch):
    from ciws.core.config import get_settings

    _run_cli(monkeypatch, ["--no-token", "--port", "9124"])
    assert get_settings().security.require_token is False
    # restore_settings puts the file back; every other test assumes the
    # secure default.


def test_home_flag_repoints_the_workspace(monkeypatch, tmp_path: Path):
    import os

    target = tmp_path / "elsewhere"
    _run_cli(monkeypatch, ["--home", str(target), "--port", "9125"])
    assert os.environ["CIWS_HOME"] == str(target)


def test_log_level_flag_is_applied(monkeypatch):
    from ciws.core.config import get_settings

    _run_cli(monkeypatch, ["--log-level", "WARNING", "--port", "9126"])
    assert get_settings().log_level == "WARNING"


def test_an_unknown_flag_exits_rather_than_starting(monkeypatch):
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, ["--not-a-real-flag"])


# ---------------------------------------------------------------------------
# Conversations and runs
# ---------------------------------------------------------------------------


async def test_conversation_crud(api):
    created = await api.post("/api/conversations", json={"title": "Helios planning"})
    assert created.status_code == 200
    conversation = created.json()
    assert conversation["id"].startswith("cnv_")

    listed = await api.get("/api/conversations")
    assert any(c["id"] == conversation["id"] for c in listed.json()["conversations"])

    fetched = await api.get(f"/api/conversations/{conversation['id']}")
    assert fetched.status_code == 200
    assert "messages" in fetched.json()

    renamed = await api.patch(
        f"/api/conversations/{conversation['id']}", json={"title": "Renamed"}
    )
    assert renamed.json()["title"] == "Renamed"

    removed = await api.delete(f"/api/conversations/{conversation['id']}")
    assert removed.status_code == 200


async def test_fetching_a_missing_conversation_is_a_typed_404(api):
    response = await api.get("/api/conversations/cnv_nope")
    assert response.status_code == 404
    assert "message" in response.json()


async def test_runs_list_is_empty_but_well_shaped(api):
    response = await api.get("/api/runs")
    assert response.status_code == 200
    assert response.json()["runs"] == []


async def test_a_missing_run_trace_is_a_typed_404(api):
    response = await api.get("/api/runs/run_nope")
    assert response.status_code == 404


async def test_a_completed_run_exposes_its_full_trace(api):
    """The trace is the provenance record the Command panel draws."""
    from ciws.agents import presets, runtime
    from ciws.gateway.registry import gateway
    from ciws.gateway.types import ModelInfo

    from tests.test_agent_loop import ScriptedProvider

    await presets.seed_presets()
    provider = ScriptedProvider([("All done.", [])])
    gateway.providers()["test"] = provider
    gateway._models = {
        "test/scripted": ModelInfo(
            id="test/scripted", provider="test", name="scripted", context_window=128_000
        )
    }
    gateway._models_loaded_at = 1e12

    result = await runtime.run_agent(
        agent_slug="analyst", prompt="say something", model_override="test/scripted"
    )
    assert result.ok, result.error

    trace = await api.get(f"/api/runs/{result.run_id}")
    assert trace.status_code == 200
    body = trace.json()
    assert body["status"] == "completed"
    assert "steps" in body and "tool_calls" in body

    listed = await api.get("/api/runs")
    assert any(r["id"] == result.run_id for r in listed.json()["runs"])


async def test_cancelling_an_unknown_run_does_not_500(api):
    response = await api.post("/api/chat/cancel/run_nope")
    assert response.status_code in (200, 404)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


async def test_task_routes_round_trip(api):
    created = await api.post("/api/tasks", json={"title": "Port the fixture recorder"})
    assert created.status_code == 200
    task = created.json()

    listed = await api.get("/api/tasks")
    assert any(t["id"] == task["id"] for t in listed.json()["tasks"])

    patched = await api.patch(f"/api/tasks/{task['id']}", json={"status": "done"})
    assert patched.json()["status"] == "done"

    removed = await api.delete(f"/api/tasks/{task['id']}")
    assert removed.status_code == 200


# ---------------------------------------------------------------------------
# Projects, credentials, system
# ---------------------------------------------------------------------------


async def test_project_routes_round_trip(api):
    created = await api.post("/api/projects", json={"name": "Helios"})
    assert created.status_code == 200
    project = created.json()

    listed = await api.get("/api/projects")
    assert any(p["id"] == project["id"] for p in listed.json()["projects"])

    patched = await api.patch(f"/api/projects/{project['id']}", json={"name": "Helios II"})
    assert patched.json()["name"] == "Helios II"

    assert (await api.delete(f"/api/projects/{project['id']}")).status_code == 200


async def test_credentials_route_never_returns_a_secret(api):
    from ciws.core import secrets

    secrets.put("openai", "sk-must-not-appear-in-the-api")
    response = await api.get("/api/credentials")
    assert response.status_code == 200
    assert "sk-must-not-appear-in-the-api" not in response.text
    secrets.delete("openai")


async def test_system_stats_and_token_routes(api):
    assert (await api.get("/api/system/stats")).status_code == 200
    assert (await api.get("/api/system/logs")).status_code == 200

    token = await api.get("/api/system/token")
    assert token.status_code == 200
    assert token.json()["token"]


async def test_vacuum_compacts_without_losing_data(api):
    await api.post("/api/memory", json={"content": "survives a vacuum"})
    assert (await api.post("/api/system/vacuum")).status_code == 200

    listed = await api.get("/api/memory")
    assert any("survives a vacuum" in m["content"] for m in listed.json()["memories"])
