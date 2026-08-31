# ciws (server)

The CIWS backend: model gateway, memory, ontology graph, agent runtime, tools, MCP client,
media studio, ingestion pipeline, workflow engine, and the HTTP/WebSocket API.

Installed as the `ciws` package. Run it with:

```bash
python -m ciws          # or: ciws
python -m ciws --open   # and open the UI in a browser
python -m ciws --help
```

Everything it stores lives under `CIWS_HOME` (default `~/.ciws`).

See the [project README](../../README.md) for setup, and
[docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) for design notes.

## Tests

```bash
python -m pytest tests/ -q
```

33 tests, no API keys and no network required — the agent loop is driven by a scripted provider.
