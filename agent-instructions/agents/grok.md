# agastya in Grok CLI

## Register

Add to `~/.grok/config.toml`:

```toml
[mcp_servers.agastya]
command = "/absolute/path/to/agastya/.venv/bin/agastya"
args = ["--mcp"]
```

Or via the CLI:

```bash
grok mcp add agastya -- /absolute/path/to/agastya/.venv/bin/agastya --mcp
```

Project-scoped servers live in `.grok/config.toml` — add `--scope project`
to `grok mcp add`.

## Verify

```bash
grok mcp list
grok mcp doctor agastya     # connectivity check
```

## Notes

- Grok also loads MCP configs from `~/.claude.json`, `.cursor/mcp.json`,
  and project `.mcp.json` files (compatibility), merged below its own
  `config.toml`.
- Working inside the agastya checkout? The bundled skill is at
  `agent-instructions/SKILL.md`.
