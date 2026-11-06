# agastya in Antigravity (CLI / IDE / 2.0)

Antigravity tools share one MCP configuration, so registering agastya once
makes it available across the suite.

## Global (all workspaces)

Edit `~/.gemini/config/mcp_config.json`:

```json
{
  "mcpServers": {
    "agastya": {
      "command": "/absolute/path/to/agastya/.venv/bin/agastya",
      "args": ["--mcp"]
    }
  }
}
```

## Per project

Same JSON in `.agents/mcp_config.json` at the project root.

## Newer CLI shortcut

Antigravity CLI 1.1.16+ has an `mcp` subcommand:

```bash
agy mcp add agastya -- /absolute/path/to/agastya/.venv/bin/agastya --mcp
```

## Skills

The bundled skill is `agent-instructions/SKILL.md`; point Antigravity at it
if you want the retrieval workflow loaded.

## Verify

Open the MCP manager (or the CLI's MCP list) and confirm agastya shows as
connected with three tools, then ask it to index a small PDF folder.

## Notes

- Antigravity's config format also supports remote servers via `serverUrl`,
  but agastya runs locally over stdio; there is no HTTP transport.
- The config lives under `~/.gemini/` (shared with older Gemini tooling);
  skills may also be placed globally at `~/.gemini/skills/`.
