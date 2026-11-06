# agastya in Qwen Code

## Register

Add to `~/.qwen/settings.json` (user scope) or `.qwen/settings.json`
(project scope):

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

`qwen mcp add` manages the same `mcpServers` entries if you prefer the CLI —
run `qwen mcp add --help` for your version's flags.

## Verify

Inside a session, run `/mcp` — agastya should show as CONNECTED with three
tools.

## Notes

- Qwen Code reads MCP servers from `settings.json` — user scope
  (`~/.qwen/`) applies to all projects; project scope (`.qwen/`) ships with
  a repo.
- Point the command at the installed console script, not at `main.py`.
