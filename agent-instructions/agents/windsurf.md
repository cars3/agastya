# agastya in Windsurf

## Register

Edit `~/.codeium/windsurf/mcp_config.json` (create it if missing):

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

## Verify

Windsurf → Cascade → MCP servers: agastya should be listed and running.
Then ask Cascade: "Use agastya to index /data/manuals and find the section
on X."

## Notes

- The command must be absolute; Windsurf does not resolve a virtualenv's
  console script from your shell PATH.
- The first `config` call is slow because it embeds the whole corpus.
