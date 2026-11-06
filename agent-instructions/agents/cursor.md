# agastya in Cursor

## Register

Add to `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

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

Cursor → Settings → MCP should list agastya as connected. Then ask the Agent:
"Use agastya to index /data/manuals and find the section on X."

## Notes

- Cursor prompts for approval on tool calls by default.
- The bundled skill lives at `agent-instructions/SKILL.md`; reference it
  from a Cursor rule if you want it loaded automatically.
