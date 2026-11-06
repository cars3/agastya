# agastya in OpenCode

## Register

OpenCode uses the `mcp` key (not `mcpServers`), and every entry needs a
`type`. Add to `opencode.json` at your project root, or to
`~/.config/opencode/opencode.json` for all projects:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "agastya": {
      "type": "local",
      "command": ["/absolute/path/to/agastya/.venv/bin/agastya", "--mcp"],
      "enabled": true
    }
  }
}
```

Working inside the agastya checkout, the skill lives at
`agent-instructions/SKILL.md`; OpenCode can load it from there when asked.

## Verify

```bash
opencode mcp list     # agastya should appear, connected
```

Then ask: "Use agastya to find the section on X in /data/manuals."

## Notes

Newer OpenCode builds may nest servers as `mcp.servers.<name>` — if your
config validator rejects the form above, check `opencode.ai/docs` for your
version's shape.
