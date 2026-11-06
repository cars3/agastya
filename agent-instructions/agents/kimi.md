# agastya in Kimi CLI

## Register

Add to `~/.kimi/mcp.json` (create it if missing; servers added via
`kimi mcp add` are stored here too):

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

`kimi mcp add` is the CLI route — run `kimi mcp add --help` for the exact
form on your version.

## Skills

Kimi merges skills from all brand directories by default
(`merge_all_available_skills: true`), so the bundled skill at
`.kimi/skills/agastya/SKILL.md` is available when working inside the repo.

## Verify

Ask: "List the tools agastya provides" — expect `config`, `query`, and
`decide`.
