# agastya in Pi

## Register

```bash
pi mcp add agastya -- /absolute/path/to/agastya/.venv/bin/agastya --mcp
```

This writes the global server list at `~/.pi/agent/mcp.json`.

## Verify

```bash
pi mcp list      # agastya should appear with 3 tools
```

## Skills

Working inside the agastya checkout, Pi auto-discovers the skill from
`agent-instructions/.agents/skills/agastya/` and exposes it as
`/skill:agastya`.
