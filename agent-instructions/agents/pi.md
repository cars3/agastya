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

The bundled skill lives at `agent-instructions/SKILL.md`; Pi can load it
directly when working inside the checkout.
