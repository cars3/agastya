# agastya in Gemini CLI (legacy)

Gemini CLI consumer access has transitioned to Antigravity CLI, which reads
the same `~/.gemini` configuration tree — see [antigravity.md](antigravity.md)
for the current setup.

If you're still on Gemini CLI, add to `~/.gemini/settings.json`:

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

Verify with `/mcp` inside a session. For new setups, prefer Antigravity.
