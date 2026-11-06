# agastya in GitHub Copilot CLI

## Register

Edit `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "agastya": {
      "type": "local",
      "command": "/absolute/path/to/agastya/.venv/bin/agastya",
      "args": ["--mcp"],
      "tools": ["*"]
    }
  }
}
```

Per-repository, Copilot CLI reads `.mcp.json` at the project root or
`.github/mcp.json` (committable). You can also manage servers in-session
with `/mcp add`.

## Verify

Run `/mcp` in a session (or the `copilot mcp` subcommands) and confirm
agastya is enabled; then ask it to index a small PDF folder.

## Notes

- Project-level servers load only after you confirm folder trust on first
  launch.
- The virtualenv must already exist; Copilot does not run the install for
  you.
