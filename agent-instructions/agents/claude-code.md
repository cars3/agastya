# agastya in Claude Code

## Register

Quickest — the CLI:

```bash
claude mcp add agastya -- /absolute/path/to/agastya/.venv/bin/agastya --mcp
```

Or add it to `.mcp.json` at your project root (alongside any servers already
listed — don't replace them):

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

## Use

Ask naturally: "Use agastya to index the PDFs in /data/manuals, then find the
section on cache set associativity." Claude Code calls `config` once, then
`query` on later questions.

Working inside the agastya checkout? The bundled skill at
`agent-instructions/SKILL.md` teaches the retrieval workflow; ask Claude to
read it.

## Verify

```bash
claude mcp list     # agastya should be listed and connected
```

Then ask: "Which agastya tools do you have?" — expect `config`, `query`, and
`decide`.

## Notes

- The command must be an absolute path; Claude Code does not inherit your
  shell's PATH for MCP servers.
- The first `config` call embeds the whole corpus and can take minutes.
