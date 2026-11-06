# Connecting agastya to your agent

agastya is a standard stdio MCP server. It runs from a local checkout, so the
command is the absolute path to the installed console script, not `uvx`.

Install it first:

```bash
cd /path/to/agastya
python3 -m venv .venv
.venv/bin/python -m pip install .
```

Then pick your agent's guide below — each has an exact config location, a
paste-ready snippet, and a verification step.

| Agent | Guide |
|---|---|
| Claude Code | [claude-code.md](claude-code.md) |
| Claude Desktop | [claude-desktop.md](claude-desktop.md) |
| Codex CLI | [codex.md](codex.md) |
| Antigravity (CLI / IDE / 2.0) | [antigravity.md](antigravity.md) |
| Cursor | [cursor.md](cursor.md) |
| OpenCode | [opencode.md](opencode.md) |
| Windsurf | [windsurf.md](windsurf.md) |
| Qwen Code | [qwen.md](qwen.md) |
| Kimi CLI | [kimi.md](kimi.md) |
| Grok CLI | [grok.md](grok.md) |
| GitHub Copilot CLI | [copilot-cli.md](copilot-cli.md) |
| Pi | [pi.md](pi.md) |
| Hermes Agent | [hermes.md](hermes.md) |
| Gemini CLI (legacy) | [gemini-cli.md](gemini-cli.md) |
| Any other MCP client | [generic.md](generic.md) |

## The universal settings

Every guide below reduces to the same three values:

- command: `/absolute/path/to/agastya/.venv/bin/agastya`
- args: `--mcp`
- transport: stdio

On Windows use `.venv\Scripts\agastya.exe` and the same `--mcp` argument.

If your agent isn't listed, use [generic.md](generic.md) — the values above
work with any MCP client once you know its config file format.

## No keys required for retrieval

PDF retrieval needs no API keys. The optional cloud decision backend needs
`TYPESAFE_API_KEY` in the server's environment; the optional local decision
backend needs a compatible Unsloth checkpoint. Neither is required for
`config` or `query`.
