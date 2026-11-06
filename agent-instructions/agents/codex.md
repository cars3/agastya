# agastya in Codex

## Register

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.agastya]
command = "/absolute/path/to/agastya/.venv/bin/agastya"
args = ["--mcp"]
```

If your Codex build ships MCP management commands, the CLI route works too
(check `codex mcp --help` for your version):

```bash
codex mcp add agastya -- /absolute/path/to/agastya/.venv/bin/agastya --mcp
```

## Use

Slash commands aren't guaranteed in Codex — phrase it in a prompt:

```text
Use agastya to index the PDFs in /data/manuals and find the section on X.
```

Headless workers use `codex exec`:

```bash
codex exec "Use agastya to query the active index for the cache replacement policy"
```

Working inside the agastya checkout, point Codex at
`agent-instructions/AGENTS.md` — that points at the agastya skill.

## Verify

Start a session and ask: "Which agastya tools do you have?" — expect
`config`, `query`, and `decide`.
